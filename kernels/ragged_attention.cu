// Chunked-prefill attention over a GPU-resident KV cache.
//
// The cache keeps keys and values for every layer and request slot in GPU
// memory, laid out like the host KVCache:
//     k, v : (n_layer, max_slots, n_head, n_pos, head_dim), float32
// It is allocated once and never copied back to the host during serving.
//
// One serving step, for one layer, works on a ragged batch: every request
// brings some new tokens (a prefill chunk, or one decode token). The host
// sends only those new tokens' q, k, v, plus two small arrays that say, for
// each new token, which slot it belongs to and its absolute position.
// ragged_cache_step() then
//   1. copies the new q/k/v to the GPU,
//   2. append_kernel: writes each new token's k/v into its slot and position,
//   3. attention_kernel: for each (new token, head) computes
//          softmax(q . K[0..pos] / sqrt(head_dim)) . V[0..pos]
//      over that token's own slot only, so no padding is needed,
//   4. copies the attention output back.
// The NumPy oracle with the same contract is model/ragged_attention.py.
//
// Build: kernels/compile_cuda.sh (produces minitorch/cuda_kernels/ragged_attention.so).

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// Threads per attention block. Must be a power of two for the reductions.
#define ATTN_THREADS 128

// Error codes returned to Python in addition to cudaError_t values (> 0).
#define RAGGED_OK 0
#define RAGGED_BAD_ARGUMENT -1
#define RAGGED_OUT_OF_MEMORY -2

struct RaggedCache {
  int n_layer, max_slots, n_head, n_pos, head_dim;
  float* k;  // device cache, (n_layer, max_slots, n_head, n_pos, head_dim)
  float* v;
  // Device scratch for one step, grown on demand to `capacity` tokens.
  int capacity;
  float* q_new;    // (capacity, n_head, head_dim)
  float* k_new;
  float* v_new;
  float* out;      // (capacity, n_head, head_dim)
  int* token_slot; // (capacity,)
  int* token_pos;  // (capacity,)
};

// Offset of (layer, slot, head, pos, 0) in the cache.
__host__ __device__ inline size_t cache_index(int layer, int slot, int head, int pos,
                                              int max_slots, int n_head, int n_pos,
                                              int head_dim) {
  return ((((size_t)layer * max_slots + slot) * n_head + head) * n_pos + pos) * head_dim;
}

// grid = (n_tokens, n_head); each block copies one token's key and value
// vectors for one head into the cache.
__global__ void append_kernel(float* k_cache, float* v_cache,
                              const float* k_new, const float* v_new,
                              const int* token_slot, const int* token_pos,
                              int layer, int max_slots, int n_head, int n_pos, int head_dim) {
  int t = blockIdx.x;
  int h = blockIdx.y;
  size_t dst = cache_index(layer, token_slot[t], h, token_pos[t], max_slots, n_head, n_pos, head_dim);
  size_t src = ((size_t)t * n_head + h) * head_dim;
  for (int e = threadIdx.x; e < head_dim; e += blockDim.x) {
    k_cache[dst + e] = k_new[src + e];
    v_cache[dst + e] = v_new[src + e];
  }
}

// Combine one value per thread into a block-wide max or sum. Every thread
// gets the result. `scratch` holds ATTN_THREADS floats of shared memory.
__device__ float block_reduce(float value, float* scratch, bool take_max) {
  int tid = threadIdx.x;
  __syncthreads();  // a previous reduction may still be reading scratch
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other = scratch[tid + stride];
      scratch[tid] = take_max ? fmaxf(scratch[tid], other) : scratch[tid] + other;
    }
    __syncthreads();
  }
  float result = scratch[0];
  __syncthreads();
  return result;
}

// grid = (n_tokens, n_head), block = ATTN_THREADS.
// Shared memory: n_pos floats for scores, head_dim for the query,
// ATTN_THREADS for reductions.
//
// Token t sits at absolute position pos in slot `slot`, so it attends to keys
// 0..pos of that slot. Its own key is already in the cache (append_kernel ran
// first), and so are the keys of earlier tokens in the same chunk, which gives
// causal attention inside the chunk without a mask.
__global__ void attention_kernel(const float* k_cache, const float* v_cache,
                                 const float* q_new, const int* token_slot, const int* token_pos,
                                 float* out, int layer, int max_slots, int n_head, int n_pos,
                                 int head_dim, float scale) {
  extern __shared__ float shared[];
  float* scores = shared;               // (n_pos,)
  float* query = scores + n_pos;        // (head_dim,)
  float* scratch = query + head_dim;    // (ATTN_THREADS,)

  int t = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;
  int n_keys = token_pos[t] + 1;
  size_t base = cache_index(layer, token_slot[t], h, 0, max_slots, n_head, n_pos, head_dim);
  const float* keys = k_cache + base;
  const float* values = v_cache + base;

  const float* q = q_new + ((size_t)t * n_head + h) * head_dim;
  for (int e = tid; e < head_dim; e += blockDim.x) query[e] = q[e];
  __syncthreads();

  // 1. Scores: thread tid handles keys tid, tid + blockDim, ...
  float local_max = -INFINITY;
  for (int j = tid; j < n_keys; j += blockDim.x) {
    const float* key = keys + (size_t)j * head_dim;
    float dot = 0.0f;
    for (int e = 0; e < head_dim; ++e) dot += query[e] * key[e];
    dot *= scale;
    scores[j] = dot;
    local_max = fmaxf(local_max, dot);
  }
  float row_max = block_reduce(local_max, scratch, true);

  // 2. Softmax weights. Subtracting the max keeps exp() from overflowing.
  // Each thread rewrites only the scores it computed above.
  float local_sum = 0.0f;
  for (int j = tid; j < n_keys; j += blockDim.x) {
    float weight = expf(scores[j] - row_max);
    scores[j] = weight;
    local_sum += weight;
  }
  // The reduction's barriers also make every weight visible to every thread.
  float inv_sum = 1.0f / block_reduce(local_sum, scratch, false);

  // 3. Output: thread e (one per output dimension) sums weight_j * V[j][e].
  // Neighbouring threads read neighbouring floats of each value row.
  for (int e = tid; e < head_dim; e += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < n_keys; ++j) acc += scores[j] * values[(size_t)j * head_dim + e];
    out[((size_t)t * n_head + h) * head_dim + e] = acc * inv_sum;
  }
}

static void free_scratch(RaggedCache* c) {
  cudaFree(c->q_new); cudaFree(c->k_new); cudaFree(c->v_new); cudaFree(c->out);
  cudaFree(c->token_slot); cudaFree(c->token_pos);
  c->q_new = c->k_new = c->v_new = c->out = NULL;
  c->token_slot = c->token_pos = NULL;
  c->capacity = 0;
}

static int ensure_capacity(RaggedCache* c, int n_tokens) {
  if (n_tokens <= c->capacity) return RAGGED_OK;
  free_scratch(c);
  size_t vec = (size_t)n_tokens * c->n_head * c->head_dim * sizeof(float);
  size_t idx = (size_t)n_tokens * sizeof(int);
  if (cudaMalloc(&c->q_new, vec) != cudaSuccess || cudaMalloc(&c->k_new, vec) != cudaSuccess ||
      cudaMalloc(&c->v_new, vec) != cudaSuccess || cudaMalloc(&c->out, vec) != cudaSuccess ||
      cudaMalloc(&c->token_slot, idx) != cudaSuccess || cudaMalloc(&c->token_pos, idx) != cudaSuccess) {
    free_scratch(c);
    cudaGetLastError();  // clear the sticky allocation error
    return RAGGED_OUT_OF_MEMORY;
  }
  c->capacity = n_tokens;
  return RAGGED_OK;
}

extern "C" {

// Allocate a zeroed cache on the GPU. Returns NULL on failure.
RaggedCache* ragged_cache_create(int n_layer, int max_slots, int n_head, int n_pos, int head_dim) {
  if (n_layer < 1 || max_slots < 1 || n_head < 1 || n_pos < 1 || head_dim < 1) return NULL;
  RaggedCache* c = (RaggedCache*)calloc(1, sizeof(RaggedCache));
  if (c == NULL) return NULL;
  c->n_layer = n_layer; c->max_slots = max_slots; c->n_head = n_head;
  c->n_pos = n_pos; c->head_dim = head_dim;
  size_t bytes = (size_t)n_layer * max_slots * n_head * n_pos * head_dim * sizeof(float);
  if (cudaMalloc(&c->k, bytes) != cudaSuccess || cudaMalloc(&c->v, bytes) != cudaSuccess ||
      cudaMemset(c->k, 0, bytes) != cudaSuccess || cudaMemset(c->v, 0, bytes) != cudaSuccess) {
    cudaFree(c->k); cudaFree(c->v); free(c);
    cudaGetLastError();
    return NULL;
  }
  return c;
}

void ragged_cache_destroy(RaggedCache* c) {
  if (c == NULL) return;
  free_scratch(c);
  cudaFree(c->k); cudaFree(c->v);
  free(c);
}

// One layer of one serving step. Host arrays:
//   q_new, k_new, v_new : (n_tokens, n_head, head_dim), C-contiguous float32
//   token_slot, token_pos : (n_tokens,) int32
//   out : (n_tokens, n_head, head_dim), written on success
// Returns 0 on success, a negative RAGGED_* code, or a cudaError_t value.
// Blocks until the result is on the host, so the step is synchronized.
int ragged_cache_step(RaggedCache* c, int layer, int n_tokens,
                      const float* q_new, const float* k_new, const float* v_new,
                      const int* token_slot, const int* token_pos, float* out) {
  if (c == NULL || layer < 0 || layer >= c->n_layer || n_tokens < 1) return RAGGED_BAD_ARGUMENT;
  for (int t = 0; t < n_tokens; ++t) {
    if (token_slot[t] < 0 || token_slot[t] >= c->max_slots ||
        token_pos[t] < 0 || token_pos[t] >= c->n_pos) return RAGGED_BAD_ARGUMENT;
  }
  int status = ensure_capacity(c, n_tokens);
  if (status != RAGGED_OK) return status;

  size_t vec = (size_t)n_tokens * c->n_head * c->head_dim * sizeof(float);
  size_t idx = (size_t)n_tokens * sizeof(int);
  cudaError_t err;
  if ((err = cudaMemcpy(c->q_new, q_new, vec, cudaMemcpyHostToDevice)) != cudaSuccess) return err;
  if ((err = cudaMemcpy(c->k_new, k_new, vec, cudaMemcpyHostToDevice)) != cudaSuccess) return err;
  if ((err = cudaMemcpy(c->v_new, v_new, vec, cudaMemcpyHostToDevice)) != cudaSuccess) return err;
  if ((err = cudaMemcpy(c->token_slot, token_slot, idx, cudaMemcpyHostToDevice)) != cudaSuccess) return err;
  if ((err = cudaMemcpy(c->token_pos, token_pos, idx, cudaMemcpyHostToDevice)) != cudaSuccess) return err;

  dim3 grid(n_tokens, c->n_head);
  int append_threads = c->head_dim < 256 ? c->head_dim : 256;
  append_kernel<<<grid, append_threads>>>(c->k, c->v, c->k_new, c->v_new, c->token_slot, c->token_pos,
                                          layer, c->max_slots, c->n_head, c->n_pos, c->head_dim);
  if ((err = cudaGetLastError()) != cudaSuccess) return err;

  // Kernels on the default stream run in order, so every new key is in the
  // cache before any attention block reads it.
  size_t shared = (size_t)(c->n_pos + c->head_dim + ATTN_THREADS) * sizeof(float);
  float scale = 1.0f / sqrtf((float)c->head_dim);
  attention_kernel<<<grid, ATTN_THREADS, shared>>>(c->k, c->v, c->q_new, c->token_slot, c->token_pos,
                                                   c->out, layer, c->max_slots, c->n_head, c->n_pos,
                                                   c->head_dim, scale);
  if ((err = cudaGetLastError()) != cudaSuccess) return err;
  if ((err = cudaMemcpy(out, c->out, vec, cudaMemcpyDeviceToHost)) != cudaSuccess) return err;
  return RAGGED_OK;
}

// Copy keys and values at positions 0..length of one slot and layer back to
// the host, each as (n_head, length, head_dim). For tests only.
int ragged_cache_read(RaggedCache* c, int layer, int slot, int length, float* k_out, float* v_out) {
  if (c == NULL || layer < 0 || layer >= c->n_layer || slot < 0 || slot >= c->max_slots ||
      length < 0 || length > c->n_pos) return RAGGED_BAD_ARGUMENT;
  size_t row = (size_t)length * c->head_dim * sizeof(float);
  cudaError_t err;
  for (int h = 0; h < c->n_head; ++h) {
    size_t src = cache_index(layer, slot, h, 0, c->max_slots, c->n_head, c->n_pos, c->head_dim);
    size_t dst = (size_t)h * length * c->head_dim;
    if ((err = cudaMemcpy(k_out + dst, c->k + src, row, cudaMemcpyDeviceToHost)) != cudaSuccess) return err;
    if ((err = cudaMemcpy(v_out + dst, c->v + src, row, cudaMemcpyDeviceToHost)) != cudaSuccess) return err;
  }
  return RAGGED_OK;
}

const char* ragged_error_string(int code) {
  if (code == RAGGED_BAD_ARGUMENT) return "bad argument (layer, slot, position, or token count out of range)";
  if (code == RAGGED_OUT_OF_MEMORY) return "out of GPU memory for step buffers";
  return cudaGetErrorString((cudaError_t)code);
}

}  // extern "C"
