"""GPT-2 decoder in MiniTorch, written for inference.

The structure mirrors MiniTorch's DecoderLM (pre-LN blocks, separate q/k/v
projections, GELU feed-forward) but differs where serving needs it:

- the layer count is configurable so GPT-2 small (12) and medium (24) both fit;
- embeddings are gathered by index instead of a one-hot matmul, which for a
  50257-token vocabulary would cost a 40 GFLOP matmul per 1024-token prompt;
- positions are absolute and start at ``start_pos`` so a later KV cache can
  feed decode steps at arbitrary offsets;
- there is no dropout; the model is only ever evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minitorch import Linear, LayerNorm1d, Module, Parameter, Tensor, TensorBackend
from minitorch.nn import GELU, softmax
from minitorch.tensor_functions import tensor_from_numpy


# Large finite negative: exp(MASK - rowmax) underflows to exactly 0 in float32
# without risking -inf arithmetic inside the softmax.
MASK_VALUE = -1e9


@dataclass(frozen=True)
class GPT2Config:
    n_vocab: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    ln_eps: float = 1e-5

    @classmethod
    def small(cls) -> "GPT2Config":
        return cls()

    @classmethod
    def medium(cls) -> "GPT2Config":
        return cls(n_embd=1024, n_head=16, n_layer=24)

    @classmethod
    def tiny(cls, n_vocab: int = 64, n_positions: int = 32) -> "GPT2Config":
        """Small random-weight configuration for fast tests."""
        return cls(n_vocab=n_vocab, n_embd=32, n_head=4, n_layer=2, n_positions=n_positions)

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head


class KVCache:
    """Preallocated keys and values for ``max_slots`` concurrent requests.

    Layout per tensor: (n_layer, max_slots, n_head, n_positions, head_dim),
    float32, host-resident numpy. MiniTorch has no slicing or in-place writes,
    so the cache lives in numpy and each attention call converts just the
    prefix it needs into a tensor. ``lengths[slot]`` is how many positions of
    that slot are filled; the scheduler-facing engine owns slot assignment.
    """

    def __init__(self, config: GPT2Config, max_slots: int) -> None:
        if type(max_slots) is not int or max_slots < 1:
            raise ValueError("max_slots must be a positive integer")
        shape = (config.n_layer, max_slots, config.n_head, config.n_positions, config.head_dim)
        self.config = config
        self.max_slots = max_slots
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.lengths = np.zeros(max_slots, dtype=np.int64)
        self._free: list[int] = list(range(max_slots - 1, -1, -1))

    @property
    def num_bytes(self) -> int:
        return self.k.nbytes + self.v.nbytes

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("no free KV-cache slot")
        slot = self._free.pop()
        self.lengths[slot] = 0
        return slot

    def free(self, slot: int) -> None:
        if slot in self._free or not 0 <= slot < self.max_slots:
            raise ValueError(f"slot {slot} is not allocated")
        self.lengths[slot] = 0
        self._free.append(slot)

    def length(self, slot: int) -> int:
        return int(self.lengths[slot])

    def append(self, layer: int, slot: int, start_pos: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write (n_head, T, head_dim) keys/values at positions start_pos..start_pos+T."""
        count = k.shape[1]
        if start_pos + count > self.config.n_positions:
            raise ValueError("KV cache slot overflow: sequence exceeds n_positions")
        self.k[layer, slot, :, start_pos:start_pos + count, :] = k
        self.v[layer, slot, :, start_pos:start_pos + count, :] = v

    def prefix(self, layer: int, slot: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Keys and values for positions 0..length of one slot, each (n_head, length, head_dim)."""
        return self.k[layer, slot, :, :length, :], self.v[layer, slot, :, :length, :]


class GatherEmbedding(Module):
    """Embedding lookup by integer index; no one-hot matmul."""

    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend) -> None:
        super().__init__()
        self.backend = backend
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        table = (np.random.randn(num_embeddings, embedding_dim) * 0.02).astype(np.float32)
        self.weights = Parameter(tensor_from_numpy(table, backend=backend))

    def table(self) -> np.ndarray:
        """Zero-copy (num_embeddings, embedding_dim) view of the parameter storage."""
        data = self.weights.value._tensor
        if not data.is_contiguous():
            raise RuntimeError("embedding table must be contiguous")
        return data._storage.reshape(self.num_embeddings, self.embedding_dim)

    def forward(self, indices: np.ndarray) -> Tensor:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.min() < 0 or indices.max() >= self.num_embeddings:
            raise ValueError("embedding index out of range")
        return tensor_from_numpy(self.table()[indices], backend=self.backend)


class CausalSelfAttention(Module):
    def __init__(self, config: GPT2Config, backend: TensorBackend) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.backend = backend
        n = config.n_embd
        self.q_projection = Linear(n, n, bias=True, backend=backend)
        self.k_projection = Linear(n, n, bias=True, backend=backend)
        self.v_projection = Linear(n, n, bias=True, backend=backend)
        self.out_projection = Linear(n, n, bias=True, backend=backend)

    def project_to_query_key_value(self, x: Tensor):
        batch_size, seq_len, n_embd = x.shape
        h, d = self.n_head, self.head_dim
        x_flat = x.view(batch_size * seq_len, n_embd)
        q = self.q_projection(x_flat).view(batch_size, seq_len, h, d).permute(0, 2, 1, 3)
        kT = self.k_projection(x_flat).view(batch_size, seq_len, h, d).permute(0, 2, 3, 1)
        v = self.v_projection(x_flat).view(batch_size, seq_len, h, d).permute(0, 2, 1, 3)
        return q, kT, v

    def causal_mask(self, seq_len: int, start_pos: int = 0) -> Tensor:
        """(1, 1, T, start_pos + T) additive mask.

        Query i sits at absolute position start_pos + i and may attend to key
        positions 0 .. start_pos + i. With start_pos = 0 this is the usual
        lower-triangular mask.
        """
        total = start_pos + seq_len
        ones = np.ones((1, 1, seq_len, total), dtype=np.float32)
        mask = MASK_VALUE * np.triu(ones, k=start_pos + 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def _attend(self, q: Tensor, kT: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = q.shape
        attn = (q @ kT) / (self.head_dim ** 0.5)          # (B, h, T, L)
        attn = softmax(attn + mask, dim=3)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous()  # (B, T, h, d)
        out = out.view(batch_size * seq_len, self.n_head * self.head_dim)
        return self.out_projection(out).view(batch_size, seq_len, self.n_head * self.head_dim)

    def forward(self, x: Tensor, cache: KVCache | None = None, layer: int | None = None,
                slot: int | None = None, start_pos: int = 0) -> Tensor:
        if cache is None:
            q, kT, v = self.project_to_query_key_value(x)
            return self._attend(q, kT, v, self.causal_mask(x.shape[1]))
        return self._forward_cached(x, cache, layer, slot, start_pos)

    def _forward_cached(self, x: Tensor, cache: KVCache, layer: int, slot: int,
                        start_pos: int) -> Tensor:
        """One request: append this chunk's K/V to its slot, attend over the full prefix."""
        batch_size, seq_len, n_embd = x.shape
        if batch_size != 1:
            raise ValueError("cached attention currently serves one slot per call")
        h, d = self.n_head, self.head_dim
        x_flat = x.view(seq_len, n_embd)
        q = self.q_projection(x_flat).view(1, seq_len, h, d).permute(0, 2, 1, 3)
        k_new = self.k_projection(x_flat).to_numpy().reshape(seq_len, h, d).transpose(1, 0, 2)
        v_new = self.v_projection(x_flat).to_numpy().reshape(seq_len, h, d).transpose(1, 0, 2)
        cache.append(layer, slot, start_pos, k_new, v_new)
        total = start_pos + seq_len
        k_all, v_all = cache.prefix(layer, slot, total)           # (h, L, d) each
        kT = tensor_from_numpy(k_all.transpose(0, 2, 1)[None], backend=self.backend)  # (1, h, d, L)
        v = tensor_from_numpy(v_all[None], backend=self.backend)                    # (1, h, L, d)
        return self._attend(q, kT, v, self.causal_mask(seq_len, start_pos))


class FeedForward(Module):
    def __init__(self, config: GPT2Config, backend: TensorBackend) -> None:
        super().__init__()
        n = config.n_embd
        self.linear_in = Linear(n, 4 * n, bias=True, backend=backend)
        self.linear_out = Linear(4 * n, n, bias=True, backend=backend)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, n_embd = x.shape
        hidden = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        return self.linear_out(hidden).view(batch_size, seq_len, n_embd)


class TransformerBlock(Module):
    def __init__(self, config: GPT2Config, backend: TensorBackend) -> None:
        super().__init__()
        self.ln_1 = LayerNorm1d(config.n_embd, config.ln_eps, backend)
        self.attention = CausalSelfAttention(config, backend)
        self.ln_2 = LayerNorm1d(config.n_embd, config.ln_eps, backend)
        self.ff = FeedForward(config, backend)

    def forward(self, x: Tensor, **attention_kwargs) -> Tensor:
        batch_size, seq_len, n_embd = x.shape
        normed = self.ln_1(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        x = x + self.attention(normed, **attention_kwargs)
        normed = self.ln_2(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        return x + self.ff(normed)


class GPT2(Module):
    def __init__(self, config: GPT2Config, backend: TensorBackend) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.token_embeddings = GatherEmbedding(config.n_vocab, config.n_embd, backend)
        self.position_embeddings = GatherEmbedding(config.n_positions, config.n_embd, backend)
        # Module stores submodules by attribute; a list would hide them from
        # named_parameters, so register each block under a numbered name.
        self.n_layer = config.n_layer
        for i in range(config.n_layer):
            setattr(self, f"block_{i}", TransformerBlock(config, backend))
        self.ln_f = LayerNorm1d(config.n_embd, config.ln_eps, backend)
        # GPT-2 ties the LM head to the token embedding; the loader fills this
        # with the transposed table so the forward stays a plain Linear.
        self.lm_head = Linear(config.n_embd, config.n_vocab, bias=False, backend=backend)
        self.eval()

    def blocks(self) -> list[TransformerBlock]:
        return [getattr(self, f"block_{i}") for i in range(self.n_layer)]

    def forward(self, idx: np.ndarray, start_pos: int = 0, cache: KVCache | None = None,
                slot: int | None = None) -> Tensor:
        """Logits of shape (batch_size, seq_len, n_vocab) for token ids ``idx``.

        ``idx`` is an integer array of shape (batch_size, seq_len). Position
        embeddings cover ``start_pos .. start_pos + seq_len``.

        With ``cache`` and ``slot`` given, ``idx`` must be one row whose
        ``start_pos`` equals the slot's current length: the row is appended to
        the slot and attends over everything cached before it. The slot length
        advances by ``seq_len`` on return.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, seq_len)")
        batch_size, seq_len = idx.shape
        if start_pos < 0 or start_pos + seq_len > self.config.n_positions:
            raise ValueError("sequence exceeds n_positions")
        if cache is not None:
            if slot is None or batch_size != 1:
                raise ValueError("cached forward takes one row and a slot")
            if cache.length(slot) != start_pos:
                raise ValueError(f"slot {slot} holds {cache.length(slot)} positions, "
                                 f"but start_pos is {start_pos}")
        positions = np.arange(start_pos, start_pos + seq_len)[None, :]
        x = self.token_embeddings(idx) + self.position_embeddings(positions)
        for layer, block in enumerate(self.blocks()):
            if cache is None:
                x = block(x)
            else:
                x = block(x, cache=cache, layer=layer, slot=slot, start_pos=start_pos)
        if cache is not None:
            cache.lengths[slot] = start_pos + seq_len
        x = self.ln_f(x.view(batch_size * seq_len, self.config.n_embd))
        return self.lm_head(x).view(batch_size, seq_len, self.config.n_vocab)


def generate_greedy_no_cache(model: GPT2, prompt: list[int], num_tokens: int) -> list[int]:
    """Reference decoder: recompute the full sequence every step, pick argmax.

    This is deliberately slow. It is the oracle that the cached, batched, and
    chunked paths must match token for token.
    """
    tokens = list(prompt)
    for _ in range(num_tokens):
        logits = model(np.array([tokens]))
        last = logits.to_numpy()[0, -1]
        tokens.append(int(np.argmax(last)))
    return tokens[len(prompt):]


def generate_greedy_cached(model: GPT2, cache: KVCache, prompt: list[int],
                           num_tokens: int) -> list[int]:
    """Greedy decode using the KV cache: one prefill pass, then one token per step."""
    slot = cache.allocate()
    try:
        logits = model(np.array([prompt]), start_pos=0, cache=cache, slot=slot)
        next_token = int(np.argmax(logits.to_numpy()[0, -1]))
        generated = [next_token]
        for _ in range(num_tokens - 1):
            logits = model(np.array([[next_token]]), start_pos=cache.length(slot),
                           cache=cache, slot=slot)
            next_token = int(np.argmax(logits.to_numpy()[0, -1]))
            generated.append(next_token)
        return generated
    finally:
        cache.free(slot)
