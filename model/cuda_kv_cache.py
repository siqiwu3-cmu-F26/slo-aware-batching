"""KV cache that lives in GPU memory, used with forward_ragged(attention="cuda").

Slot bookkeeping (allocate, free, lengths) stays on the host exactly as in
KVCache. The keys and values themselves are allocated once on the GPU by
kernels/ragged_attention.cu and never copied back during serving. Each layer
of each step sends only the new tokens' q/k/v to the GPU, where the kernel
writes the new k/v into the cache and computes attention, and receives only
the attention output.

Build the library with kernels/compile_cuda.sh on a machine with nvcc.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Sequence

import numpy as np

from .gpt2 import GPT2Config, KVCache

LIBRARY_PATH = Path(__file__).resolve().parent.parent / "minitorch" / "cuda_kernels" / "ragged_attention.so"

_FLOATS = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_INTS = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_library: ctypes.CDLL | None = None


def load_library(path: Path = LIBRARY_PATH) -> ctypes.CDLL:
    """Load the compiled kernel once; raises OSError if it is missing."""
    global _library
    if _library is None:
        if not path.exists():
            raise OSError(f"{path} not found; run kernels/compile_cuda.sh on a CUDA machine")
        lib = ctypes.CDLL(str(path))
        lib.ragged_cache_create.argtypes = [ctypes.c_int] * 5
        lib.ragged_cache_create.restype = ctypes.c_void_p
        lib.ragged_cache_destroy.argtypes = [ctypes.c_void_p]
        lib.ragged_cache_destroy.restype = None
        lib.ragged_cache_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                          _FLOATS, _FLOATS, _FLOATS, _INTS, _INTS, _FLOATS]
        lib.ragged_cache_step.restype = ctypes.c_int
        lib.ragged_cache_read.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          _FLOATS, _FLOATS]
        lib.ragged_cache_read.restype = ctypes.c_int
        lib.ragged_error_string.argtypes = [ctypes.c_int]
        lib.ragged_error_string.restype = ctypes.c_char_p
        _library = lib
    return _library


def cuda_kernel_available() -> bool:
    """True when the library loads and a GPU cache can be allocated."""
    try:
        lib = load_library()
    except OSError:
        return False
    handle = lib.ragged_cache_create(1, 1, 1, 1, 1)
    if not handle:
        return False
    lib.ragged_cache_destroy(handle)
    return True


class CudaKVCache(KVCache):
    """Same slot interface as KVCache, with keys and values on the GPU."""

    on_device = True

    def __init__(self, config: GPT2Config, max_slots: int) -> None:
        # Deliberately not calling KVCache.__init__: it would allocate the
        # full host arrays this class exists to avoid.
        if type(max_slots) is not int or max_slots < 1:
            raise ValueError("max_slots must be a positive integer")
        self.config = config
        self.max_slots = max_slots
        self.lengths = np.zeros(max_slots, dtype=np.int64)
        self._free: list[int] = list(range(max_slots - 1, -1, -1))
        self._lib = load_library()
        self._handle = self._lib.ragged_cache_create(config.n_layer, max_slots, config.n_head,
                                                     config.n_positions, config.head_dim)
        if not self._handle:
            raise RuntimeError("could not allocate the KV cache on the GPU")

    @property
    def num_bytes(self) -> int:
        c = self.config
        return 2 * c.n_layer * self.max_slots * c.n_head * c.n_positions * c.head_dim * 4

    def close(self) -> None:
        """Free the GPU memory. Safe to call more than once."""
        if getattr(self, "_handle", None):
            self._lib.ragged_cache_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def append(self, *args, **kwargs) -> None:
        raise NotImplementedError("CudaKVCache is written by append_and_attend (forward_ragged, attention='cuda')")

    def prefix(self, *args, **kwargs):
        raise NotImplementedError("CudaKVCache is read on the GPU; use read_prefix for tests")

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"ragged attention kernel failed: {self._lib.ragged_error_string(code).decode()}")

    def append_and_attend(self, layer: int, q_new: np.ndarray, k_new: np.ndarray, v_new: np.ndarray,
                          slots: Sequence[int], starts: Sequence[int], counts: Sequence[int]) -> np.ndarray:
        """Write new k/v into their slots, then attend; returns (n_tokens, n_head, head_dim).

        q_new, k_new, v_new are (n_tokens, n_head, head_dim) with rows in the
        same order as ``slots``/``starts``/``counts``. Matches
        ragged_attention_reference run after KVCache.append.
        """
        if not self._handle:
            raise RuntimeError("CudaKVCache is closed")
        n_tokens = int(sum(counts))
        shape = (n_tokens, self.config.n_head, self.config.head_dim)
        if q_new.shape != shape or k_new.shape != shape or v_new.shape != shape:
            raise ValueError(f"q/k/v must have shape {shape}")
        # Per-token slot and absolute position, so each GPU block finds its row.
        token_slot = np.repeat(np.asarray(slots, dtype=np.int32), counts)
        token_pos = np.concatenate([np.arange(start, start + count, dtype=np.int32)
                                    for start, count in zip(starts, counts)])
        if token_pos.max() >= self.config.n_positions:
            raise ValueError("KV cache slot overflow: sequence exceeds n_positions")
        flat = [np.ascontiguousarray(a, dtype=np.float32).reshape(-1) for a in (q_new, k_new, v_new)]
        out = np.empty(n_tokens * self.config.n_head * self.config.head_dim, dtype=np.float32)
        self._check(self._lib.ragged_cache_step(self._handle, layer, n_tokens, *flat,
                                                token_slot, token_pos, out))
        return out.reshape(shape)

    def read_prefix(self, layer: int, slot: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Copy keys and values 0..length of one slot back to the host, each (n_head, length, head_dim)."""
        h, d = self.config.n_head, self.config.head_dim
        k = np.empty(h * length * d, dtype=np.float32)
        v = np.empty(h * length * d, dtype=np.float32)
        self._check(self._lib.ragged_cache_read(self._handle, layer, slot, length, k, v))
        return k.reshape(h, length, d), v.reshape(h, length, d)
