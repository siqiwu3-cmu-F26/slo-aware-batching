"""Attention for a ragged batch, computed without padding.

A ragged batch has one row per request. Row b brings ``counts[b]`` new tokens
whose keys and values have already been written to ``slots[b]`` of the KV
cache at positions ``starts[b] .. starts[b] + counts[b]``. New token i of row b
sits at absolute position ``p = starts[b] + i`` and attends to that slot's keys
at positions ``0 .. p`` (its cached prefix, the earlier new tokens of its own
row, and itself). It never sees another slot or a later position.

This NumPy version defines exactly what the chunked-prefill CUDA kernel must
compute. It is the CPU fallback and the oracle the kernel is tested against.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def ragged_attention_reference(q_new: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
                               slots: Sequence[int], starts: Sequence[int],
                               counts: Sequence[int]) -> np.ndarray:
    """softmax(q k^T / sqrt(d)) v for every new token, over its own slot only.

    Args:
        q_new: (n_tokens, n_head, head_dim) queries of all rows, row after row.
        k_cache, v_cache: (max_slots, n_head, n_positions, head_dim) for one
            layer, already holding this step's new keys and values.
        slots, starts, counts: one entry per row.

    Returns:
        (n_tokens, n_head, head_dim) attention output, in the same token order
        as ``q_new``.
    """
    n_tokens, n_head, head_dim = q_new.shape
    if sum(counts) != n_tokens:
        raise ValueError("counts must add up to the number of query tokens")
    scale = 1.0 / np.sqrt(head_dim)
    out = np.empty_like(q_new)
    offset = 0
    for slot, start, count in zip(slots, starts, counts):
        total = start + count
        q = q_new[offset:offset + count].transpose(1, 0, 2)      # (h, count, d)
        k = k_cache[slot, :, :total, :]                           # (h, total, d)
        v = v_cache[slot, :, :total, :]                           # (h, total, d)
        scores = (q @ k.transpose(0, 2, 1)) * scale               # (h, count, total)
        # New token i is at position start + i and may see keys 0 .. start + i.
        query_pos = start + np.arange(count)[:, None]             # (count, 1)
        key_pos = np.arange(total)[None, :]                       # (1, total)
        scores = np.where(key_pos <= query_pos, scores, -np.inf)
        # Subtract each row's maximum before exp so large scores cannot overflow.
        # Every query sees at least key 0, so the maximum is always finite.
        scores = scores - scores.max(axis=2, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=2, keepdims=True)
        out[offset:offset + count] = (weights @ v).transpose(1, 0, 2)   # (count, h, d)
        offset += count
    return out.astype(np.float32, copy=False)
