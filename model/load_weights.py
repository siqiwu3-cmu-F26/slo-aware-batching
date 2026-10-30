"""Map HuggingFace GPT-2 checkpoints onto the MiniTorch GPT2 module.

HuggingFace stores attention and MLP weights as Conv1D with shape (in, out),
which is exactly MiniTorch Linear's layout, so no transposes are needed. The
fused ``c_attn`` (in, 3*n_embd) is split column-wise into q, k, v.
"""

from __future__ import annotations

import numpy as np

from minitorch import Parameter
from minitorch.tensor_functions import tensor_from_numpy

from .gpt2 import GPT2, GPT2Config


def hf_state_dict(name_or_path: str = "gpt2") -> tuple[GPT2Config, dict[str, np.ndarray]]:
    """Load a HuggingFace GPT-2 checkpoint as float32 numpy arrays.

    Requires ``torch`` and ``transformers``; both are development dependencies
    used only for loading weights and for the reference test.
    """
    import torch
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(name_or_path, torch_dtype=torch.float32)
    cfg = hf.config
    config = GPT2Config(
        n_vocab=cfg.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head, n_layer=cfg.n_layer,
        n_positions=cfg.n_positions, ln_eps=cfg.layer_norm_epsilon)
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in hf.state_dict().items()}
    return config, state


def _set(param: Parameter, array: np.ndarray, backend) -> None:
    expected = param.value.shape
    if tuple(array.shape) != tuple(expected):
        raise ValueError(f"shape mismatch: checkpoint {array.shape} vs model {tuple(expected)}")
    param.update(tensor_from_numpy(np.ascontiguousarray(array, dtype=np.float32), backend=backend))
    # Inference only: skip autograd bookkeeping on every forward op.
    param.value.requires_grad_(False)


def load_hf_weights(model: GPT2, state: dict[str, np.ndarray]) -> None:
    """Copy a HuggingFace GPT-2 state dict (numpy arrays) into ``model`` in place."""
    backend = model.backend
    n = model.config.n_embd
    _set(model.token_embeddings.weights, state["transformer.wte.weight"], backend)
    _set(model.position_embeddings.weights, state["transformer.wpe.weight"], backend)
    for i, block in enumerate(model.blocks()):
        p = f"transformer.h.{i}."
        _set(block.ln_1.weights, state[p + "ln_1.weight"], backend)
        _set(block.ln_1.bias, state[p + "ln_1.bias"], backend)
        w, b = state[p + "attn.c_attn.weight"], state[p + "attn.c_attn.bias"]
        attn = block.attention
        for j, proj in enumerate((attn.q_projection, attn.k_projection, attn.v_projection)):
            _set(proj.weights, w[:, j * n:(j + 1) * n], backend)
            _set(proj.bias, b[j * n:(j + 1) * n], backend)
        _set(attn.out_projection.weights, state[p + "attn.c_proj.weight"], backend)
        _set(attn.out_projection.bias, state[p + "attn.c_proj.bias"], backend)
        _set(block.ln_2.weights, state[p + "ln_2.weight"], backend)
        _set(block.ln_2.bias, state[p + "ln_2.bias"], backend)
        _set(block.ff.linear_in.weights, state[p + "mlp.c_fc.weight"], backend)
        _set(block.ff.linear_in.bias, state[p + "mlp.c_fc.bias"], backend)
        _set(block.ff.linear_out.weights, state[p + "mlp.c_proj.weight"], backend)
        _set(block.ff.linear_out.bias, state[p + "mlp.c_proj.bias"], backend)
    _set(model.ln_f.weights, state["transformer.ln_f.weight"], backend)
    _set(model.ln_f.bias, state["transformer.ln_f.bias"], backend)
    # Tied head: Linear expects (in=n_embd, out=n_vocab), i.e. the transposed table.
    _set(model.lm_head.weights, state["transformer.wte.weight"].T, backend)
