"""GPT-2 model correctness: shapes, determinism, and HuggingFace logit match."""

import unittest

import numpy as np

from minitorch import FastOps, TensorBackend
from model import GPT2, GPT2Config, generate_greedy_no_cache, hf_state_dict, load_hf_weights

BACKEND = TensorBackend(FastOps)
PROMPT = "The quick brown fox jumps over the lazy dog because"


def _reference_or_none():
    """HuggingFace GPT-2 small plus tokenizer, or None when unavailable offline."""
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        hf = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).eval()
    except Exception:  # missing dependency or no cached weights
        return None
    return torch, tok, hf


class TinyGPT2Tests(unittest.TestCase):
    def test_forward_shape_and_greedy_is_deterministic(self):
        np.random.seed(0)
        model = GPT2(GPT2Config.tiny(), BACKEND)
        idx = np.array([[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]])
        logits = model(idx)
        self.assertEqual(logits.shape, (2, 5, 64))
        first = generate_greedy_no_cache(model, [1, 2, 3], 4)
        second = generate_greedy_no_cache(model, [1, 2, 3], 4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_rejects_sequences_past_context(self):
        model = GPT2(GPT2Config.tiny(n_positions=8), BACKEND)
        with self.assertRaises(ValueError):
            model(np.zeros((1, 9), dtype=np.int64))
        with self.assertRaises(ValueError):
            model(np.zeros((1, 4), dtype=np.int64), start_pos=5)


class PretrainedGPT2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = _reference_or_none()
        if ref is None:
            raise unittest.SkipTest("HuggingFace GPT-2 reference unavailable")
        cls.torch, cls.tok, cls.hf = ref
        config, state = hf_state_dict("gpt2")
        cls.model = GPT2(config, BACKEND)
        load_hf_weights(cls.model, state)
        cls.ids = cls.tok(PROMPT)["input_ids"]

    def test_logits_match_huggingface(self):
        ours = self.model(np.array([self.ids])).to_numpy()[0]
        with self.torch.no_grad():
            theirs = self.hf(self.torch.tensor([self.ids])).logits[0].numpy()
        self.assertEqual(ours.shape, theirs.shape)
        max_abs = float(np.max(np.abs(ours - theirs)))
        self.assertLess(max_abs, 2e-2, f"max abs logit diff {max_abs}")
        np.testing.assert_array_equal(ours.argmax(-1), theirs.argmax(-1))

    def test_greedy_continuation_matches_huggingface(self):
        steps = 6
        ours = generate_greedy_no_cache(self.model, self.ids, steps)
        tokens = list(self.ids)
        with self.torch.no_grad():
            for _ in range(steps):
                logits = self.hf(self.torch.tensor([tokens])).logits[0, -1]
                tokens.append(int(logits.argmax()))
        self.assertEqual(ours, tokens[len(self.ids):])
