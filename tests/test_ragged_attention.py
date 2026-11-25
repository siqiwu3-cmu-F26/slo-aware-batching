"""Unpadded ragged attention: NumPy reference on CPU, CUDA kernel on a GPU.

The CPU tests run everywhere. The CUDA tests skip unless
minitorch/cuda_kernels/ragged_attention.so is built (kernels/compile_cuda.sh)
and a GPU is visible. Run on a GPU node with:

    python -m unittest tests.test_ragged_attention -v
"""

import unittest

import numpy as np

from engine.minitorch_engine import MiniTorchEngine
from interfaces import Request
from minitorch import FastOps, TensorBackend
from model import (GPT2, GPT2Config, CudaKVCache, KVCache, cuda_kernel_available,
                   generate_greedy_no_cache, ragged_attention_reference)
from scheduler import FixedBudget, OrcaScheduler, SarathiScheduler, StaticScheduler
from sim import run_simulation

BACKEND = TensorBackend(FastOps)
SCHEDULERS = (lambda: StaticScheduler(2), lambda: OrcaScheduler(2),
              lambda: SarathiScheduler(2, FixedBudget(3)))
PROMPTS = {"a": [1, 2, 3, 4, 5, 6, 7], "b": [8, 9], "c": [10, 11, 12, 13, 14], "d": [15, 16, 17]}
OUTPUT_LENGTHS = {"a": 4, "b": 1, "c": 3, "d": 2}


def _random_qkv(rng, n_tokens, config):
    shape = (n_tokens, config.n_head, config.head_dim)
    return tuple(rng.standard_normal(shape).astype(np.float32) for _ in range(3))


def _append_host(cache, layer, slots, starts, counts, k_new, v_new):
    offset = 0
    for slot, start, count in zip(slots, starts, counts):
        rows = slice(offset, offset + count)
        cache.append(layer, slot, start, k_new[rows].transpose(1, 0, 2), v_new[rows].transpose(1, 0, 2))
        cache.lengths[slot] = start + count
        offset += count


def _engine_outputs(model, attention, make_scheduler):
    engine = MiniTorchEngine(model, 2, PROMPTS, attention=attention)
    requests = [Request(rid, 0, len(prompt), OUTPUT_LENGTHS[rid]) for rid, prompt in PROMPTS.items()]
    run_simulation(requests, make_scheduler(), engine)
    return engine


class ReferenceAttentionTests(unittest.TestCase):
    """The unpadded NumPy reference computes the same thing as the padded path."""

    def setUp(self):
        np.random.seed(4)
        self.model = GPT2(GPT2Config.tiny(n_positions=32), BACKEND)
        self.config = self.model.config
        self.rng = np.random.default_rng(11)

    def _prompt(self, length):
        return [int(t) for t in self.rng.integers(0, 64, size=length)]

    def test_reference_matches_padded_attention(self):
        # Rows: fresh 5-token prefill, decode after 4, 3-token chunk after 6,
        # decode after 9; includes a chunk size of 1 and one longer than others.
        cache = KVCache(self.config, max_slots=4)
        slots = [cache.allocate() for _ in range(4)]
        starts, counts = [0, 4, 6, 9], [5, 1, 3, 1]
        for slot, start in zip(slots, starts):
            if start:
                _, k, v = _random_qkv(self.rng, start, self.config)
                _append_host(cache, 0, [slot], [0], [start], k, v)
        q, k, v = _random_qkv(self.rng, sum(counts), self.config)
        _append_host(cache, 0, slots, starts, counts, k, v)
        attention = self.model.blocks()[0].attention
        padded = attention._padded_ragged_attention(q, cache, 0, slots, starts, counts)
        reference = ragged_attention_reference(q, cache.k[0], cache.v[0], slots, starts, counts)
        self.assertEqual(reference.shape, (10, self.config.n_head, self.config.head_dim))
        np.testing.assert_allclose(reference, padded, rtol=1e-5, atol=1e-5)

    def test_unpadded_forward_matches_padded_forward(self):
        histories = [[], self._prompt(4), self._prompt(6), self._prompt(9)]
        rows = [self._prompt(5), self._prompt(1), self._prompt(3), self._prompt(1)]
        results = {}
        for attention in ("padded", "unpadded"):
            cache = KVCache(self.config, max_slots=4)
            slots = [cache.allocate() for _ in rows]
            for history, slot in zip(histories, slots):
                if history:
                    self.model.forward_ragged([history], [0], cache, [slot], attention=attention)
            starts = [len(h) for h in histories]
            results[attention] = self.model.forward_ragged(rows, starts, cache, slots, attention=attention)
            self.assertEqual([cache.length(s) for s in slots], [5, 5, 9, 10])
        np.testing.assert_allclose(results["unpadded"], results["padded"], rtol=1e-4, atol=1e-4)

    def test_unpadded_greedy_matches_no_cache_oracle(self):
        prompts = [self._prompt(2), self._prompt(7), self._prompt(4)]
        expected = [generate_greedy_no_cache(self.model, p, 5) for p in prompts]
        cache = KVCache(self.config, max_slots=3)
        slots = [cache.allocate() for _ in prompts]
        logits = self.model.forward_ragged(prompts, [0, 0, 0], cache, slots, attention="unpadded")
        generated = [[int(np.argmax(row))] for row in logits]
        for _ in range(4):
            starts = [cache.length(s) for s in slots]
            logits = self.model.forward_ragged([[g[-1]] for g in generated], starts, cache, slots,
                                               attention="unpadded")
            for b, row in enumerate(logits):
                generated[b].append(int(np.argmax(row)))
        self.assertEqual(generated, expected)

    def test_attention_mode_must_match_cache(self):
        cache = KVCache(self.config, max_slots=1)
        slot = cache.allocate()
        with self.assertRaises(ValueError):
            self.model.forward_ragged([[1]], [0], cache, [slot], attention="cuda")
        with self.assertRaises(ValueError):
            self.model.forward_ragged([[1]], [0], cache, [slot], attention="flash")
        self.assertEqual(0, cache.length(slot))

    def test_engine_unpadded_matches_oracle_under_all_schedulers(self):
        expected = {rid: generate_greedy_no_cache(self.model, prompt, OUTPUT_LENGTHS[rid])
                    for rid, prompt in PROMPTS.items()}
        for make in SCHEDULERS:
            engine = _engine_outputs(self.model, "unpadded", make)
            with self.subTest(scheduler=type(make()).__name__):
                self.assertEqual(expected, engine.output_tokens)
                self.assertEqual(2, engine.cache.free_slots)
                self.assertEqual("unpadded", engine.metadata["attention"])
                self.assertEqual("host_numpy", engine.metadata["kv_storage"])

    def test_engine_rejects_unpadded_with_grouped_execution(self):
        with self.assertRaises(ValueError):
            MiniTorchEngine(self.model, 2, PROMPTS, execution_mode="grouped", attention="unpadded")


@unittest.skipUnless(cuda_kernel_available(), "ragged_attention.so not built or no GPU visible")
class CudaKernelTests(unittest.TestCase):
    """The CUDA kernel matches the NumPy reference, and the model/engine match the oracle."""

    def setUp(self):
        self.rng = np.random.default_rng(5)

    def _compare_steps(self, config, steps, max_slots):
        """Run the same steps on a GPU cache and a host cache; compare every output and the caches."""
        device = CudaKVCache(config, max_slots)
        host = KVCache(config, max_slots)
        try:
            for layer, slots, starts, counts in steps:
                q, k, v = _random_qkv(self.rng, sum(counts), config)
                out = device.append_and_attend(layer, q, k, v, slots, starts, counts)
                _append_host(host, layer, slots, starts, counts, k, v)
                expected = ragged_attention_reference(q, host.k[layer], host.v[layer], slots, starts, counts)
                np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
                for slot, start, count in zip(slots, starts, counts):
                    k_dev, v_dev = device.read_prefix(layer, slot, start + count)
                    k_host, v_host = host.prefix(layer, slot, start + count)
                    np.testing.assert_array_equal(k_dev, k_host)
                    np.testing.assert_array_equal(v_dev, v_host)
        finally:
            device.close()

    def test_kernel_matches_reference_on_mixed_steps(self):
        config = GPT2Config.tiny(n_positions=32)
        self._compare_steps(config, [
            # Fresh prefills of 7, 1, and 3 tokens (chunk size 1 included).
            (0, [0, 1, 2], [0, 0, 0], [7, 1, 3]),
            # Decode, a 5-token chunk after 1 cached, decode: one ragged step.
            (0, [0, 1, 2], [7, 1, 3], [1, 5, 1]),
            # A different layer and a subset of slots in a different order.
            (1, [2, 0], [0, 0], [4, 2]),
        ], max_slots=3)

    def test_kernel_matches_reference_with_long_prefixes(self):
        # More keys than threads per block (128) and close to the context limit.
        config = GPT2Config.tiny(n_positions=1024)
        self._compare_steps(config, [
            (0, [0, 1], [0, 0], [600, 129]),
            (0, [0, 1], [600, 129], [1, 300]),
            (0, [0], [601], [423]),   # fills the slot to exactly 1024 positions
        ], max_slots=2)

    def test_gpt2_head_size(self):
        # GPT-2 small/medium use head_dim 64; tiny uses 8.
        config = GPT2Config(n_vocab=64, n_embd=768, n_head=12, n_layer=1, n_positions=256)
        self._compare_steps(config, [
            (0, [0, 1, 2, 3], [0, 0, 0, 0], [64, 1, 33, 200]),
            (0, [0, 1, 2, 3], [64, 1, 33, 200], [1, 1, 16, 1]),
        ], max_slots=4)

    def test_overflow_and_closed_cache_raise(self):
        config = GPT2Config.tiny(n_positions=8)
        cache = CudaKVCache(config, 1)
        q, k, v = _random_qkv(self.rng, 9, config)
        with self.assertRaises(ValueError):
            cache.append_and_attend(0, q, k, v, [0], [0], [9])
        with self.assertRaises(NotImplementedError):
            cache.prefix(0, 0, 1)
        cache.close()
        q, k, v = _random_qkv(self.rng, 1, config)
        with self.assertRaises(RuntimeError):
            cache.append_and_attend(0, q, k, v, [0], [0], [1])

    def test_model_greedy_with_cuda_attention_matches_no_cache_oracle(self):
        np.random.seed(4)
        model = GPT2(GPT2Config.tiny(n_positions=32), BACKEND)
        prompts = [[int(t) for t in self.rng.integers(0, 64, size=n)] for n in (2, 7, 4)]
        expected = [generate_greedy_no_cache(model, p, 5) for p in prompts]
        cache = CudaKVCache(model.config, max_slots=3)
        try:
            slots = [cache.allocate() for _ in prompts]
            logits = model.forward_ragged(prompts, [0, 0, 0], cache, slots, attention="cuda")
            generated = [[int(np.argmax(row))] for row in logits]
            for _ in range(4):
                starts = [cache.length(s) for s in slots]
                logits = model.forward_ragged([[g[-1]] for g in generated], starts, cache, slots,
                                              attention="cuda")
                for b, row in enumerate(logits):
                    generated[b].append(int(np.argmax(row)))
        finally:
            cache.close()
        self.assertEqual(generated, expected)

    def test_engine_cuda_attention_matches_oracle_under_all_schedulers(self):
        np.random.seed(4)
        model = GPT2(GPT2Config.tiny(n_positions=32), BACKEND)
        expected = {rid: generate_greedy_no_cache(model, prompt, OUTPUT_LENGTHS[rid])
                    for rid, prompt in PROMPTS.items()}
        for make in SCHEDULERS:
            engine = _engine_outputs(model, "cuda", make)
            with self.subTest(scheduler=type(make()).__name__):
                self.assertEqual(expected, engine.output_tokens)
                self.assertEqual(2, engine.cache.free_slots)
                self.assertEqual("device_cuda", engine.metadata["kv_storage"])
            engine.cache.close()


@unittest.skipUnless(cuda_kernel_available(), "ragged_attention.so not built or no GPU visible")
class PretrainedCudaKernelTests(unittest.TestCase):
    """GPT-2 small: CUDA attention gives the same logits as the padded host path."""

    @classmethod
    def setUpClass(cls):
        try:
            from model import hf_state_dict, load_hf_weights
            config, state = hf_state_dict("gpt2")
        except Exception as error:  # no transformers or no cached weights
            raise unittest.SkipTest(f"pretrained GPT-2 unavailable: {error}")
        cls.model = GPT2(config, BACKEND)
        load_hf_weights(cls.model, state)

    def test_mixed_batch_logits_match_padded_path(self):
        rng = np.random.default_rng(2)
        histories = [[], list(rng.integers(0, 50257, size=40)), list(rng.integers(0, 50257, size=17))]
        rows = [list(rng.integers(0, 50257, size=24)), [int(rng.integers(0, 50257))],
                list(rng.integers(0, 50257, size=9))]
        results = {}
        for attention, cache in (("padded", KVCache(self.model.config, 3)),
                                 ("cuda", CudaKVCache(self.model.config, 3))):
            slots = [cache.allocate() for _ in rows]
            for history, slot in zip(histories, slots):
                if history:
                    self.model.forward_ragged([history], [0], cache, [slot], attention=attention)
            results[attention] = self.model.forward_ragged(rows, [len(h) for h in histories], cache, slots,
                                                           attention=attention)
            if cache.on_device:
                cache.close()
        max_abs = float(np.max(np.abs(results["cuda"] - results["padded"])))
        self.assertLess(max_abs, 1e-2, f"max abs logit diff {max_abs}")
        np.testing.assert_array_equal(results["cuda"].argmax(-1), results["padded"].argmax(-1))


if __name__ == "__main__":
    unittest.main()
