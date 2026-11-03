"""Real numerical integration gates; tiny random weights, no network/GPU needed."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmark import BenchmarkConfig, run_one
from engine import EngineConfig, create_engine
from interfaces import BatchPlan, EventKind, Request, RequestState
from scheduler import FixedBudget, OrcaScheduler, SarathiScheduler, StaticScheduler
from sim import run_simulation


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("numpy", "numba"))
if HAS_RUNTIME:
    from engine.minitorch_engine import MiniTorchEngine
    from model import generate_greedy_no_cache


@unittest.skipUnless(HAS_RUNTIME, "install requirements-engine-cpu.txt for real-engine tests")
class MiniTorchEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_engine(EngineConfig(engine="minitorch", model_name="tiny", warmup_steps=0), 2).model

    def test_all_schedulers_match_independent_no_cache_outputs(self):
        prompts = {"a": [1, 2, 3, 4, 5, 6, 7], "b": [8, 9],
                   "c": [10, 11, 12, 13, 14], "d": [15, 16, 17]}
        lengths = {"a": 4, "b": 1, "c": 3, "d": 2}
        expected = {rid: generate_greedy_no_cache(self.model, prompt, lengths[rid])
                    for rid, prompt in prompts.items()}
        for scheduler in (StaticScheduler(2), OrcaScheduler(2), SarathiScheduler(2, FixedBudget(3))):
            with self.subTest(scheduler=type(scheduler).__name__):
                engine = MiniTorchEngine(self.model, 2, prompts)
                requests = [Request(rid, 0, len(prompt), lengths[rid]) for rid, prompt in prompts.items()]
                result = run_simulation(requests, scheduler, engine)
                self.assertEqual(expected, engine.output_tokens)
                self.assertEqual(2, engine.cache.free_slots)
                self.assertTrue(all(request.state == RequestState.FINISHED for request in requests))
                self.assertEqual(sorted(event.timestamp for event in result.events),
                                 [event.timestamp for event in result.events])
                for rid, output in expected.items():
                    indices = [event.token_index for event in result.events
                               if event.kind == EventKind.TOKEN and event.request_id == rid]
                    self.assertEqual(list(range(1, len(output) + 1)), indices)

    def test_chunk_sizes_one_nondivisor_and_full_match_whole_prefill(self):
        prompt = [1, 5, 9, 13, 17, 21, 25]
        expected = generate_greedy_no_cache(self.model, prompt, 3)
        for chunk in (1, 3, len(prompt)):
            engine = MiniTorchEngine(self.model, 1, {"a": prompt})
            engine.register(Request("a", 0, len(prompt), 3))
            for start in range(0, len(prompt), chunk):
                count = min(chunk, len(prompt) - start)
                result = engine.execute(BatchPlan([("a", start, count)]))
                self.assertEqual(("a",) if start + count == len(prompt) else (), result.emitted_tokens)
            for position in (len(prompt), len(prompt) + 1):
                result = engine.execute(BatchPlan([("a", position, 1)]))
            self.assertEqual(("a",), result.finished_requests)
            self.assertEqual(expected, engine.output_tokens["a"])
            self.assertEqual(1, engine.cache.free_slots)

    def test_equal_spans_batch_and_mixed_spans_use_separate_forwards(self):
        engine = MiniTorchEngine(self.model, 3, {"a": [1, 2], "b": [3, 4], "c": [5, 6, 7]})
        for rid, length in (("a", 2), ("b", 2), ("c", 3)):
            engine.register(Request(rid, 0, length, 3))
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            engine.execute(BatchPlan([("a", 0, 2), ("b", 0, 2)]))
            self.assertEqual(1, forward.call_count)
            self.assertEqual((2, 2), forward.call_args.args[0].shape)
            forward.reset_mock()
            engine.execute(BatchPlan([("a", 2, 1), ("b", 2, 1), ("c", 0, 3)], token_budget=5))
            self.assertEqual([(2, 1), (1, 3)], [call.args[0].shape for call in forward.call_args_list])

    def test_failed_later_forward_rolls_back_earlier_groups(self):
        engine = MiniTorchEngine(self.model, 3, {"a": [1, 2], "b": [3], "c": [5, 6, 7]})
        for rid, length, output in (("a", 2, 1), ("b", 1, 2), ("c", 3, 1)):
            engine.register(Request(rid, 0, length, output))
        engine.execute(BatchPlan([("b", 0, 1)]))
        lengths, calls = engine.cache.lengths.copy(), []
        original = self.model.forward

        def fail_second(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("injected")
            return original(*args)

        plan = BatchPlan([("a", 0, 2), ("b", 1, 1), ("c", 0, 3)])
        with patch.object(self.model, "forward", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                engine.execute(plan)
        self.assertEqual(lengths.tolist(), engine.cache.lengths.tolist())
        self.assertEqual(2, engine.cache.free_slots)
        self.assertEqual({"a": [], "b": engine.output_tokens["b"][:1], "c": []}, engine.output_tokens)
        result = engine.execute(plan)  # the same plan is still valid afterwards
        self.assertEqual(("a", "b", "c"), result.finished_requests)
        self.assertEqual(3, engine.cache.free_slots)
        for rid, prompt, output in (("a", [1, 2], 1), ("b", [3], 2), ("c", [5, 6, 7], 1)):
            self.assertEqual(generate_greedy_no_cache(self.model, prompt, output), engine.output_tokens[rid])

    def test_factory_reuses_loaded_model_with_fresh_cache(self):
        config = EngineConfig(engine="minitorch", model_name="tiny", warmup_steps=0)
        first, second = create_engine(config, 2), create_engine(config, 2)
        self.assertIs(first.model, second.model)
        self.assertIsNot(first.cache, second.cache)
        self.assertEqual(first.metadata["weights_sha256"], second.metadata["weights_sha256"])

    def test_invalid_plans_do_not_consume_slots_or_modify_cache(self):
        engine = MiniTorchEngine(self.model, 1, {"a": [1, 2, 3, 4], "b": [5, 6]})
        engine.register(Request("a", 0, 4, 1))
        engine.register(Request("b", 0, 2, 1))
        self.assertEqual(1, engine.cache.free_slots)  # registration allocates nothing
        for plan in (BatchPlan([("a", 1, 2)]), BatchPlan([("a", 0, 2), ("b", 0, 2)]),
                     BatchPlan([("a", 0, 2)], token_budget=1)):
            with self.assertRaises(ValueError):
                engine.execute(plan)
            self.assertEqual(1, engine.cache.free_slots)
        engine.execute(BatchPlan([("a", 0, 2)]))
        with self.assertRaises(ValueError):
            engine.execute(BatchPlan([("b", 0, 2)]))
        engine.execute(BatchPlan([("a", 2, 2)]))
        engine.execute(BatchPlan([("b", 0, 2)]))
        self.assertEqual(generate_greedy_no_cache(self.model, [5, 6], 1), engine.output_tokens["b"])
        with self.assertRaises(ValueError):
            engine.execute(BatchPlan([("a", 0, 1)]))

    def test_registration_rejects_bad_prompts_and_context_overflow(self):
        for prompt in ([1], [1, 64], [1, True]):
            with self.assertRaises(ValueError):
                MiniTorchEngine(self.model, 1, {"a": prompt}).register(Request("a", 0, 2, 1))
        with self.assertRaises(ValueError):
            MiniTorchEngine(self.model, 1, {}).register(Request("a", 0, 2, 1))
        with self.assertRaises(ValueError):
            MiniTorchEngine(self.model, 1).register(Request("a", 0, 31, 2))

    def test_warmup_releases_slots_and_keeps_real_generation_clean(self):
        sync_calls = []
        engine = MiniTorchEngine(self.model, 2, {"a": [1, 2]}, synchronize=lambda: sync_calls.append(1))
        engine.warmup(35)  # exercise context reset as well as slot release
        self.assertEqual(2, engine.cache.free_slots)
        self.assertEqual({}, engine.output_tokens)
        engine.register(Request("a", 0, 2, 1))
        with self.assertRaises(ValueError):
            engine.warmup(1)
        engine.execute(BatchPlan([("a", 0, 2)]))
        self.assertEqual(3, len(sync_calls))  # warmup, pre-execute, post-execute
        self.assertEqual(generate_greedy_no_cache(self.model, [1, 2], 1), engine.output_tokens["a"])

    def test_synthetic_prompts_do_not_depend_on_registration_order(self):
        engines = [MiniTorchEngine(self.model, 2, prompt_seed=23) for _ in range(2)]
        for engine, ids in zip(engines, (("a", "b"), ("b", "a"))):
            for rid in ids:
                engine.register(Request(rid, 0, 5, 1))
        self.assertEqual(engines[0].prompt_tokens, engines[1].prompt_tokens)

    def test_real_benchmark_archives_inputs_outputs_and_does_not_resume_mock_data(self):
        outputs = []
        with TemporaryDirectory() as directory:
            for scheduler in ("static", "orca", "sarathi"):
                config = BenchmarkConfig(scheduler, 1000, 0, 3, engine="minitorch", model_name="tiny",
                                         max_slots=2, token_budget=3 if scheduler == "sarathi" else None,
                                         prompt_max=5, output_max=3, max_total_tokens=32, warmup_steps=2)
                summary = run_one(config, directory)
                run_dir = Path(directory) / "runs" / config.run_id
                metadata = json.loads((run_dir / "metadata.json").read_text())
                prompts = json.loads((run_dir / "prompts.json").read_text())
                generated = json.loads((run_dir / "outputs.json").read_text())
                self.assertEqual(3, summary["completed_requests"])
                self.assertEqual("wall", summary["clock_mode"])
                self.assertEqual("random_smoke_only", metadata["weights"])
                self.assertEqual("grouped_by_input_length", summary["execution_mode"])
                self.assertEqual("synthetic_length_replay", summary["prompt_source"])
                for rid, tokens in generated.items():
                    self.assertEqual(generate_greedy_no_cache(self.model, prompts[rid], len(tokens)), tokens)
                outputs.append(generated)
            self.assertTrue(all(output == outputs[0] for output in outputs))
            path = run_dir / "summary.json"
            saved = json.loads(path.read_text())
            saved["stale_marker"] = True
            path.write_text(json.dumps(saved))
            self.assertNotIn("stale_marker", run_one(config, directory, resume=True))


if __name__ == "__main__":
    unittest.main()
