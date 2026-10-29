"""Scheduler invariants on deterministic traces; no real engine/harness required."""

from dataclasses import dataclass
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from engine import MockEngine
from interfaces import BatchPlan, EventKind, Request, RequestState, StepResult
from metrics import compute_metrics
from scheduler import FixedBudget, OrcaScheduler, SarathiScheduler, StaticScheduler
from scheduler.budget import BudgetContext, BudgetController
from sim import run_simulation


class CheckingEngine(MockEngine):
    def __init__(self, max_slots: int, scheduler, **kwargs):
        super().__init__(max_slots, **kwargs)
        self.scheduler = scheduler
        self.plans: list[BatchPlan] = []

    def execute(self, plan: BatchPlan) -> StepResult:
        assert len(self.scheduler.active) <= self.max_slots
        assert len(plan.items) <= self.max_slots
        assert all(count > 0 for _, _, count in plan.items)
        if plan.token_budget is not None:
            assert plan.total_tokens <= plan.token_budget
            decoding = {rid for rid, request in self.scheduler.active.items()
                        if request.state == RequestState.DECODING}
            assert decoding <= {rid for rid, _, count in plan.items if count == 1}
        self.plans.append(plan)
        return super().execute(plan)


class EngineOnlyLengthRequest(Request):
    hide_output = False

    def __getattribute__(self, name):
        if name == "output_len" and self.hide_output:
            raise AssertionError("scheduler read engine-only output_len")
        return super().__getattribute__(name)


class SchedulerTests(unittest.TestCase):
    def schedulers(self, slots=3):
        return [StaticScheduler(slots), OrcaScheduler(slots),
                SarathiScheduler(slots, FixedBudget(slots))]

    def test_finite_trace_finishes_without_starvation_or_capacity_violation(self):
        # Budgets equal to slots stress decode reservation and prefill progress.
        for seed in range(5):
            rng = random.Random(seed)
            trace = [(str(i), rng.randrange(12) / 100, rng.randrange(1, 80), rng.randrange(1, 15))
                     for i in range(35)]
            for scheduler in self.schedulers():
                with self.subTest(seed=seed, scheduler=type(scheduler).__name__):
                    requests = [Request(*row) for row in trace]
                    engine = CheckingEngine(3, scheduler)
                    result = run_simulation(requests, scheduler, engine)
                    self.assertTrue(all(request.state == RequestState.FINISHED for request in requests))
                    self.assertEqual(len(trace), sum(event.kind == EventKind.FINISH for event in result.events))
                    self.assertEqual(sum(row[3] for row in trace),
                                     sum(event.kind == EventKind.TOKEN for event in result.events))
                    self.assertEqual(0, result.rejected_requests)
                    self.assertEqual(sorted(event.timestamp for event in result.events),
                                     [event.timestamp for event in result.events])

    def test_scheduler_never_reads_output_length(self):
        for scheduler in self.schedulers(2):
            engine = MockEngine(2)
            requests = [EngineOnlyLengthRequest("a", 0, 7, 3),
                        EngineOnlyLengthRequest("b", 0, 2, 1)]
            for request in requests:
                engine.register(request)
                request.hide_output = True
                scheduler.submit(request)
            now = 0.0
            while scheduler.has_pending:
                plan = scheduler.plan(now)
                result = engine.execute(plan)
                now += result.duration
                scheduler.update(plan, result, now)
            self.assertEqual([3, 1], [request.generated_tokens for request in requests])

    def test_orca_beats_static_on_heterogeneous_trace(self):
        def run(cls):
            # Two long requests overlap under Orca, but occupy separate static batches.
            requests = [Request("long1", 0, 1, 20), Request("short", 0, 1, 1),
                        Request("long2", 0, 1, 20)]
            result = run_simulation(requests, cls(2), MockEngine(2, a=1, b=0))
            return result, compute_metrics(requests, result.events)
        static, static_metrics = run(StaticScheduler)
        orca, orca_metrics = run(OrcaScheduler)
        self.assertEqual(40, static.end_time)
        self.assertEqual(21, orca.end_time)
        self.assertGreater(orca_metrics.requests_per_second, static_metrics.requests_per_second)
        self.assertLess(orca_metrics.ttft_p99, static_metrics.ttft_p99)

    def test_static_membership_is_frozen_until_batch_drains(self):
        scheduler = StaticScheduler(2)
        engine = CheckingEngine(2, scheduler, a=1, b=0)
        run_simulation([Request("long", 0, 1, 3), Request("short", 0, 1, 1),
                        Request("next", 0, 1, 1)], scheduler, engine)
        self.assertEqual([["long", "short"], ["long"], ["long"], ["next"]],
                         [[rid for rid, _, _ in plan.items] for plan in engine.plans])

    def test_sarathi_decodes_first_and_packs_prefill_fcfs(self):
        scheduler = SarathiScheduler(3, FixedBudget(5))
        engine = CheckingEngine(3, scheduler)
        run_simulation([Request("a", 0, 2, 3), Request("b", 0, 8, 2),
                        Request("c", 0, 1, 1)], scheduler, engine)
        self.assertEqual([("a", 0, 2), ("b", 0, 3)], engine.plans[0].items)
        self.assertEqual([("a", 2, 1), ("b", 3, 4)], engine.plans[1].items)
        self.assertEqual([("a", 3, 1), ("b", 7, 1), ("c", 0, 1)], engine.plans[2].items)
        self.assertEqual(1, scheduler.deferred_requests)
        self.assertEqual(2, scheduler.deferred_decisions)

    def test_arrivals_during_step_keep_timestamps_and_wait_for_next_plan(self):
        scheduler = OrcaScheduler(2)
        engine = CheckingEngine(2, scheduler, a=1, b=0)
        requests = [Request("a", 0, 1, 2), Request("b", 0.25, 1, 1)]
        result = run_simulation(requests, scheduler, engine)
        self.assertEqual([("a", 0, 1)], engine.plans[0].items)
        self.assertIn(("b", 0, 1), engine.plans[1].items)
        self.assertEqual(0.25, next(event.timestamp for event in result.events
                                    if event.kind == EventKind.ARRIVAL and event.request_id == "b"))
        self.assertEqual(2, next(event.timestamp for event in result.events
                                if event.kind == EventKind.FIRST_TOKEN and event.request_id == "b"))

    def test_rejection_and_deferral_counters_and_jsonl(self):
        scheduler = OrcaScheduler(1, max_queue_size=2)
        requests = [Request(str(i), 0, 1, 2) for i in range(4)]
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            result = run_simulation(requests, scheduler, MockEngine(1), log_path)
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual(2, result.rejected_requests)
        self.assertEqual(1, result.deferred_requests)
        self.assertEqual(2, result.deferred_decisions)
        self.assertEqual(2, sum(row["kind"] == "reject" for row in rows))
        self.assertEqual(2, compute_metrics(requests, result.events).completed_requests)
        self.assertEqual(0.5, compute_metrics(requests, result.events).rejection_rate)

    def test_metrics_known_latencies_slos_and_one_token_output(self):
        requests = [Request("a", 2, 1, 3, ttft_slo=1, tpot_slo=1),
                    Request("b", 2, 1, 1, ttft_slo=0.5, tpot_slo=0.5)]
        result = run_simulation(requests, OrcaScheduler(2), MockEngine(2, a=1, b=0))
        metrics = compute_metrics(requests, result.events)
        self.assertEqual(3, metrics.duration)
        self.assertEqual(1, metrics.ttft_p50)
        self.assertEqual(1, metrics.ttft_p99)
        self.assertEqual(0.5, metrics.tpot_p50)
        self.assertAlmostEqual(0.99, metrics.tpot_p99)
        self.assertAlmostEqual(2 / 3, metrics.requests_per_second)
        self.assertAlmostEqual(4 / 3, metrics.tokens_per_second)
        self.assertAlmostEqual(1 / 3, metrics.goodput)

    def test_empty_trace_and_idle_clock_jump(self):
        result = run_simulation([], OrcaScheduler(1), MockEngine(1))
        self.assertEqual([], result.events)
        metrics = compute_metrics([], result.events)
        self.assertIsNone(metrics.ttft_p99)
        self.assertEqual(0, metrics.goodput)
        requests = [Request("later", 100, 1, 1)]
        result = run_simulation(requests, OrcaScheduler(1), MockEngine(1, a=1, b=0))
        self.assertEqual(101, result.end_time)
        self.assertEqual(1, compute_metrics(requests, result.events).duration)

    def test_duplicate_ids_and_invalid_limits_fail_early(self):
        with self.assertRaises(ValueError):
            run_simulation([Request("a", 0, 1, 1), Request("a", 1, 1, 1)],
                           OrcaScheduler(1), MockEngine(1))
        with self.assertRaises(ValueError):
            run_simulation([], OrcaScheduler(2), MockEngine(1))
        for policy in (FixedBudget(1), FixedBudget(5, min_budget=1, max_budget=8)):
            with self.assertRaises(ValueError):
                SarathiScheduler(2, policy)
        for factory in (lambda: OrcaScheduler(0), lambda: FixedBudget(0),
                        lambda: FixedBudget(5, 6, 10), lambda: FixedBudget(5, 1, 10, -1),
                        lambda: Request("bad", 0, 0, 1), lambda: MockEngine(1, 0, 0)):
            with self.assertRaises(ValueError):
                factory()

    def test_mock_rejects_bad_plans_without_partial_execution(self):
        engine = MockEngine(1)
        engine.register(Request("a", 0, 4, 2))
        engine.register(Request("b", 0, 4, 2))
        with self.assertRaises(ValueError):
            engine.execute(BatchPlan([("a", 1, 2)]))
        engine.execute(BatchPlan([("a", 0, 2)]))
        with self.assertRaises(ValueError):
            engine.execute(BatchPlan([("b", 0, 1)]))  # a still occupies a slot
        with self.assertRaises(ValueError):
            engine.execute(BatchPlan([("a", 2, 2)], token_budget=1))
        result = engine.execute(BatchPlan([("a", 2, 2)]))
        self.assertEqual(("a",), result.emitted_tokens)
        self.assertEqual((), result.finished_requests)


class BudgetTests(unittest.TestCase):
    def test_custom_policy_is_clamped_and_rate_limited(self):
        @dataclass
        class Policy:
            min_budget: int = 3
            max_budget: int = 10
            max_change_per_step: int = 2
            desired: int = 100

            def propose(self, context: BudgetContext) -> int:
                return self.desired

        policy = Policy()
        controller = BudgetController(policy, max_slots=3)
        context = BudgetContext(0, (), ())
        self.assertEqual(10, controller.choose(context))
        policy.desired = -100
        self.assertEqual([8, 6, 4, 3], [controller.choose(context) for _ in range(4)])
        policy.desired = 100
        self.assertEqual(5, controller.choose(context))

    def test_fixed_budget_stays_fixed_with_wider_bounds(self):
        controller = BudgetController(FixedBudget(7, 3, 10, 2), 3)
        self.assertEqual([7] * 5, [controller.choose(BudgetContext(i, (), ())) for i in range(5)])


if __name__ == "__main__":
    unittest.main()
