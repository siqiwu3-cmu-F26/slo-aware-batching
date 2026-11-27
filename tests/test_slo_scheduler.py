"""Stage 4 SLO-aware scheduler, its step-time model, and the P99 TPOT SLO metric."""

import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from engine import MockEngine
from interfaces import Event, EventKind, Request, RequestState
from metrics import compute_metrics, compute_request_metrics
from scheduler import (FixedBudget, OrcaScheduler, SarathiScheduler, SloAwareBudget,
                       SloAwareScheduler, StaticScheduler, StepTimeModel)
from scheduler.budget import BudgetContext, BudgetController
from sim import run_simulation
from tests.test_scheduler import CheckingEngine, EngineOnlyLengthRequest

A, B = 0.01, 0.001     # mock step cost: A + B * tokens seconds


def _trace(seed, count=60, rate=40.0, ttft_slo=0.5, tpot_slo=0.03):
    """Heterogeneous Poisson trace that overloads 3 slots, so every path is exercised."""
    rng = random.Random(seed)
    now, rows = 0.0, []
    for i in range(count):
        now += rng.expovariate(rate)
        rows.append(Request(str(i), now, rng.randrange(1, 200), rng.randrange(1, 30),
                            ttft_slo=ttft_slo, tpot_slo=tpot_slo))
    return rows


def _gaps(events):
    times = {}
    for event in events:
        if event.kind == EventKind.TOKEN:
            times.setdefault(event.request_id, []).append(event.timestamp)
    return {rid: [b - a for a, b in zip(t, t[1:])] for rid, t in times.items()}


class StepTimeModelTests(unittest.TestCase):
    def test_recovers_exact_line_after_two_token_counts(self):
        model = StepTimeModel()
        self.assertFalse(model.ready)
        with self.assertRaises(RuntimeError):
            model.predict(1)
        for tokens in (4, 4, 36, 100):
            model.observe(tokens, A + B * tokens)
        self.assertTrue(model.fitted)
        self.assertAlmostEqual(A, model.a)
        self.assertAlmostEqual(B, model.b)
        self.assertEqual(12, model.max_tokens_within(A + B * 12 + 1e-12))
        self.assertEqual(-1, model.max_tokens_within(A / 2))
        # 300 tokens in chunks of 256: two steps' fixed cost plus every token.
        self.assertAlmostEqual(2 * A + 300 * B, model.prefill_time(300, 256))

    def test_single_token_count_is_an_upper_bound(self):
        model = StepTimeModel()
        model.observe(10, A + 10 * B)
        self.assertTrue(model.ready)
        self.assertFalse(model.fitted)
        for tokens in (1, 5, 10, 50, 500):
            self.assertGreaterEqual(model.predict(tokens) + 1e-12, A + B * tokens)
            fits = model.max_tokens_within(A + B * tokens)
            if fits >= 1:
                self.assertLessEqual(model.predict(fits), A + B * tokens + 1e-12)
        with self.assertRaises(RuntimeError):
            model.prefill_time(10, 4)    # no lower bound without a fitted line

    def test_rejects_bad_observations(self):
        model = StepTimeModel()
        for tokens, seconds in ((0, 1.0), (1, 0.0), (1, math.inf), (1.5, 1.0)):
            with self.assertRaises(ValueError):
                model.observe(tokens, seconds)


class SloAwareBudgetTests(unittest.TestCase):
    def setUp(self):
        self.model = StepTimeModel()
        for tokens in (4, 64):
            self.model.observe(tokens, A + B * tokens)
        self.policy = SloAwareBudget(min_budget=4, max_budget=256)

    def context(self, deadlines, now=1.0, model=True):
        return BudgetContext(now, (), (), tuple(deadlines), self.model if model else None)

    def test_decodes_get_min_budget_until_step_time_is_fitted(self):
        early = StepTimeModel()
        early.observe(128, A + 128 * B)
        context = BudgetContext(1.0, (), (), (1.03,), early)
        self.assertEqual(4, self.policy.propose(context))
        self.assertEqual(256, self.policy.propose(BudgetContext(1.0, (), (), (), early)))

    def test_largest_budget_that_meets_the_tightest_decode_deadline(self):
        self.assertEqual(256, self.policy.propose(self.context([])))
        self.assertEqual(256, self.policy.propose(self.context([1.0], model=False)))
        # 1.03 - 1.0 = 30 ms allows 20 tokens; the 50 ms deadline allows 40.
        self.assertEqual(20, self.policy.propose(self.context([1.05, 1.03 + 1e-12])))

    def test_unreachable_deadline_does_not_shrink_the_budget(self):
        # Even the minimum budget (14 ms) misses a 5 ms deadline, so it is ignored.
        self.assertEqual(40, self.policy.propose(self.context([1.005, 1.05 + 1e-12])))

    def test_controller_can_change_budget_every_step(self):
        controller = BudgetController(self.policy, max_slots=4)
        chosen = [controller.choose(self.context(d)) for d in ([], [1.03 + 1e-12], [])]
        self.assertEqual([256, 20, 256], chosen)


class SloAwareSchedulerTests(unittest.TestCase):
    def run_trace(self, scheduler, requests, slots=3):
        engine = CheckingEngine(slots, scheduler, a=A, b=B)
        return run_simulation(requests, scheduler, engine), engine

    def test_invariants_on_overloaded_traces_for_every_ablation(self):
        variants = (dict(), dict(order="fcfs"), dict(reject=False), dict(tpot_target="token"),
                    dict(budget_policy=FixedBudget(64), max_budget=64))
        for seed in range(3):
            for options in variants:
                with self.subTest(seed=seed, options=options):
                    requests = _trace(seed)
                    scheduler = SloAwareScheduler(3, max_budget=options.pop("max_budget", 64), **options)
                    result, engine = self.run_trace(scheduler, requests)
                    terminal = [event for event in result.events
                                if event.kind in (EventKind.FINISH, EventKind.REJECT)]
                    self.assertEqual(sorted(r.id for r in requests), sorted(e.request_id for e in terminal))
                    self.assertTrue(all(r.state in (RequestState.FINISHED, RequestState.REJECTED)
                                        for r in requests))
                    self.assertEqual([e.timestamp for e in result.events],
                                     sorted(e.timestamp for e in result.events))
                    self.assertEqual(result.rejected_requests,
                                     sum(e.kind == EventKind.REJECT for e in result.events))
                    # Rejection only ever drops requests that never received work.
                    scheduled = {rid for plan in engine.plans for rid, _, _ in plan.items}
                    rejected = {e.request_id for e in result.events if e.kind == EventKind.REJECT}
                    self.assertFalse(rejected & scheduled)
                    if not scheduler.reject:
                        self.assertEqual(0, result.rejected_requests)

    def test_rejected_requests_could_not_have_met_their_ttft(self):
        requests = _trace(7, count=120, rate=80.0)
        result, _ = self.run_trace(SloAwareScheduler(3, max_budget=64), requests)
        by_id = {request.id: request for request in requests}
        rejections = [e for e in result.events if e.kind == EventKind.REJECT]
        self.assertGreater(len(rejections), 0)
        for event in rejections:
            request = by_id[event.request_id]
            self.assertEqual("ttft_slo_unreachable", event.reason)
            fastest = math.ceil(request.prompt_len / 64) * A + B * request.prompt_len
            self.assertGreater(event.timestamp + fastest, request.arrival_time + request.ttft_slo)

    def test_token_target_keeps_every_gap_within_tpot_slo(self):
        # Feasible SLO: a step with only the 3 decodes takes 13 ms < 30 ms.
        requests = _trace(1)
        result, engine = self.run_trace(SloAwareScheduler(3, max_budget=128, tpot_target="token"), requests)
        gaps = [gap for values in _gaps(result.events).values() for gap in values]
        self.assertGreater(len(gaps), 100)
        self.assertLessEqual(max(gaps), 0.03 + 1e-9)
        # The budget really adapted: tight while decoding, wide when not.
        budgets = {plan.token_budget for plan in engine.plans}
        self.assertIn(128, budgets)
        self.assertLess(min(budgets), 128)

    def test_mean_target_keeps_every_completed_mean_tpot_within_slo(self):
        requests = _trace(2)
        result, _ = self.run_trace(SloAwareScheduler(3, max_budget=128), requests)
        rows = [row for row in compute_request_metrics(requests, result.events) if row.status == "completed"]
        self.assertGreater(len(rows), 10)
        self.assertTrue(all(row.meets_tpot_slo for row in rows))

    def test_beats_sarathi_goodput_when_overloaded(self):
        def goodput(scheduler):
            requests = _trace(3, count=150, rate=25.0)
            result, _ = self.run_trace(scheduler, requests)
            return compute_metrics(requests, result.events).goodput
        self.assertGreater(goodput(SloAwareScheduler(3, max_budget=128)),
                           goodput(SarathiScheduler(3, FixedBudget(128))))

    def test_slack_order_admits_feasible_requests_before_hopeless_ones(self):
        scheduler = SloAwareScheduler(1, max_budget=8)
        for tokens in (4, 8):
            scheduler.step_time.observe(tokens, A + B * tokens)
        # "late" can no longer meet its TTFT; "ok" can. FCFS would admit "late".
        late = Request("late", 0.0, 40, 1, ttft_slo=0.1)
        ok = Request("ok", 0.05, 8, 1, ttft_slo=1.0)
        scheduler.reject = False
        for request in (late, ok):
            scheduler.submit(request)
        plan = scheduler.plan(0.1)
        self.assertEqual([("ok", 0, 8)], plan.items)
        self.assertEqual(["late"], [request.id for request in scheduler.waiting])

    def test_never_reads_output_length(self):
        scheduler = SloAwareScheduler(2, max_budget=8)
        engine = MockEngine(2, a=A, b=B)
        requests = [EngineOnlyLengthRequest("a", 0, 7, 3, ttft_slo=1, tpot_slo=1),
                    EngineOnlyLengthRequest("b", 0, 2, 1, ttft_slo=1, tpot_slo=1),
                    EngineOnlyLengthRequest("c", 0, 9, 2, ttft_slo=1, tpot_slo=1)]
        for request in requests:
            engine.register(request)      # the engine may read output_len; the scheduler may not
            request.hide_output = True
            scheduler.submit(request)
        now = 0.0
        while scheduler.has_pending:
            self.assertEqual([], scheduler.expire(now))
            plan = scheduler.plan(now)
            result = engine.execute(plan)
            now += result.duration
            scheduler.update(plan, result, now)
        self.assertEqual([3, 1, 2], [request.generated_tokens for request in requests])

    def test_baselines_never_expire_requests(self):
        for scheduler in (StaticScheduler(2), OrcaScheduler(2), SarathiScheduler(2, FixedBudget(2))):
            scheduler.submit(Request("a", 0, 5, 1, ttft_slo=1e-6))
            self.assertEqual([], scheduler.expire(100.0))
            self.assertEqual(1, len(scheduler.waiting))

    def test_rejects_unknown_options(self):
        with self.assertRaises(ValueError):
            SloAwareScheduler(2, order="lifo")
        with self.assertRaises(ValueError):
            SloAwareScheduler(2, tpot_target="p50")
        with self.assertRaises(ValueError):
            SloAwareScheduler(4, max_budget=2)     # budget below the decode reservation


class P99TpotMetricTests(unittest.TestCase):
    def test_per_request_p99_gap_and_strict_goodput(self):
        requests = [Request("bursty", 0, 1, 4, ttft_slo=1, tpot_slo=0.5),
                    Request("steady", 0, 1, 3, ttft_slo=1, tpot_slo=0.5),
                    Request("single", 0, 1, 1, ttft_slo=1, tpot_slo=0.5)]
        token_times = {"bursty": [0.1, 0.2, 0.3, 1.3], "steady": [0.1, 0.3, 0.5], "single": [0.1]}
        events = [Event(EventKind.ARRIVAL, rid, 0.0) for rid in token_times]
        for rid, times in token_times.items():
            events.append(Event(EventKind.FIRST_TOKEN, rid, times[0], 1))
            events += [Event(EventKind.TOKEN, rid, t, i) for i, t in enumerate(times, 1)]
            events.append(Event(EventKind.FINISH, rid, times[-1]))
        events.sort(key=lambda event: event.timestamp)
        rows = {row.request_id: row for row in compute_request_metrics(requests, events)}
        # Gaps 0.1, 0.1, 1.0: the mean (0.4) passes, the P99 (0.982) does not.
        self.assertAlmostEqual(0.4, rows["bursty"].mean_tpot)
        self.assertAlmostEqual(0.982, rows["bursty"].p99_tpot)
        self.assertEqual((True, False), (rows["bursty"].meets_slo, rows["bursty"].meets_slo_p99_tpot))
        self.assertAlmostEqual(0.2, rows["steady"].p99_tpot)
        self.assertTrue(rows["steady"].meets_slo_p99_tpot)
        self.assertEqual(0.0, rows["single"].p99_tpot)
        self.assertTrue(rows["single"].meets_slo_p99_tpot)
        metrics = compute_metrics(requests, events)
        self.assertAlmostEqual(1.0, metrics.slo_attainment_rate)
        self.assertAlmostEqual(2 / 3, metrics.slo_attainment_rate_p99_tpot)
        self.assertAlmostEqual(2 / 1.3, metrics.goodput_p99_tpot)


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_slo_scheduler_runs_through_benchmark_and_plots(self):
        from benchmark import BenchmarkConfig, run_benchmarks
        from plots import load_summary
        common = dict(arrival_rate=4.0, seed=0, num_requests=30, ttft_slo=2.0, tpot_slo=0.05,
                      prompt_median=64, output_median=8)
        configs = [BenchmarkConfig(scheduler="slo", slo_max_budget=64, **common),
                   BenchmarkConfig(scheduler="slo", slo_max_budget=64, slo_reject=False,
                                   slo_order="fcfs", slo_adaptive_budget=False, **common),
                   BenchmarkConfig(scheduler="sarathi", token_budget=64, **common)]
        self.assertEqual(3, len({config.run_id for config in configs}))
        with TemporaryDirectory() as directory:
            summaries = run_benchmarks(configs, directory)
            for summary in summaries:
                self.assertEqual(30, summary["completed_requests"] + summary["rejected_requests"])
                self.assertIn("goodput_p99_tpot", summary)
                self.assertLessEqual(summary["slo_attainment_rate_p99_tpot"], summary["slo_attainment_rate"])
            rows = load_summary(Path(directory) / "summary.csv")
            self.assertEqual(["SLO-aware", "SLO-aware", "Sarathi (B=64)"], [row.series for row in rows])
        for bad in (dict(token_budget=64), dict(slo_max_budget=2), dict(slo_order="lifo"),
                    dict(slo_tpot_target="p50")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BenchmarkConfig(scheduler="slo", **common, **bad)


if __name__ == "__main__":
    unittest.main()
