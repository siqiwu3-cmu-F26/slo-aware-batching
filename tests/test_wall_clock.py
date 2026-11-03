"""Wall-clock replay includes scheduler work and arrivals during execution."""

import unittest

from interfaces import EventKind, Request, StepResult
from scheduler import OrcaScheduler
from sim import run_simulation


class WallClockTests(unittest.TestCase):
    def test_scheduler_overhead_idle_wait_and_execution_are_counted_once(self):
        class Time:
            now = 10.0

            def clock(self):
                return self.now

            def advance(self, seconds):
                self.now += seconds

        timer = Time()

        class Scheduler(OrcaScheduler):
            def plan(self, now):
                timer.advance(0.2)
                return super().plan(now)

        class Engine:
            max_slots = 1

            def register(self, request):
                pass

            def execute(self, plan):
                timer.advance(0.7)
                ids = tuple(rid for rid, _, _ in plan.items)
                return StepResult(0.7, ids, ids)

        result = run_simulation([Request("a", 5, 1, 1), Request("b", 5.5, 1, 1)],
                                Scheduler(1), Engine(), clock_mode="wall",
                                clock=timer.clock, wait=timer.advance)
        first = {event.request_id: event.timestamp for event in result.events
                 if event.kind == EventKind.FIRST_TOKEN}
        self.assertAlmostEqual(5.9, first["a"])
        self.assertAlmostEqual(6.8, first["b"])
        self.assertAlmostEqual(6.8, result.end_time)
        self.assertEqual("wall", result.clock_mode)
        self.assertEqual(sorted(event.timestamp for event in result.events),
                         [event.timestamp for event in result.events])


if __name__ == "__main__":
    unittest.main()
