"""Small virtual-clock driver; replace MockEngine with an Engine adapter later."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from engine import MockEngine
from interfaces import Engine, Event, EventKind, Request, RequestState, Scheduler
from metrics import compute_metrics
from scheduler import FixedBudget, OrcaScheduler, SarathiScheduler, StaticScheduler


@dataclass
class SimulationResult:
    events: list[Event]
    end_time: float
    steps: int
    rejected_requests: int
    deferred_requests: int
    deferred_decisions: int


def run_simulation(requests: Iterable[Request], scheduler: Scheduler, engine: Engine,
                   log_path: str | Path | None = None) -> SimulationResult:
    # Assumption: ties preserve input order, establishing deterministic FCFS.
    requests = sorted(requests, key=lambda request: request.arrival_time)
    if scheduler.max_slots != engine.max_slots:
        raise ValueError("scheduler and engine must agree on max_slots")
    if len({request.id for request in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    if scheduler.has_pending or any(request.state != RequestState.WAITING
                                    or request.prefilled_tokens or request.generated_tokens
                                    for request in requests):
        raise ValueError("simulation requires fresh requests and an empty scheduler")
    events: list[Event] = []
    index = 0
    now = 0.0
    steps = 0

    def feed_arrivals(until: float) -> None:
        nonlocal index
        while index < len(requests) and requests[index].arrival_time <= until:
            request = requests[index]
            events.append(Event(EventKind.ARRIVAL, request.id, request.arrival_time))
            if scheduler.submit(request):
                engine.register(request)
            else:
                events.append(Event(EventKind.REJECT, request.id, request.arrival_time,
                                    reason="waiting_queue_full"))
            index += 1

    while index < len(requests) or scheduler.has_pending:
        if not scheduler.has_pending:
            now = max(now, requests[index].arrival_time)
        feed_arrivals(now)
        if not scheduler.has_pending:
            continue
        plan = scheduler.plan(now)
        if not plan.items:
            raise RuntimeError("scheduler has pending work but produced an empty plan")
        result = engine.execute(plan)
        if not math.isfinite(result.duration) or result.duration <= 0:
            raise ValueError("engine duration must be finite and positive")
        now += result.duration
        # Assumption: steps cannot be preempted. Arrivals during execution retain
        # their exact timestamps but can only be admitted on the next iteration.
        # Feed before recording completions to keep the event log chronological.
        feed_arrivals(now)
        events.extend(scheduler.update(plan, result, now))
        steps += 1
    if log_path is not None:
        with Path(log_path).open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(asdict(event)) + "\n")
    return SimulationResult(events, now, steps, scheduler.rejected_requests,
                            scheduler.deferred_requests, scheduler.deferred_decisions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=("static", "orca", "sarathi"), default="sarathi")
    parser.add_argument("--trace", type=Path, help="JSON array of Request constructor fields")
    parser.add_argument("--log", type=Path, default=Path("events.jsonl"))
    parser.add_argument("--max-slots", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=32)
    parser.add_argument("--min-budget", type=int)
    parser.add_argument("--max-budget", type=int)
    parser.add_argument("--max-change-per-step", type=int, default=0)
    parser.add_argument("--max-queue-size", type=int)
    parser.add_argument("--a", type=float, default=0.01, help="fixed step overhead in seconds")
    parser.add_argument("--b", type=float, default=0.001, help="seconds per planned input token")
    args = parser.parse_args()
    # Only a tiny smoke fixture; workload generation/benchmarking belong to the team.
    if args.trace is None:
        requests = [Request("long", 0, 48, 12), Request("short", 0, 8, 2),
                    Request("later", 0.03, 16, 4)]
    else:
        requests = [Request(**row) for row in json.loads(args.trace.read_text(encoding="utf-8"))]
    engine = MockEngine(args.max_slots, args.a, args.b)
    scheduler: Scheduler
    if args.scheduler == "sarathi":
        policy = FixedBudget(args.token_budget, args.min_budget, args.max_budget,
                             args.max_change_per_step)
        scheduler = SarathiScheduler(args.max_slots, policy, args.max_queue_size)
    else:
        scheduler_class = StaticScheduler if args.scheduler == "static" else OrcaScheduler
        scheduler = scheduler_class(args.max_slots, args.max_queue_size)
    result = run_simulation(requests, scheduler, engine, args.log)
    summary = compute_metrics(requests, result.events).to_dict()
    summary.update(steps=result.steps, deferred_requests=result.deferred_requests,
                   deferred_decisions=result.deferred_decisions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
