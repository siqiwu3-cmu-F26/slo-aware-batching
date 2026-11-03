"""Trace driver with virtual mock timing or wall-clock MiniTorch execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Callable, Iterable, cast

if TYPE_CHECKING:
    from engine.minitorch_engine import MiniTorchEngine

from engine import create_engine
from engine.factory import add_engine_arguments, config_from_args
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
    clock_mode: str = "virtual"


def run_simulation(requests: Iterable[Request], scheduler: Scheduler, engine: Engine,
                   log_path: str | Path | None = None, *, clock_mode: str = "virtual",
                   clock: Callable[[], float] = perf_counter,
                   wait: Callable[[float], None] = sleep) -> SimulationResult:
    if clock_mode not in {"virtual", "wall"}:
        raise ValueError("clock_mode must be virtual or wall")
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
    # Assumption: model/trace loading and warmup happen before this clock origin.
    # Wall mode includes scheduling, registration, host work, and execution;
    # StepResult.duration remains diagnostic and is never added a second time.
    origin = clock() if clock_mode == "wall" else 0.0

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
        if clock_mode == "wall":
            now = clock() - origin
        if not scheduler.has_pending:
            if clock_mode == "wall":
                remaining = requests[index].arrival_time - now
                if remaining > 0:
                    wait(remaining)
                now = clock() - origin
            else:
                now = max(now, requests[index].arrival_time)
        feed_arrivals(now)
        if not scheduler.has_pending:
            continue
        plan = scheduler.plan(clock() - origin if clock_mode == "wall" else now)
        if not plan.items:
            raise RuntimeError("scheduler has pending work but produced an empty plan")
        result = engine.execute(plan)
        if not math.isfinite(result.duration) or result.duration <= 0:
            raise ValueError("engine duration must be finite and positive")
        now = clock() - origin if clock_mode == "wall" else now + result.duration
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
                            scheduler.deferred_requests, scheduler.deferred_decisions, clock_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_arguments(parser)
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
        if args.engine == "minitorch" and args.model_name == "tiny":
            requests = [Request("long", 0, 7, 4), Request("short", 0, 2, 1),
                        Request("later", 0.03, 5, 2)]
        else:
            requests = [Request("long", 0, 48, 12), Request("short", 0, 8, 2),
                        Request("later", 0.03, 16, 4)]
    else:
        requests = [Request(**row) for row in json.loads(args.trace.read_text(encoding="utf-8"))]
    clock_mode = args.clock_mode or ("wall" if args.engine == "minitorch" else "virtual")
    if args.engine == "mock" and clock_mode != "virtual":
        parser.error("mock engine requires --clock virtual")
    # Validate flags before the potentially slow model load and warmup.
    engine = create_engine(config_from_args(args), args.max_slots, args.a, args.b)
    scheduler: Scheduler
    if args.scheduler == "sarathi":
        policy = FixedBudget(args.token_budget, args.min_budget, args.max_budget,
                             args.max_change_per_step)
        scheduler = SarathiScheduler(args.max_slots, policy, args.max_queue_size)
    else:
        scheduler_class = StaticScheduler if args.scheduler == "static" else OrcaScheduler
        scheduler = scheduler_class(args.max_slots, args.max_queue_size)
    result = run_simulation(requests, scheduler, engine, args.log, clock_mode=clock_mode)
    summary = compute_metrics(requests, result.events).to_dict()
    summary.update(steps=result.steps, deferred_requests=result.deferred_requests,
                   deferred_decisions=result.deferred_decisions, engine=args.engine,
                   clock_mode=result.clock_mode)
    if args.engine == "minitorch":
        real_engine = cast("MiniTorchEngine", engine)
        # Preserve actual tokens and execution limitations next to the event log.
        artifact = {"metadata": real_engine.metadata, "prompts": real_engine.prompt_tokens,
                    "outputs": real_engine.output_tokens, "forward_calls": real_engine.forward_calls}
        args.log.with_suffix(".engine.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
