"""Run reproducible benchmark matrices against the current serving engines."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from engine import MockEngine
from interfaces import Request, Scheduler
from metrics import compute_metrics
from scheduler import FixedBudget, OrcaScheduler, SarathiScheduler, StaticScheduler
from sim import run_simulation
from workload import generate_sharegpt_trace, generate_synthetic_trace, write_trace


@dataclass(frozen=True)
class BenchmarkConfig:
    scheduler: str
    arrival_rate: float
    seed: int
    num_requests: int
    arrival_mode: str = "poisson"
    token_budget: int | None = None
    max_slots: int = 4
    max_queue_size: int | None = None
    ttft_slo: float | None = None
    tpot_slo: float | None = None
    engine_a: float = 0.01
    engine_b: float = 0.001
    source: str = "synthetic"
    dataset: str | None = None
    prompt_median: float = 256
    prompt_sigma: float = 1.0
    prompt_min: int = 1
    prompt_max: int = 2048
    output_median: float = 64
    output_sigma: float = 0.8
    output_min: int = 1
    output_max: int = 512
    burst_factor: float = 5.0
    burst_fraction: float = 0.1
    burst_period: float = 10.0

    def __post_init__(self) -> None:
        if self.scheduler not in {"static", "orca", "sarathi"}:
            raise ValueError("unsupported scheduler")
        if self.scheduler == "sarathi" and self.token_budget is None:
            raise ValueError("Sarathi requires a token budget")
        if self.scheduler != "sarathi" and self.token_budget is not None:
            raise ValueError("only Sarathi uses a token budget")
        if self.source not in {"synthetic", "sharegpt"}:
            raise ValueError("source must be synthetic or sharegpt")
        if self.source == "sharegpt" and self.dataset is None:
            raise ValueError("ShareGPT source requires a dataset")

    @property
    def run_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def build_matrix(
    schedulers: Iterable[str],
    arrival_rates: Iterable[float],
    seeds: Iterable[int],
    token_budgets: Iterable[int],
    **common: Any,
) -> list[BenchmarkConfig]:
    configs = []
    budgets = list(token_budgets)
    for arrival_rate in arrival_rates:
        for seed in seeds:
            for scheduler in schedulers:
                scheduler_budgets = budgets if scheduler == "sarathi" else [None]
                for token_budget in scheduler_budgets:
                    configs.append(BenchmarkConfig(
                        scheduler=scheduler,
                        arrival_rate=arrival_rate,
                        seed=seed,
                        token_budget=token_budget,
                        **common,
                    ))
    return configs


def _generate_rows(config: BenchmarkConfig) -> list[dict[str, Any]]:
    common = dict(
        count=config.num_requests,
        arrival_rate=config.arrival_rate,
        seed=config.seed,
        arrival_mode=config.arrival_mode,
        ttft_slo=config.ttft_slo,
        tpot_slo=config.tpot_slo,
        burst_factor=config.burst_factor,
        burst_fraction=config.burst_fraction,
        burst_period=config.burst_period,
    )
    if config.source == "sharegpt":
        return generate_sharegpt_trace(config.dataset, **common)
    return generate_synthetic_trace(
        **common,
        prompt_median=config.prompt_median,
        prompt_sigma=config.prompt_sigma,
        prompt_min=config.prompt_min,
        prompt_max=config.prompt_max,
        output_median=config.output_median,
        output_sigma=config.output_sigma,
        output_min=config.output_min,
        output_max=config.output_max,
    )


def _make_scheduler(config: BenchmarkConfig) -> Scheduler:
    if config.scheduler == "static":
        return StaticScheduler(config.max_slots, config.max_queue_size)
    if config.scheduler == "orca":
        return OrcaScheduler(config.max_slots, config.max_queue_size)
    assert config.token_budget is not None
    return SarathiScheduler(
        config.max_slots, FixedBudget(config.token_budget), config.max_queue_size
    )


def _summary(config: BenchmarkConfig, metrics: dict[str, Any], result: Any) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        **asdict(config),
        **metrics,
        "steps": result.steps,
        "deferred_requests": result.deferred_requests,
        "deferred_decisions": result.deferred_decisions,
    }


def run_one(config: BenchmarkConfig, output_dir: str | Path, resume: bool = True) -> dict[str, Any]:
    run_dir = Path(output_dir) / "runs" / config.run_id
    summary_path = run_dir / "summary.json"
    if resume and summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("run_id") != config.run_id:
            raise ValueError(f"invalid saved summary in {run_dir}")
        return saved

    run_dir.mkdir(parents=True, exist_ok=True)
    rows = _generate_rows(config)
    requests = [Request(**row) for row in rows]
    scheduler = _make_scheduler(config)
    engine = MockEngine(config.max_slots, config.engine_a, config.engine_b)

    write_trace(rows, run_dir / "trace.json")
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    result = run_simulation(requests, scheduler, engine, run_dir / "events.jsonl")
    metrics = compute_metrics(requests, result.events).to_dict()
    summary = _summary(config, metrics, result)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_aggregate(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for summary in summaries:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")

    fieldnames = list(summaries[0]) if summaries else []
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(summaries)


def run_benchmarks(
    configs: Iterable[BenchmarkConfig], output_dir: str | Path, resume: bool = True
) -> list[dict[str, Any]]:
    configs = list(configs)
    run_ids = [config.run_id for config in configs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("benchmark matrix contains duplicate configurations")

    output_dir = Path(output_dir)
    summaries = []
    for index, config in enumerate(configs, 1):
        print(
            f"[{index}/{len(configs)}] {config.scheduler} "
            f"rate={config.arrival_rate:g} seed={config.seed} "
            f"budget={config.token_budget if config.token_budget is not None else '-'}"
        )
        summaries.append(run_one(config, output_dir, resume))
    _write_aggregate(summaries, output_dir)
    return summaries


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedulers", nargs="+", choices=("static", "orca", "sarathi"),
                        default=("static", "orca", "sarathi"))
    parser.add_argument("--arrival-rates", nargs="+", type=_positive_float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--token-budgets", nargs="+", type=_positive_int, default=(16, 32, 64))
    parser.add_argument("--num-requests", type=_positive_int, default=1000)
    parser.add_argument("--arrival-mode", choices=("poisson", "bursty"), default="poisson")
    parser.add_argument("--max-slots", type=_positive_int, default=4)
    parser.add_argument("--max-queue-size", type=_positive_int)
    parser.add_argument("--ttft-slo", type=_positive_float)
    parser.add_argument("--tpot-slo", type=_positive_float)
    parser.add_argument("--engine-a", type=float, default=0.01)
    parser.add_argument("--engine-b", type=float, default=0.001)
    parser.add_argument("--source", choices=("synthetic", "sharegpt"), default="synthetic")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prompt-median", type=_positive_float, default=256)
    parser.add_argument("--prompt-sigma", type=float, default=1.0)
    parser.add_argument("--prompt-min", type=_positive_int, default=1)
    parser.add_argument("--prompt-max", type=_positive_int, default=2048)
    parser.add_argument("--output-median", type=_positive_float, default=64)
    parser.add_argument("--output-sigma", type=float, default=0.8)
    parser.add_argument("--output-min", type=_positive_int, default=1)
    parser.add_argument("--output-max", type=_positive_int, default=512)
    parser.add_argument("--burst-factor", type=_positive_float, default=5.0)
    parser.add_argument("--burst-fraction", type=float, default=0.1)
    parser.add_argument("--burst-period", type=_positive_float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--no-resume", action="store_true",
                        help="rerun configurations that already have a summary")
    args = parser.parse_args()

    if args.source == "sharegpt" and args.dataset is None:
        parser.error("--dataset is required when --source=sharegpt")
    if any(budget < args.max_slots for budget in args.token_budgets):
        parser.error("every token budget must be at least --max-slots")

    common = dict(
        num_requests=args.num_requests,
        arrival_mode=args.arrival_mode,
        max_slots=args.max_slots,
        max_queue_size=args.max_queue_size,
        ttft_slo=args.ttft_slo,
        tpot_slo=args.tpot_slo,
        engine_a=args.engine_a,
        engine_b=args.engine_b,
        source=args.source,
        dataset=str(args.dataset) if args.dataset is not None else None,
        prompt_median=args.prompt_median,
        prompt_sigma=args.prompt_sigma,
        prompt_min=args.prompt_min,
        prompt_max=args.prompt_max,
        output_median=args.output_median,
        output_sigma=args.output_sigma,
        output_min=args.output_min,
        output_max=args.output_max,
        burst_factor=args.burst_factor,
        burst_fraction=args.burst_fraction,
        burst_period=args.burst_period,
    )
    configs = build_matrix(
        args.schedulers, args.arrival_rates, args.seeds, args.token_budgets, **common
    )
    summaries = run_benchmarks(configs, args.output_dir, resume=not args.no_resume)
    print(f"Wrote {len(summaries)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
