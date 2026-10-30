"""Generate deterministic synthetic or ShareGPT-style serving traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Callable, Iterable

from interfaces import Request


TraceRow = dict[str, str | int | float | None]


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


def generate_arrival_times(
    count: int,
    rate: float,
    rng: random.Random,
    mode: str = "poisson",
    burst_factor: float = 5.0,
    burst_fraction: float = 0.1,
    burst_period: float = 10.0,
) -> list[float]:
    """Return arrival times for a Poisson or periodic bursty Poisson process.

    The bursty process spends ``burst_fraction`` of each period at
    ``rate * burst_factor`` and chooses its low rate so the long-run mean is
    ``rate``. A randomized initial phase avoids starting every trace in a burst.
    """
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    if mode not in {"poisson", "bursty"}:
        raise ValueError("mode must be 'poisson' or 'bursty'")
    if count == 0:
        return []

    if mode == "poisson":
        now = 0.0
        arrivals = []
        for _ in range(count):
            now += rng.expovariate(rate)
            arrivals.append(now)
        return arrivals

    parameters = (burst_factor, burst_fraction, burst_period)
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("burst parameters must be finite")
    if burst_factor <= 1 or not 0 < burst_fraction < 1 or burst_period <= 0:
        raise ValueError("invalid burst parameters")
    if burst_factor * burst_fraction >= 1:
        raise ValueError("burst_factor * burst_fraction must be less than 1")

    high_rate = rate * burst_factor
    low_rate = rate * (1 - burst_factor * burst_fraction) / (1 - burst_fraction)
    phase_offset = rng.random() * burst_period
    now = 0.0
    arrivals = []
    # Thinning a process at high_rate gives the desired piecewise Poisson rate.
    while len(arrivals) < count:
        now += rng.expovariate(high_rate)
        phase = (now + phase_offset) % burst_period
        current_rate = high_rate if phase < burst_period * burst_fraction else low_rate
        if rng.random() <= current_rate / high_rate:
            arrivals.append(now)
    return arrivals


def _bounded_lognormal(
    rng: random.Random, median: float, sigma: float, minimum: int, maximum: int
) -> int:
    if not math.isfinite(median) or median <= 0:
        raise ValueError("median must be finite and positive")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum:
        raise ValueError("length bounds must be positive integers with min <= max")
    sampled = round(rng.lognormvariate(math.log(median), sigma))
    return max(minimum, min(maximum, sampled))


def _trace_rows(
    lengths: Iterable[tuple[int, int]],
    arrivals: Iterable[float],
    ttft_slo: float | None,
    tpot_slo: float | None,
) -> list[TraceRow]:
    rows = []
    for index, ((prompt_len, output_len), arrival_time) in enumerate(zip(lengths, arrivals)):
        row: TraceRow = {
            "id": f"request-{index:06d}",
            "arrival_time": arrival_time,
            "prompt_len": prompt_len,
            "output_len": output_len,
        }
        if ttft_slo is not None:
            row["ttft_slo"] = ttft_slo
        if tpot_slo is not None:
            row["tpot_slo"] = tpot_slo
        # Constructing Request keeps generated traces aligned with the simulator.
        Request(**row)
        rows.append(row)
    return rows


def generate_synthetic_trace(
    count: int,
    arrival_rate: float,
    seed: int,
    arrival_mode: str = "poisson",
    prompt_median: float = 256,
    prompt_sigma: float = 1.0,
    prompt_min: int = 1,
    prompt_max: int = 2048,
    output_median: float = 64,
    output_sigma: float = 0.8,
    output_min: int = 1,
    output_max: int = 512,
    ttft_slo: float | None = None,
    tpot_slo: float | None = None,
    burst_factor: float = 5.0,
    burst_fraction: float = 0.1,
    burst_period: float = 10.0,
) -> list[TraceRow]:
    rng = random.Random(seed)
    arrivals = generate_arrival_times(
        count, arrival_rate, rng, arrival_mode, burst_factor, burst_fraction, burst_period
    )
    lengths = [
        (
            _bounded_lognormal(rng, prompt_median, prompt_sigma, prompt_min, prompt_max),
            _bounded_lognormal(rng, output_median, output_sigma, output_min, output_max),
        )
        for _ in range(count)
    ]
    return _trace_rows(lengths, arrivals, ttft_slo, tpot_slo)


def extract_sharegpt_lengths(
    records: Iterable[object], count_tokens: Callable[[str], int]
) -> list[tuple[int, int]]:
    """Extract the first adjacent user/assistant pair from each conversation."""
    pairs = []
    for record in records:
        if not isinstance(record, dict):
            continue
        messages = record.get("conversations", record.get("messages"))
        if not isinstance(messages, list):
            continue
        for user, assistant in zip(messages, messages[1:]):
            if not isinstance(user, dict) or not isinstance(assistant, dict):
                continue
            user_role = user.get("from", user.get("role"))
            assistant_role = assistant.get("from", assistant.get("role"))
            prompt = user.get("value", user.get("content"))
            output = assistant.get("value", assistant.get("content"))
            if (
                user_role in {"human", "user"}
                and assistant_role in {"gpt", "assistant"}
                and isinstance(prompt, str)
                and isinstance(output, str)
                and prompt
                and output
            ):
                prompt_len, output_len = count_tokens(prompt), count_tokens(output)
                if prompt_len > 0 and output_len > 0:
                    pairs.append((prompt_len, output_len))
                break
    return pairs


def _gpt2_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "ShareGPT generation requires tiktoken; install it with "
            "'python3 -m pip install tiktoken'"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def generate_sharegpt_trace(
    dataset_path: str | Path,
    count: int,
    arrival_rate: float,
    seed: int,
    arrival_mode: str = "poisson",
    ttft_slo: float | None = None,
    tpot_slo: float | None = None,
    burst_factor: float = 5.0,
    burst_fraction: float = 0.1,
    burst_period: float = 10.0,
    count_tokens: Callable[[str], int] | None = None,
) -> list[TraceRow]:
    path = Path(dataset_path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("ShareGPT dataset must be a JSON array")
    pairs = extract_sharegpt_lengths(records, count_tokens or _gpt2_token_counter())
    if not pairs:
        raise ValueError("ShareGPT dataset contains no valid user/assistant pairs")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    lengths = [pairs[index % len(pairs)] for index in range(count)]
    arrivals = generate_arrival_times(
        count, arrival_rate, rng, arrival_mode, burst_factor, burst_fraction, burst_period
    )
    return _trace_rows(lengths, arrivals, ttft_slo, tpot_slo)


def write_trace(rows: list[TraceRow], output: str | Path) -> None:
    Path(output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-requests", type=_positive_int, required=True)
    parser.add_argument("--arrival-rate", type=_positive_float, required=True,
                        help="long-run mean requests per second")
    parser.add_argument("--arrival-mode", choices=("poisson", "bursty"), default="poisson")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ttft-slo", type=_positive_float)
    parser.add_argument("--tpot-slo", type=_positive_float)
    parser.add_argument("--burst-factor", type=_positive_float, default=5.0)
    parser.add_argument("--burst-fraction", type=float, default=0.1)
    parser.add_argument("--burst-period", type=_positive_float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source", required=True)
    synthetic = subparsers.add_parser("synthetic", help="sample bounded lognormal lengths")
    _add_common_arguments(synthetic)
    synthetic.add_argument("--prompt-median", type=_positive_float, default=256)
    synthetic.add_argument("--prompt-sigma", type=float, default=1.0)
    synthetic.add_argument("--prompt-min", type=_positive_int, default=1)
    synthetic.add_argument("--prompt-max", type=_positive_int, default=2048)
    synthetic.add_argument("--output-median", type=_positive_float, default=64)
    synthetic.add_argument("--output-sigma", type=float, default=0.8)
    synthetic.add_argument("--output-min", type=_positive_int, default=1)
    synthetic.add_argument("--output-max", type=_positive_int, default=512)

    sharegpt = subparsers.add_parser("sharegpt", help="sample lengths from ShareGPT JSON")
    _add_common_arguments(sharegpt)
    sharegpt.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    common = dict(
        count=args.num_requests,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
        arrival_mode=args.arrival_mode,
        ttft_slo=args.ttft_slo,
        tpot_slo=args.tpot_slo,
        burst_factor=args.burst_factor,
        burst_fraction=args.burst_fraction,
        burst_period=args.burst_period,
    )
    if args.source == "synthetic":
        rows = generate_synthetic_trace(
            **common,
            prompt_median=args.prompt_median,
            prompt_sigma=args.prompt_sigma,
            prompt_min=args.prompt_min,
            prompt_max=args.prompt_max,
            output_median=args.output_median,
            output_sigma=args.output_sigma,
            output_min=args.output_min,
            output_max=args.output_max,
        )
    else:
        rows = generate_sharegpt_trace(args.dataset, **common)
    write_trace(rows, args.output)
    print(f"Wrote {len(rows)} requests to {args.output}")


if __name__ == "__main__":
    main()
