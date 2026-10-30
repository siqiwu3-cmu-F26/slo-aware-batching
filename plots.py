"""Aggregate benchmark seeds and render publication-ready comparison plots."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Iterable


METRICS = {
    "requests_per_second": "Throughput (requests/s)",
    "tokens_per_second": "Throughput (tokens/s)",
    "ttft_p99": "P99 TTFT (s)",
    "token_tpot_p99": "P99 inter-token latency (s)",
    "slo_attainment_rate": "SLO attainment rate",
    "goodput": "Goodput (requests/s)",
}


@dataclass(frozen=True)
class SummaryRow:
    scheduler: str
    arrival_rate: float
    seed: int
    token_budget: int | None
    values: dict[str, float | None]

    @property
    def series(self) -> str:
        if self.scheduler == "sarathi" and self.token_budget is not None:
            return f"Sarathi (B={self.token_budget})"
        return self.scheduler.capitalize()


@dataclass(frozen=True)
class AggregatePoint:
    series: str
    x: float
    mean: float
    ci95: float
    samples: int


def _finite_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"row {row_number}: invalid {field}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return parsed


def load_summary(path: str | Path) -> list[SummaryRow]:
    """Load the aggregate CSV produced by benchmark.py."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"scheduler", "arrival_rate", "seed", "token_budget", *METRICS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"summary is missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row_number, row in enumerate(reader, 2):
            scheduler = row["scheduler"]
            if scheduler not in {"static", "orca", "sarathi"}:
                raise ValueError(f"row {row_number}: unsupported scheduler")
            budget_text = row["token_budget"].strip()
            token_budget = int(budget_text) if budget_text else None
            values = {
                metric: (
                    _finite_float(row[metric], metric, row_number)
                    if row[metric].strip() else None
                )
                for metric in METRICS
            }
            rows.append(SummaryRow(
                scheduler=scheduler,
                arrival_rate=_finite_float(
                    row["arrival_rate"], "arrival_rate", row_number
                ),
                seed=int(row["seed"]),
                token_budget=token_budget,
                values=values,
            ))
    if not rows:
        raise ValueError("summary contains no benchmark rows")
    return rows


def _aggregate(
    rows: Iterable[SummaryRow],
    metric: str,
    x_value: Callable[[SummaryRow], float],
    series_name: Callable[[SummaryRow], str],
) -> list[AggregatePoint]:
    if metric not in METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        value = row.values[metric]
        if value is not None:
            grouped.setdefault((series_name(row), x_value(row)), []).append(value)

    points = []
    for (series, x), values in grouped.items():
        ci95 = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        points.append(AggregatePoint(series, x, mean(values), ci95, len(values)))
    return sorted(points, key=lambda point: (_series_sort_key(point.series), point.x))


def aggregate_by_arrival_rate(
    rows: Iterable[SummaryRow], metric: str
) -> list[AggregatePoint]:
    return _aggregate(rows, metric, lambda row: row.arrival_rate, lambda row: row.series)


def aggregate_budget_ablation(
    rows: Iterable[SummaryRow], metric: str
) -> list[AggregatePoint]:
    sarathi = [
        row for row in rows
        if row.scheduler == "sarathi" and row.token_budget is not None
    ]
    return _aggregate(
        sarathi,
        metric,
        lambda row: float(row.token_budget),
        lambda row: f"Arrival rate {row.arrival_rate:g}",
    )


def _series_sort_key(series: str) -> tuple[int, float | str]:
    if series == "Static":
        return (0, 0)
    if series == "Orca":
        return (1, 0)
    if series.startswith("Sarathi (B="):
        return (2, float(series.removeprefix("Sarathi (B=").removesuffix(")")))
    if series.startswith("Arrival rate "):
        return (3, float(series.removeprefix("Arrival rate ")))
    return (4, series)


def _pyplot():
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "plot generation requires matplotlib; install it with "
            "'python3 -m pip install -r requirements-plots.txt'"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_series(ax, points: list[AggregatePoint]) -> None:
    markers = ("o", "s", "^", "D", "v", "P", "X")
    series_names = sorted({point.series for point in points}, key=_series_sort_key)
    for index, series in enumerate(series_names):
        selected = [point for point in points if point.series == series]
        ax.errorbar(
            [point.x for point in selected],
            [point.mean for point in selected],
            yerr=[point.ci95 for point in selected],
            marker=markers[index % len(markers)],
            linewidth=1.8,
            capsize=3,
            label=series,
        )
    ax.grid(True, alpha=0.25)


def _save_rate_plot(
    plt,
    points: list[AggregatePoint],
    metric: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _draw_series(ax, points)
    ax.set_xlabel("Arrival rate (requests/s)")
    ax.set_ylabel(METRICS[metric])
    ax.set_ylim(bottom=0)
    if metric == "slo_attainment_rate":
        ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _save_budget_plot(
    plt,
    rows: list[SummaryRow],
    output: Path,
) -> list[AggregatePoint]:
    metrics = (
        "requests_per_second",
        "ttft_p99",
        "token_tpot_p99",
        "goodput",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    all_points = []
    for ax, metric in zip(axes.flat, metrics):
        points = aggregate_budget_ablation(rows, metric)
        all_points.extend(points)
        _draw_series(ax, points)
        ax.set_xlabel("Token budget")
        ax.set_ylabel(METRICS[metric])
        ax.set_ylim(bottom=0)
    if all_points:
        axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return all_points


def _write_aggregates(path: Path, rows: Iterable[tuple[str, AggregatePoint]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ("chart", "series", "x", "mean", "ci95", "samples")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for chart, point in rows:
            writer.writerow({
                "chart": chart,
                "series": point.series,
                "x": point.x,
                "mean": point.mean,
                "ci95": point.ci95,
                "samples": point.samples,
            })


def generate_plots(
    summary_path: str | Path,
    output_dir: str | Path,
    image_format: str = "png",
) -> list[Path]:
    if image_format not in {"png", "pdf", "svg"}:
        raise ValueError("image_format must be png, pdf, or svg")
    rows = load_summary(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()

    charts = (
        ("throughput_requests", "requests_per_second"),
        ("throughput_tokens", "tokens_per_second"),
        ("p99_ttft", "ttft_p99"),
        ("p99_tpot", "token_tpot_p99"),
        ("slo_attainment", "slo_attainment_rate"),
        ("goodput", "goodput"),
    )
    outputs = []
    aggregates: list[tuple[str, AggregatePoint]] = []
    for chart, metric in charts:
        points = aggregate_by_arrival_rate(rows, metric)
        if not points:
            continue
        output = output_dir / f"{chart}.{image_format}"
        _save_rate_plot(plt, points, metric, output)
        outputs.append(output)
        aggregates.extend((chart, point) for point in points)

    if any(row.scheduler == "sarathi" for row in rows):
        output = output_dir / f"budget_ablation.{image_format}"
        points = _save_budget_plot(plt, rows, output)
        if points:
            outputs.append(output)
            aggregates.extend(("budget_ablation", point) for point in points)

    _write_aggregates(output_dir / "aggregated_plot_data.csv", aggregates)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("benchmark-results/summary.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    args = parser.parse_args()
    outputs = generate_plots(args.summary, args.output_dir, args.format)
    print(f"Wrote {len(outputs)} plots and aggregated data to {args.output_dir}")


if __name__ == "__main__":
    main()
