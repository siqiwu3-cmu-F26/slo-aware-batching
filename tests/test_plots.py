import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from plots import (
    METRICS,
    aggregate_budget_ablation,
    aggregate_by_arrival_rate,
    generate_plots,
    load_summary,
)


class PlotTests(unittest.TestCase):
    def write_summary(self, path: Path) -> None:
        fieldnames = ["scheduler", "arrival_rate", "seed", "token_budget", *METRICS]
        rows = [
            ("static", 1, 0, "", 2),
            ("static", 1, 1, "", 4),
            ("sarathi", 1, 0, 4, 4),
            ("sarathi", 1, 1, 4, 6),
            ("sarathi", 1, 0, 8, 5),
            ("sarathi", 1, 1, 8, 7),
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for scheduler, rate, seed, budget, throughput in rows:
                writer.writerow({
                    "scheduler": scheduler,
                    "arrival_rate": rate,
                    "seed": seed,
                    "token_budget": budget,
                    "requests_per_second": throughput,
                    "tokens_per_second": throughput * 10,
                    "ttft_p99": throughput / 10,
                    "token_tpot_p99": throughput / 100,
                    "slo_attainment_rate": throughput / 10,
                    "goodput": throughput / 2,
                })

    def test_loads_and_aggregates_seed_mean_and_ci(self):
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.csv"
            self.write_summary(summary)
            rows = load_summary(summary)
        points = aggregate_by_arrival_rate(rows, "requests_per_second")
        static = next(point for point in points if point.series == "Static")
        self.assertEqual(3, static.mean)
        self.assertAlmostEqual(1.96, static.ci95)
        self.assertEqual(2, static.samples)
        self.assertEqual(
            ["Sarathi (B=4)", "Sarathi (B=8)"],
            sorted({
                point.series
                for point in points
                if point.series.startswith("Sarathi")
            }),
        )

    def test_budget_ablation_uses_budget_as_x_axis(self):
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.csv"
            self.write_summary(summary)
            rows = load_summary(summary)
        points = aggregate_budget_ablation(rows, "goodput")
        self.assertEqual([4, 8], [point.x for point in points])
        self.assertTrue(all(point.series == "Arrival rate 1" for point in points))
        self.assertEqual([2.5, 3], [point.mean for point in points])

    def test_rejects_summary_with_missing_columns(self):
        with TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.csv"
            summary.write_text("scheduler,arrival_rate\norca,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_summary(summary)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib is an optional dependency"
    )
    def test_renders_all_expected_plots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            output = root / "figures"
            self.write_summary(summary)
            paths = generate_plots(summary, output, "png")
            self.assertEqual(7, len(paths))
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertTrue((output / "aggregated_plot_data.csv").exists())


if __name__ == "__main__":
    unittest.main()
