import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark import BenchmarkConfig, build_matrix, run_benchmarks


class BenchmarkTests(unittest.TestCase):
    def test_matrix_applies_budgets_only_to_sarathi(self):
        configs = build_matrix(
            schedulers=["static", "orca", "sarathi"],
            arrival_rates=[1, 2],
            seeds=[3, 4],
            token_budgets=[4, 8],
            num_requests=5,
        )
        self.assertEqual(16, len(configs))
        self.assertEqual(
            8,
            sum(config.scheduler == "sarathi" for config in configs),
        )
        self.assertTrue(all(
            (config.token_budget is not None) == (config.scheduler == "sarathi")
            for config in configs
        ))
        self.assertEqual(len(configs), len({config.run_id for config in configs}))

    def test_benchmark_writes_complete_run_and_aggregate_outputs(self):
        configs = [
            BenchmarkConfig("orca", 2, 0, 8, prompt_max=8, output_max=4),
            BenchmarkConfig(
                "sarathi", 2, 0, 8, token_budget=4, prompt_max=8, output_max=4
            ),
        ]
        with TemporaryDirectory() as directory:
            summaries = run_benchmarks(configs, directory)
            root = Path(directory)
            jsonl_rows = [
                json.loads(line)
                for line in (root / "summary.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            with (root / "summary.csv").open(encoding="utf-8", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            for config in configs:
                run_dir = root / "runs" / config.run_id
                self.assertTrue((run_dir / "config.json").exists())
                self.assertTrue((run_dir / "trace.json").exists())
                self.assertTrue((run_dir / "events.jsonl").exists())
                self.assertTrue((run_dir / "summary.json").exists())
                with (run_dir / "request_metrics.csv").open(
                    encoding="utf-8", newline=""
                ) as stream:
                    request_rows = list(csv.DictReader(stream))
                self.assertEqual(8, len(request_rows))
                self.assertIn("ttft", request_rows[0])
                self.assertIn("mean_tpot", request_rows[0])
                self.assertIn("e2e_latency", request_rows[0])
        self.assertEqual(2, len(summaries))
        self.assertEqual(summaries, jsonl_rows)
        self.assertEqual(2, len(csv_rows))
        self.assertTrue(all(summary["completed_requests"] == 8 for summary in summaries))

    def test_resume_reuses_completed_summary(self):
        config = BenchmarkConfig(
            "sarathi", 1, 5, 4, token_budget=4, prompt_max=4, output_max=2
        )
        with TemporaryDirectory() as directory:
            first = run_benchmarks([config], directory)
            summary_path = Path(directory) / "runs" / config.run_id / "summary.json"
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            saved["resume_marker"] = True
            summary_path.write_text(json.dumps(saved), encoding="utf-8")
            second = run_benchmarks([config], directory, resume=True)
        self.assertNotIn("resume_marker", first[0])
        self.assertTrue(second[0]["resume_marker"])

    def test_invalid_scheduler_budget_combinations_are_rejected(self):
        with self.assertRaises(ValueError):
            BenchmarkConfig("sarathi", 1, 0, 1)
        with self.assertRaises(ValueError):
            BenchmarkConfig("orca", 1, 0, 1, token_budget=4)


if __name__ == "__main__":
    unittest.main()
