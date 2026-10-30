import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from workload import (
    extract_sharegpt_lengths,
    generate_arrival_times,
    generate_sharegpt_trace,
    generate_synthetic_trace,
    write_trace,
)


class WorkloadTests(unittest.TestCase):
    def test_synthetic_trace_is_deterministic_and_simulator_compatible(self):
        kwargs = dict(
            count=20,
            arrival_rate=4,
            seed=17,
            prompt_min=8,
            prompt_max=64,
            output_min=2,
            output_max=16,
            ttft_slo=1.5,
            tpot_slo=0.2,
        )
        first = generate_synthetic_trace(**kwargs)
        second = generate_synthetic_trace(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(20, len(first))
        self.assertEqual(20, len({row["id"] for row in first}))
        self.assertEqual(
            sorted(row["arrival_time"] for row in first),
            [row["arrival_time"] for row in first],
        )
        self.assertTrue(all(8 <= row["prompt_len"] <= 64 for row in first))
        self.assertTrue(all(2 <= row["output_len"] <= 16 for row in first))
        self.assertTrue(all(
            row["prompt_len"] + row["output_len"] <= 1024 for row in first
        ))
        self.assertTrue(all(row["ttft_slo"] == 1.5 for row in first))

    def test_context_limit_filters_or_rejects_lengths(self):
        rows = generate_synthetic_trace(
            count=20,
            arrival_rate=1,
            seed=4,
            prompt_median=8,
            prompt_min=4,
            prompt_max=16,
            output_median=4,
            output_min=2,
            output_max=8,
            max_total_tokens=10,
        )
        self.assertTrue(all(
            row["prompt_len"] + row["output_len"] <= 10 for row in rows
        ))
        with self.assertRaises(ValueError):
            generate_synthetic_trace(
                count=1,
                arrival_rate=1,
                seed=0,
                prompt_min=8,
                output_min=4,
                max_total_tokens=10,
            )

    def test_poisson_and_bursty_arrivals_are_reproducible(self):
        for mode in ("poisson", "bursty"):
            first = generate_arrival_times(100, 5, random.Random(3), mode)
            second = generate_arrival_times(100, 5, random.Random(3), mode)
            self.assertEqual(first, second)
            self.assertEqual(sorted(first), first)
            self.assertTrue(all(time > 0 for time in first))

    def test_invalid_burst_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_arrival_times(
                10, 2, random.Random(0), "bursty", burst_factor=5, burst_fraction=0.2
            )

    def test_extracts_first_valid_sharegpt_pair_in_both_common_schemas(self):
        records = [
            {
                "conversations": [
                    {"from": "system", "value": "ignored"},
                    {"from": "human", "value": "one two"},
                    {"from": "gpt", "value": "three"},
                    {"from": "human", "value": "later turn"},
                    {"from": "gpt", "value": "not selected"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "four"},
                    {"role": "assistant", "content": "five six"},
                ]
            },
            {"conversations": [{"from": "human", "value": "missing response"}]},
        ]
        self.assertEqual(
            [(2, 1), (1, 2)],
            extract_sharegpt_lengths(records, lambda text: len(text.split())),
        )

    def test_sharegpt_sampling_and_json_output(self):
        dataset = [
            {
                "conversations": [
                    {"from": "human", "value": "a b"},
                    {"from": "gpt", "value": "c"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "d"},
                    {"from": "gpt", "value": "e f g"},
                ]
            },
        ]
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "sharegpt.json"
            trace_path = Path(directory) / "trace.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            rows = generate_sharegpt_trace(
                dataset_path,
                count=5,
                arrival_rate=2,
                seed=9,
                count_tokens=lambda text: len(text.split()),
            )
            write_trace(rows, trace_path)
            loaded = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(rows, loaded)
        self.assertEqual(5, len(rows))
        self.assertEqual({(2, 1), (1, 3)}, {
            (row["prompt_len"], row["output_len"]) for row in rows
        })

        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "sharegpt.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            filtered = generate_sharegpt_trace(
                dataset_path,
                count=3,
                arrival_rate=2,
                seed=9,
                max_total_tokens=3,
                count_tokens=lambda text: len(text.split()),
            )
        self.assertEqual(
            {(2, 1)},
            {(row["prompt_len"], row["output_len"]) for row in filtered},
        )


if __name__ == "__main__":
    unittest.main()
