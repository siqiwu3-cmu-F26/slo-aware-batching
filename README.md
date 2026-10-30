# slo-aware-batching

GPT-2 serving scheduler for MiniTorch. Goal: maximize requests completed within
latency targets.

Static, Orca, and Sarathi scheduling currently run on a mock engine.
The real MiniTorch engine and Stage 4 SLO-aware adaptation are pending.

## Structure

```text
├── interfaces.py       # Request, batch plan, engine API, events
├── scheduler/          # Static / Orca / Sarathi + budget policies
├── engine/             # Mock engine; real engine plugs in here
├── workload.py         # Synthetic and ShareGPT trace generation
├── benchmark.py        # Reproducible experiment matrix runner
├── sim.py              # Virtual-clock simulation
├── metrics.py          # Temporary latency and throughput metrics
├── tests/              # Scheduler tests
└── docs/proposal.md    # Project proposal
```

## Run

Python 3.10+, no extra dependencies. Run from the repository root:

```bash
python3 sim.py --scheduler sarathi --max-slots 4 --token-budget 32
python3 -m unittest discover -s tests -v
```

Use `--scheduler static` or `--scheduler orca` to compare baselines.
Simulation prints metrics and saves events to `events.jsonl`.
Use `python3 sim.py --help` for other options.

Generate a deterministic synthetic Poisson trace and run it with:

```bash
python3 workload.py synthetic --num-requests 1000 --arrival-rate 5 \
  --seed 0 --output trace.json
python3 sim.py --trace trace.json --scheduler sarathi
```

Use `--arrival-mode bursty` for periodic load bursts. To sample token lengths
from a ShareGPT JSON dataset, install `tiktoken` and run:

```bash
python3 -m pip install tiktoken
python3 workload.py sharegpt --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-requests 1000 --arrival-rate 5 --seed 0 --output trace.json
```

Run `python3 workload.py synthetic --help` or
`python3 workload.py sharegpt --help` for length, SLO, and burst parameters.

Run a benchmark matrix with:

```bash
python3 benchmark.py --arrival-rates 1 2 4 8 --seeds 0 1 2 \
  --token-budgets 16 32 64 --num-requests 1000 \
  --ttft-slo 2 --tpot-slo 0.1 --output-dir benchmark-results
```

Each configuration gets a deterministic run ID and its own config, trace,
event log, and summary under `benchmark-results/runs/`. Aggregate
`summary.jsonl` and `summary.csv` files are written at the top level. Completed
runs are reused by default; pass `--no-resume` to rerun them.

## Key interface assumptions

- Only the engine reads `output_len`; it reports tokens and completion to the scheduler.
- Memory is capped by `max_slots` (concurrent requests), not sequence length.
- Final prefill emits the first output token; each later decode emits one token.
- Sarathi's minimum token budget must be at least `max_slots` to fit all decodes.
- Times use seconds. TPOT is the mean interval between a request's output tokens.

Full interface assumptions are in [interfaces.py](interfaces.py).
