# slo-aware-batching

GPT-2 serving scheduler for MiniTorch. Goal: maximize requests completed within
latency targets.

Static, Orca, and Sarathi run with either the mock engine or a MiniTorch adapter.
The adapter currently groups inputs by length; the mixed-batch kernel and
Stage 4 SLO-aware adaptation are pending.

## Structure

```text
├── interfaces.py       # Request, batch plan, engine API, events
├── scheduler/          # Static / Orca / Sarathi + budget policies
├── engine/             # Mock engine + MiniTorch adapter and loading
├── minitorch/          # Vendored MiniTorch tensor runtime (CPU numba + CUDA backends)
├── kernels/            # CUDA sources for MiniTorch's fused kernels + build script
├── model/              # GPT-2 decoder in MiniTorch + HuggingFace weight loader
├── workload.py         # Synthetic and ShareGPT trace generation
├── benchmark.py        # Reproducible experiment matrix runner
├── plots.py            # Multi-seed aggregation and paper-ready plots
├── sim.py              # Virtual-clock simulation
├── metrics.py          # Per-request and aggregate serving metrics
├── tests/              # Scheduler tests
├── docs/proposal.md    # Project proposal
└── docs/experiment_plan.md # Fixed evaluation protocol
```

## Run

Python 3.11. The scheduler, mock engine, and benchmark harness need no extra
dependencies. Run from the repository root:

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
  --max-total-tokens 1024 --seed 0 --output trace.json
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
The generator rejects samples whose prompt plus output exceeds the configured
model context limit (1024 tokens by default).

Run a benchmark matrix with:

```bash
python3 benchmark.py --arrival-rates 1 2 4 8 --seeds 0 1 2 \
  --token-budgets 16 32 64 --num-requests 1000 \
  --ttft-slo 2 --tpot-slo 0.1 --output-dir benchmark-results
```

Each configuration gets a deterministic run ID and its own config, trace,
event log, per-request metrics CSV, and summary under `benchmark-results/runs/`.
Aggregate `summary.jsonl` and `summary.csv` files are written at the top level.
Completed runs are reused by default; pass `--no-resume` to rerun them.

Generate plots from the benchmark summary with:

```bash
python3 -m pip install -r requirements-plots.txt
python3 plots.py --summary benchmark-results/summary.csv \
  --output-dir figures --format png
```

The plotting pipeline reports the mean and 95% confidence interval across
seeds. It writes throughput, P99 latency, SLO attainment, goodput, and token
budget ablation figures, plus `aggregated_plot_data.csv` for auditing.

## MiniTorch integration

CPU smoke test with a tiny random-weight model (no downloads or training):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-engine-cpu.txt
NUMBA_NUM_THREADS=1 .venv/bin/python sim.py --engine minitorch --model tiny \
  --scheduler sarathi --max-slots 2 --token-budget 4
NUMBA_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

For pretrained GPT-2, install `requirements-engine.txt` and use `--model gpt2`
or `--model gpt2-medium`. CUDA additionally requires `pycuda` and
`./kernels/compile_cuda.sh`; select it with `--backend cuda`.

Both `sim.py` and `benchmark.py` accept `--engine minitorch`. Model loading and
20 warmup steps precede wall-clock measurement. `--prompts prompts.json` supplies
a mapping of request IDs to token-ID lists; otherwise inputs are deterministic
synthetic tokens replaying the trace's lengths. The adapter generates exactly
`output_len` tokens and ignores EOS. Actual inputs, outputs, and execution
metadata are saved alongside results.

Unequal input lengths use separate forwards; this path has CPU correctness
coverage and still needs GPU validation and the optimized mixed-batch kernel.
The KV cache is host-resident numpy (`kv_storage: host_numpy` in metadata), so
`--backend cuda` timings include per-step host/device cache transfers; label
GPU numbers accordingly until the cache moves to the device. A benchmark
process loads the model once and gives each run a fresh KV cache and warmup.
See [integration status and midterm handoff](docs/midterm_progress.md).

## Key interface assumptions

- Only the engine reads `output_len`; it reports tokens and completion to the scheduler.
- Memory is capped by `max_slots` (concurrent requests), not sequence length.
- Final prefill emits the first output token; each later decode emits one token.
- Sarathi's minimum token budget must be at least `max_slots` to fit all decodes.
- Times use seconds. `tpot_p50/p99` summarize per-request mean TPOT;
  `token_tpot_p50/p99` summarize all individual inter-token intervals.
- SLO attainment uses all arrived requests as its denominator, so rejected and
  incomplete requests do not attain their SLO.

Full interface assumptions are in [interfaces.py](interfaces.py).
