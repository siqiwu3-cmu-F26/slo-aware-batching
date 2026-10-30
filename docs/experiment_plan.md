# Experiment Plan

This document fixes the evaluation protocol before the real MiniTorch engine
and adaptive scheduler are connected. Hardware-dependent values are selected
by the calibration procedure below rather than guessed from the mock engine.

## 1. Questions and hypotheses

The evaluation answers four questions:

1. Does iteration-level scheduling improve throughput and tail latency over
   static request-level batching on heterogeneous requests?
2. Does chunked prefill reduce decode stalls compared with whole-prompt
   prefill?
3. Does SLO-aware budget adaptation improve goodput and SLO attainment over
   the best fixed token budget?
4. How do the policies behave as offered load, SLO tightness, and request
   burstiness change?

Expected outcomes:

- Orca should outperform static batching when output lengths vary.
- Sarathi should reduce P99 inter-token latency relative to Orca when long
  prefills arrive during decoding.
- No single fixed token budget should be best at every load and SLO.
- The adaptive policy should improve goodput near saturation; it need not win
  raw throughput at low load.

These are hypotheses, not acceptance criteria. Unexpected results must be
reported rather than filtered.

## 2. Fixed metric definitions

All times are in seconds and include queueing after request arrival.

- **TTFT:** first output token timestamp minus arrival timestamp.
- **Per-request mean TPOT:** mean interval between consecutive output tokens.
  A one-token output has TPOT 0 because it has no inter-token interval.
- **Token-level TPOT:** each individual interval between consecutive output
  tokens. `token_tpot_p99` is the primary tail decode-latency metric.
- **E2E latency:** finish timestamp minus arrival timestamp.
- **Throughput:** completed requests/s and emitted tokens/s.
- **SLO attainment:** fraction of all arrived requests that complete while
  satisfying both TTFT and mean-TPOT SLOs. Rejected and incomplete requests
  count as failures.
- **Goodput:** completed requests satisfying both SLOs per second.

The benchmark stores both per-request mean TPOT and token-level TPOT so the
report can state the distinction explicitly.

## 3. Workload construction

### Dataset

Use a pinned ShareGPT JSON snapshot and record its filename and checksum in
the final report. For each conversation, use the first adjacent user/assistant
pair. Tokenize both strings with the GPT-2 tokenizer.

Filter out empty pairs and pairs where:

```text
prompt_tokens + output_tokens > model_context_length
```

GPT-2 small and medium both use a 1024-token context limit. The workload
generator now enforces this limit by default. Do not silently truncate an
over-length sample.

For deterministic serving comparisons, each engine should generate exactly
the trace's `output_len` tokens. If EOS stopping is enabled instead, all
systems must use the same rule and actual output lengths must be reported.

### Arrivals

Primary workload: Poisson arrivals.

Stress workload: periodic bursty Poisson arrivals using the existing
`--arrival-mode bursty` generator. Keep burst factor, fraction, and period
identical across systems and record them in each run config.

Use the same generated trace for every scheduler at a given:

```text
(model, workload type, arrival rate, seed)
```

### Seeds

- Calibration and tuning: seed `99`.
- Reported runs: seeds `0`, `1`, and `2`.

Never select policy parameters using the reported seeds.

## 4. Systems

Evaluate:

1. Static batching.
2. Orca-style continuous batching with FCFS admission.
3. Sarathi-style chunked prefill with a fixed token budget.
4. SLO-aware adaptive scheduling.

All systems must use the same:

- GPT-2 weights and tokenizer
- precision
- GPU
- maximum resident slots / KV-cache capacity
- workload trace
- stopping rule
- warmup procedure

The static and Orca baselines do not receive a token-budget sweep. Sarathi's
main-result budget is chosen on calibration seed 99, then frozen.

## 5. Correctness gates

Do not run the full matrix until all gates pass:

1. Cached single-request output matches no-cache GPT-2 token by token.
2. Variable-length batched output matches independent single-request output.
3. Chunked-prefill output matches whole-prefill output for chunk sizes
   including 1, a non-divisor of prompt length, and the full prompt.
4. Every scheduler respects resident-slot and token-budget limits.
5. Every submitted request reaches exactly one terminal state: finished or
   rejected.
6. Event timestamps are monotonic and token indices are consecutive.
7. Repeating one run with the same seed produces the same trace and outputs.

## 6. Timing protocol

- Load model weights before measurement.
- Run at least 20 warmup iterations before accepting requests.
- Synchronize CUDA before starting and stopping GPU timing.
- Exclude model loading, dataset loading, and plot generation.
- Include scheduler overhead, queueing, prefill, and decode in serving
  latency.
- Run one benchmark configuration per process if allocator or KV-cache state
  cannot be reliably reset.
- Record GPU model, CUDA version, MiniTorch commit, precision, model, and
  maximum slots with every experiment batch.

`StepResult.duration` from the real engine must represent synchronized elapsed
execution time, not asynchronous kernel launch time.

## 7. Calibration

Calibration uses seed 99 only.

### 7.1 Unloaded latency

Run requests serially (`max_slots=1`) to measure unloaded P99 TTFT and
token-level TPOT for each model.

Evaluate SLO multipliers:

```text
1.5x, 2x, and 3x unloaded P99
```

Use 2x as the primary SLO and the other two as the SLO-tightness ablation.
Record the resulting absolute TTFT and TPOT targets.

### 7.2 Capacity and arrival rates

Use a saturation run to estimate maximum completed request throughput for each
model on the primary ShareGPT length distribution.

Set the six reported arrival rates to:

```text
0.25, 0.50, 0.75, 0.90, 1.00, and 1.10
```

times the measured capacity. Store both normalized load and absolute
requests/s. This avoids carrying mock-engine arrival rates onto real hardware.

### 7.3 Fixed token budget

On calibration seed 99, sweep powers of two that fit the implementation and
GPU memory. Include at least four values spanning decode-protective to
prefill-aggressive behavior.

Choose the fixed Sarathi budget with the highest goodput, breaking ties by
lower P99 token-level TPOT. Freeze it before running seeds 0–2. Plot the full
sweep as the token-budget ablation.

## 8. Reported experiment matrix

### Main Poisson experiment

```text
4 systems × 6 arrival rates × 2 models × 3 seeds = 144 runs
```

### Bursty stress experiment

Run at normalized loads 0.75 and 1.00:

```text
4 systems × 2 loads × 2 models × 3 seeds = 48 runs
```

### Ablations

- Fixed token-budget sweep on calibration and reported seeds.
- SLO multipliers 1.5x, 2x, and 3x.
- Adaptive policy components:
  - adaptive budget disabled
  - slack ordering disabled
  - rejection/defer rule disabled
- Optional chunk-size sweep if chunk size is independent of token budget.

If GPU time is limited, prioritize the main Poisson matrix, token-budget
ablation, and one adaptive-policy ablation. Do not reduce the number of seeds
for only one system.

## 9. Required figures

For each model:

1. Arrival rate vs completed requests/s.
2. Arrival rate vs emitted tokens/s.
3. Arrival rate vs P99 TTFT.
4. Arrival rate vs P99 token-level TPOT.
5. Arrival rate vs SLO attainment.
6. Arrival rate vs goodput.
7. Token budget vs throughput, P99 TTFT, P99 TPOT, and goodput.

Plot the mean across seeds with 95% confidence intervals. Keep identical axis
limits when comparing systems across panels. The plotting pipeline also emits
`aggregated_plot_data.csv`; retain it with the figures.

## 10. Commands

Generate a ShareGPT trace:

```bash
python3 workload.py sharegpt \
  --dataset data/ShareGPT.json \
  --num-requests 1000 \
  --arrival-rate <requests-per-second> \
  --max-total-tokens 1024 \
  --seed 0 \
  --output trace.json
```

Current mock-engine pipeline smoke test:

```bash
python3 benchmark.py \
  --schedulers static orca sarathi \
  --arrival-rates 1 2 4 8 \
  --seeds 0 1 2 \
  --token-budgets 16 32 64 \
  --num-requests 1000 \
  --max-total-tokens 1024 \
  --ttft-slo 2 \
  --tpot-slo 0.1 \
  --output-dir benchmark-results
```

The numeric rates and SLOs above are smoke-test values only. Replace them with
calibrated GPU values for reported experiments.

Generate figures:

```bash
python3 -m pip install -r requirements-plots.txt
python3 plots.py \
  --summary benchmark-results/summary.csv \
  --output-dir figures \
  --format pdf
```

## 11. Integration checklist

When teammates deliver the real engine and adaptive scheduler:

- Add real-engine selection to `benchmark.py`; keep mock mode for tests.
- Add `adaptive` to benchmark scheduler choices and plotting labels.
- Confirm real engine implements `register()` and `execute()` from
  `interfaces.py`.
- Confirm CUDA-synchronized duration reporting.
- Run correctness gates before calibration.
- Run calibration and freeze SLO/rate/budget values in a committed config.
- Run the full matrix without changing policy parameters between systems.
- Archive configs, traces, events, per-request CSVs, summaries, and figures.
