# Engine integration and midterm handoff

## Implemented

- `MiniTorchEngine` connects the existing GPT2/KVCache to the unchanged schedulers.
- Registration stores prompts; first execution allocates a slot; completion frees it.
- Prefill/chunks consume prompt tokens; decodes consume the previous generated token.
- Equal input lengths batch together. Unequal lengths use separate forwards.
- Real mode uses wall-clock replay, including scheduler/host overhead. Loading
  and warmup occur before timing; CUDA execution has a synchronization hook.
- Both CLI drivers select mock or MiniTorch. Real benchmark runs archive prompts,
  generated tokens, hardware/runtime metadata, and a weight checksum.
- CPU tiny-model tests compare all three schedulers and chunk sizes 1, a
  non-divisor, and the whole prompt against independent no-cache greedy outputs.

## Small end-to-end check

After installing `requirements-engine-cpu.txt`, run:

```bash
NUMBA_NUM_THREADS=1 python benchmark.py --engine minitorch --model tiny \
  --schedulers static orca sarathi --arrival-rates 1000 --seeds 0 \
  --token-budgets 2 4 --max-slots 2 --num-requests 4 \
  --prompt-max 7 --output-max 4 --max-total-tokens 32 \
  --output-dir benchmark-results/integration-smoke
```

This uses random tiny weights and synthetic tokens. Its numbers are smoke-test
measurements, not pretrained GPT-2/GPU results. Real runs are always rerun;
automatic resume remains enabled for mock runs. Keep each model/backend/clock
in a separate output directory. `--clock virtual` is useful for debugging real
execution without arrival waits, but excludes scheduler overhead from latency.

## Remaining work by owner

| Owner | Next work |
|---|---|
| Scheduler | Review integrated baselines; implement Stage 4 after baseline measurements. |
| Engine | Validate the CUDA path; replace grouped execution with variable-query-length mixed batching/chunked-prefill kernel; optimize the host-resident KV path. |
| Workload/benchmark | Supply actual ShareGPT prompt token IDs through `--prompts`; calibrate rates/SLOs/budgets on the target GPU; run the experiment matrix and plots. |
| Everyone | Review interface assumptions, correctness evidence, and midterm claims. |

The current ShareGPT generator exports lengths. Without a prompt sidecar, real
execution is explicitly labeled `synthetic_length_replay`, even for ShareGPT
length distributions. All systems use fixed output lengths with EOS ignored.

## Midterm report material

1. Problem: throughput versus TTFT/inter-token latency; goodput objective.
2. Architecture: workload → scheduler → Engine adapter → MiniTorch GPT-2.
3. Implementation: Stage 1 model/cache, Stage 2 continuous batching, Stage 3
   budget packing and current grouped execution; state the kernel's actual status.
4. Evidence: numerical correctness tests, reproducible configs, and real GPU
   throughput/P99 TTFT/P99 token-level TPOT once available. Label mock/CPU smoke
   results explicitly; pretrained reference tests must actually run, not skip.
5. Remaining work: optimized mixed kernel, calibrated experiments, and Stage 4.
