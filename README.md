# slo-aware-batching

Single-GPU LLM serving engine built on MiniTorch (GPT-2).

## Goal
Maximize goodput — requests/s that meet both TTFT and TPOT SLOs.

## Systems
1. Static batching (baseline)
2. Orca-style continuous batching, FCFS
3. Sarathi-Serve chunked prefill, fixed token budget
4. Ours: SLO-aware adaptive token budget + slack-ordered prefill

## Layout
engine/      KV cache, batched attention, chunked-prefill kernel
scheduler/   request queue, batch planning, budget policies
workload/    ShareGPT-based traces, Poisson / bursty arrivals
bench/       harness, metrics (TTFT, TPOT, goodput), plots
interfaces.py  Request, BatchPlan, engine API, event log format

## Status
- [ ] Stage 1: KV cache + batched attention
- [ ] Stage 2: continuous batching
- [ ] Stage 3: chunked prefill + fixed budget
- [ ] Stage 4: SLO-aware adaptive scheduler
