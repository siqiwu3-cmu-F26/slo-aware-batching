# SLO-Aware Adaptive Continuous Batching for Efficient LLM Serving

Polaris Wu, Siqi Zhu, Anmol Kumari — Carnegie Mellon University

## Abstract

LLM serving systems must balance throughput against per-request latency targets, measured as time-to-first-token (TTFT) and time-per-output-token (TPOT). We propose to reimplement two recent serving techniques in MiniTorch: iteration-level continuous batching from Orca (Yu et al., 2022) and chunked prefill with a fixed per-iteration token budget from Sarathi-Serve (Agrawal et al., 2024). On top of this engine we add an SLO-aware adaptive scheduler that adjusts the token budget and batch composition every iteration based on each request's remaining slack to its TTFT and TPOT deadlines. We will evaluate on GPT-2 with ShareGPT-style request traces under varying arrival rates, reporting throughput, tail latency, SLO attainment, and goodput against static batching, Orca-style scheduling, and Sarathi-Serve's fixed budget.

## 1. Problem and System Challenges

Serving an LLM means running many concurrent generation requests on one GPU. Each request has a compute-heavy prefill phase over its prompt, followed by many memory-bound decode steps that produce one token each. How requests are grouped into batches decides both throughput and latency, which creates four system challenges.

- **Batching granularity.** Static (request-level) batching waits for the longest request in a batch to finish, so short requests idle and new requests queue. Scheduling per decoding iteration removes this waste but requires attention to handle requests of different lengths in the same batch.
- **Prefill–decode interference.** A long prompt prefill in the same iteration stalls the decode steps of every other request, causing TPOT spikes. Delaying prefills to protect decodes instead raises TTFT.
- **Static budgets.** Chunked prefill bounds per-iteration work with a fixed token budget, but the best budget depends on load and on how close each request is to its deadline, both of which change over time.
- **Memory limits.** The KV cache grows with every token, so GPU memory caps how many requests can run at once. The scheduler must decide admission under this cap.

## 2. Existing Methods

- **Orca** (Yu et al., 2022) introduced iteration-level scheduling (continuous batching) and selective batching. No official code is released.
- **Sarathi-Serve** (Agrawal et al., 2024) splits long prefills into chunks and mixes them with decodes under a fixed token budget (stall-free batching) to bound TPOT while keeping throughput high. Code is open source.
- **vLLM** (Kwon et al., 2023) adds paged KV cache management and continuous batching; it is the most widely used open-source serving engine (Apache-2.0).
- **DistServe** (Zhong et al., 2024) disaggregates prefill and decode onto different GPUs and optimizes goodput under TTFT/TPOT SLOs. Code is open source.
- **SGLang** (Zheng et al., 2024) adds prefix sharing across requests and a fast scheduler. Code is open source.

We reimplement Orca and Sarathi-Serve on a single GPU, and borrow the goodput-under-SLO objective from DistServe without disaggregation.

## 3. Proposed Approach

We build the engine inside MiniTorch in four stages, each adding one layer on top of the previous one.

- **Stage 1: KV cache and batched attention.** Extend the MiniTorch GPT-2 (Radford et al., 2019) decoder with a KV cache preallocated per request slot, so each decode step reuses cached keys and values. Support batches of requests with different lengths through attention masks.
- **Stage 2: Continuous batching (Orca).** Add a scheduler loop that, at every iteration, admits waiting requests up to the memory cap, runs one step for all running requests, and retires finished ones immediately.
- **Stage 3: Chunked prefill (Sarathi-Serve).** Split prompts into chunks and fill each iteration with all running decodes plus prefill chunks up to a fixed token budget. Implement a chunked-prefill attention kernel in MiniTorch's CUDA backend, where a query chunk attends to its cached prefix and to itself.
- **Stage 4: SLO-aware adaptive scheduling (our extension).** Each iteration, compute every request's slack to its TTFT or TPOT deadline. Shrink the token budget when decoding requests are close to their TPOT deadline; grow it when waiting requests are close to their TTFT deadline. Order prefills by slack instead of FCFS, and reject or defer requests that can no longer meet their SLO.
- **Stretch goals.** Prefix caching across requests with a shared prompt, and a comparison against vLLM on the same model.

## 4. Evaluation Plan

- **Model.** GPT-2 small and medium implemented in MiniTorch, loaded with pretrained weights. We verify outputs match the no-cache reference token by token.
- **Workload.** Prompt and output lengths sampled from the ShareGPT conversation dataset; requests arrive by a Poisson process at several rates, plus a bursty trace to stress adaptation.
- **Metrics.** Throughput (requests/s and tokens/s); P50/P99 TTFT and TPOT; SLO attainment and goodput (requests/s that meet both SLOs).
- **Baselines.** (1) static batching; (2) Orca-style continuous batching with FCFS; (3) Sarathi-Serve with a fixed token budget; (4) ours with SLO-aware adaptive budget.
- **Ablations.** Fixed budget size, adaptation policy components, and SLO tightness.

## 5. Team and Work Split

| Member | Responsibility |
|---|---|
| Polaris Wu | Scheduler: continuous batching, token budget, SLO-aware adaptive policy |
| Anmol Kumari | KV cache, variable-length batched attention, chunked-prefill CUDA kernel |
| Siqi Zhu | Workload generator, benchmark harness, baselines, plots |

All members co-write the reports and review each other's code. Stage 1 is done jointly so everyone shares the same inference path.

## 6. Timeline

| Date | Milestone |
|---|---|
| 9/30 | Proposal submitted |
| 10/1–10/11 | Read Orca and Sarathi-Serve; Stage 1 KV cache and batched attention; benchmark harness and ShareGPT workload |
| 10/19–11/3 | Stage 2 continuous batching; Stage 3 chunked prefill and kernel |
| 11/4 | Mid-term report: Orca and Sarathi baselines reproduced, first throughput and latency numbers |
| 11/5–11/22 | Stage 4 SLO-aware adaptive scheduler; stretch goals if time allows |
| 11/23–12/3 | Full experiments and ablations |
| 12/4 | Final presentation |
| 12/6 | Final report |

## 7. Compute Requirements

- **GPU.** One NVIDIA GPU with at least 24 GB memory (A10G, L4, or A100).
- **CPU/RAM.** 4–8 cores and 32 GB RAM for the workload generator and scheduler.
- **Storage.** About 50 GB for model weights, datasets, and logs.
- **Time estimate.** Development and debugging: about 50 GPU-hours across the team. Experiments: 4 systems × 6 arrival rates × 2 models × 3 seeds at roughly 10 minutes per run is about 24 GPU-hours; with ablations, about 35 GPU-hours. Total: roughly 85 GPU-hours.

## References

- Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B., Tumanov, A., and Ramjee, R. Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve. OSDI 2024, pp. 117–134.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with PagedAttention. SOSP 2023, pp. 611–626.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. OpenAI Technical Report, 2019.
- Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.-G. Orca: A distributed serving system for Transformer-based generative models. OSDI 2022, pp. 521–538.
- Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., and Sheng, Y. SGLang: Efficient execution of structured language model programs. NeurIPS 2024.
- Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., and Zhang, H. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. OSDI 2024, pp. 193–210.
