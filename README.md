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

## Key interface assumptions

- Only the engine reads `output_len`; it reports tokens and completion to the scheduler.
- Memory is capped by `max_slots` (concurrent requests), not sequence length.
- Final prefill emits the first output token; each later decode emits one token.
- Sarathi's minimum token budget must be at least `max_slots` to fit all decodes.
- Times use seconds. TPOT is the mean interval between a request's output tokens.

Full interface assumptions are in [interfaces.py](interfaces.py).
