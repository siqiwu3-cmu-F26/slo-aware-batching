"""Provisional scheduler/engine boundary; every contract below is an assumption."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Protocol


# Assumption: string IDs are unique within one simulation; times/SLOs are seconds.
class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    REJECTED = "rejected"


@dataclass
class Request:
    id: str
    arrival_time: float
    prompt_len: int
    # Assumption: trace supplies output_len for the mock only; schedulers never read it.
    output_len: int = field(repr=False)
    state: RequestState = RequestState.WAITING
    # Assumption: scheduler owns these counters and updates them after execute().
    prefilled_tokens: int = 0
    generated_tokens: int = 0
    ttft_slo: float | None = None
    tpot_slo: float | None = None

    def __post_init__(self) -> None:
        # Assumption: empty prompts/outputs are unsupported; use a BOS prompt token.
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a nonempty string")
        if not math.isfinite(self.arrival_time) or self.arrival_time < 0:
            raise ValueError("arrival_time must be finite and nonnegative")
        for name in ("prompt_len", "output_len"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for slo in (self.ttft_slo, self.tpot_slo):
            if slo is not None and (not math.isfinite(slo) or slo <= 0):
                raise ValueError("SLOs must be finite and positive")


# Assumption: one contiguous input span per request per iteration, no padding cost.
# start_pos is a zero-based absolute input/KV position. The final prefill emits the
# first output token; the next decode consumes it at prompt_len + generated - 1.
PlanItem = tuple[str, int, int]


@dataclass
class BatchPlan:
    items: list[PlanItem] = field(default_factory=list)
    # Assumption: only chunked schedulers impose a token budget; others use None.
    token_budget: int | None = None

    @property
    def total_tokens(self) -> int:
        return sum(num_tokens for _, _, num_tokens in self.items)


@dataclass(frozen=True)
class StepResult:
    # Assumption: execute is synchronous; all emissions/finishes occur at step end.
    duration: float
    # Assumption: at most one output token per request per iteration; EOS counts.
    emitted_tokens: tuple[str, ...] = ()
    finished_requests: tuple[str, ...] = ()


class Engine(Protocol):
    # Assumption: memory is a request-slot cap, independent of sequence length.
    max_slots: int

    # Assumption: registration supplies engine-owned metadata but allocates no slot.
    # The real engine can attach prompt tensors/KV handles behind this boundary.
    def register(self, request: Request) -> None: ...

    # Assumption: engine owns private execution state, never mutates Request, and
    # releases finished slots automatically. No CUDA/kernel implementation here.
    def execute(self, plan: BatchPlan) -> StepResult: ...


class EventKind(str, Enum):
    # Assumption: first_token is also recorded as a token event (index 1).
    ARRIVAL = "arrival"
    FIRST_TOKEN = "first_token"
    TOKEN = "token"
    FINISH = "finish"
    REJECT = "reject"


@dataclass(frozen=True)
class Event:
    # Assumption: JSONL events use seconds, one-based output token indices, and an
    # optional rejection reason; equal timestamps retain causal insertion order.
    kind: EventKind
    request_id: str
    timestamp: float
    token_index: int | None = None
    reason: str | None = None


class Scheduler(Protocol):
    # Assumption: submit only at/after arrival; one fresh Request object per run.
    max_slots: int
    rejected_requests: int
    deferred_decisions: int

    @property
    def deferred_requests(self) -> int: ...

    @property
    def has_pending(self) -> bool: ...

    def submit(self, request: Request) -> bool: ...

    def plan(self, now: float) -> BatchPlan: ...

    # Assumption: call exactly once per executed plan, at the step-end timestamp.
    def update(self, plan: BatchPlan, result: StepResult, now: float) -> list[Event]: ...
