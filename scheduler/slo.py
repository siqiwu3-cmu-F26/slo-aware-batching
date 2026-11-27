"""Stage 4: SLO-aware adaptive chunked-prefill scheduling."""

from __future__ import annotations

from collections import deque
import math

from interfaces import BatchPlan, Event, EventKind, Request, RequestState, StepResult
from .budget import BudgetContext, BudgetPolicy, SloAwareBudget
from .core import SarathiScheduler
from .step_time import StepTimeModel

ORDERS = ("slack", "fcfs")
TPOT_TARGETS = ("mean", "token")


class SloAwareScheduler(SarathiScheduler):
    """Sarathi-style stall-free batching driven by per-request SLO slack.

    Each iteration it
    1. fits step time against tokens per step from the steps it has timed;
    2. picks the largest token budget whose predicted step still lets every
       decoding request emit its next token by its TPOT deadline
       (``SloAwareBudget``), so the budget shrinks under decode pressure and
       grows to ``max_budget`` when decodes have slack or none are running;
    3. admits and prefills requests in order of TTFT slack (deadline minus now
       minus predicted remaining prefill time), least slack first, with
       requests that can no longer make their deadline moved to the back;
    4. rejects waiting requests whose TTFT deadline is out of reach even if
       they started now with the whole budget. That is a lower bound on their
       TTFT, so a rejected request could not have met its SLO.

    Ablation switches: pass ``FixedBudget(max_budget)`` as ``budget_policy`` to
    disable (2), ``order="fcfs"`` to disable (3), and ``reject=False`` to
    disable (4).

    ``tpot_target`` selects the TPOT SLO the decode deadlines protect:
    "mean" (the request's mean inter-token gap, which lets a request that is
    ahead of schedule absorb one long step) or "token" (every gap, matching
    the per-request P99 TPOT SLO).

    Like every scheduler here it never reads ``output_len``.
    """

    def __init__(self, max_slots: int, budget_policy: BudgetPolicy | None = None,
                 max_queue_size: int | None = None, *, max_budget: int = 256,
                 order: str = "slack", reject: bool = True, tpot_target: str = "mean") -> None:
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        if tpot_target not in TPOT_TARGETS:
            raise ValueError(f"tpot_target must be one of {TPOT_TARGETS}")
        policy = budget_policy if budget_policy is not None else SloAwareBudget(max_slots, max_budget)
        super().__init__(max_slots, policy, max_queue_size)
        self.order = order
        self.reject = reject
        self.tpot_target = tpot_target
        self.step_time = StepTimeModel()
        self._first_token: dict[str, float] = {}
        self._last_token: dict[str, float] = {}
        self._plan_time: float | None = None

    # Deadlines and slack -------------------------------------------------

    def _next_token_deadline(self, request: Request) -> float | None:
        if request.tpot_slo is None or request.id not in self._last_token:
            return None
        if self.tpot_target == "token":
            return self._last_token[request.id] + request.tpot_slo
        # Mean TPOT over n gaps stays within the SLO while the next token comes
        # no later than first_token + n * tpot_slo, with n = tokens so far.
        return self._first_token[request.id] + request.generated_tokens * request.tpot_slo

    def _ttft_slack(self, request: Request, now: float) -> float:
        """TTFT deadline minus now minus the fastest possible remaining prefill."""
        if request.ttft_slo is None or not self.step_time.fitted:
            return math.inf
        remaining = request.prompt_len - request.prefilled_tokens
        fastest = self.step_time.prefill_time(remaining, self.budget.policy.max_budget)
        return request.arrival_time + request.ttft_slo - now - fastest

    def _priority(self, request: Request, now: float) -> tuple[bool, float]:
        slack = self._ttft_slack(request, now)
        # Requests that cannot make their deadline go last, so they only use
        # capacity no salvageable request wants. sorted() keeps FCFS on ties.
        return (slack < 0, slack)

    # Scheduler interface -------------------------------------------------

    def expire(self, now: float) -> list[Event]:
        if not self.reject or not self.step_time.fitted:
            return []
        events, kept = [], deque()
        for request in self.waiting:
            if self._ttft_slack(request, now) < 0:
                request.state = RequestState.REJECTED
                self.rejected_requests += 1
                events.append(Event(EventKind.REJECT, request.id, now, reason="ttft_slo_unreachable"))
            else:
                kept.append(request)
        self.waiting = kept
        return events

    def _admit_by_priority(self, now: float) -> None:
        if self.order == "fcfs":
            self._admit()
            return
        free = self.max_slots - len(self.active)
        if free <= 0 or not self.waiting:
            return
        chosen = sorted(self.waiting, key=lambda request: self._priority(request, now))[:free]
        chosen_ids = {request.id for request in chosen}
        self.waiting = deque(request for request in self.waiting if request.id not in chosen_ids)
        for request in chosen:
            request.state = RequestState.PREFILLING
            self.active[request.id] = request

    def plan(self, now: float) -> BatchPlan:
        self._admit_by_priority(now)
        decodes = [request for request in self.active.values() if request.state == RequestState.DECODING]
        deadlines = tuple(deadline for deadline in map(self._next_token_deadline, decodes)
                          if deadline is not None)
        budget = self.budget.choose(BudgetContext(now, tuple(self.active.values()), tuple(self.waiting),
                                                  deadlines, self.step_time))
        items = [self._whole_step(request) for request in decodes]
        remaining = budget - len(items)
        prefilling = [request for request in self.active.values()
                      if request.state == RequestState.PREFILLING]
        if self.order == "slack":
            prefilling.sort(key=lambda request: self._priority(request, now))
        for request in prefilling:
            if not remaining:
                break
            count = min(remaining, request.prompt_len - request.prefilled_tokens)
            items.append((request.id, request.prefilled_tokens, count))
            remaining -= count
        self._plan_time = now
        return self._record_deferrals(BatchPlan(items, budget))

    def update(self, plan: BatchPlan, result: StepResult, now: float) -> list[Event]:
        events = super().update(plan, result, now)
        # Time the step as the scheduler sees it: plan to update. On the virtual
        # clock that is exactly the engine duration; on the wall clock it also
        # includes host overhead, which deadlines must account for too.
        elapsed = now - self._plan_time if self._plan_time is not None else 0.0
        self.step_time.observe(plan.total_tokens, elapsed if elapsed > 0 else result.duration)
        self._plan_time = None
        for event in events:
            if event.kind == EventKind.TOKEN:
                self._first_token.setdefault(event.request_id, event.timestamp)
                self._last_token[event.request_id] = event.timestamp
            elif event.kind == EventKind.FINISH:
                self._first_token.pop(event.request_id, None)
                self._last_token.pop(event.request_id, None)
        return events
