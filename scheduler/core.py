"""Scheduling only: no output-length inspection, tensors, or timing model."""

from collections import deque
import math

from interfaces import BatchPlan, Event, EventKind, Request, RequestState, StepResult
from .budget import BudgetContext, BudgetController, BudgetPolicy, FixedBudget


class _BaseScheduler:
    def __init__(self, max_slots: int, max_queue_size: int | None = None) -> None:
        if type(max_slots) is not int or max_slots < 1:
            raise ValueError("max_slots must be a positive integer")
        if max_queue_size is not None and (type(max_queue_size) is not int or max_queue_size < 1):
            raise ValueError("max_queue_size must be positive or None")
        self.max_slots = max_slots
        self.max_queue_size = max_queue_size
        self.waiting: deque[Request] = deque()
        self.active: dict[str, Request] = {}
        self._seen: set[str] = set()
        self._deferred: set[str] = set()
        self.rejected_requests = 0
        self.deferred_decisions = 0

    @property
    def deferred_requests(self) -> int:
        return len(self._deferred)

    @property
    def has_pending(self) -> bool:
        return bool(self.waiting or self.active)

    def submit(self, request: Request) -> bool:
        if request.id in self._seen:
            raise ValueError(f"duplicate scheduler request: {request.id}")
        if request.state != RequestState.WAITING or request.prefilled_tokens or request.generated_tokens:
            raise ValueError("submit requires a fresh waiting request")
        self._seen.add(request.id)
        # Assumption: optional queue cap counts waiting requests only, at arrival;
        # no implicit SLO rejection is performed by these baseline schedulers.
        if self.max_queue_size is not None and len(self.waiting) >= self.max_queue_size:
            request.state = RequestState.REJECTED
            self.rejected_requests += 1
            return False
        self.waiting.append(request)
        return True

    def _admit(self) -> None:
        while self.waiting and len(self.active) < self.max_slots:
            request = self.waiting.popleft()
            request.state = RequestState.PREFILLING
            self.active[request.id] = request

    def _record_deferrals(self, plan: BatchPlan) -> BatchPlan:
        # Assumption: defer means an arrived, unfinished request received no work
        # in this iteration. Count unique requests and request-iteration decisions.
        scheduled = {rid for rid, _, _ in plan.items}
        for request in (*self.waiting, *self.active.values()):
            if request.state != RequestState.FINISHED and request.id not in scheduled:
                self._deferred.add(request.id)
                self.deferred_decisions += 1
        return plan

    @staticmethod
    def _whole_step(request: Request) -> tuple[str, int, int]:
        if request.prefilled_tokens < request.prompt_len:
            return (request.id, request.prefilled_tokens, request.prompt_len - request.prefilled_tokens)
        return (request.id, request.prompt_len + request.generated_tokens - 1, 1)

    def update(self, plan: BatchPlan, result: StepResult, now: float) -> list[Event]:
        if not math.isfinite(result.duration) or result.duration <= 0:
            raise ValueError("engine duration must be finite and positive")
        emitted = set(result.emitted_tokens)
        finished = set(result.finished_requests)
        planned = {rid for rid, _, _ in plan.items}
        expected_emissions = {
            rid for rid, _, count in plan.items
            if self.active[rid].prefilled_tokens + count >= self.active[rid].prompt_len
        }
        if (len(emitted) != len(result.emitted_tokens)
                or len(finished) != len(result.finished_requests)
                or emitted != expected_emissions or not finished <= emitted or not emitted <= planned):
            raise ValueError("engine returned inconsistent emissions/completions")
        events = []
        for rid, _, count in plan.items:
            request = self.active[rid]
            if request.prefilled_tokens < request.prompt_len:
                request.prefilled_tokens += count
            if rid in emitted:
                request.generated_tokens += 1
                if request.generated_tokens == 1:
                    events.append(Event(EventKind.FIRST_TOKEN, rid, now, 1))
                events.append(Event(EventKind.TOKEN, rid, now, request.generated_tokens))
            request.state = (RequestState.DECODING if request.prefilled_tokens == request.prompt_len
                             else RequestState.PREFILLING)
            if rid in finished:
                request.state = RequestState.FINISHED
                events.append(Event(EventKind.FINISH, rid, now))
        self._retire(finished)
        return events

    def _retire(self, finished: set[str]) -> None:
        for rid in finished:
            del self.active[rid]


class StaticScheduler(_BaseScheduler):
    """Freeze membership until every member of the current batch has finished."""

    def plan(self, now: float) -> BatchPlan:
        if not self.active:
            self._admit()
        # Assumption: finished members reserve their batch slots but consume no work.
        return self._record_deferrals(BatchPlan([
            self._whole_step(request) for request in self.active.values()
            if request.state != RequestState.FINISHED
        ]))

    def _retire(self, finished: set[str]) -> None:
        if all(request.state == RequestState.FINISHED for request in self.active.values()):
            self.active.clear()


class OrcaScheduler(_BaseScheduler):
    """FCFS admission each iteration, with whole-prompt prefill."""

    def plan(self, now: float) -> BatchPlan:
        self._admit()
        return self._record_deferrals(BatchPlan([
            self._whole_step(request) for request in self.active.values()
        ]))


class SarathiScheduler(_BaseScheduler):
    """All decodes first; fill the remaining token budget with FCFS prefills."""

    def __init__(self, max_slots: int, budget_policy: BudgetPolicy | None = None,
                 max_queue_size: int | None = None) -> None:
        super().__init__(max_slots, max_queue_size)
        self.budget = BudgetController(
            budget_policy if budget_policy is not None else FixedBudget(32), max_slots)

    def plan(self, now: float) -> BatchPlan:
        self._admit()
        budget = self.budget.choose(BudgetContext(now, tuple(self.active.values()), tuple(self.waiting)))
        items = [self._whole_step(request) for request in self.active.values()
                 if request.state == RequestState.DECODING]
        remaining = budget - len(items)
        for request in self.active.values():
            if request.state == RequestState.PREFILLING and remaining:
                count = min(remaining, request.prompt_len - request.prefilled_tokens)
                items.append((request.id, request.prefilled_tokens, count))
                remaining -= count
        return self._record_deferrals(BatchPlan(items, budget))
