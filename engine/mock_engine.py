"""Deterministic timing/completion stub, with no model or kernel code."""

from dataclasses import dataclass
import math

from interfaces import BatchPlan, Request, StepResult


@dataclass
class _Progress:
    prompt_len: int
    output_len: int
    prefilled: int = 0
    generated: int = 0


class MockEngine:
    def __init__(self, max_slots: int = 4, a: float = 0.01, b: float = 0.001) -> None:
        if type(max_slots) is not int or max_slots < 1:
            raise ValueError("max_slots must be a positive integer")
        if not all(math.isfinite(x) and x >= 0 for x in (a, b)) or a + b <= 0:
            raise ValueError("a and b must be finite, nonnegative, and not both zero")
        self.max_slots = max_slots
        self.a, self.b = a, b
        self._requests: dict[str, _Progress] = {}
        self._resident: set[str] = set()

    def register(self, request: Request) -> None:
        if request.id in self._requests:
            raise ValueError(f"duplicate engine request: {request.id}")
        self._requests[request.id] = _Progress(request.prompt_len, request.output_len)

    def execute(self, plan: BatchPlan) -> StepResult:
        ids = [rid for rid, _, _ in plan.items]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("plan must be nonempty with one span per request")
        if len(self._resident | set(ids)) > self.max_slots:
            raise ValueError("plan exceeds resident request-slot capacity")
        if plan.token_budget is not None and plan.total_tokens > plan.token_budget:
            raise ValueError("plan exceeds token budget")
        # Validate the whole plan before changing any execution state.
        for rid, start, count in plan.items:
            if rid not in self._requests:
                raise ValueError(f"unregistered request: {rid}")
            req = self._requests[rid]
            if type(start) is not int or type(count) is not int or count <= 0:
                raise ValueError("positions/counts must be integers and counts positive")
            if req.generated == req.output_len:
                raise ValueError(f"request already finished: {rid}")
            if req.prefilled < req.prompt_len:
                if start != req.prefilled or count > req.prompt_len - req.prefilled:
                    raise ValueError("invalid prefill span")
            elif start != req.prompt_len + req.generated - 1 or count != 1:
                raise ValueError("decode must consume one token at the next KV position")
        emitted, finished = [], []
        for rid, _, count in plan.items:
            req = self._requests[rid]
            self._resident.add(rid)
            if req.prefilled < req.prompt_len:
                req.prefilled += count
                emits = req.prefilled == req.prompt_len
            else:
                emits = True
            if emits:
                req.generated += 1
                emitted.append(rid)
                if req.generated == req.output_len:
                    finished.append(rid)
                    self._resident.remove(rid)
        return StepResult(self.a + self.b * plan.total_tokens, tuple(emitted), tuple(finished))
