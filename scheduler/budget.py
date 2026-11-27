"""Token-budget policies and the controller that clamps them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from interfaces import Request

if TYPE_CHECKING:
    from .step_time import StepTimeModel


@dataclass(frozen=True)
class BudgetContext:
    # Assumption: policies see scheduler-visible state and time, never output_len.
    now: float
    active: Sequence[Request]
    waiting: Sequence[Request]
    # Filled by the SLO-aware scheduler only: the time by which each decoding
    # request must emit its next token, and the online step-time estimate.
    next_token_deadlines: Sequence[float] = ()
    step_time: StepTimeModel | None = None


class BudgetPolicy(Protocol):
    # Assumption: policy limits stay fixed during a run; proposals may vary.
    @property
    def min_budget(self) -> int: ...

    @property
    def max_budget(self) -> int: ...

    @property
    def max_change_per_step(self) -> int: ...

    def propose(self, context: BudgetContext) -> int: ...


@dataclass(frozen=True, init=False)
class FixedBudget:
    tokens: int
    min_budget: int
    max_budget: int
    max_change_per_step: int

    def __init__(self, tokens: int, min_budget: int | None = None,
                 max_budget: int | None = None, max_change_per_step: int = 0) -> None:
        # Assumption: omitted bounds pin the policy to its fixed token count.
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "min_budget", tokens if min_budget is None else min_budget)
        object.__setattr__(self, "max_budget", tokens if max_budget is None else max_budget)
        object.__setattr__(self, "max_change_per_step", max_change_per_step)
        validate_bounds(self)
        if type(self.tokens) is not int or not self.min_budget <= self.tokens <= self.max_budget:
            raise ValueError("fixed tokens must be an integer within budget bounds")

    def propose(self, context: BudgetContext) -> int:
        return self.tokens


@dataclass(frozen=True)
class SloAwareBudget:
    """Largest budget whose predicted step keeps every decode on its TPOT deadline.

    With no decoding request it proposes ``max_budget`` so prefills (and TTFT)
    get the most room; while step time is not yet fitted from two different
    step sizes, decodes get ``min_budget``. A decode whose
    deadline is already out of reach even at ``min_budget`` does not constrain
    the budget: shrinking cannot save it, and would only slow everyone else.
    """

    min_budget: int
    max_budget: int
    # None allows any change per step; the controller treats 0 as "frozen".
    max_change_per_step: int | None = None

    def __post_init__(self) -> None:
        if self.max_change_per_step is None:
            object.__setattr__(self, "max_change_per_step", self.max_budget - self.min_budget)
        validate_bounds(self)

    def propose(self, context: BudgetContext) -> int:
        model = context.step_time
        if model is None or not context.next_token_deadlines:
            return self.max_budget
        if not model.fitted:
            # One step size cannot tell a late decode from a loose estimate, so
            # protect decodes; the smaller step also completes the fit.
            return self.min_budget
        budget = self.max_budget
        for deadline in context.next_token_deadlines:
            fits = model.max_tokens_within(deadline - context.now)
            if fits >= self.min_budget:
                budget = min(budget, fits)
        return budget


def validate_bounds(policy: BudgetPolicy) -> None:
    values = (policy.min_budget, policy.max_budget, policy.max_change_per_step)
    if any(type(value) is not int for value in values):
        raise ValueError("budget bounds and change limit must be integers")
    if not 1 <= policy.min_budget <= policy.max_budget or policy.max_change_per_step < 0:
        raise ValueError("invalid budget bounds or change limit")


class BudgetController:
    """Clamp even custom policies to bounds and an absolute per-step change cap."""

    def __init__(self, policy: BudgetPolicy, max_slots: int) -> None:
        validate_bounds(policy)
        # Assumption: reserve enough budget for every possible resident decode.
        # This also leaves progress capacity whenever any resident is prefilling.
        if policy.min_budget < max_slots:
            raise ValueError("min_budget must be >= max_slots to include all decodes")
        self.policy = policy
        self.current: int | None = None

    def choose(self, context: BudgetContext) -> int:
        proposed = self.policy.propose(context)
        if type(proposed) is not int:
            raise ValueError("budget policy must propose an integer")
        chosen = max(self.policy.min_budget, min(self.policy.max_budget, proposed))
        if self.current is not None:
            delta = self.policy.max_change_per_step
            chosen = max(self.current - delta, min(self.current + delta, chosen))
        # Assumption: the first iteration has no previous budget/change constraint.
        self.current = chosen
        return chosen
