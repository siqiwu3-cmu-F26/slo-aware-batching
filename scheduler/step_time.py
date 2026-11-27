"""Online estimate of step time as a function of tokens per step."""

from __future__ import annotations

import math


class StepTimeModel:
    """Least-squares fit of ``seconds ≈ a + b * tokens`` over observed steps.

    The scheduler only sees (planned tokens, elapsed seconds) pairs, so the fit
    works the same on the mock engine (exactly linear) and the real engine
    (where it is an approximation). Before any step it is not ready.

    While every step so far had the same token count x0 (mean time y0), it
    predicts max(y0, y0 * x / x0). For any true a, b >= 0 that is an upper
    bound on a + b * x, so an early budget cap can only be too cautious.
    ``fitted`` turns true once two token counts have been seen.
    """

    def __init__(self) -> None:
        self.count = 0
        self._sum_x = self._sum_y = self._sum_xx = self._sum_xy = 0.0
        self._first_x: int | None = None
        self._distinct = False
        self.a = 0.0
        self.b = 0.0

    @property
    def ready(self) -> bool:
        return self.count > 0

    @property
    def fitted(self) -> bool:
        return self._distinct

    def observe(self, tokens: int, seconds: float) -> None:
        if type(tokens) is not int or tokens < 1:
            raise ValueError("tokens must be a positive integer")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be finite and positive")
        self.count += 1
        self._sum_x += tokens
        self._sum_y += seconds
        self._sum_xx += tokens * tokens
        self._sum_xy += tokens * seconds
        if self._first_x is None:
            self._first_x = tokens
        elif tokens != self._first_x:
            self._distinct = True
        self._refit()

    def _refit(self) -> None:
        n = self.count
        if not self._distinct:
            return
        denominator = n * self._sum_xx - self._sum_x ** 2
        b = (n * self._sum_xy - self._sum_x * self._sum_y) / denominator
        # Step time cannot shrink with more tokens or be negative at zero tokens.
        b = max(b, 0.0)
        a = max((self._sum_y - b * self._sum_x) / n, 0.0)
        self.a, self.b = a, b

    def predict(self, tokens: int) -> float:
        if not self.ready:
            raise RuntimeError("no steps observed yet")
        if not self._distinct:
            y0 = self._sum_y / self.count
            return max(y0, y0 * tokens / self._first_x)
        return self.a + self.b * tokens

    def max_tokens_within(self, seconds: float) -> int:
        """Largest token count whose predicted step fits in ``seconds``; -1 if none."""
        if not self.ready:
            raise RuntimeError("no steps observed yet")
        if not self._distinct:
            y0 = self._sum_y / self.count
            return math.floor(seconds * self._first_x / y0) if seconds >= y0 else -1
        if seconds < self.a:
            return -1
        if self.b == 0:
            return 2 ** 31 - 1
        return math.floor((seconds - self.a) / self.b)

    def prefill_time(self, tokens: int, chunk: int) -> float:
        """Predicted time to prefill ``tokens`` alone in chunks of at most ``chunk``.

        Needs ``fitted``: with a fitted line this is a lower bound whenever the
        line is exact (and no other request shares the steps).
        """
        if not self._distinct:
            raise RuntimeError("needs steps with two different token counts")
        if chunk < 1:
            raise ValueError("chunk must be positive")
        steps = math.ceil(tokens / chunk)
        return steps * self.a + self.b * tokens
