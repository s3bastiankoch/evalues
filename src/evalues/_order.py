"""Tests about medians and other percentiles. They need no bounds on the data.

Each observation only counts as "at or below the limit" or "above it", so
these tests work on heavy-tailed data such as latencies with rare large
spikes.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._common import Reporting, choice, error_rate, number, same_length
from ._mean import MeanTest

_FLIP = {"either": "either", "above": "below", "below": "above", "undecided": "undecided"}
_SIGN_WORDS = {"above": "increase", "below": "decrease", "undecided": "undecided"}


class _IndicatorTest(Reporting):
    """A MeanTest on 0/1 indicators; shared by QuantileTest and SignTest."""

    _test: MeanTest

    @property
    def confidence(self) -> float:
        return self._test.confidence

    @property
    def deadline(self) -> int | None:
        return self._test.deadline

    @property
    def n(self) -> int:
        """Number of observations so far."""
        return self._test.n

    @property
    def done(self) -> bool:
        """True once a verdict has been reached or the deadline has passed."""
        return self._test.done

    @property
    def verdict(self) -> str:
        """ "above", "below" or "undecided". The first verdict is final."""
        return _FLIP[self._test.verdict]

    @property
    def decided_at(self) -> int | None:
        """Number of observations at which the verdict was reached."""
        return self._test.decided_at

    @property
    def log_e_value(self) -> float:
        """Natural logarithm of `e_value`."""
        return self._test.log_e_value

    @property
    def e_value(self) -> float:
        """Current strength of evidence. 1 means none."""
        return self._test.e_value

    @property
    def p_value(self) -> float:
        """A p-value that is valid at every moment."""
        return self._test.p_value

    def _head(self) -> str | None:
        if self.n == 0:
            return "No observations yet."
        if self._test.out_of_time:
            return f"Stopped at the deadline of {self.deadline} observations without a decision."
        if not self.done:
            return f"After {self.n} observations: no decision yet."
        return None


class QuantileTest(_IndicatorTest):
    _estimate = "share_at_or_below"

    """Is a percentile (the median, the p99, ...) above or below `limit`?

        test = QuantileTest(limit=120.0, quantile=0.99, direction="above")
        for latency in stream:                 # is the p99 above the 120 ms limit?
            test.update(latency)
            if test.done:
                break

    `limit` must be a fixed number you chose beforehand, such as a service
    level objective. Do not use a percentile measured on another sample (for
    example the old build's p99): its own uncertainty would be ignored and
    false alarms become frequent. To compare two builds use `DominanceTest`.

    `quantile` is the percentile as a fraction: 0.5 is the median, 0.99 the p99.

    `direction` says what you are looking for:
      "either"  the percentile differs from the limit
      "above"   the percentile is above the limit
      "below"   the percentile is below the limit

    `verdict` is "above", "below" or "undecided". "above" means strictly
    above the limit; "below" means at or below it (with whole-number data a
    percentile can sit exactly on the limit). The chance that a wrong verdict
    ever appears is at most 1 - confidence.

    Assumes that the distribution does not drift while you measure: each
    observation, given the earlier ones, falls at or below the limit with
    the same probability. `deadline=N` works as in `MeanTest`.
    """

    def __init__(
        self,
        limit: float,
        quantile: float = 0.5,
        *,
        confidence: float = 0.95,
        direction: str = "either",
        deadline: int | None = None,
    ):
        self.limit = number("limit", limit)
        self.quantile = number("quantile", quantile)
        if not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be between 0 and 1, for example 0.99; got {self.quantile:g}")
        self.direction = choice("direction", direction, ("either", "above", "below"))
        # a higher percentile means fewer observations at or below the limit
        self._test = MeanTest(self.quantile, (0.0, 1.0), confidence=confidence,
                              direction=_FLIP[self.direction], deadline=deadline)

    def update(self, x: float) -> None:
        """Add one observation. Any number is allowed."""
        self._test.update(1.0 if number("observation", x) <= self.limit else 0.0)

    def update_many(self, xs: Iterable[float]) -> None:
        """Add several observations in order (up to the deadline, if one is set)."""
        for x in xs:
            if self.deadline is not None and self.n >= self.deadline:
                break
            self.update(x)

    @property
    def share_at_or_below(self) -> float:
        """Share of observations at or below the limit so far."""
        return self._test.mean

    def summary(self) -> str:
        """One plain sentence about where the test stands."""
        seen = f"{100 * self.share_at_or_below:.3g} % of observations were at or below {self.limit:g}."
        head = self._head()
        if head:
            return head if self.n == 0 else f"{head} {seen}"
        name = "median" if self.quantile == 0.5 else f"{100 * self.quantile:g}th percentile"
        side = "above" if self.verdict == "above" else "at or below"
        return (f"Decided after {self.decided_at} observations: the {name} is {side} "
                f"{self.limit:g} ({error_rate(self._test._alpha)}). {seen}")

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state."""
        return {"kind": "QuantileTest", "limit": self.limit, "quantile": self.quantile,
                "direction": self.direction, "test": self._test.to_dict()}

    @classmethod
    def from_dict(cls, state: dict) -> QuantileTest:
        """Continue from a snapshot made by `to_dict`."""
        test = cls(state["limit"], state["quantile"], direction=state["direction"])
        test._test = MeanTest.from_dict(state["test"])
        return test


class SignTest(_IndicatorTest):
    """Is new larger than old more often than it is smaller, or the reverse?

        test = SignTest()
        for new, old in pairs:
            test.update(new, old)              # or test.update(new - old)

    Only the sign of each difference counts, not its size, so outliers cannot
    dominate. Ties count half for each side. (The sequential sign test.)

    `direction="increase"` looks for new being larger more often, "decrease"
    for new being smaller more often; `verdict` is "increase", "decrease" or
    "undecided". With a single argument, `update(x)` compares x against
    `center` (0 by default).

    The chance that a wrong verdict ever appears is at most 1 - confidence.
    `deadline=N` works as in `MeanTest`.
    """

    _estimate = "share_above"

    def __init__(
        self,
        *,
        center: float = 0.0,
        confidence: float = 0.95,
        direction: str = "either",
        deadline: int | None = None,
    ):
        self.center = number("center", center)
        self.direction = choice("direction", direction, ("either", "increase", "decrease"))
        inner = {"either": "either", "increase": "below", "decrease": "above"}[self.direction]
        self._test = MeanTest(0.5, (0.0, 1.0), confidence=confidence, direction=inner, deadline=deadline)

    @property
    def verdict(self) -> str:
        """ "increase" (new is larger more often), "decrease" or "undecided"."""
        return _SIGN_WORDS[_FLIP[self._test.verdict]]

    def update(self, new: float, old: float | None = None) -> None:
        """Add one pair `(new, old)`, or one difference."""
        x = number("observation", new) if old is None else number("new", new) - number("old", old)
        self._test.update(0.5 if x == self.center else 1.0 if x < self.center else 0.0)

    def update_many(self, new: Iterable[float], old: Iterable[float] | None = None) -> None:
        """Add several pairs, or several differences, in order. Items beyond
        the deadline, if one is set, are ignored."""
        if old is not None:
            same_length(new, old)
        pairs = ((x, None) for x in new) if old is None else zip(new, old, strict=True)
        for a, b in pairs:
            if self.deadline is not None and self.n >= self.deadline:
                break
            self.update(a, b)

    @property
    def share_above(self) -> float:
        """Share of differences above the centre so far (ties count half)."""
        return 1.0 - self._test.mean

    def summary(self) -> str:
        """One plain sentence about where the test stands."""
        seen = f"{100 * self.share_above:.3g} % of differences were above {self.center:g}."
        head = self._head()
        if head:
            return head if self.n == 0 else f"{head} {seen}"
        side = "above" if self.verdict == "increase" else "below"
        return (f"Decided after {self.decided_at} observations: differences are {side} "
                f"{self.center:g} more often than not ({error_rate(self._test._alpha)}). {seen}")

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state."""
        return {"kind": "SignTest", "center": self.center, "direction": self.direction,
                "test": self._test.to_dict()}

    @classmethod
    def from_dict(cls, state: dict) -> SignTest:
        """Continue from a snapshot made by `to_dict`."""
        test = cls(center=state["center"], direction=state["direction"])
        test._test = MeanTest.from_dict(state["test"])
        return test
