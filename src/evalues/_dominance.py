"""Comparing two whole distributions, not only their means."""

from __future__ import annotations

import math

import numpy as np

from ._common import Reporting, alpha_from, error_rate, number, same_length, whole
from ._games import safe_exp

STATE_VERSION = 1
_MAX_STAKE = 1.0 - 1e-4


class _Sorted:
    """A growing sorted array of floats."""

    def __init__(self, values=()):
        self.n = len(values)
        self._a = np.empty(max(64, 2 * self.n))
        self._a[: self.n] = np.sort(np.asarray(values, dtype=float))

    def add(self, v: float) -> None:
        if self.n == self._a.size:
            self._a = np.concatenate([self._a, np.empty(self._a.size)])
        i = int(np.searchsorted(self._a[: self.n], v))
        self._a[i + 1 : self.n + 1] = self._a[i : self.n]
        self._a[i] = v
        self.n += 1

    @property
    def values(self) -> np.ndarray:
        return self._a[: self.n]

    def count_at_or_below(self, z: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.values, z, side="right")


class DominanceTest(Reporting):
    _estimate = "share_new_higher"

    """Does the new measurement exceed some level more often than the old one?

        test = DominanceTest()
        for new, old in paired_latencies:
            test.update(new, old)
            if test.done:
                break            # new is higher somewhere in the distribution

    A test of averages misses changes that leave the average alone, such as a
    heavier tail. This test looks at every level at once and asks whether new
    lands above it more often than old does, for any level.

    `verdict` is "higher_somewhere" or "undecided". "higher_somewhere" does
    not mean that new is higher everywhere, only that there is at least one
    level it exceeds more often; `where` gives the level the test was betting
    on most when it decided, as a hint where to look. If old is at least as large as new at every level
    (statisticians: old first-order stochastically dominates new), the chance
    that this verdict ever appears is at most 1 - confidence. The test looks
    one way only; to ask whether new is lower somewhere, swap the arguments.
    It does not say which percentile moved: a verdict can come from the
    median as well as from the tail.

    Needs no bounds and no assumption about the shape of the data. The pairs
    must be independent draws from one joint distribution; new and old within
    a pair may depend on each other in any way, so paired measurements on the
    same machine are fine.

    Each round bets on a set of levels (percentiles of the data seen so far),
    weighting levels and sizing bets by how the two samples differed in the
    past. Every pair is kept, so memory and the size of `to_dict()` grow with
    the number of pairs. `deadline=N` stops the test after N pairs.

    Arnold, Choe, Scarsini and Tsetlin (2026), "Betting on bets: anytime-valid
    tests for stochastic dominance", arXiv:2604.21851. Bet sizes and level
    weights follow the defaults of the authors' reference code.
    """

    def __init__(self, *, confidence: float = 0.95, deadline: int | None = None,
                 levels: int = 101, sharpness: float = 2.0):
        if whole("levels", levels, at_least=2) is None:
            raise ValueError("levels must be a whole number of at least 2")
        if number("sharpness", sharpness) < 0:
            raise ValueError("sharpness must not be negative")
        self.confidence = float(confidence)
        self._alpha = alpha_from(confidence)
        self.deadline = whole("deadline", deadline, at_least=2)
        self._where: float | None = None
        self._higher = 0.0
        self._decided_at: int | None = None
        self.levels = int(levels)
        self.sharpness = float(sharpness)
        self._x, self._y, self._max, self._pooled = _Sorted(), _Sorted(), _Sorted(), _Sorted()
        self._log_e = 0.0
        self._max_log_e = 0.0

    @property
    def n(self) -> int:
        """Number of pairs so far."""
        return self._x.n

    def _plan(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Levels, weights and stakes for the next pair, from past pairs only.
        Internally x is the old and y the new measurement."""
        pooled = self._pooled.values
        if pooled.size <= self.levels:
            z = np.unique(pooled)
        else:
            idx = np.round(np.linspace(0, pooled.size - 1, self.levels)).astype(int)
            z = np.unique(pooled[idx])
        both = self._max.count_at_or_below(z)
        up = self._x.count_at_or_below(z) - both       # past pairs with x <= z < y
        down = self._y.count_at_or_below(z) - both     # past pairs with y <= z < x
        total = up + down
        stakes = np.clip((up - down) / (total + 1.0), 0.0, _MAX_STAKE)
        score = np.where(total > 0, (up - down) / np.sqrt(np.maximum(total, 1)), 0.0)
        weights = np.exp(self.sharpness * (score - score.max()))
        return z, weights / weights.sum(), stakes

    def update(self, new: float, old: float) -> None:
        """Add one pair."""
        if self.deadline is not None and self.n >= self.deadline:
            raise ValueError(f"the deadline of {self.deadline} pairs has been reached")
        x, y = number("old", old), number("new", new)
        if self.n > 0:
            z, weights, stakes = self._plan()
            d = (x <= z).astype(float) - (y <= z).astype(float)
            self._log_e += math.log(float(np.sum(weights * (1.0 + stakes * d))))
            self._max_log_e = max(self._max_log_e, self._log_e)
            if self._decided_at is None:
                self._where = float(z[int(np.argmax(weights * stakes))])
                if self._log_e >= math.log(1.0 / self._alpha):
                    self._decided_at = self.n + 1
        self._higher += 1.0 if y > x else 0.5 if y == x else 0.0
        self._x.add(x)
        self._y.add(y)
        self._max.add(max(x, y))
        self._pooled.add(x)
        self._pooled.add(y)

    def update_many(self, new, old) -> None:
        """Add several pairs in order. Pairs beyond the deadline, if one is
        set, are ignored."""
        same_length(new, old)
        for a, b in zip(new, old, strict=True):
            if self.deadline is not None and self.n >= self.deadline:
                break
            self.update(a, b)

    @property
    def log_e_value(self) -> float:
        """Natural logarithm of `e_value`."""
        return self._log_e

    @property
    def e_value(self) -> float:
        """Current strength of evidence that new is higher somewhere. 1 means none."""
        return safe_exp(self._log_e)

    @property
    def p_value(self) -> float:
        """A p-value that is valid at every moment: 1 / (largest e-value so far)."""
        return min(1.0, math.exp(-self._max_log_e))

    @property
    def done(self) -> bool:
        """True once the verdict has been reached or the deadline has passed."""
        return self._decided_at is not None or (self.deadline is not None and self.n >= self.deadline)

    @property
    def verdict(self) -> str:
        """ "higher_somewhere" or "undecided"."""
        return "higher_somewhere" if self._decided_at is not None else "undecided"

    @property
    def decided_at(self) -> int | None:
        """Number of pairs at which the verdict was reached."""
        return self._decided_at

    @property
    def share_new_higher(self) -> float:
        """Share of pairs in which new was higher than old (ties count half)."""
        return self._higher / self.n if self.n else math.nan

    @property
    def where(self) -> float | None:
        """The level the test was betting on most when the verdict was reached
        (before that: in the latest round). A hint where to look, not a finding."""
        return self._where

    @property
    def where_percentile(self) -> float | None:
        """Share of old values at or below `where`: 0.95 means the hint points
        at about the 95th percentile of old."""
        if self._where is None or self.n == 0:
            return None
        return float(self._x.count_at_or_below(np.array([self._where]))[0]) / self.n

    def summary(self) -> str:
        """One plain sentence about where the test stands."""
        if self.n == 0:
            return "No pairs yet."
        if self._decided_at is not None:
            return (f"Decided after {self._decided_at} pairs: new exceeds some level more often than old "
                    f"does ({error_rate(self._alpha)}). The largest bet at that moment was on level "
                    f"{self._where:.4g}, about the {100 * self.where_percentile:.3g}th percentile of old "
                    "(a hint where to look, not a finding).")
        if self.deadline is not None and self.n >= self.deadline:
            return (f"Stopped at the deadline of {self.deadline} pairs without a decision. "
                    "This is not evidence that new is nowhere higher.")
        return (f"After {self.n} pairs: no decision yet; evidence {self.e_value:.3g} "
                f"of the {1 / self._alpha:.3g} needed.")

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state. It holds every pair."""
        # x and y are stored sorted; their pairing is not needed again, only
        # the per-pair maximum, which is kept separately
        return {
            "kind": "DominanceTest",
            "version": STATE_VERSION,
            "confidence": self.confidence,
            "deadline": self.deadline,
            "where": self._where,
            "higher": self._higher,
            "decided_at": self._decided_at,
            "levels": self.levels,
            "sharpness": self.sharpness,
            "x": self._x.values.tolist(),
            "y": self._y.values.tolist(),
            "max": self._max.values.tolist(),
            "log_e": self._log_e,
            "max_log_e": self._max_log_e,
        }

    @classmethod
    def from_dict(cls, state: dict) -> DominanceTest:
        """Continue from a snapshot made by `to_dict`."""
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported state version {state.get('version')!r}")
        test = cls(confidence=state["confidence"], deadline=state["deadline"],
                   levels=state["levels"], sharpness=state["sharpness"])
        test._where, test._decided_at = state["where"], state["decided_at"]
        test._higher = float(state["higher"])
        test._x, test._y, test._max = _Sorted(state["x"]), _Sorted(state["y"]), _Sorted(state["max"])
        test._pooled = _Sorted(state["x"] + state["y"])
        test._log_e = float(state["log_e"])
        test._max_log_e = float(state["max_log_e"])
        return test
