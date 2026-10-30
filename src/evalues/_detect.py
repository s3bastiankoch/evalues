"""Watching a stream for the moment its average changes."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ._common import Reporting, choice, number, pair

# stakes as shares of the largest stake that keeps capital positive
_STAKES = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
STATE_VERSION = 1


class ChangeDetector(Reporting):
    _estimate = "mean"

    """Raise an alarm when the average of a stream moves away from `baseline`.

        # error rate normally 1 %; 100,000 requests a day; about one false alarm a month
        detector = ChangeDetector(baseline=0.01, false_alarm_every=30 * 100_000, direction="increase")
        for failed in stream:
            detector.update(failed)
            if detector.alarm:
                print(detector.summary())
                detector.reset()

    A test asks one question and stops. A detector runs indefinitely, so its
    promise is about time and there is no `confidence`.

    `false_alarm_every` is counted in observations: while the average stays
    at the baseline, false alarms come on average at most once per that many
    observations, and the chance of a false alarm within the first t
    observations is at most t / false_alarm_every. A larger value means fewer
    false alarms and a slower reaction. (Statisticians: the average run
    length.)

    How fast a real change is caught depends on its size. In a simulation
    with baseline 1 % and one false alarm per 3,000,000, a jump to 5 % was
    caught after about 260 observations and a jump to 2 % after about 15,000
    (medians of 20 runs). Small shifts take long.

    `direction`:
      "either"    alarm on any change; assumes the average is at the baseline before
      "increase"  alarm on an increase; assumes the average is at most the baseline before
      "decrease"  alarm on a decrease; assumes the average is at least the baseline before
    If the normal level is only known as "at most the baseline", use "increase";
    with "either" a service running below its baseline will alarm.

    `bounds` are the hard limits of one observation, (0, 1) for pass/fail.

    Shin, Ramdas and Rinaldo (2024), "E-detectors"; this is the
    Shiryaev-Roberts form averaged over a few fixed stakes.
    """

    def __init__(
        self,
        baseline: float,
        bounds: tuple[float, float] = (0.0, 1.0),
        *,
        false_alarm_every: float,
        direction: str = "either",
        clip: bool = False,
    ):
        lo, hi = pair("bounds", bounds)
        self.baseline = number("baseline", baseline)
        if not lo < self.baseline < hi:
            raise ValueError(f"baseline {self.baseline:g} must lie strictly inside bounds ({lo:g}, {hi:g})")
        self.false_alarm_every = number(
            "false_alarm_every", false_alarm_every, "; it is a number of observations"
        )
        if not self.false_alarm_every > 1:
            raise ValueError(f"false_alarm_every must be larger than 1, got {self.false_alarm_every:g}")
        self.direction = choice("direction", direction, ("either", "increase", "decrease"))
        self.bounds = (lo, hi)
        self.clip = bool(clip)
        self._nu = (self.baseline - lo) / (hi - lo)
        up = _STAKES / self._nu if self.direction in ("either", "increase") else np.array([])
        down = -_STAKES / (1.0 - self._nu) if self.direction in ("either", "decrease") else np.array([])
        self._stakes = np.concatenate([up, down])
        self._n_up = up.size
        self.reset()

    def reset(self) -> None:
        """Start over, for example after an alarm has been handled."""
        self._sums = np.zeros(self._stakes.size)
        self._n = 0
        self._total = 0.0
        self._alarm_at: int | None = None
        self._verdict = "undecided"

    def update(self, x: float) -> None:
        """Add one observation."""
        x = number("observation", x)
        lo, hi = self.bounds
        if x < lo or x > hi:
            if not self.clip:
                raise ValueError(
                    f"observation {x:g} is outside the bounds ({lo:g}, {hi:g}). "
                    "Widen the bounds or pass clip=True."
                )
            x = min(max(x, lo), hi)
        u = (x - lo) / (hi - lo)
        # For each stake: the summed capital of bets started at every past moment.
        self._sums = np.minimum((self._sums + 1.0) * (1.0 + self._stakes * (u - self._nu)), 1e300)
        self._n += 1
        self._total += x
        if self._alarm_at is None and self.statistic >= self.false_alarm_every:
            self._alarm_at = self._n
            up, down = self._sums[: self._n_up].sum(), self._sums[self._n_up :].sum()
            self._verdict = "increase" if up >= down else "decrease"

    def update_many(self, xs: Iterable[float]) -> None:
        """Add several observations in order."""
        for x in xs:
            self.update(x)

    @property
    def n(self) -> int:
        """Number of observations since the start or the last `reset()`."""
        return self._n

    @property
    def mean(self) -> float:
        """Plain average since the start or the last `reset()`."""
        return self._total / self._n if self._n else math.nan

    @property
    def statistic(self) -> float:
        """The detector's internal value. The alarm goes off when it reaches
        `false_alarm_every`."""
        return float(self._sums.mean())

    @property
    def alarm(self) -> bool:
        """True once the alarm has gone off. Stays True until `reset()`."""
        return self._alarm_at is not None

    @property
    def alarm_at(self) -> int | None:
        """Observation number at which the alarm went off."""
        return self._alarm_at

    @property
    def done(self) -> bool:
        """Same as `alarm`."""
        return self.alarm

    @property
    def decided_at(self) -> int | None:
        """Same as `alarm_at`."""
        return self._alarm_at

    @property
    def verdict(self) -> str:
        """ "increase" or "decrease" after an alarm (the direction that set it
        off), "undecided" before."""
        return self._verdict

    def summary(self) -> str:
        """One plain sentence about where the detector stands."""
        if self.alarm:
            side = "risen above" if self._verdict == "increase" else "fallen below"
            return (f"Alarm at observation {self._alarm_at}: the average appears to have {side} "
                    f"{self.baseline:g}. False alarms like this come at most about once per "
                    f"{self.false_alarm_every:,.0f} observations.")
        return f"After {self._n} observations: no alarm."

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state."""
        return {
            "kind": "ChangeDetector",
            "version": STATE_VERSION,
            "baseline": self.baseline,
            "bounds": list(self.bounds),
            "false_alarm_every": self.false_alarm_every,
            "direction": self.direction,
            "clip": self.clip,
            "sums": self._sums.tolist(),
            "n": self._n,
            "total": self._total,
            "alarm_at": self._alarm_at,
            "verdict": self._verdict,
        }

    @classmethod
    def from_dict(cls, state: dict) -> ChangeDetector:
        """Continue from a snapshot made by `to_dict`."""
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported state version {state.get('version')!r}")
        d = cls(state["baseline"], tuple(state["bounds"]), false_alarm_every=state["false_alarm_every"],
                direction=state["direction"], clip=state["clip"])
        d._sums = np.asarray(state["sums"], dtype=float)
        d._n = int(state["n"])
        d._total = float(state["total"])
        d._alarm_at = state["alarm_at"]
        d._verdict = state["verdict"]
        return d
