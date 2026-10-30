"""Tests and intervals for an average that stay valid while you watch.

Every class works the same way: create one, call `update(...)` for each new
observation, and read `.done`, `.verdict` or `.summary()` whenever you like.
Checking after every observation and stopping as soon as you are satisfied
does not break the guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ._common import Reporting, alpha_from, choice, error_rate, number, pair, same_length, whole
from ._edo import edo_stake
from ._games import _EDGE, Games, log_mean_exp, safe_exp


def rel_diff(new: float, old: float) -> float:
    """Relative difference (new - old) / (new + old) of two positive measurements.

    It always lies in [-1, 1]. For small differences it is about half the
    percentage change: new = 1.02 * old gives roughly 0.01. `Comparison` does
    this conversion for you.
    """
    if new < 0 or old < 0 or new + old <= 0:
        raise ValueError(f"rel_diff needs non-negative values with a positive sum, got {new}, {old}")
    return (new - old) / (new + old)


class _Deadline:
    """Shared handling of an optional maximum number of observations."""

    deadline: int | None

    def _check_open(self) -> None:
        if self.deadline is not None and self.n >= self.deadline:
            raise ValueError(
                f"the deadline of {self.deadline} observations has been reached; "
                "start a new object to keep measuring"
            )

    @property
    def out_of_time(self) -> bool:
        """True if the deadline was reached without a verdict."""
        return self.deadline is not None and self.n >= self.deadline and self.verdict == "undecided"


class MeanTest(Reporting, _Deadline):
    """Is the average above or below `baseline`?

        test = MeanTest(baseline=0.5, bounds=(0, 1))
        for x in stream:
            test.update(x)
            if test.done:
                break
        print(test.summary())

    `baseline` is a fixed number you chose beforehand (a target, last
    quarter's rate), not something estimated from the same data.

    `bounds` are the hard limits of what you measure: (0, 1) for pass/fail,
    (0, timeout) for a duration. Do not pick them from the data you are about
    to test; that breaks the guarantee. Tighter true limits give faster
    answers.

    `direction` says what you are looking for:
      "either"  the average differs from the baseline, whichever way
      "above"   the average is above the baseline
      "below"   the average is below the baseline

    `verdict` is "above", "below" or "undecided". If the average really is at
    the baseline (or on the side you are not looking for), the chance that a
    wrong verdict ever appears is at most 1 - confidence, no matter how long
    you watch. "undecided" is not evidence that the average equals the
    baseline; use `DifferenceTest` or `MeanInterval` for that.

    Options:

    `deadline=N`: you will take at most N observations. The test stops there,
    and bets grow bolder as N approaches, which helps a test that is close to
    a decision get there in time (Taga, Oymak and Shekhar 2026; their hedge
    over six schedules, with the random choice between stakes replaced by its
    average).

    `expected=m`: you know roughly what average to expect if the baseline is
    wrong. The test then looks only in that direction and uses one fixed
    stake: the one with the best guaranteed growth over every distribution
    whose average is `m` or further from the baseline (Arnold and Clerico
    2026). It is fast when the guess is right or cautious, and slow when the
    true average is much closer to the baseline than `m`; the default needs
    no guess and adapts.

    `expected=m, hoped_within=T`: in addition you hope for the decision within
    about T observations. The stake becomes bolder, the
    exponential-decay-optimal bet of Clerico, Wegel, Azangulov and Rebeschini
    (2026). It is optimal for 0/1 data; for other bounded data it is valid and
    no longer optimal.

    `population_size=N`: the observations are drawn without repeats from a
    finite set of N items, for example the questions of a benchmark in random
    order. The test is then about the average of the whole set and sharpens
    as the set is used up.

    `clip=True`: observations outside the bounds are clipped instead of
    raising an error; the test is then about the average of the clipped
    values.
    """

    def __init__(
        self,
        baseline: float,
        bounds: tuple[float, float] = (0.0, 1.0),
        *,
        confidence: float = 0.95,
        direction: str = "either",
        deadline: int | None = None,
        expected: float | None = None,
        hoped_within: float | None = None,
        population_size: int | None = None,
        clip: bool = False,
    ):
        lo, hi = pair("bounds", bounds)
        self.baseline = number("baseline", baseline)
        if not lo <= self.baseline <= hi:
            raise ValueError(
                f"baseline {self.baseline:g} is outside bounds ({lo:g}, {hi:g}). If it is a "
                "percentage, pass it as a fraction or widen the bounds, for example bounds=(0, 100)."
            )
        self.bounds = (lo, hi)
        self.confidence = float(confidence)
        self._alpha = alpha_from(confidence)
        self.direction = choice("direction", direction, ("either", "above", "below"))
        self.deadline = whole("deadline", deadline, at_least=2)
        if hoped_within is not None and expected is None:
            raise ValueError("hoped_within needs expected, the average you expect if the baseline is wrong")
        if population_size is not None and (deadline is not None or expected is not None):
            raise ValueError("population_size cannot be combined with deadline or expected")
        if deadline is not None and expected is not None:
            raise ValueError("deadline and expected cannot be combined; pick one")
        fixed = None
        if expected is not None:
            expected = number("expected", expected)
            if not (lo <= expected <= hi) or expected == self.baseline:
                raise ValueError("expected must lie within the bounds and differ from the baseline")
            side = "above" if expected > self.baseline else "below"
            if self.direction not in ("either", side):
                raise ValueError(
                    f"expected={expected:g} is {side} the baseline, but direction={self.direction!r}"
                )
            self.direction = side
            if hoped_within is not None and not number("hoped_within", hoped_within) > 0:
                raise ValueError("hoped_within must be a positive number of observations")
            p0 = min(max((self.baseline - lo) / (hi - lo), _EDGE), 1.0 - _EDGE)
            p1 = (expected - lo) / (hi - lo)
            if side == "below":                      # mirror, so that p1 > p0 below
                p0, p1 = 1.0 - p0, 1.0 - p1
            if not p1 > p0:
                raise ValueError(
                    "expected is too close to the baseline, or both are too close to a bound, "
                    "to bet on the difference"
                )
            if hoped_within is None:
                stake = (p1 - p0) / (p0 * (1.0 - p0))      # worst-case-optimal fixed stake
            else:
                stake = edo_stake(p0, p1, hoped_within)
            stake = min(stake, (1.0 - 1e-3) / p0)
            fixed = (stake, 0.0) if side == "above" else (0.0, stake)
        self._games = Games([self.baseline], lo, hi, clip, population_size, self.deadline, fixed)
        self._max_log_e = 0.0
        self._verdict = "undecided"
        self._decided_at: int | None = None

    def update(self, x: float) -> None:
        """Add one observation."""
        self._check_open()
        self._games.update(number("observation", x))
        log_e = self.log_e_value
        self._max_log_e = max(self._max_log_e, log_e)
        if self._verdict == "undecided" and log_e >= math.log(1.0 / self._alpha):
            up, down = float(self._games.log_up[0]), float(self._games.log_down[0])
            if self.direction == "either":
                self._verdict = "above" if up >= down else "below"
            else:
                self._verdict = self.direction
            self._decided_at = self.n

    def update_many(self, xs: Iterable[float]) -> None:
        """Add several observations in order (up to the deadline, if one is set)."""
        for x in xs:
            if self.deadline is not None and self.n >= self.deadline:
                break
            self.update(x)

    @property
    def n(self) -> int:
        """Number of observations so far."""
        return self._games.n

    @property
    def mean(self) -> float:
        """Plain average of the observations. An estimate, not a guarantee."""
        return self._games.mean

    @property
    def done(self) -> bool:
        """True once a verdict has been reached or the deadline has passed."""
        return self._verdict != "undecided" or self.out_of_time

    @property
    def verdict(self) -> str:
        """ "above", "below" or "undecided". The first verdict is final."""
        return self._verdict

    @property
    def decided_at(self) -> int | None:
        """Number of observations at which the verdict was reached."""
        return self._decided_at

    @property
    def log_e_value(self) -> float:
        """Natural logarithm of `e_value`."""
        up, down = float(self._games.log_up[0]), float(self._games.log_down[0])
        if self.direction == "above":
            return up
        if self.direction == "below":
            return down
        return log_mean_exp(up, down)

    @property
    def e_value(self) -> float:
        """Current strength of evidence against the baseline. 1 means none;
        a verdict is reached at 1 / (1 - confidence), which is 20 by default."""
        return safe_exp(self.log_e_value)

    @property
    def p_value(self) -> float:
        """A p-value that is valid at every moment: 1 / (largest e-value so far)."""
        return min(1.0, math.exp(-self._max_log_e))

    def summary(self) -> str:
        """One plain sentence about where the test stands."""
        if self.n == 0:
            return "No observations yet."
        seen = f"Observed average {self.mean:.4g}"
        if self._verdict != "undecided":
            text = (f"Decided after {self._decided_at} observations: the average is {self._verdict} "
                    f"{self.baseline:g} ({error_rate(self._alpha)}).")
            later = self.n - self._decided_at
            if later == 0:
                return f"{text} {seen}."
            text += f" {later} more were added since; the observed average is now {self.mean:.4g}."
            if (self.mean > self.baseline) != (self._verdict == "above"):
                text += " Later data point the other way, so the average probably moved during the test."
            return text
        if self.out_of_time:
            return (f"Stopped at the deadline of {self.deadline} observations without a decision. "
                    f"This is not evidence that the average equals {self.baseline:g}. {seen}.")
        return (f"After {self.n} observations: no decision yet. {seen}; "
                f"evidence {self.e_value:.3g} of the {1 / self._alpha:.3g} needed.")

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state, to store or to send elsewhere."""
        return {
            "kind": "MeanTest",
            "baseline": self.baseline,
            "confidence": self.confidence,
            "direction": self.direction,
            "max_log_e": self._max_log_e,
            "verdict": self._verdict,
            "decided_at": self._decided_at,
            "games": self._games.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> MeanTest:
        """Continue from a snapshot made by `to_dict`."""
        g = state["games"]
        test = cls(state["baseline"], (g["lo"], g["hi"]), confidence=state["confidence"],
                   direction=state["direction"])
        test._games = Games.from_dict(g)
        test.deadline = test._games.deadline
        test._max_log_e = float(state["max_log_e"])
        test._verdict = state["verdict"]
        test._decided_at = state["decided_at"]
        return test


class MeanInterval(Reporting):
    """A range for the average that is valid at every moment.

        estimate = MeanInterval(bounds=(0, 1), margin=0.03)     # to within +/- 3 points
        for x in stream:
            estimate.update(x)
            if estimate.done:
                break
        print(estimate.low, estimate.high)

    If the average does not move while you measure, the true average is
    inside the interval at every moment simultaneously, in at least
    `confidence` of all uses. So you can stop when it is narrow enough. An
    ordinary confidence interval only promises this for one sample size chosen
    in advance. (Statisticians call this a confidence sequence.)

    `done` becomes True when the interval is at most `2 * margin` wide (or
    `target_width`, if you prefer to give the full width), and stays True.

    `bounds` are the hard limits of what you measure, as in `MeanTest`.

    If the average drifts while you measure, the data can rule out every
    single value. `consistent` then turns False and `interval` becomes
    (nan, nan).

    The interval is computed on a grid of candidate values, so it cannot get
    narrower than one grid step. `resolution` sets the number of grid steps
    and is chosen automatically from `margin`, up to 20,000; a margin below
    1/20,000 of the bounds' width cannot be reached.

    `population_size` and `clip` work as in `MeanTest`.
    """

    def __init__(
        self,
        bounds: tuple[float, float] = (0.0, 1.0),
        *,
        confidence: float = 0.95,
        margin: float | None = None,
        target_width: float | None = None,
        population_size: int | None = None,
        resolution: int | None = None,
        clip: bool = False,
    ):
        lo, hi = pair("bounds", bounds)
        self.bounds = (lo, hi)
        self.confidence = float(confidence)
        self._alpha = alpha_from(confidence)
        if margin is not None and target_width is not None:
            raise ValueError("give margin or target_width, not both (target_width = 2 * margin)")
        if margin is not None:
            target_width = 2.0 * number("margin", margin)
        if target_width is not None:
            target_width = number("target_width", target_width)
            if not 0 < target_width < hi - lo:
                raise ValueError(
                    f"the requested width {target_width:g} must be positive and smaller than the "
                    f"bounds ({lo:g}, {hi:g}) are wide; for +/-3 points on a 0 to 1 scale use margin=0.03"
                )
        self.target_width = target_width
        if resolution is None:
            resolution = 200
            if target_width is not None:
                resolution = min(20_000, max(200, math.ceil(4 * (hi - lo) / target_width)))
                if target_width < 2 * (hi - lo) / resolution:
                    raise ValueError(
                        f"the requested width {target_width:g} is too fine for bounds ({lo:g}, {hi:g}): "
                        f"the smallest reachable width is {2 * (hi - lo) / resolution:g}. "
                        "Use tighter bounds or a larger margin."
                    )
        if whole("resolution", resolution, at_least=2) is None:
            raise ValueError("resolution must be a whole number of at least 2")
        self._step = (hi - lo) / resolution
        self._games = Games(np.linspace(lo, hi, resolution + 1), lo, hi, clip, population_size)
        # A candidate is ruled out as too low once the game betting "the average
        # is above it" has won, and as too high once the opposite game has. Each
        # side gets half of the error budget. The interval runs from the top of
        # the unbroken run of too-low candidates to the bottom of the unbroken
        # run of too-high ones, so it is wrong only if a grid neighbour of the
        # true average was ruled out, which has probability at most alpha.
        self._too_low = np.zeros(resolution + 1, dtype=bool)
        self._too_high = np.zeros(resolution + 1, dtype=bool)
        self._log_threshold = math.log(2.0 / self._alpha)
        self._done_at: int | None = None

    def update(self, x: float) -> None:
        """Add one observation."""
        g = self._games
        g.update(number("observation", x))
        self._too_low |= g.log_up >= self._log_threshold
        self._too_high |= g.log_down >= self._log_threshold
        if self._done_at is None and self.target_width is not None and self.width <= self.target_width:
            self._done_at = self.n

    def update_many(self, xs: Iterable[float]) -> None:
        """Add several observations in order."""
        for x in xs:
            self.update(x)

    @property
    def n(self) -> int:
        """Number of observations so far."""
        return self._games.n

    @property
    def mean(self) -> float:
        """Plain average of the observations. An estimate, not a guarantee."""
        return self._games.mean

    def _edges(self) -> tuple[float, float]:
        g = self._games
        n_low = len(self._too_low) if self._too_low.all() else int(np.argmin(self._too_low))
        n_high = len(self._too_high) if self._too_high.all() else int(np.argmin(self._too_high[::-1]))
        low = float(g.candidates[n_low - 1]) if n_low else g.lo
        high = float(g.candidates[len(g.candidates) - n_high]) if n_high else g.hi
        sure_lo, sure_hi = g.feasible()
        if g.population is not None and (g.n == g.population or low > sure_hi or high < sure_lo):
            # what is known for certain about a finite set overrides the bets
            return sure_lo, sure_hi
        return max(low, sure_lo), min(high, sure_hi)

    @property
    def consistent(self) -> bool:
        """False once the data have ruled out every value for the average,
        which means it probably moved while you measured."""
        low, high = self._edges()
        return low <= high

    @property
    def interval(self) -> tuple[float, float]:
        """Current (low, high) range for the average. It never widens; it
        becomes (nan, nan) if `consistent` is False."""
        low, high = self._edges()
        return (low, high) if low <= high else (math.nan, math.nan)

    @property
    def low(self) -> float:
        """Lower end of the interval."""
        return self.interval[0]

    @property
    def high(self) -> float:
        """Upper end of the interval."""
        return self.interval[1]

    @property
    def width(self) -> float:
        """Width of the interval, high - low."""
        low, high = self.interval
        return high - low

    @property
    def done(self) -> bool:
        """True once the interval has been as narrow as requested. Stays True
        unless the data later turn out not to be `consistent`."""
        return self._done_at is not None and self.consistent

    @property
    def decided_at(self) -> int | None:
        """Number of observations at which the requested width was reached."""
        return self._done_at

    def excludes(self, value: float) -> bool:
        """True if `value` is outside the current interval."""
        low, high = self.interval
        return not low <= value <= high

    def summary(self) -> str:
        """One plain sentence about where the estimate stands."""
        if self.n == 0:
            return "No observations yet."
        seen = f"Observed average {self.mean:.4g}."
        if not self.consistent:
            return (f"After {self.n} observations: the data are not consistent with one fixed average "
                    f"(it probably moved while you measured), so no range can be given. {seen}")
        low, high = self.interval
        return (f"After {self.n} observations: the average is between {low:.4g} and {high:.4g} "
                f"({error_rate(self._alpha)}). {seen}")

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state."""
        return {
            "kind": "MeanInterval",
            "confidence": self.confidence,
            "target_width": self.target_width,
            "too_low": self._too_low.tolist(),
            "too_high": self._too_high.tolist(),
            "done_at": self._done_at,
            "games": self._games.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> MeanInterval:
        """Continue from a snapshot made by `to_dict`."""
        g = state["games"]
        estimate = cls((g["lo"], g["hi"]), confidence=state["confidence"],
                       target_width=state["target_width"], resolution=len(g["candidates"]) - 1)
        estimate._games = Games.from_dict(g)
        estimate._too_low = np.asarray(state["too_low"], dtype=bool)
        estimate._too_high = np.asarray(state["too_high"], dtype=bool)
        estimate._done_at = state["done_at"]
        return estimate


_DIFFERENCE_VERDICTS = {
    "either": ("increase", "decrease", "no_change"),
    "increase": ("increase", "no_increase"),
    "decrease": ("decrease", "no_decrease"),
}


class DifferenceTest(Reporting, _Deadline):
    """Is the average difference larger than `min_difference`, or safely smaller?

        test = DifferenceTest(min_difference=0.5, bounds=(-10, 10))
        for difference in stream:                # e.g. new score minus old score
            test.update(difference)
            if test.done:
                break
        print(test.verdict)

    Feed it differences. For two positive measurements per round, such as run
    times, `Comparison` is easier and needs no bounds.

    `min_difference` is the smallest difference you would act on. The closer
    the true difference is to it, the more observations are needed; exactly
    on it, the test can stay undecided forever, so consider a `deadline`.

    With `direction="either"` the verdict is
      "increase"   the average difference is above +min_difference
      "decrease"   the average difference is below -min_difference
      "no_change"  the average difference is within +/-min_difference
      "undecided"  not enough evidence yet

    With `direction="increase"` only one side matters, which needs fewer
    observations: the verdict is "increase", "no_increase" (the average
    difference is not above +min_difference) or "undecided".
    `direction="decrease"` mirrors this with "decrease" and "no_decrease".

    Whatever the truth, the chance that a wrong verdict ever appears is at
    most 1 - confidence. The first verdict reached is final.

    `bounds` are the hard limits of a single difference, as in `MeanTest`.
    `deadline=N` stops the test after N observations.
    """

    def __init__(
        self,
        min_difference: float,
        bounds: tuple[float, float] = (-1.0, 1.0),
        *,
        confidence: float = 0.95,
        direction: str = "either",
        deadline: int | None = None,
        clip: bool = False,
    ):
        self.min_difference = number("min_difference", min_difference)
        if not self.min_difference > 0:
            raise ValueError("min_difference must be positive; use MeanTest to test against one value")
        lo, hi = pair("bounds", bounds)
        if not (lo < -self.min_difference and self.min_difference < hi):
            raise ValueError(
                f"bounds ({lo:g}, {hi:g}) must extend beyond +/-min_difference ({self.min_difference:g})"
            )
        self.bounds = (lo, hi)
        self.confidence = float(confidence)
        self._alpha = alpha_from(confidence)
        self.direction = choice("direction", direction, ("either", "increase", "decrease"))
        self.deadline = whole("deadline", deadline, at_least=2)
        self._games = Games([-self.min_difference, self.min_difference], lo, hi, clip)
        # Looking both ways, four games can end the test and, whatever the truth,
        # two of them would end it wrongly, so each gets half of the error budget.
        # Looking one way, only one game can be wrong for any given truth.
        self._log_threshold = math.log((2.0 if self.direction == "either" else 1.0) / self._alpha)
        self._max_log_e = 0.0
        self._verdict = "undecided"
        self._decided_at: int | None = None

    def _judge(self) -> str:
        g, thr = self._games, self._log_threshold
        up_hi, down_hi = g.log_up[1], g.log_down[1]        # games against +min_difference
        up_lo, down_lo = g.log_up[0], g.log_down[0]        # games against -min_difference
        if self.direction == "increase":
            return "increase" if up_hi >= thr else "no_increase" if down_hi >= thr else "undecided"
        if self.direction == "decrease":
            return "decrease" if down_lo >= thr else "no_decrease" if up_lo >= thr else "undecided"
        if up_hi >= thr:
            return "increase"
        if down_lo >= thr:
            return "decrease"
        if down_hi >= thr and up_lo >= thr:
            return "no_change"
        return "undecided"

    def update(self, x: float) -> None:
        """Add one difference."""
        self._check_open()
        self._games.update(number("observation", x))
        self._max_log_e = max(self._max_log_e, self.log_e_value)
        if self._verdict == "undecided":
            self._verdict = self._judge()
            if self._verdict != "undecided":
                self._decided_at = self.n

    def update_many(self, xs: Iterable[float]) -> None:
        """Add several differences in order (up to the deadline, if one is set)."""
        for x in xs:
            if self.deadline is not None and self.n >= self.deadline:
                break
            self.update(x)

    @property
    def n(self) -> int:
        """Number of observations so far."""
        return self._games.n

    @property
    def mean(self) -> float:
        """Plain average of the differences. An estimate, not a guarantee."""
        return self._games.mean

    @property
    def verdict(self) -> str:
        """The verdict, or "undecided". The first verdict is final."""
        return self._verdict

    @property
    def decided_at(self) -> int | None:
        """Number of observations at which the verdict was reached."""
        return self._decided_at

    @property
    def done(self) -> bool:
        """True once a verdict has been reached or the deadline has passed."""
        return self._verdict != "undecided" or self.out_of_time

    @property
    def log_e_value(self) -> float:
        """Natural logarithm of `e_value`."""
        g = self._games
        if self.direction == "increase":
            return float(g.log_up[1])
        if self.direction == "decrease":
            return float(g.log_down[0])
        return log_mean_exp(float(g.log_up[1]), float(g.log_down[0]))

    @property
    def e_value(self) -> float:
        """Strength of evidence for a difference beyond min_difference, in the
        direction(s) the test looks at. This is what `combine` and `flag` read."""
        return safe_exp(self.log_e_value)

    @property
    def p_value(self) -> float:
        """A p-value for "no difference beyond min_difference", valid at every moment."""
        return min(1.0, math.exp(-self._max_log_e))

    def _claim(self) -> str:
        d = f"{self.min_difference:g}"
        return {
            "increase": f"the average difference is above +{d}",
            "decrease": f"the average difference is below -{d}",
            "no_change": f"the average difference is within +/-{d}",
            "no_increase": f"the average difference is not above +{d}",
            "no_decrease": f"the average difference is not below -{d}",
        }[self._verdict]

    def _reversal(self) -> str:
        d, mean = self.min_difference, self.mean
        against = {
            "increase": mean < d, "decrease": mean > -d, "no_change": abs(mean) > d,
            "no_increase": mean > d, "no_decrease": mean < -d,
        }[self._verdict]
        if not against:
            return ""
        return " Data added since the verdict point the other way, so things probably moved during the test."

    def summary(self) -> str:
        """One plain sentence about where the test stands."""
        if self.n == 0:
            return "No observations yet."
        seen = f"Observed average difference {self.mean:.4g} (an estimate)."
        if self._verdict != "undecided":
            text = (f"Decided after {self._decided_at} observations: {self._claim()} "
                    f"({error_rate(self._alpha)}). {seen}")
            return text + (self._reversal() if self.n > self._decided_at else "")
        if self.out_of_time:
            return (f"Stopped at the deadline of {self.deadline} observations without a decision. "
                    f"This is not evidence of no difference. {seen}")
        return f"After {self.n} observations: no decision yet. {seen}"

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state."""
        return {
            "kind": "DifferenceTest",
            "min_difference": self.min_difference,
            "confidence": self.confidence,
            "direction": self.direction,
            "deadline": self.deadline,
            "max_log_e": self._max_log_e,
            "verdict": self._verdict,
            "decided_at": self._decided_at,
            "games": self._games.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: dict) -> DifferenceTest:
        """Continue from a snapshot made by `to_dict`."""
        g = state["games"]
        test = cls(state["min_difference"], (g["lo"], g["hi"]), confidence=state["confidence"],
                   direction=state["direction"], deadline=state["deadline"], clip=g["clip"])
        test._games = Games.from_dict(g)
        test._max_log_e = float(state["max_log_e"])
        test._verdict = state["verdict"]
        test._decided_at = state["decided_at"]
        return test


class Comparison(Reporting):
    _estimate = "average_change"

    """Is the new measurement larger or smaller than the old one, by enough to matter?

        comparison = Comparison(min_change=0.02)        # 2 % or more matters to me
        for new, old in paired_latencies:
            comparison.update(new, old)
            if comparison.done:
                break
        print(comparison.summary())

    The easiest entry point for before/after measurements of positive
    quantities such as run times, costs or scores. Give one new and one old
    measurement per round, taken close together so that they share the same
    conditions (alternate which goes first, if you can).

    `min_change` is the smallest change you would act on, as a fraction: 0.02
    means 2 %. The closer the true change is to it, the more pairs are needed.

    `direction`:
      "either"    verdicts "increase", "decrease", "no_change"
      "increase"  only a slowdown matters: verdicts "increase", "no_increase"
      "decrease"  only a drop matters: verdicts "decrease", "no_decrease"
    Looking one way needs fewer pairs, and `flag` then lists only that kind.

    Whatever the truth, the chance that a wrong verdict ever appears is at
    most 1 - confidence. `deadline=N` stops the comparison after N pairs.

    What "change" means here. Each pair is turned into
    (new - old) / (new + old), and the verdict is about the average of that
    over pairs. A doubling and a halving count the same, and no single pair
    counts more than +/-100 %, however extreme. So the verdict is about the
    change averaged over pairs with outliers held in check. It is not the
    change of the total, and it is not "most pairs": a tenth of the pairs
    being three times slower gives "increase" although most are unchanged.
    For "slower more often than not" use `SignTest`; for a heavier tail use
    `DominanceTest`. `min_change` is converted to the same scale, so 0.02
    stands for new = 1.02 x old on the way up and new = old / 1.02 (about
    -1.96 %) on the way down.
    """

    def __init__(
        self,
        min_change: float,
        *,
        confidence: float = 0.95,
        direction: str = "either",
        deadline: int | None = None,
    ):
        self.min_change = number("min_change", min_change, "; use a fraction such as 0.02 for 2 %")
        if not self.min_change > 0:
            raise ValueError("min_change must be positive, for example 0.02 for 2 %")
        if self.min_change >= 1:
            raise ValueError(
                f"min_change is a fraction: 0.02 means 2 %. Got {self.min_change:g}, "
                f"which would mean {100 * self.min_change:g} %."
            )
        self._test = DifferenceTest(self.min_change / (2.0 + self.min_change), (-1.0, 1.0),
                                    confidence=confidence, direction=direction, deadline=deadline)

    def update(self, new: float, old: float) -> None:
        """Add one pair of measurements."""
        if self.deadline is not None and self.n >= self.deadline:
            raise ValueError(f"the deadline of {self.deadline} pairs has been reached; "
                             "start a new object to keep measuring")
        new, old = number("new", new), number("old", old)
        if new < 0 or old < 0 or new + old <= 0:
            raise ValueError(
                f"Comparison needs positive measurements, got new={new:g}, old={old:g}. "
                "For values that can be negative use DifferenceTest on new - old."
            )
        self._test.update((new - old) / (new + old))

    def update_many(self, new: Iterable[float], old: Iterable[float]) -> None:
        """Add several pairs in order. Pairs beyond the deadline, if one is
        set, are ignored."""
        same_length(new, old)
        for a, b in zip(new, old, strict=True):
            if self.deadline is not None and self.n >= self.deadline:
                break
            self.update(a, b)

    @property
    def confidence(self) -> float:
        return self._test.confidence

    @property
    def direction(self) -> str:
        return self._test.direction

    @property
    def deadline(self) -> int | None:
        return self._test.deadline

    @property
    def n(self) -> int:
        """Number of pairs so far."""
        return self._test.n

    @property
    def verdict(self) -> str:
        """The verdict, or "undecided". The first verdict is final."""
        return self._test.verdict

    @property
    def decided_at(self) -> int | None:
        """Number of pairs at which the verdict was reached."""
        return self._test.decided_at

    @property
    def done(self) -> bool:
        """True once a verdict has been reached or the deadline has passed."""
        return self._test.done

    @property
    def average_change(self) -> float:
        """Change of new against old averaged over the pairs so far, as a
        fraction (0.05 means +5 %), with each pair's influence capped as
        described above. An estimate, not a guarantee."""
        d = self._test.mean
        return 2.0 * d / (1.0 - d) if d < 1.0 else math.inf

    @property
    def log_e_value(self) -> float:
        """Natural logarithm of `e_value`."""
        return self._test.log_e_value

    @property
    def e_value(self) -> float:
        """Strength of evidence for a change beyond `min_change`, in the
        direction(s) the comparison looks at."""
        return self._test.e_value

    @property
    def p_value(self) -> float:
        """A p-value for "no change beyond min_change", valid at every moment."""
        return self._test.p_value

    def summary(self) -> str:
        """One plain sentence about where the comparison stands."""
        if self.n == 0:
            return "No pairs yet."
        up = f"{100 * self.min_change:.3g} %"
        down = f"{100 * (1 - 1 / (1 + self.min_change)):.3g} %"
        if math.isfinite(self.average_change):
            seen = f"Average change so far {100 * self.average_change:+.2f} % (an estimate)."
        else:
            seen = "The average change cannot be given because old was 0 in every pair."
        claim = {
            "increase": f"new is more than {up} larger than old",
            "decrease": f"new is more than {down} smaller than old",
            "no_change": f"new is within {up} of old",
            "no_increase": f"new is not more than {up} larger than old",
            "no_decrease": f"new is not more than {down} smaller than old",
        }.get(self.verdict)
        if claim:
            text = (f"Decided after {self.decided_at} pairs: averaged over pairs, {claim} "
                    f"({error_rate(self._test._alpha)}). {seen}")
            if self.verdict.startswith("no_"):
                text += " Each pair counts at most +/-100 %, so rare spikes barely show; use DominanceTest for the tail."
            return text + (self._test._reversal() if self.n > self.decided_at else "")
        if self._test.out_of_time:
            return (f"Stopped at the deadline of {self.deadline} pairs without a decision. "
                    f"This is not evidence of no change. {seen}")
        return f"After {self.n} pairs: no decision yet. {seen}"

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full state."""
        return {"kind": "Comparison", "min_change": self.min_change, "test": self._test.to_dict()}

    @classmethod
    def from_dict(cls, state: dict) -> Comparison:
        """Continue from a snapshot made by `to_dict`."""
        comparison = cls(state["min_change"])
        comparison._test = DifferenceTest.from_dict(state["test"])
        return comparison
