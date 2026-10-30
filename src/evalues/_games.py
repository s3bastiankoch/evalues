"""The betting engine behind every test in this package.

To test a candidate mean, play a game against it. Start with capital 1. Before
each observation, stake part of the capital on the observation landing above
the candidate (the "up" game) or below it (the "down" game). If the candidate
really is the mean, neither game can be won on average, so the chance that
capital ever reaches 1/alpha is at most alpha (Ville's inequality). Capital
that does get there is evidence against the candidate, and looking after every
observation costs nothing.

The stake is sized from past observations only (aGRAPA, Waudby-Smith and
Ramdas 2024). That rule only affects how fast evidence grows. Validity holds
for any stake that ignores the current observation.
"""

from __future__ import annotations

import math

import numpy as np

_EDGE = 1e-4        # candidates are kept this far from 0 and 1
_VAR_FLOOR = 1e-6
_STAKE_CAP = 0.9    # largest stake, as a share of what keeps capital positive
_ALL_IN = 1.0 - 1e-3
_LOG_CAP = 700.0    # log capital is reported no higher than this, so nothing overflows
# (onset as a share of the deadline, exponent) of the six schedules that move
# the stake from the usual one towards all-in as a deadline approaches
_SCHEDULES = [(onset, power) for onset in (0.25, 0.5, 0.75) for power in (1, 2)]
STATE_VERSION = 1


class Games:
    """Up and down betting games against a set of candidate means.

    Observations live in [lo, hi] and are rescaled to [0, 1] internally.
    `log_up[j]` is the log capital of the game betting that the mean is above
    candidate j, `log_down[j]` of the game betting that it is below.
    """

    def __init__(self, candidates, lo: float, hi: float, clip: bool = False,
                 population: int | None = None, deadline: int | None = None,
                 fixed_stakes: tuple[float, float] | None = None):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"bounds must be finite with lo < hi, got ({lo}, {hi})")
        candidates = np.atleast_1d(np.asarray(candidates, dtype=float))
        if candidates.size == 0:
            raise ValueError("need at least one candidate mean")
        if np.any(candidates < lo) or np.any(candidates > hi):
            raise ValueError(f"candidate means must lie within the bounds ({lo}, {hi})")
        self.lo, self.hi = float(lo), float(hi)
        if population is not None and (int(population) != population or population < 1):
            raise ValueError(f"population_size must be a positive integer, got {population}")
        if deadline is not None and (int(deadline) != deadline or deadline < 2):
            raise ValueError(f"deadline must be an integer of at least 2, got {deadline}")
        self.clip = bool(clip)
        self.deadline = None if deadline is None else int(deadline)
        self.fixed_stakes = None if fixed_stakes is None else (float(fixed_stakes[0]),
                                                                float(fixed_stakes[1]))
        self.population = None if population is None else int(population)
        self.candidates = candidates
        self._raw = (candidates - lo) / (hi - lo)
        # one row of capital per betting schedule; a single row unless a deadline is set
        rows = len(_SCHEDULES) if self.deadline is not None else 1
        self._log_up = np.zeros((rows, candidates.size))
        self._log_down = np.zeros((rows, candidates.size))
        # sampling without replacement can make a candidate impossible outright
        self._too_low = np.zeros(candidates.size, dtype=bool)
        self._too_high = np.zeros(candidates.size, dtype=bool)
        self.n = 0
        self.total = 0.0     # sum of observations on the original scale
        # the same sum on the [0, 1] scale, kept with a running correction term so
        # that rounding does not build up over long streams (Neumaier summation)
        self._seen = 0.0
        self._seen_fix = 0.0
        # rounding one observation can suffer when it is rescaled to [0, 1]
        self._grain = 4.0 * np.finfo(float).eps * (1.0 + max(abs(lo), abs(hi)) / (hi - lo))
        # running mean and variance on the [0, 1] scale; used only to size stakes
        self._mean = 0.5
        self._m2 = 0.0
        self._var = 1.0 / 12.0

    def update(self, x: float) -> None:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"observation must be a finite number, got {x}")
        if x < self.lo or x > self.hi:
            if not self.clip:
                raise ValueError(
                    f"observation {x} is outside the bounds ({self.lo}, {self.hi}). "
                    "Widen the bounds, or pass clip=True to test the mean of the "
                    "clipped values instead."
                )
            x = min(max(x, self.lo), self.hi)
        u = (x - self.lo) / (self.hi - self.lo)

        # `nu` is the candidate the games are played against. `room_up` and
        # `room_down` are kept away from zero and only limit the stakes, so that
        # capital stays positive even for a candidate on the edge of the bounds.
        nu = self._raw
        if self.population is not None:
            # Without replacement the next draw is not centred on the population
            # mean but on what is left of it: (N * mean - sum so far) / (N - n).
            # Waudby-Smith and Ramdas (2020).
            if self.n >= self.population:
                raise ValueError(f"all {self.population} items of the population were already seen")
            remaining = self.population - self.n
            owed = self.population * self._raw - (self._seen + self._seen_fix)
            # a candidate is impossible only if what is left cannot make up for it,
            # by more than rounding could explain
            slack = self._slack()
            self._too_low |= owed < -slack
            self._too_high |= owed > remaining + slack
            left = owed / remaining
            nu = np.clip(left, 0.0, 1.0)
        room_up = np.maximum(nu, _EDGE)
        room_down = np.maximum(1.0 - nu, _EDGE)

        var = max(self._var, _VAR_FLOOR)
        diff = u - nu
        gap = self._mean - nu
        base = var + gap * gap
        if self.fixed_stakes is not None:
            stake_up = np.full(nu.shape, self.fixed_stakes[0])
            stake_down = np.full(nu.shape, self.fixed_stakes[1])
        else:
            stake_up = np.clip(gap / base, 0.0, _STAKE_CAP / room_up)
            stake_down = np.clip(-gap / base, 0.0, _STAKE_CAP / room_down)
        if self.deadline is not None:
            # Hedge over six schedules (Taga, Oymak and Shekhar 2026, App. C):
            # each moves from the usual stake to all-in, in the direction the
            # data favour, as the deadline nears. The six capitals are averaged.
            t = self.n + 1
            share = np.array([
                min(1.0, max(0.0, (t - onset * self.deadline) / ((1.0 - onset) * self.deadline)))
                ** power
                for onset, power in _SCHEDULES
            ])[:, None]
            all_in_up = np.where(gap > 0, _ALL_IN / room_up, stake_up)
            all_in_down = np.where(gap < 0, _ALL_IN / room_down, stake_down)
            stake_up = (1.0 - share) * stake_up + share * all_in_up
            stake_down = (1.0 - share) * stake_down + share * all_in_down
        self._log_up += np.log1p(stake_up * diff)
        self._log_down += np.log1p(-stake_down * diff)

        # update the running statistics only after the bets are placed
        self.n += 1
        self.total += x
        t = self._seen + u
        self._seen_fix += (self._seen - t) + u if abs(self._seen) >= abs(u) else (u - t) + self._seen
        self._seen = t
        delta = u - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (u - self._mean)
        if self.n >= 2:
            self._var = max(self._m2 / self.n, _VAR_FLOOR)

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else float("nan")

    def feasible(self) -> tuple[float, float]:
        """Range the mean can still be in for certain. Narrower than the bounds
        only when sampling without replacement."""
        if self.population is None:
            return self.lo, self.hi
        width, slack = self.hi - self.lo, self._slack()
        seen = self._seen + self._seen_fix
        low = self.lo + width * (seen - slack) / self.population
        high = self.lo + width * (seen + (self.population - self.n) + slack) / self.population
        return max(self.lo, low), min(self.hi, high)

    def _slack(self) -> float:
        """How far rounding can have moved the running sum on the [0, 1] scale."""
        return self._grain * (self.n + 1 + (self.population or 0))

    @property
    def log_up(self) -> np.ndarray:
        """Log capital of the games betting that the mean is above each candidate."""
        return np.where(self._too_low, _LOG_CAP, np.minimum(_log_mean_exp_rows(self._log_up), _LOG_CAP))

    @property
    def log_down(self) -> np.ndarray:
        """Log capital of the games betting that the mean is below each candidate."""
        return np.where(self._too_high, _LOG_CAP, np.minimum(_log_mean_exp_rows(self._log_down), _LOG_CAP))

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "lo": self.lo,
            "hi": self.hi,
            "clip": self.clip,
            "population": self.population,
            "deadline": self.deadline,
            "fixed_stakes": None if self.fixed_stakes is None else list(self.fixed_stakes),
            "candidates": self.candidates.tolist(),
            "log_up": self._log_up.tolist(),
            "log_down": self._log_down.tolist(),
            "too_low": self._too_low.tolist(),
            "too_high": self._too_high.tolist(),
            "n": self.n,
            "total": self.total,
            "seen": self._seen,
            "seen_fix": self._seen_fix,
            "mean": self._mean,
            "m2": self._m2,
            "var": self._var,
        }

    @classmethod
    def from_dict(cls, state: dict) -> Games:
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported state version {state.get('version')!r}")
        games = cls(state["candidates"], state["lo"], state["hi"], state["clip"],
                    state["population"], state["deadline"], state["fixed_stakes"])
        games._log_up = np.asarray(state["log_up"], dtype=float)
        games._log_down = np.asarray(state["log_down"], dtype=float)
        games._too_low = np.asarray(state["too_low"], dtype=bool)
        games._too_high = np.asarray(state["too_high"], dtype=bool)
        games.n = int(state["n"])
        games.total = float(state["total"])
        games._seen, games._seen_fix = float(state["seen"]), float(state["seen_fix"])
        games._mean = float(state["mean"])
        games._m2 = float(state["m2"])
        games._var = float(state["var"])
        return games


def _log_mean_exp_rows(a: np.ndarray) -> np.ndarray:
    if a.shape[0] == 1:
        return a[0]
    m = a.max(axis=0)
    return m + np.log(np.exp(a - m).mean(axis=0))


def log_mean_exp(a: float, b: float) -> float:
    """log((e^a + e^b) / 2) without overflow."""
    m = max(a, b)
    if math.isinf(m):
        return m
    return m + math.log((math.exp(a - m) + math.exp(b - m)) / 2.0)


def safe_exp(log_value: float) -> float:
    return math.exp(min(log_value, 700.0))
