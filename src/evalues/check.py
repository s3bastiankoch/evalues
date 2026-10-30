"""Check the guarantees by simulation.

Use these to convince yourself, or a reviewer, that a test keeps its promise
on data that looks like yours.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from ._common import VERDICTS_ALL, VERDICTS_FOUND


def false_alarm_rate(
    make_test: Callable[[], object],
    sample: Callable[[np.random.Generator], object],
    n_obs: int,
    n_runs: int = 1000,
    seed: int = 0,
    wrong: Iterable[str] = VERDICTS_FOUND,
) -> float:
    """Share of runs in which a test reaches a wrong verdict at any point.

    `make_test` returns a fresh object. `sample(rng)` draws one observation
    (or a tuple such as `(new, old)`) from a situation you control. `wrong`
    lists the verdicts that would be wrong in that situation; by default
    every verdict that claims to have found something, which fits data where
    nothing is going on. For data with a real increase you would pass
    `wrong={"decrease", "no_change"}`.

    The result should be at most 1 - confidence, up to simulation noise. For
    a `ChangeDetector` the bound is n_obs / false_alarm_every instead. For a
    `MeanInterval` use `coverage`.
    """
    if isinstance(wrong, str):
        raise ValueError(f"wrong must be a set of verdicts such as {{'increase'}}, got the text {wrong!r}")  # noqa: TRY004
    wrong = set(wrong)
    unknown = wrong - VERDICTS_ALL
    if unknown:
        raise ValueError(f"unknown verdicts {sorted(unknown)}; choose from {sorted(VERDICTS_ALL)}")
    if not hasattr(make_test(), "verdict"):
        raise ValueError("this object has no verdict; for a MeanInterval use evalues.check.coverage")
    rng = np.random.default_rng(seed)
    errors = 0
    for _ in range(n_runs):
        test = make_test()
        for _ in range(n_obs):
            observation = sample(rng)
            if isinstance(observation, tuple | list):
                test.update(*observation)
            else:
                test.update(observation)
            if test.done:
                break
        errors += test.verdict in wrong
    return errors / n_runs


def coverage(
    make_interval: Callable[[], object],
    sample: Callable[[np.random.Generator], float],
    true_mean: float,
    n_obs: int,
    n_runs: int = 1000,
    seed: int = 0,
) -> float:
    """Share of runs in which the interval contains `true_mean` at every step.

    `make_interval` returns a fresh `MeanInterval`. The result should be at
    least its confidence, up to simulation noise.
    """
    rng = np.random.default_rng(seed)
    covered = 0
    for _ in range(n_runs):
        estimate = make_interval()
        ok = True
        for _ in range(n_obs):
            estimate.update(sample(rng))
            lo, hi = estimate.interval
            if not lo <= true_mean <= hi:
                ok = False
                break
        covered += ok
    return covered / n_runs
