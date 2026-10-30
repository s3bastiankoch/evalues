"""Fixed stakes that are optimal in the worst case over a composite alternative.

Arnold and Clerico (2026), "Optimal e-values for testing the mean of a bounded
random variable against a composite alternative", arXiv:2601.11347, Theorem 1.
The e-value of one observation x is 1 + stake * (x - baseline).
"""

from __future__ import annotations

from ._common import number, pair


def _check(baseline, expected, bounds) -> tuple[float, float, float, float]:
    lo, hi = pair("bounds", bounds)
    baseline = number("baseline", baseline)
    expected = baseline if expected is None else number("expected", expected)
    if not lo < baseline < hi:
        raise ValueError(f"baseline {baseline:g} must lie strictly inside bounds ({lo:g}, {hi:g})")
    if not lo < expected < hi:
        raise ValueError(f"expected {expected:g} must lie strictly inside bounds ({lo:g}, {hi:g})")
    return baseline, expected, lo, hi


def grow_stake(baseline: float, expected: float, bounds: tuple[float, float] = (0.0, 1.0)) -> float:
    """The stake with the best guaranteed growth when the mean is `expected` or beyond.

    Among all e-values for "the mean is at most `baseline`" (or at least, if
    `expected` is below it), 1 + stake * (x - baseline) has the largest
    expected log-growth in the worst case over every distribution on `bounds`
    whose mean is `expected` or further from the baseline (GROW; Theorem 1(a)
    and (b), equation 7). The worst case is the distribution that puts all its
    mass on the two bounds, so this is the Kelly stake for that coin, and the
    growth is positive for every distribution in the alternative.

    Used repeatedly as a fixed stake it is what `MeanTest(expected=...)` does.
    """
    baseline, expected, lo, hi = _check(baseline, expected, bounds)
    if expected == baseline:
        raise ValueError("expected must differ from the baseline")
    return (expected - baseline) / ((baseline - lo) * (hi - baseline))


def regrow_stake(baseline: float, expected: float | None = None,
                 bounds: tuple[float, float] = (0.0, 1.0)) -> float:
    """The stake with the smallest worst-case shortfall against an oracle, for ONE e-value.

    For the null "the mean is at most `baseline`" against "the mean is above
    `expected`", this stake minimises the worst-case gap in expected
    log-growth to the e-value an oracle knowing the true distribution would
    use (REGROW; Theorem 1(b), equation 8). `expected` may equal the baseline
    (the default), where the GROW stake is zero and this one is not. Pass an
    `expected` below the baseline for the mirrored problem; the stake is then
    negative.

    This is a single-round result, and the paper leaves the multi-round case
    open. It is meant for turning one bounded number (say, one batch average)
    into one e-value. Repeating it as a fixed stake over many observations is
    not recommended: it is tuned to do well against extreme alternatives and
    loses capital on moderate ones. In a simulation with a coin of bias 0.65
    against baseline 0.5 it reached a verdict in 40 % of runs of 1,000 flips,
    where the adaptive default and `grow_stake` both reached 100 %.
    """
    baseline, expected, lo, hi = _check(baseline, expected, bounds)
    if expected < baseline:
        return -regrow_stake(lo + hi - baseline, lo + hi - expected, (lo, hi))
    r = ((expected - lo) / (hi - lo)) ** ((expected - lo) / (hi - expected))
    k = (hi - expected) / (hi - lo) * r
    return ((hi - baseline) - (baseline - lo) * k) / ((baseline - lo) * (hi - baseline) * (1.0 + k))
