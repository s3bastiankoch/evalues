"""Cases an adversarial audit found. Each one used to give a wrong or broken result."""

import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from evalues import Comparison, DominanceTest, MeanInterval, MeanTest, flag


def _exact_closed_ebh(e, alpha):
    """Closed e-BH by its definition, over all subsets, in exact arithmetic."""
    values = [Fraction(x) for x in e]
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i], reverse=True)
    subsets = [s for k in range(1, m + 1) for s in itertools.combinations(range(m), k)]
    for r in range(m, 0, -1):
        chosen = set(order[:r])
        if all(sum(values[i] for i in s) * r * Fraction(alpha) >= len(s) * len(chosen & set(s))
               for s in subsets):
            return sorted(chosen)
    return []


def test_flag_is_exact_next_to_enormous_e_values():
    assert flag([1e17, 9, 9, 9]) == [0]                   # used to flag all four
    assert flag([1e17] + [8.01] * 4) == [0]
    assert flag([1e300, 1e6]) == [0, 1]                   # used to flag only the first
    assert flag([1e18, 60]) == [0, 1]
    rng = np.random.default_rng(0)
    for _ in range(600):
        m = int(rng.integers(2, 8))
        e = [float(x) for x in np.round(rng.exponential(1.0, m) * rng.choice([1, 20, 200, 1e17, 1e30], m), 2)]
        assert flag(e, max_false_share=0.1) == _exact_closed_ebh(e, Fraction(1, 10))


def test_fixed_stake_is_never_negative_near_a_bound():
    with pytest.raises(ValueError, match="too close"):
        MeanTest(5e-5, expected=5.0001e-5)
    with pytest.raises(ValueError, match="too close"):
        MeanTest(3.0, (0, 60000), expected=3.5)
    test = MeanTest(0.01, expected=0.02)                  # an ordinary case still works
    assert test._games.fixed_stakes[0] > 0


def test_interval_accepts_awkward_bounds_and_refuses_unreachable_margins():
    MeanInterval((-0.1, 0.6), resolution=1329)            # the last grid point used to overshoot
    rng = np.random.default_rng(1)
    for _ in range(300):
        lo = float(rng.normal(0, 10))
        MeanInterval((lo, lo + float(rng.exponential(5)) + 1e-3), resolution=int(rng.integers(2, 3000)))
    with pytest.raises(ValueError, match="too fine"):
        MeanInterval(margin=1e-5)


@pytest.mark.parametrize("lo, width", [(0.0, 1.0), (1000.0, 1.0), (1e9, 1e-3), (-3.0, 10.0)])
def test_finite_population_arithmetic_survives_large_offsets(lo, width):
    rng = np.random.default_rng(2)
    population = lo + width * np.round(rng.random(40), 1)
    truth = math.fsum(population) / 40
    wrong = missed = 0
    for _ in range(200):
        test = MeanTest(truth, (lo, lo + width), population_size=40)
        estimate = MeanInterval((lo, lo + width), population_size=40)
        covered = True
        for x in rng.permutation(population):
            test.update(x)
            estimate.update(x)
            covered &= estimate.low <= truth <= estimate.high
        assert estimate.low <= truth <= estimate.high       # exact once everything was seen
        assert estimate.high - estimate.low <= 0.02 * width
        wrong += test.done
        missed += not covered
    limit = 0.05 + 3 * (0.05 * 0.95 / 200) ** 0.5
    assert wrong / 200 <= limit and missed / 200 <= limit


def test_nothing_infinite_reaches_the_snapshot_or_the_result():
    test = MeanTest(0.2, population_size=4)
    test.update_many([1, 1, 1])                           # the baseline is now impossible
    assert test.done and math.isfinite(test.log_e_value) and test.p_value > 0
    json.dumps(test.to_dict(), allow_nan=False)
    json.dumps(test.result(), allow_nan=False)
    comparison = Comparison(0.02)
    comparison.update(1.0, 0.0)
    json.dumps(comparison.result(), allow_nan=False)
    assert "cannot be given" in comparison.summary()
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            DominanceTest(sharpness=bad)
