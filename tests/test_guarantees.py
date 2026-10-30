"""The promises the library makes, checked by simulation.

If one of these fails the library is statistically wrong. Fix that first.
Tolerances are the nominal level plus about three standard errors.
"""

import itertools

import numpy as np
import pytest

import evalues
from evalues import DifferenceTest, MeanInterval, MeanTest

ALPHA = 0.05
RUNS = 600
SLACK = 3 * (ALPHA * (1 - ALPHA) / RUNS) ** 0.5

NULLS = {
    "coin": (lambda rng: float(rng.random() < 0.3), 0.3),
    "beta": (lambda rng: float(rng.beta(2, 5)), 2 / 7),
    "rare spike": (lambda rng: 1.0 if rng.random() < 0.02 else 0.0, 0.02),
    "uniform": (lambda rng: float(rng.random()), 0.5),
}


@pytest.mark.parametrize("name", NULLS)
def test_mean_test_rarely_rejects_a_true_null(name):
    sample, mean = NULLS[name]
    rate = evalues.check.false_alarm_rate(lambda: MeanTest(mean), sample, n_obs=400, n_runs=RUNS)
    assert rate <= ALPHA + SLACK


@pytest.mark.parametrize("direction, null", [("above", 0.35), ("below", 0.25)])
def test_one_sided_test_rarely_rejects_a_true_null(direction, null):
    sample, _ = NULLS["coin"]  # true mean 0.3
    rate = evalues.check.false_alarm_rate(
        lambda: MeanTest(null, direction=direction), sample, n_obs=400, n_runs=RUNS
    )
    assert rate <= ALPHA + SLACK


@pytest.mark.parametrize("name", NULLS)
def test_interval_contains_the_mean_at_every_step(name):
    sample, mean = NULLS[name]
    cov = evalues.check.coverage(lambda: MeanInterval(), sample, mean, n_obs=400, n_runs=300)
    assert cov >= 1 - ALPHA - 3 * (ALPHA * (1 - ALPHA) / 300) ** 0.5


@pytest.mark.parametrize("true_mean, wrong", [
    (0.0, {"increase", "decrease"}),
    (0.02, {"increase", "decrease"}),          # on the edge counts as no change
    (0.05, {"decrease", "no_change"}),
    (-0.05, {"increase", "no_change"}),
])
def test_change_test_rarely_reaches_a_wrong_verdict(true_mean, wrong):
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(RUNS):
        test = DifferenceTest(min_difference=0.02, bounds=(-0.5, 0.5))
        test.update_many(np.clip(rng.normal(true_mean, 0.1, 500), -0.5, 0.5))
        errors += test.verdict in wrong
    assert errors / RUNS <= ALPHA + SLACK


def test_mean_test_finds_a_real_difference():
    rng = np.random.default_rng(2)
    test = MeanTest(baseline=0.5)
    test.update_many(rng.random(2000) < 0.6)
    assert test.done
    assert test.p_value <= ALPHA


def test_change_test_reaches_each_verdict():
    rng = np.random.default_rng(3)
    for true_mean, expected in [(0.08, "increase"), (-0.08, "decrease"), (0.0, "no_change")]:
        test = DifferenceTest(min_difference=0.03, bounds=(-0.5, 0.5))
        test.update_many(np.clip(rng.normal(true_mean, 0.1, 20000), -0.5, 0.5))
        assert test.verdict == expected


def test_interval_shrinks_around_the_mean():
    rng = np.random.default_rng(4)
    cs = MeanInterval()
    widths = []
    for x in rng.beta(2, 5, 3000):
        cs.update(x)
        widths.append(cs.width)
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(widths))
    lo, hi = cs.interval
    assert lo <= 2 / 7 <= hi
    assert hi - lo < 0.1


@pytest.mark.parametrize("make, sample, n_obs, n_runs", [
    (lambda: MeanTest(baseline=1e-5), lambda rng: float(rng.random() < 1e-5), 45_000, 40),
    (lambda: MeanTest(baseline=0.5, bounds=(0, 10_000)), lambda rng: float(rng.uniform(0.3, 0.7)), 90_000, 20),
    (lambda: MeanTest(baseline=0.0, direction="above"), lambda rng: 0.0, 2_000, 20),
    (lambda: MeanTest(baseline=1.0, direction="below"), lambda rng: 1.0, 2_000, 20),
])
def test_baseline_on_the_edge_of_the_bounds_keeps_the_level(make, sample, n_obs, n_runs):
    # The game must be played against the baseline itself, however close to a
    # bound. Playing against a nudged value gave 65 % false verdicts here.
    rate = evalues.check.false_alarm_rate(make, sample, n_obs=n_obs, n_runs=n_runs)
    assert rate <= 0.2


def test_interval_stays_valid_when_the_truth_lies_between_grid_points():
    rng = np.random.default_rng(5)
    misses = 0
    for _ in range(60):
        estimate = MeanInterval(bounds=(0, 100))
        for x in np.clip(rng.normal(42.3, 0.5, 4000), 0, 100):
            estimate.update(x)
            if not estimate.low <= 42.3 <= estimate.high:      # also catches (nan, nan)
                misses += 1
                break
        else:
            assert estimate.consistent and estimate.width <= 1.0
    assert misses / 60 <= ALPHA + 3 * (ALPHA * (1 - ALPHA) / 60) ** 0.5
