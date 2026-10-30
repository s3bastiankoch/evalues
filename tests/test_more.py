"""Multiple testing, calibrators, sampling without replacement, quantiles, detection."""

import itertools
import json
import math

import numpy as np
import pytest
from scipy import integrate

import evalues
from evalues import ChangeDetector, MeanInterval, MeanTest, QuantileTest, SignTest
from evalues.expert import e_to_p, ebh, holm, p_to_e

ALPHA = 0.05
RUNS = 600
SLACK = 3 * (ALPHA * (1 - ALPHA) / RUNS) ** 0.5


# ---------------------------------------------------------------- multiple testing
def _subsets(m):
    for k in range(1, m + 1):
        yield from itertools.combinations(range(m), k)


def _mean_consistent(e, rejected, alpha):
    """Definition of the closed e-BH collection, checked over all subsets."""
    r = len(rejected)
    return all(
        np.mean([e[i] for i in s]) >= len(set(s) & set(rejected)) / (r * alpha) - 1e-9
        for s in _subsets(len(e))
    )


def test_closed_ebh_matches_its_definition_on_all_subsets():
    rng = np.random.default_rng(0)
    for _ in range(300):
        m = int(rng.integers(1, 8))
        e = list(np.round(rng.exponential(1.0, m) * rng.choice([1, 20, 200], m), 2))
        got = ebh(e, alpha=0.1)
        if got:
            assert _mean_consistent(e, got, 0.1)
        order = sorted(range(m), key=lambda i: e[i], reverse=True)
        for r in range(len(got) + 1, m + 1):       # no larger top-r set qualifies
            assert not _mean_consistent(e, order[:r], 0.1)


def test_closed_ebh_never_rejects_less_than_the_original():
    rng = np.random.default_rng(1)
    more = 0
    for _ in range(500):
        e = list(rng.exponential(1.0, 12) * rng.choice([1, 50, 500], 12))
        base, closed = ebh(e, closed=False), ebh(e)
        assert len(closed) >= len(base)
        more += len(closed) > len(base)
    assert more > 0


def test_closed_ebh_example_from_the_paper():
    m, alpha = 6, 0.05
    e = [(m - 0.5) / alpha, 1 / (2 * alpha)] + [0.0] * (m - 2)
    assert ebh(e, alpha, closed=False) == []
    assert ebh(e, alpha) == [0]


def test_holm_is_closed_testing_with_the_average():
    rng = np.random.default_rng(2)
    for _ in range(300):
        m = int(rng.integers(1, 8))
        e = list(rng.exponential(1.0, m) * rng.choice([1, 30, 300], m))
        expected = [
            i for i in range(m)
            if all(np.mean([e[j] for j in s]) >= 1 / ALPHA for s in _subsets(m) if i in s)
        ]
        assert holm(e) == expected
        assert set(holm(e)) >= {i for i in range(m) if e[i] >= m / ALPHA}
        assert set(holm(e)) <= set(ebh(e))


def test_false_discoveries_stay_rare_under_strong_dependence():
    rng = np.random.default_rng(3)
    fdp, any_false = [], 0
    for _ in range(2000):
        shared = rng.exponential(1.0)                      # one e-value copied to 15 true nulls
        e = [shared] * 15 + list(rng.exponential(1.0, 5) * 400)
        rejected = ebh(e)
        fdp.append(sum(i < 15 for i in rejected) / max(len(rejected), 1))
        any_false += any(i < 15 for i in holm(e))
    assert np.mean(fdp) <= ALPHA + 0.01
    assert any_false / 2000 <= ALPHA + 0.015


# ---------------------------------------------------------------- calibrators
def test_default_p_to_e_is_a_valid_e_value():
    # E[e(U)] for uniform U, integrated in t = -ln p where the integrand is tame
    # up to T = 600; beyond that the integrand is 1 / t^2 to machine precision
    body, _ = integrate.quad(lambda t: p_to_e(math.exp(-t)) * math.exp(-t), 0, 600, limit=500)
    assert math.isclose(body + 1 / 600, 1.0, abs_tol=1e-6)
    assert math.isclose(p_to_e(1 - 1.0001e-4), p_to_e(1 - 0.9999e-4), rel_tol=1e-6)
    assert math.isclose(p_to_e(0.01), 4.451, abs_tol=1e-3)


def test_randomised_e_to_p():
    assert e_to_p(40, u=0.5) == 0.0125 and e_to_p(40) == 0.025
    rng = np.random.default_rng(4)
    e = rng.exponential(1.0, 200_000)                      # valid e-values with mean 1
    p = np.minimum(1.0, rng.random(200_000) / e)
    assert np.mean(p <= 0.05) <= 0.05 + 0.002


# ---------------------------------------------------------------- without replacement
def test_without_replacement_keeps_the_level_and_ends_in_certainty():
    rng = np.random.default_rng(6)
    population = rng.beta(2, 5, 300)
    truth = float(population.mean())
    alarms = 0
    for _ in range(RUNS):
        test = MeanTest(baseline=truth, population_size=300)
        for x in rng.permutation(population):
            test.update(x)
            if test.done:
                alarms += 1
                break
    assert alarms / RUNS <= ALPHA + SLACK

    cs = MeanInterval(population_size=300)
    cs.update_many(rng.permutation(population))
    lo, hi = cs.interval
    assert lo - 1e-9 <= truth <= hi + 1e-9 and hi - lo <= 0.02           # the whole set was seen
    with pytest.raises(ValueError, match="already seen"):
        cs.update(0.5)


def test_without_replacement_is_sharper_than_with():
    rng = np.random.default_rng(7)
    population = rng.random(400)
    plain, finite = MeanInterval(), MeanInterval(population_size=400)
    for x in rng.permutation(population)[:360]:
        plain.update(x)
        finite.update(x)
    assert finite.width < plain.width


# ---------------------------------------------------------------- quantiles and signs
def test_quantile_and_sign_tests_need_no_bounds():
    def cauchy_median_0(rng):
        return float(rng.standard_cauchy())

    rate = evalues.check.false_alarm_rate(lambda: SignTest(), cauchy_median_0, n_obs=400, n_runs=RUNS)
    assert rate <= ALPHA + SLACK
    p90 = math.tan(math.pi * (0.9 - 0.5))                  # 0.9-quantile of the Cauchy
    rate = evalues.check.false_alarm_rate(
        lambda: QuantileTest(limit=p90, quantile=0.9), cauchy_median_0, n_obs=400, n_runs=RUNS
    )
    assert rate <= ALPHA + SLACK


def test_quantile_test_directions():
    xs = np.random.default_rng(8).standard_cauchy(3000) + 1.0      # median 1
    for direction, expect in [("above", True), ("below", False), ("either", True)]:
        test = QuantileTest(limit=0.0, quantile=0.5, direction=direction)
        test.update_many(xs)
        assert test.done is expect
    up = SignTest(direction="increase")
    up.update_many(xs)
    assert up.done
    ties = SignTest()
    ties.update_many([0.0] * 500)
    assert ties.e_value == 1.0


# ---------------------------------------------------------------- change detection
def test_detector_false_alarms_are_rare_and_real_changes_are_found():
    rng = np.random.default_rng(9)
    arl, cap = 200, 4000
    times = []
    for _ in range(300):
        d = ChangeDetector(baseline=0.3, false_alarm_every=arl)
        for x in rng.random(cap) < 0.3:
            d.update(x)
            if d.alarm:
                break
        times.append(d.alarm_at or cap)
    assert np.mean(times) >= arl                                    # average run length
    assert np.mean(np.array(times) <= 20) <= 20 / arl + 0.03        # P(alarm by t) <= t / arl

    delays = []
    for _ in range(100):
        d = ChangeDetector(baseline=0.3, false_alarm_every=1000)
        d.update_many(rng.random(500) < 0.3)
        if d.alarm:
            d.reset()
        start = d.n
        for x in rng.random(5000) < 0.5:
            d.update(x)
            if d.alarm:
                break
        assert d.alarm
        delays.append(d.n - start)
    assert np.median(delays) < 200


def test_detector_direction_reset_and_snapshot():
    xs = np.random.default_rng(10).random(3000) < 0.1
    up = ChangeDetector(0.3, false_alarm_every=1000, direction="increase")
    down = ChangeDetector(0.3, false_alarm_every=1000, direction="decrease")
    up.update_many(xs)
    down.update_many(xs)
    assert not up.alarm and down.alarm
    resumed = ChangeDetector.from_dict(json.loads(json.dumps(down.to_dict())))
    assert resumed.to_dict() == down.to_dict()
    down.reset()
    assert not down.alarm and down.n == 0
    with pytest.raises(ValueError):
        ChangeDetector(baseline=0.0, false_alarm_every=1000)


@pytest.mark.parametrize("make", [
    lambda: MeanTest(baseline=0.3, population_size=500),
    lambda: MeanInterval(resolution=40, population_size=500),
    lambda: QuantileTest(limit=0.25, quantile=0.4),
    lambda: SignTest(center=0.3),
])
def test_new_snapshots_continue_exactly(make):
    xs = np.random.default_rng(11).random(200) * 0.6
    whole, first = make(), make()
    whole.update_many(xs)
    first.update_many(xs[:80])
    resumed = type(first).from_dict(json.loads(json.dumps(first.to_dict())))
    resumed.update_many(xs[80:])
    assert resumed.to_dict() == whole.to_dict()
