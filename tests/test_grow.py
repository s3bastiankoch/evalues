"""Worst-case-optimal fixed stakes (Arnold and Clerico 2026)."""

import json
import math

import numpy as np
import pytest

import evalues
from evalues import MeanTest
from evalues._edo import edo_stake
from evalues.expert import grow_stake, regrow_stake


def _worst_case_growth(stake, baseline, mean, lo, hi):
    """Expected log e-value under the two-point distribution on the bounds with this mean."""
    q = (hi - mean) / (hi - lo)
    return q * math.log1p(stake * (lo - baseline)) + (1 - q) * math.log1p(stake * (hi - baseline))


def test_grow_is_the_kelly_stake_for_the_coin_on_the_bounds():
    assert math.isclose(grow_stake(0.5, 0.65), 0.6)
    assert math.isclose(grow_stake(0.5, 0.65), edo_stake(0.5, 0.65, 1e12), rel_tol=1e-6)   # the patient limit
    assert math.isclose(grow_stake(40, 55, bounds=(0, 100)), 0.00625)
    assert math.isclose(grow_stake(0.5, 0.35), -0.6)
    best = grow_stake(3.0, 5.0, bounds=(1, 9))
    here = _worst_case_growth(best, 3.0, 5.0, 1, 9)
    assert here > 0
    for other in (0.8 * best, 1.2 * best):                 # no other stake does better in the worst case
        assert _worst_case_growth(other, 3.0, 5.0, 1, 9) < here
    rng = np.random.default_rng(0)                         # and every distribution with that mean does at least as well
    xs = 1 + 8 * rng.beta(2, 2, 200_000)
    assert np.mean(np.log1p(best * (xs - 3.0))) >= here


def test_regrow_matches_the_papers_coin_example_and_equalises_its_two_worst_cases():
    for mu0 in (0.1, 0.3, 0.5, 0.8):
        eps = 1 / (1 + (1 - mu0) * mu0 ** (mu0 / (1 - mu0)))
        assert math.isclose(regrow_stake(mu0), (eps - mu0) / (mu0 * (1 - mu0)), rel_tol=1e-12)
    # shortfall against the oracle is equal at the two worst cases: all mass on the upper
    # bound, and the two-point distribution with mean `expected` (equation 8, Remark 1)
    baseline, expected, lo, hi = 3.0, 4.0, 1.0, 9.0
    stake, kelly, top = regrow_stake(baseline, expected, (lo, hi)), grow_stake(baseline, expected, (lo, hi)), 1 / (baseline - lo)
    at_bound = math.log1p(stake * (hi - baseline)) - math.log1p(top * (hi - baseline))
    two_point = (_worst_case_growth(stake, baseline, expected, lo, hi)
                 - _worst_case_growth(kelly, baseline, expected, lo, hi))
    assert math.isclose(at_bound, two_point, rel_tol=1e-9)
    assert kelly < stake < top
    assert math.isclose(regrow_stake(0.7, 0.6), -regrow_stake(0.3, 0.4))      # mirror image
    for bad in (lambda: grow_stake(0.5, 0.5), lambda: grow_stake(0.0, 0.5), lambda: regrow_stake(0.5, 1.0)):
        with pytest.raises(ValueError):
            bad()


def test_one_stake_gives_a_valid_e_value():
    rng = np.random.default_rng(1)
    for stake in (grow_stake(0.3, 0.5), regrow_stake(0.3), regrow_stake(0.3, 0.5)):
        for xs in (rng.random(200_000) < 0.3, rng.beta(3, 7, 200_000), rng.beta(1, 9, 200_000)):
            e = 1 + stake * (xs - 0.3)
            assert e.min() >= 0 and e.mean() <= 1 + 0.01


def test_mean_test_with_expected_alone_uses_the_grow_stake():
    assert MeanTest(0.5, expected=0.65)._games.fixed_stakes == (pytest.approx(0.6), 0.0)
    assert MeanTest(0.5, expected=0.35)._games.fixed_stakes == (0.0, pytest.approx(0.6))
    assert MeanTest(0.5, expected=0.65).direction == "above"
    rate = evalues.check.false_alarm_rate(
        lambda: MeanTest(0.3, expected=0.45), lambda r: float(r.random() < 0.3), n_obs=400, n_runs=1000
    )
    assert rate <= 0.05 + 3 * (0.05 * 0.95 / 1000) ** 0.5
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(400):
        test = MeanTest(0.5, expected=0.65)
        test.update_many(rng.random(100) < 0.65)
        hits += test.verdict == "above"
    assert hits / 400 > 0.7                                # the adaptive default manages about 0.54
    again = evalues.load(json.loads(json.dumps(test.to_dict())))
    assert again.to_dict() == test.to_dict()
    with pytest.raises(ValueError):
        MeanTest(0.5, hoped_within=50)
