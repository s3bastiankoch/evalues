"""Stake rules for tests that must finish soon. They change power, never validity."""

import json
import math

import numpy as np
import pytest

from evalues import MeanTest
from evalues._edo import edo_stake

ALPHA = 0.05
RUNS = 1000
SLACK = 3 * (ALPHA * (1 - ALPHA) / RUNS) ** 0.5


def _reject_rate(make, sample, n, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(RUNS):
        test = make()
        for _ in range(n):
            test.update(sample(rng))
            if test.done:
                break
        hits += test.verdict != "undecided"
    return hits / RUNS


@pytest.mark.parametrize("make", [
    lambda: MeanTest(0.3, deadline=100),
    lambda: MeanTest(0.3, deadline=100, direction="above"),
    lambda: MeanTest(0.3, expected=0.45, hoped_within=50),
    lambda: MeanTest(0.3, expected=0.2, hoped_within=20),
])
def test_stake_rules_keep_the_level(make):
    for sample in (lambda r: float(r.random() < 0.3), lambda r: float(r.beta(3, 7))):
        # run past the deadline on purpose: the guarantee does not depend on stopping there
        assert _reject_rate(make, sample, n=100) <= ALPHA + SLACK


def test_edo_stake_matches_the_paper_and_tends_to_kelly():
    assert math.isclose(edo_stake(0.5, 2 / 3, 30), 1.015, abs_tol=2e-3)
    kelly = (2 / 3 - 0.5) / (0.5 * 0.5)
    assert math.isclose(edo_stake(0.5, 2 / 3, 1e9), kelly, rel_tol=1e-4)
    assert edo_stake(0.5, 2 / 3, 30) > kelly                       # impatience bets more
    assert edo_stake(0.5, 0.6, 1.0) == 2.0                         # hopelessly short: all in
    # defining equation of the criterion: E_1[(1 + stake (X - p0))^eta] = exp(1 / T)
    p0, p1, t = 0.3, 0.5, 40.0
    stake = edo_stake(p0, p1, t)
    etas = np.linspace(1e-3, 1 - 1e-3, 20001)
    values = p1 * (1 + stake * (1 - p0)) ** etas + (1 - p1) * (1 - stake * p0) ** etas
    assert np.min(np.abs(values - math.exp(1 / t))) < 1e-4


def test_options_validate_and_snapshot():
    for bad in (
        lambda: MeanTest(0.3, expected=0.3, hoped_within=10),
        lambda: MeanTest(0.3, expected=0.5, hoped_within=10, direction="below"),
        lambda: MeanTest(0.3, deadline=1),
        lambda: MeanTest(0.3, deadline=50, population_size=100),
    ):
        with pytest.raises(ValueError):
            bad()
    xs = np.random.default_rng(1).random(90) < 0.5
    for make in (lambda: MeanTest(0.3, deadline=100), lambda: MeanTest(0.3, expected=0.5, hoped_within=40)):
        whole, first = make(), make()
        whole.update_many(xs)
        first.update_many(xs[:40])
        resumed = MeanTest.from_dict(json.loads(json.dumps(first.to_dict())))
        resumed.update_many(xs[40:])
        assert resumed.to_dict() == whole.to_dict()
