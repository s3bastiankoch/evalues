import json
import math

import numpy as np
import pytest

from evalues import DifferenceTest, MeanInterval, MeanTest
from evalues.expert import e_to_p, ebh, merge, merge_independent, p_to_e, rel_diff


def test_nothing_seen_means_no_evidence():
    test = MeanTest(baseline=0.5)
    assert test.n == 0 and test.e_value == 1.0 and test.p_value == 1.0 and not test.done
    assert MeanInterval(bounds=(-2, 3)).interval == (-2, 3)
    assert DifferenceTest(min_difference=0.1).verdict == "undecided"


def test_out_of_bounds_raises_unless_clip():
    with pytest.raises(ValueError, match="outside the bounds"):
        MeanTest(baseline=0.5).update(1.5)
    with pytest.raises(ValueError, match="finite"):
        MeanTest(baseline=0.5).update(float("nan"))
    clipped, at_edge = MeanTest(baseline=0.5, clip=True), MeanTest(baseline=0.5)
    clipped.update_many([1.5, 0.2])
    at_edge.update_many([1.0, 0.2])
    assert clipped.e_value == at_edge.e_value


@pytest.mark.parametrize("bad", [
    lambda: MeanTest(baseline=2.0),
    lambda: MeanTest(baseline=0.5, confidence=1.0),
    lambda: MeanTest(baseline=0.5, confidence=0.05),
    lambda: MeanTest(baseline=0.5, bounds=1),
    lambda: MeanTest(baseline="0.5"),
    lambda: MeanInterval(target_width=5),
    lambda: MeanInterval(margin=0.1, target_width=0.2),
    lambda: MeanTest(baseline=0.5, direction="bigger"),
    lambda: MeanInterval(bounds=(1, 0)),
    lambda: DifferenceTest(min_difference=0),
    lambda: DifferenceTest(min_difference=0.5, bounds=(-0.5, 0.5)),
])
def test_bad_arguments_raise(bad):
    with pytest.raises(ValueError):
        bad()


def test_rejection_is_sticky_and_p_value_never_rises():
    rng = np.random.default_rng(0)
    test = MeanTest(baseline=0.5)
    test.update_many(rng.random(300) < 0.8)
    assert test.done
    p = test.p_value
    test.update_many(rng.random(2000) < 0.5)
    assert test.done and test.p_value <= p


def test_one_sided_only_reacts_to_its_direction():
    xs = np.random.default_rng(1).random(500) < 0.8
    up, down = MeanTest(0.5, direction="above"), MeanTest(0.5, direction="below")
    up.update_many(xs)
    down.update_many(xs)
    assert up.done and not down.done and down.e_value <= 1.0


def test_bounds_are_only_a_change_of_units():
    xs = np.random.default_rng(2).random(200)
    unit, scaled = MeanTest(baseline=0.4), MeanTest(baseline=40.0, bounds=(0, 100))
    unit.update_many(xs)
    scaled.update_many(100 * xs)
    assert math.isclose(unit.log_e_value, scaled.log_e_value, rel_tol=1e-9)


@pytest.mark.parametrize("make", [
    lambda: MeanTest(baseline=0.4, direction="above"),
    lambda: MeanInterval(resolution=50),
    lambda: DifferenceTest(min_difference=0.05, bounds=(-1, 1)),
    lambda: DifferenceTest(min_difference=0.05, direction="increase"),
])
def test_snapshot_continues_exactly(make):
    xs = np.random.default_rng(3).random(200) * 0.6
    whole, first = make(), make()
    whole.update_many(xs)
    first.update_many(xs[:80])
    resumed = type(first).from_dict(json.loads(json.dumps(first.to_dict())))
    resumed.update_many(xs[80:])
    assert resumed.to_dict() == whole.to_dict()


def test_rel_diff():
    assert rel_diff(3, 1) == 0.5 and rel_diff(1, 3) == -0.5 and rel_diff(2, 2) == 0
    with pytest.raises(ValueError):
        rel_diff(0, 0)


def test_merge():
    assert merge([1, 3]) == 2
    assert merge([1, 3], weights=[0.25, 0.75]) == 2.5
    assert merge([10, 30], weights=[0.25, 0.25]) == 10.5      # the other half counts as 1
    assert merge_independent([2, 3, 4]) == 24
    for bad in (lambda: merge([]), lambda: merge([1, -1]), lambda: merge([1, 2], weights=[1, 1])):
        with pytest.raises(ValueError):
            bad()


def test_ebh():
    # 4 hypotheses, alpha 0.05: thresholds 80, 40, 26.7, 20 for ranks 1..4
    assert ebh([100, 1, 50, 2], closed=False) == [0, 2]
    assert ebh([30, 30, 30, 1], closed=False) == [0, 1, 2]
    assert ebh([10, 10, 10, 10], closed=False) == []
    assert ebh([]) == []


def test_calibrators():
    assert e_to_p(20) == 0.05 and e_to_p(0.5) == 1.0 and e_to_p(0) == 1.0
    assert math.isclose(p_to_e(0.01, kappa=0.5), 5.0)
    assert p_to_e(1.0) == 0.5 and p_to_e(1.0, kappa=0.5) == 0.5
    with pytest.raises(ValueError):
        p_to_e(0.0)
