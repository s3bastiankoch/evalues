"""The plain-language surface: done, verdict, summary, result, Comparison, flag, load."""

import json
import math

import numpy as np
import pytest

import evalues
from evalues import (
    ChangeDetector,
    Comparison,
    DifferenceTest,
    DominanceTest,
    MeanInterval,
    MeanTest,
    QuantileTest,
    SignTest,
    combine,
    flag,
    flag_strict,
)

ALPHA = 0.05
RUNS = 500
SLACK = 3 * (ALPHA * (1 - ALPHA) / RUNS) ** 0.5


def _pairs(rng, ratio, noise=0.05, n=4000):
    old = rng.lognormal(0.0, noise, n)
    return ratio * rng.lognormal(0.0, noise, n), old


@pytest.mark.parametrize("direction, ratio, wrong, expected", [
    ("either", 1.0, {"increase", "decrease"}, "no_change"),
    ("either", 1.08, {"decrease", "no_change"}, "increase"),
    ("either", 0.93, {"increase", "no_change"}, "decrease"),
    ("increase", 1.0, {"increase"}, "no_increase"),
    ("increase", 1.02, {"increase"}, None),                # exactly on the threshold
    ("increase", 1.08, {"no_increase"}, "increase"),
    ("decrease", 1.0, {"decrease"}, "no_decrease"),
    ("decrease", 0.93, {"no_decrease"}, "decrease"),
])
def test_comparison_reaches_the_right_verdict_and_rarely_a_wrong_one(direction, ratio, wrong, expected):
    rng = np.random.default_rng(0)
    verdicts = []
    for _ in range(RUNS):
        comparison = Comparison(min_change=0.02, direction=direction)
        news, olds = _pairs(rng, ratio, n=1500)
        for new, old in zip(news, olds, strict=True):
            comparison.update(new, old)
            if comparison.done:
                break
        verdicts.append(comparison.verdict)
    assert np.mean([v in wrong for v in verdicts]) <= ALPHA + SLACK
    if expected:
        assert np.mean([v == expected for v in verdicts]) > 0.9


def test_looking_one_way_is_faster_and_flags_only_that_way():
    rng = np.random.default_rng(1)
    both, one = [], []
    for _ in range(100):
        a, b = Comparison(0.02), Comparison(0.02, direction="increase")
        for new, old in zip(*_pairs(rng, 1.06), strict=True):
            if not a.done:
                a.update(new, old)
            if not b.done:
                b.update(new, old)
        both.append(a.decided_at)
        one.append(b.decided_at)
    assert np.mean(one) < np.mean(both)
    faster, slower = Comparison(0.02, direction="increase"), Comparison(0.02, direction="increase")
    faster.update_many(*_pairs(rng, 0.9))
    slower.update_many(*_pairs(rng, 1.1))
    assert flag({"faster": faster, "slower": slower}) == ["slower"]


def test_comparison_wording_and_input_checks():
    rng = np.random.default_rng(2)
    comparison = Comparison(min_change=0.01)
    comparison.update_many(*_pairs(rng, 1.06))            # keeps going past the verdict
    assert comparison.verdict == "increase" and comparison.done and comparison.n == 4000
    assert comparison.decided_at < 4000
    assert abs(comparison.average_change - 0.06) < 0.01
    assert "averaged over pairs, new is more than 1 % larger than old" in comparison.summary()
    assert "error rate at most 5 %" in comparison.summary() and "estimate" in comparison.summary()
    down = Comparison(min_change=0.5)
    down.update_many(*_pairs(rng, 0.3))
    assert "more than 33.3 % smaller" in down.summary()   # the real limit on the way down
    for bad in (
        lambda: Comparison(),
        lambda: Comparison(min_change=0),
        lambda: Comparison(min_change=2),
        lambda: Comparison(min_change="2%"),
        lambda: Comparison(0.02, direction="up"),
        lambda: Comparison(0.02, confidence=95),
        lambda: comparison.update(-1.0, 2.0),
        lambda: comparison.update(None, 2.0),
        lambda: comparison.update("12.5", 11),
        lambda: comparison.update([1.0, 2.0], [1.0, 2.0]),
        lambda: comparison.update(float("inf"), 1.0),
        lambda: comparison.update_many([1.0, 2.0, 3.0], [1.0, 2.0]),
    ):
        with pytest.raises((ValueError, TypeError)):
            bad()
    assert comparison.n == 4000                            # nothing was half-added


def test_comparison_does_not_claim_more_than_it_tests():
    rng = np.random.default_rng(3)
    comparison = Comparison(min_change=0.02)
    for _ in range(20_000):
        old = rng.lognormal(0, 0.05)
        new = rng.lognormal(0, 0.05) * (100.0 if rng.random() < 0.002 else 1.0)   # rare 100x spikes
        comparison.update(new, old)
        if comparison.done:
            break
    assert comparison.verdict == "no_change"
    assert "averaged over pairs" in comparison.summary() and "DominanceTest" in comparison.summary()


def test_mean_test_names_the_side_and_reports_later_reversals():
    rng = np.random.default_rng(4)
    wrong = 0
    for _ in range(RUNS):
        test = MeanTest(baseline=0.5)
        test.update_many(rng.random(400) < 0.45)          # truly below the baseline
        wrong += test.verdict == "above"
    assert wrong / RUNS <= ALPHA + SLACK
    one_way = MeanTest(baseline=0.5, direction="above")
    one_way.update_many(rng.random(3000) < 0.4)
    assert one_way.verdict == "undecided" and not one_way.done
    flipped = MeanTest(baseline=0.5)
    flipped.update_many([1] * 12 + [0] * 40)
    assert flipped.verdict == "above" and flipped.decided_at <= 12
    assert "Decided after" in flipped.summary() and "point the other way" in flipped.summary()


def test_deadline_ends_every_test_without_a_verdict():
    for obj, feed in [
        (MeanTest(0.5, deadline=20), lambda o: o.update(0.5)),
        (DifferenceTest(0.1, deadline=20), lambda o: o.update(0.1)),
        (Comparison(0.02, deadline=20), lambda o: o.update(1.02, 1.0)),
        (QuantileTest(1.0, deadline=20), lambda o: o.update(1.0 if o.n % 2 else 2.0)),
        (SignTest(deadline=20), lambda o: o.update(0.0)),
        (DominanceTest(deadline=20), lambda o: o.update(1.0, 1.0)),
    ]:
        for _ in range(20):
            assert not obj.done
            feed(obj)
        assert obj.done and obj.verdict == "undecided"
        assert "deadline of 20" in obj.summary() and "without a decision" in obj.summary()
        with pytest.raises(ValueError, match="deadline"):
            feed(obj)
    limited = Comparison(0.02, deadline=10)
    limited.update_many(new=np.ones(50), old=np.ones(50))          # stops quietly at the deadline
    assert limited.n == 10


def test_interval_target_drift_and_arguments():
    rng = np.random.default_rng(5)
    estimate = MeanInterval(margin=0.05)
    assert not estimate.done
    for x in rng.beta(2, 5, 5000):
        estimate.update(x)
        if estimate.done:
            break
    assert estimate.done and estimate.high - estimate.low <= 0.1
    assert estimate.low <= 2 / 7 <= estimate.high
    assert not MeanInterval().done                        # no target, never done
    fine = MeanInterval(bounds=(0, 1000), target_width=5)
    fine.update_many(np.full(3000, 500.0) + rng.normal(0, 1, 3000))
    assert fine.done                                      # the grid adapts to the target
    drifting = MeanInterval(margin=0.05)
    drifting.update_many(np.r_[rng.random(3000) < 0.3, rng.random(3000) < 0.7])
    assert not drifting.consistent and not drifting.done
    assert math.isnan(drifting.low) and "not consistent" in drifting.summary()
    with pytest.raises(TypeError):
        MeanInterval((0, 1), 0.06)                        # only bounds may be positional


def test_sign_test_takes_pairs_or_differences():
    rng = np.random.default_rng(6)
    new, old = rng.normal(0.3, 1, 600), rng.normal(0, 1, 600)
    paired, plain = SignTest(), SignTest()
    paired.update_many(new, old)
    plain.update_many(new - old)
    assert paired.to_dict() == plain.to_dict()
    assert paired.verdict == "increase" and paired.share_above > 0.5


def _fed(rng):
    """One object of every kind, fed some data."""
    objects = [
        Comparison(min_change=0.02),
        MeanTest(baseline=0.4),
        MeanInterval(resolution=40),
        DifferenceTest(min_difference=0.1),
        QuantileTest(limit=0.5, quantile=0.9),
        SignTest(center=0.3),
        DominanceTest(levels=20),
        ChangeDetector(baseline=0.3, false_alarm_every=200),
    ]
    xs = rng.random(120) * 0.8
    for obj in objects:
        if isinstance(obj, Comparison | DominanceTest):
            obj.update_many(xs + 0.5, xs[::-1] + 0.4)
        else:
            obj.update_many(xs)
    return objects


def test_every_object_speaks_the_same_language():
    rng = np.random.default_rng(7)
    for fresh in (Comparison(0.02), MeanTest(0.5), MeanInterval(), DifferenceTest(0.1), QuantileTest(1.0),
                  SignTest(), DominanceTest(), ChangeDetector(0.5, false_alarm_every=100)):
        assert fresh.done is False and fresh.n == 0
        assert isinstance(fresh.summary(), str) and fresh.summary()
    for obj in _fed(rng):
        assert isinstance(obj.done, bool)
        assert obj.summary()[0].isupper() and obj.summary().endswith(".")
        assert type(obj).__name__ in repr(obj)
        result = obj.result()
        assert result["kind"] == type(obj).__name__ and result["n"] == obj.n
        assert result["summary"] == obj.summary() and obj._estimate in result or "low" in result
        json.dumps(result)
        if not isinstance(obj, MeanInterval):
            assert result["verdict"] == obj.verdict and result["decided_at"] == obj.decided_at


def test_load_restores_any_snapshot():
    rng = np.random.default_rng(8)
    for obj in _fed(rng):
        again = evalues.load(json.loads(json.dumps(obj.to_dict())))
        assert type(again) is type(obj)
        assert again.to_dict() == obj.to_dict() and again.summary() == obj.summary()
    for junk in ({}, {"kind": "Nope"}, 3, MeanTest(0.5).result()):
        with pytest.raises(ValueError):
            evalues.load(junk)
    with pytest.raises(ValueError, match=r"json\.loads"):
        evalues.load(json.dumps(MeanTest(0.5).to_dict()))


def test_flag_and_combine_take_numbers_tests_or_named_tests():
    rng = np.random.default_rng(9)
    tests = {name: MeanTest(baseline=0.5) for name in ("parse", "render", "io", "net")}
    for name, test in tests.items():
        test.update_many(rng.random(800) < (0.62 if name in ("render", "net") else 0.5))
    numbers = [t.e_value for t in tests.values()]
    assert flag(tests) == ["render", "net"]
    assert flag(list(tests.values())) == flag(numbers) == [1, 3]
    assert flag_strict(tests) == ["render", "net"]
    assert combine(tests) == combine(numbers) == sum(numbers) / 4
    assert flag([]) == [] and flag({}) == [] and flag_strict([]) == []
    with pytest.raises(ValueError):
        flag(numbers, max_false_share=0)
    with pytest.raises(ValueError, match="did you mean 0.05"):
        flag(numbers, max_false_share=0.95)
    with pytest.raises(TypeError):
        flag(numbers, 0.95)                                # keyword only, so it cannot be mixed up
    with pytest.raises(ValueError):
        flag_strict(numbers, confidence=1.0)
    with pytest.raises(ValueError, match="e_value"):
        combine([MeanInterval()])


def test_statistical_names_live_in_expert():
    from evalues import expert

    e = [100.0, 1.0, 50.0, 2.0, 400.0]
    assert expert.ebh(e, 0.1) == flag(e, max_false_share=0.1)
    assert expert.ebh(e, closed=False) == flag(e, closed=False)
    assert expert.holm(e, 0.1) == flag_strict(e, confidence=0.9)
    assert expert.merge(e) == combine(e)
    assert expert.merge_independent(e) == evalues.combine_independent(e)
    assert not hasattr(evalues, "ebh") and not hasattr(evalues, "rel_diff")


def test_detector_says_which_way_it_went():
    rng = np.random.default_rng(10)
    with pytest.raises(TypeError):
        ChangeDetector(baseline=0.3)                       # false_alarm_every has no default
    detector = ChangeDetector(baseline=0.3, false_alarm_every=500)
    assert detector.verdict == "undecided"
    detector.update_many(rng.random(2000) < 0.1)
    assert detector.alarm and detector.done and detector.verdict == "decrease"
    assert "appears to have fallen below 0.3" in detector.summary()
    assert detector.decided_at == detector.alarm_at
    assert "once per 500 observations" in detector.summary()


def test_dominance_points_to_where():
    rng = np.random.default_rng(11)
    test = DominanceTest()
    for _ in range(5000):
        test.update(new=rng.normal(100, 5) + (60 if rng.random() < 0.1 else 0), old=rng.normal(100, 5))
        if test.done:
            break
    assert test.verdict == "higher_somewhere" and test.decided_at == test.n
    assert 100 < test.where < 170
    assert "hint where to look" in test.summary() and test.where_percentile > 0.5
