import json

import numpy as np
import pytest

from evalues import DominanceTest

ALPHA = 0.05
RUNS = 400
SLACK = 3 * (ALPHA * (1 - ALPHA) / RUNS) ** 0.5


def _rate(draw, n=400, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(RUNS):
        test = DominanceTest()
        xs, ys = draw(rng, n)
        for x, y in zip(xs, ys, strict=True):
            test.update(new=y, old=x)
            if test.done:
                hits += 1
                break
    return hits / RUNS


def _shared_noise(r, n):
    shared = r.lognormal(0, 1, n)
    return shared * r.lognormal(0, 0.2, n), shared * r.lognormal(0, 0.2, n)


NULLS = {
    "same distribution, independent": lambda r, n: (r.lognormal(0, 1, n), r.lognormal(0, 1, n)),
    "same distribution, same machine noise": lambda r, n: _shared_noise(r, n),
    "y smaller everywhere": lambda r, n: (r.normal(0.3, 1, n), r.normal(0, 1, n)),
    "few distinct values": lambda r, n: (r.integers(0, 4, n), r.integers(0, 4, n)),
    "heavy tails": lambda r, n: (r.standard_cauchy(n), r.standard_cauchy(n)),
}


@pytest.mark.parametrize("name", NULLS)
def test_rarely_rejects_when_y_has_no_upside(name):
    assert _rate(NULLS[name]) <= ALPHA + SLACK


def test_finds_a_shift_and_a_heavier_tail_with_equal_means():
    shift = _rate(lambda r, n: (r.normal(0, 1, n), r.normal(0.5, 1, n)), n=600)
    tail = _rate(lambda r, n: (r.normal(0, 1, n), r.normal(0, 2, n)), n=1500)
    assert shift > 0.9 and tail > 0.9


def test_direction_matters():
    rng = np.random.default_rng(1)
    x, y = rng.normal(0, 1, 800), rng.normal(0.5, 1, 800)
    forward, backward = DominanceTest(), DominanceTest()
    forward.update_many(y, x)
    backward.update_many(x, y)
    assert forward.done and not backward.done


def test_snapshot_continues_exactly_and_bad_input_raises():
    rng = np.random.default_rng(2)
    x, y = rng.normal(0, 1, 300), rng.normal(0.2, 1.5, 300)
    whole, first = DominanceTest(levels=20), DominanceTest(levels=20)
    whole.update_many(y, x)
    first.update_many(y[:120], x[:120])
    resumed = DominanceTest.from_dict(json.loads(json.dumps(first.to_dict())))
    resumed.update_many(y[120:], x[120:])
    assert resumed.to_dict() == whole.to_dict()
    with pytest.raises(ValueError):
        whole.update(float("nan"), 1.0)
    with pytest.raises(ValueError):
        whole.update_many([1.0, 2.0], [1.0])
