"""The README snippets on simulated data.  uv run python examples/quickstart.py"""

import json

import numpy as np

import evalues
from evalues import (
    ChangeDetector,
    Comparison,
    DominanceTest,
    MeanInterval,
    MeanTest,
    QuantileTest,
    SignTest,
    combine,
    flag,
    flag_strict,
)

rng = np.random.default_rng(0)


def latency_pairs(ratio, n):
    """(new, old) latencies; new is `ratio` times old on a typical run."""
    return ratio * rng.lognormal(0.0, 0.05, n), rng.lognormal(0.0, 0.05, n)


# Is the new version slower? It is, by 6 %.
comparison = Comparison(min_change=0.02)
for new, old in zip(*latency_pairs(1.06, 100_000), strict=True):
    comparison.update(new, old)
    if comparison.done:
        break
print("Comparison    ", comparison.summary())

# Nothing changed: the comparison can say so and stop.
same = Comparison(min_change=0.02)
for new, old in zip(*latency_pairs(1.0, 100_000), strict=True):
    same.update(new, old)
    if same.done:
        break
print("Comparison    ", same.summary())

# Only slowdowns matter, and there are at most 200 pairs.
budget = Comparison(min_change=0.02, direction="increase", deadline=200)
budget.update_many(*latency_pairs(1.03, 200))
print("Comparison    ", budget.summary())

# Is the pass rate still 50 %? It is 60 %.
test = MeanTest(baseline=0.5, bounds=(0, 1))
for passed in rng.random(10_000) < 0.6:
    test.update(passed)
    if test.done:
        break
print("MeanTest      ", test.summary())

# Accuracy on a benchmark of 2,000 questions in random order, to within +/- 3 points.
answers = rng.permutation(np.r_[np.ones(1420), np.zeros(580)])
estimate = MeanInterval(bounds=(0, 1), margin=0.03, population_size=2000)
for correct in answers:
    estimate.update(correct)
    if estimate.done:
        break
print("MeanInterval  ", estimate.summary())

# Is the p99 latency above the 120 ms limit? Heavy-tailed data, no bounds needed.
p99 = QuantileTest(limit=120.0, quantile=0.99, direction="above")
for latency in rng.lognormal(4.5, 0.3, 20_000):
    p99.update(latency)
    if p99.done:
        break
print("QuantileTest  ", p99.summary())

# Is new slower than old more often than not?
sign = SignTest(direction="increase")
for new, old in zip(*latency_pairs(1.02, 20_000), strict=True):
    sign.update(new, old)
    if sign.done:
        break
print("SignTest      ", sign.summary())

# Same typical latency, but 5 % of new requests are three times slower.
dominance = DominanceTest()
for new, old in zip(*latency_pairs(1.0, 20_000), strict=True):
    dominance.update(new * (3.0 if rng.random() < 0.05 else 1.0), old)
    if dominance.done:
        break
print("DominanceTest ", dominance.summary())

# An error rate that jumps from 1 % to 4 % at request 20,000.
detector = ChangeDetector(baseline=0.01, false_alarm_every=1_000_000, direction="increase")
for i in range(100_000):
    detector.update(rng.random() < (0.01 if i < 20_000 else 0.04))
    if detector.alarm:
        break
print("ChangeDetector", detector.summary())

# Thirty benchmarks; five got slower, five got faster. Which regressed?
benchmarks = {}
for i in range(30):
    ratio = 1.08 if i < 5 else 0.92 if i < 10 else 1.0
    benchmarks[f"bench{i:02d}"] = Comparison(min_change=0.02, direction="increase")
    benchmarks[f"bench{i:02d}"].update_many(*latency_pairs(ratio, 300))
print(f"flag           {flag(benchmarks)}")
print(f"flag_strict    {flag_strict(benchmarks)}")

# Four machines measured the same change with few pairs each.
machines = [Comparison(min_change=0.02, direction="increase") for _ in range(4)]
for machine in machines:
    machine.update_many(*latency_pairs(1.08, 60))
print(f"combine        single machines {[round(m.e_value, 1) for m in machines]} -> pooled {combine(machines):.1f}")

# Save a running test and continue it elsewhere.
restored = evalues.load(json.loads(json.dumps(comparison.to_dict())))
print(f"load           restored: {restored.summary() == comparison.summary()}")
print(f"result         {comparison.result()['verdict']!r} after {comparison.result()['decided_at']} pairs")

# The guarantee, checked by simulation: identical builds, how often is a change claimed?
rate = evalues.check.false_alarm_rate(
    make_test=lambda: Comparison(min_change=0.02),
    sample=lambda r: (r.lognormal(0.0, 0.05), r.lognormal(0.0, 0.05)),
    n_obs=600,
    n_runs=300,
)
print(f"check          wrong verdicts on identical builds: {rate:.3f} (limit 0.05)")
