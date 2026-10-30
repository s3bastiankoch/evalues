# Guide

Every object with an example, and what to watch out for. The [README](../README.md) has the short version; the statistics behind each object are in [statistics.md](statistics.md).

## Start here: is the new one different from the old one?

```python
from evalues import Comparison

comparison = Comparison(min_change=0.02)        # a change below 2 % does not matter to me
for new_ms, old_ms in paired_measurements:      # e.g. run time after and before a code change
    comparison.update(new=new_ms, old=old_ms)
    if comparison.done:
        break

print(comparison.summary())
# Decided after 226 pairs: averaged over pairs, new is more than 2 % larger than old
# (error rate at most 5 %). Average change so far +5.98 % (an estimate).
```

Have all the data already? `comparison.update_many(new=new_values, old=old_values)`.

`comparison.verdict` is `"increase"`, `"decrease"`, `"no_change"` or
`"undecided"`. `"no_change"` is a real result: averaged over pairs, the change
is not more than `min_change` (it can still be 1.9 % at a 2 % threshold), so
you can stop measuring.

**What the 5 % means.** Out of all comparisons you run, on average at most
5 % ever show a wrong verdict, however often you look, provided the conditions
under "Things to know" hold. That is a promise about the method, not a 95 %
probability that this particular verdict is right.

**What `Comparison` looks at.** It answers "is new slower on average over
pairs?", with every pair counted by its relative difference and no single
pair counting more than ±100 %. So rare huge spikes barely show, and a few
much slower pairs can give "increase" although most pairs are unchanged. Use
`SignTest` for "slower more often than not" and `DominanceTest` for a heavier
tail, on the same pairs.

**Only slowdowns matter?** `Comparison(min_change=0.02, direction="increase")`
needs fewer pairs and answers `"increase"` or `"no_increase"`.

## One pattern

Create an object, call `update(...)` for every new observation, and read the
result whenever you want:

| | |
|---|---|
| `.done` | is there a result (or has the deadline passed)? |
| `.verdict` | what was found, or `"undecided"` (not on `MeanInterval`, which gives a range) |
| `.summary()` | one plain sentence |
| `.result()` | the same as a small dict, for logs |
| `.n`, `.decided_at` | observations so far, and how many it took |

## Pick by the question you have

| Question | Use | Feed it | Verdicts |
|---|---|---|---|
| Is new larger or smaller than old, by enough to matter? | `Comparison` | pairs of positive numbers | increase, decrease, no_change |
| Is the average above or below a fixed value? | `MeanTest` | numbers within known bounds | above, below |
| Where is the average? | `MeanInterval` | numbers within known bounds | none: gives `.low`, `.high` |
| Is the average difference larger than X, or safely smaller? | `DifferenceTest` | differences within known bounds | increase, decrease, no_change |
| Is the median or the p99 above or below a fixed limit? | `QuantileTest` | any numbers | above, below |
| Is new larger than old more often than it is smaller? | `SignTest` | pairs of any numbers | increase, decrease |
| Is new higher anywhere in the distribution, such as the tail? | `DominanceTest` | pairs of any numbers | higher_somewhere |
| Tell me when the average of a stream moves | `ChangeDetector` | numbers within known bounds | alarm: increase, decrease |

All take `confidence=0.95`, meaning at most 5 % of uses end in a wrong
verdict, except `ChangeDetector`, whose promise is about time. Tests take
`direction` to look one way only (`DominanceTest` looks one way by nature;
swap its arguments for the other) and `deadline=N` to stop after at most N
observations. "no_change" and "no_increase" mean "not more than the threshold
you set", not "nothing at all".

**Against a fixed value**

```python
from evalues import MeanTest

test = MeanTest(baseline=0.5, bounds=(0, 1))    # is the pass rate still 50 %?
for passed in stream:
    test.update(passed)
    if test.done:
        break
test.verdict                                    # "above", "below" or "undecided"
```

**A range instead of a yes or no**

```python
from evalues import MeanInterval

estimate = MeanInterval(bounds=(0, 1), margin=0.03)     # accuracy to within +/- 3 points
for correct in answers:
    estimate.update(correct)
    if estimate.done:
        break
estimate.low, estimate.high
```

If the observations are the items of a finite set in random order, such as
the 2,000 questions of a benchmark, add `population_size=2000`: the range then
closes in as the set is used up. `MeanTest` takes the same option.

**Medians and percentiles, no bounds needed.** Each observation only counts
as above or below a limit, so rare huge outliers cannot break anything.

```python
from evalues import QuantileTest, SignTest

p99 = QuantileTest(limit=120.0, quantile=0.99, direction="above")       # is the p99 above the 120 ms limit?
sign = SignTest()                                                       # is new slower more often than not?
for new, old in paired_latencies:
    p99.update(new)
    sign.update(new, old)
```

The `limit` must be a fixed number you chose beforehand, such as a service
level objective. Do not use a percentile measured on the old build; to compare
the distributions of two builds use `DominanceTest`.

**Whole distributions, not only the average**

```python
from evalues import DominanceTest

test = DominanceTest()
for new, old in paired_latencies:
    test.update(new, old)
    if test.done:               # new exceeds some level more often than old does
        print(test.summary())   # includes a hint at which level to look
        break
```

**Watching a stream indefinitely.** A test answers once. A detector keeps
running and promises that false alarms are rare in time:

```python
from evalues import ChangeDetector

# error rate normally 1 %, 100,000 requests a day, about one false alarm a month
detector = ChangeDetector(baseline=0.01, false_alarm_every=30 * 100_000, direction="increase")
for failed in stream:
    detector.update(failed)
    if detector.alarm:
        print(detector.summary())
        detector.reset()
```

## If it stays undecided

The closer the true change is to your threshold, the longer a decision takes;
exactly on the threshold it may never come. Median number of pairs until a
verdict in a simulation with `min_change=0.03` and 5 % noise per measurement:

| true change | 0 % | +2 % | +4 % | +5 % | +6 % | +8 % | +10 % | +20 % |
|---|---|---|---|---|---|---|---|---|
| `direction="either"` | 320 | 814 | 900 | 532 | 312 | 184 | 134 | 61 |
| `direction="increase"` | 234 | 770 | 854 | 369 | 245 | 140 | 104 | 49 |

So: take more pairs, raise `min_change` to the smallest change you would act
on, look one way only, or set a `deadline` and accept "undecided" as an
answer. "Undecided" is never evidence that nothing changed. On `Comparison`,
`DifferenceTest` and `DominanceTest` a deadline only stops the test.

On `MeanTest`, `QuantileTest` and `SignTest` a deadline also makes the bets
bolder as it approaches, and `MeanTest` can bet for a known alternative:

```python
MeanTest(baseline=0.5, deadline=100)                    # at most 100 observations: bets get bolder near the end
MeanTest(baseline=0.5, expected=0.65)                   # you expect about 0.65 if the baseline is wrong
MeanTest(baseline=0.5, expected=0.65, hoped_within=50)  # ... and hope for an answer within about 50
```

In a simulation with 100 coin flips, baseline 0.5 and true bias 0.65, the
chance of a verdict in time was 44 % by default and 51 % with `deadline=100`.
Looking one way only (`direction="above"`) gave 53 %. Telling the test the
true bias, which also makes it look one way, gave 80 % with `expected=0.65`
and 75 % with `expected=0.65, hoped_within=50`. A wrong guess costs: with
`expected=0.65`, a true bias of 0.55 and 1,000 flips the chance was 34 %,
against 54 % for the default, which needs no guess. With a distant deadline
(500 flips, true bias 0.58) `deadline` gained little: 62 % against 60 %.

## Many tests at once

Every test carries an `e_value`, its strength of evidence: 1 means none, and
if nothing is going on the chance that it ever reaches 20 is at most 5 %.
E-values can be averaged, so results from several machines or teams combine
without sharing raw data. The functions take numbers, the tests themselves,
or a dict of them.

```python
from evalues import combine, flag, flag_strict

benchmarks = {name: Comparison(min_change=0.02, direction="increase") for name in names}
...                                                 # feed each its own pairs

flag(benchmarks)                            # which regressed? on average at most 5 % of the list is false
flag(benchmarks, max_false_share=0.10)      # accept 10 % instead
flag_strict(benchmarks)                     # at most a 5 % chance of even one false entry

combine(machines) >= 20                     # several machines, one question: is the averaged evidence enough?
```

Things to keep straight:

- `flag` may be called at any moment; act on the latest list only. Pass the
  tests as they are now, never each test's best value so far.
- `combine` averages. That protects against one lucky machine and never gives
  more evidence than the best single machine. To gain from four machines,
  feed all their pairs into one `Comparison` in the order they arrive.
- `combine_independent` multiplies and is only for sources that truly are
  independent. Two CI jobs on the same host are not.

## Saving and moving a test

```python
import json, evalues

saved = json.dumps(test.to_dict())              # the full state as plain JSON
test = evalues.load(json.loads(saved))          # any object; continues exactly where it stopped
```

## Checking the guarantee yourself

```python
import evalues

evalues.check.false_alarm_rate(
    make_test=lambda: evalues.Comparison(min_change=0.02),
    sample=lambda rng: (rng.lognormal(0, 0.05), rng.lognormal(0, 0.05)),   # identical builds
    n_obs=600,
)                                                                          # stays below 0.05
```

[`tests/test_guarantees.py`](../tests/test_guarantees.py) and [`tests/test_surface.py`](../tests/test_surface.py) run such checks on
every commit.

## Things to know

- **Nothing may drift while you measure.** Warm-up, a machine getting busier
  or traffic changing during the test break the guarantee. For `Comparison`,
  take each new/old pair close together and alternate which goes first; then
  drift cancels within the pair.
- **Bounds are hard limits, not what you happened to see.** `MeanTest`,
  `MeanInterval`, `DifferenceTest` and `ChangeDetector` need the range a
  single observation can possibly take: 0 and 1 for pass/fail, 0 and the
  timeout for a duration. Choosing them from the data you test breaks the
  guarantee. True limits that are tight give faster answers. Values outside
  raise an error; with `clip=True` they are clipped and the result is about
  the clipped values. The other four classes need no bounds.
- **`min_change` is the smallest change you would act on,** as a fraction:
  0.02 is 2 %. On the way down the same threshold is new = old / 1.02, about
  -1.96 %.
- **"undecided" is not "no change".** Only `Comparison`, `DifferenceTest` and
  a narrow `MeanInterval` can tell you that a change is smaller than the
  threshold you set.
- **The first verdict is final.** Later data do not revise it; `summary()`
  says when it was reached.
- **A fixed-sample test is more efficient** if you really do look only once.
  The cost of being allowed to look is a somewhat larger sample for the same
  decision.

