# Changelog

## 0.1.0 (2026-10-30)

First release.

- The published methods only, with the fixes of an internal audit (closed e-BH summation order, fixed stakes near a bound, finite-population rounding, capped log capital).
- One pattern for every object: `update(...)`, `.done`, `.verdict`, `.summary()`, `.result()`, `to_dict()` and `evalues.load()`.
- `Comparison`: paired before/after measurements with a three-way verdict, or `direction="increase"` to look one way only.
- `MeanTest`, `MeanInterval`: betting test and confidence sequence for the average of bounded observations, optionally for sampling without replacement (`population_size`).
- `DifferenceTest`: three-way or one-directional verdict against a minimum difference.
- `QuantileTest`, `SignTest`: tests for percentiles and signs that need no bounds.
- `DominanceTest`: first-order stochastic dominance between paired samples.
- `ChangeDetector`: e-detector with a guaranteed average time between false alarms.
- `deadline=N` on every test; `MeanTest` also sizes its bets for it. `MeanTest(expected=...)` for a known alternative (worst-case-optimal fixed stake), optionally with `hoped_within=...`.
- `combine`, `combine_independent`, `flag` (closed e-BH), `flag_strict` (e-Holm), accepting numbers, tests or dicts.
- `evalues.expert`: `ebh`, `holm`, `merge`, `merge_independent`, `p_to_e`, `e_to_p`, `rel_diff`, `grow_stake`, `regrow_stake`.
- `evalues.check`: simulation helpers for wrong-verdict rate and coverage.
