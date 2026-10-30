<div align="center">

# evalues

**Statistical tests you can check after every observation**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/s3bastiankoch/evalues/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)](https://github.com/s3bastiankoch/evalues/blob/main/pyproject.toml)
[![Dependencies: numpy](https://img.shields.io/badge/dependencies-numpy-4dabcf)](https://github.com/s3bastiankoch/evalues/blob/main/pyproject.toml)
[![CI](https://github.com/s3bastiankoch/evalues/actions/workflows/ci.yml/badge.svg)](https://github.com/s3bastiankoch/evalues/actions/workflows/ci.yml)

</div>

A classical test fixes the sample size in advance. Look at the result early, or keep measuring until it turns significant, and its error rate is no longer what it claims. Benchmarks, A/B tests and model evaluations are run exactly like that.

evalues implements tests built for looking. Each one is a betting game against the hypothesis it tests, and its capital, the **e-value**, can be checked after every observation: stop whenever you like and the error rate stays below the level you set. A test also says when it is done, so you measure until the data decide instead of for a fixed time. Every method is published; [docs/statistics.md](https://github.com/s3bastiankoch/evalues/blob/main/docs/statistics.md) cites the source of each.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/s3bastiankoch/evalues/main/docs/peeking-dark.png">
  <img src="https://raw.githubusercontent.com/s3bastiankoch/evalues/main/docs/peeking-light.png" width="100%" alt="Checking after every pair, on identical builds, 2,000 runs of up to 1,000 pairs each. A t-test checked after every pair ends in a wrong verdict in 46 % of runs although it promises 5 %; evalues Comparison in 0 of 2,000 runs.">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/s3bastiankoch/evalues/main/docs/evidence-dark.png">
  <img src="https://raw.githubusercontent.com/s3bastiankoch/evalues/main/docs/evidence-light.png" width="100%" alt="E-value of a Comparison with min_change 0.02 after every pair, 5 % noise. A +6 % change reaches the decision threshold 20 after 266 pairs; a +3 % change, close to the 2 % threshold, after 862.">
</picture>

## Quickstart

```bash
pip install evalues        # needs only numpy
```

Is the new build slower than the old one, by more than 2 %?

```python
from evalues import Comparison

comparison = Comparison(min_change=0.02)        # a change below 2 % does not matter
for new_ms, old_ms in paired_measurements:      # run time after and before a change
    comparison.update(new=new_ms, old=old_ms)
    if comparison.done:
        break

print(comparison.summary())
```

```
Decided after 226 pairs: averaged over pairs, new is more than 2 % larger than old
(error rate at most 5 %). Average change so far +5.98 % (an estimate).
```

`comparison.verdict` is `"increase"`, `"decrease"`, `"no_change"` or `"undecided"`. `"no_change"` is a result too: the change is below `min_change`, so you can stop measuring. `uv run python examples/quickstart.py` runs every object on simulated data.

## Pick by the question you have

| Question | Use | Feed it |
|---|---|---|
| Is new larger or smaller than old, by enough to matter? | `Comparison` | pairs of positive numbers |
| Is the average above or below a fixed value? | `MeanTest` | numbers within known bounds |
| Where is the average? | `MeanInterval` | numbers within known bounds |
| Is the average difference larger than X, or safely smaller? | `DifferenceTest` | differences within known bounds |
| Is the median or the p99 above or below a fixed limit? | `QuantileTest` | any numbers |
| Is new larger than old more often than it is smaller? | `SignTest` | pairs of any numbers |
| Is new higher anywhere in the distribution, such as the tail? | `DominanceTest` | pairs of any numbers |
| Tell me when the average of a stream moves | `ChangeDetector` | numbers within known bounds |
| Many benchmarks: which ones changed? | `flag`, `flag_strict` | the tests above |
| Several machines, one question | `combine` | the tests above |

Every object has `update(...)`, `.done`, `.summary()`, `.result()` and `to_dict()`, and all but `MeanInterval` (which gives a range) have `.verdict`; `evalues.load()` continues a saved test where it stopped. Examples for each row, deadlines, and what to watch out for are in [docs/guide.md](https://github.com/s3bastiankoch/evalues/blob/main/docs/guide.md).

## Two things to know

- **Nothing may drift while you measure.** Take each new/old pair close together and alternate which goes first; then drift cancels within the pair.
- **"undecided" is not "no change".** Only `Comparison`, `DifferenceTest` and a narrow `MeanInterval` can tell you that a change is below your threshold.

## Repository layout

```
src/evalues/         the library; the betting engine is _games.py (about 250 lines)
tests/               unit tests and simulation checks of every guarantee (uv run pytest, a few minutes)
examples/            quickstart.py
docs/                guide.md (usage), statistics.md (methods and references), README figures
scripts/figures/     make.py regenerates the README figures
```

## Citation

If you use evalues, please cite it as software, together with the papers of the methods you use ([docs/statistics.md](https://github.com/s3bastiankoch/evalues/blob/main/docs/statistics.md)):

```bibtex
@software{koch2026evalues,
  author = {Koch, Sebastian},
  title  = {{evalues}: Statistical Tests You Can Check after Every Observation},
  year   = {2026},
  url    = {https://github.com/s3bastiankoch/evalues},
  note   = {Version 0.1.0}
}
```

## License

[MIT](https://github.com/s3bastiankoch/evalues/blob/main/LICENSE).
