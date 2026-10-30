# Statistics

What the objects are, for readers who know e-values. Every method here is published; the references are at the end.

## The objects

The objects are e-processes and confidence sequences for bounded means built
by betting (aGRAPA stakes, separate up and down games per candidate mean).
`e_value`, `log_e_value` and an anytime-valid `p_value` are available on every
test. `MeanTest(direction="above")` tests the null mean <= baseline.
`MeanInterval` is a confidence sequence on a grid: a candidate is excluded
from below once its up game reaches 2/alpha and from above once its down game
does, and the interval runs between the unbroken excluded runs, which needs no
assumption about the shape of the capital in the candidate. `DifferenceTest`
runs one-sided games
at ±`min_difference`: four games at threshold 2/alpha when looking both ways,
two at 1/alpha when looking one way. `Comparison` is a `DifferenceTest` on
(new - old)/(new + old). `flag` is the closed e-BH procedure, `flag_strict` is
e-Holm, `combine` is arithmetic-mean merging. `evalues.expert` exports them as
`ebh`, `holm`, `merge` and `merge_independent` with an `alpha` argument, along
with the calibrators `p_to_e` and `e_to_p`, and the single-round GROW and
REGROW stakes of Arnold and Clerico as `grow_stake` and `regrow_stake`.

## How it works

To test a candidate average, play a betting game against it: start with
capital 1 and stake part of it, before each observation, on the observation
landing above the candidate. If the candidate is the true average the game is
fair and capital only reaches 20 with probability 1/20, however long you play
(Ville's inequality). That capital is the e-value. The stake is sized from
past observations only, which is what makes looking safe. The whole engine is
about 250 lines in [`src/evalues/_games.py`](../src/evalues/_games.py).

## References

- Waudby-Smith and Ramdas (2024). Estimating means of bounded random variables by betting. JRSS-B.
- Vovk and Wang (2021). E-values: calibration, combination and applications. Annals of Statistics.
- Wang and Ramdas (2022). False discovery rate control with e-values. JRSS-B.
- Wang, Dandapanthula and Ramdas (2025). Anytime-valid FDR control with the stopped e-BH procedure. arXiv:2502.08539.
- Xu, Solari, Fischer, de Heide, Ramdas and Goeman (2025). Bringing closure to false discovery rate control. arXiv:2509.02517.
- Hartog and Lei (2025). Family-wise error rate control with e-values. arXiv:2501.09015.
- Wang (2025). The only admissible way of merging arbitrary e-values. Biometrika.
- Waudby-Smith and Ramdas (2020). Confidence sequences for sampling without replacement. NeurIPS. arXiv:2006.04347.
- Shin, Ramdas and Rinaldo (2024). E-detectors: a nonparametric framework for sequential change detection. arXiv:2203.03532.
- Arnold, Choe, Scarsini and Tsetlin (2026). Betting on bets: anytime-valid tests for stochastic dominance. arXiv:2604.21851.
- Taga, Oymak and Shekhar (2026). Learning to bet for horizon-aware anytime-valid testing. arXiv:2603.19551.
- Clerico, Wegel, Azangulov and Rebeschini (2026). Time-sensitive anytime-valid testing. arXiv:2605.06521.
- Arnold and Clerico (2026). Optimal e-values for testing the mean of a bounded random variable against a composite alternative. arXiv:2601.11347.
- Clerico (2025). On the optimality of coin-betting for mean estimation. arXiv:2412.02640.
- Ramdas and Wang (2025). Hypothesis testing with e-values. Foundations and Trends in Statistics.

