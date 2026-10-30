"""Combining evidence from several tests.

Every test here carries an `e_value`: a strength of evidence on a scale where
1 means "nothing" and 1 / (1 - confidence) means "decided" (20 at 95 %).
Unlike p-values, e-values can be averaged, which is what makes evidence from
many machines, benchmarks or teams easy to combine.

Every function accepts plain numbers, test objects (their current `e_value`
is used), or a dict of either, in which case the dict's keys are returned
instead of positions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np

from ._common import alpha_from


def _read(evidence) -> tuple[list, list[float]]:
    """Labels and e-values from numbers, test objects, or a dict of either."""
    if isinstance(evidence, Mapping):
        labels, items = list(evidence.keys()), list(evidence.values())
    else:
        items = list(evidence)
        labels = list(range(len(items)))
    values = []
    for item in items:
        value = getattr(item, "e_value", item)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(  # noqa: TRY004
                f"combine and flag need e-values or tests that have an e_value, got "
                f"{type(item).__name__} (MeanInterval and ChangeDetector have none)"
            )
        values.append(float(value))
    if any(math.isnan(e) or e < 0 for e in values):
        raise ValueError("e-values must be non-negative numbers")
    return labels, values


def _need_some(values: list[float]) -> None:
    if not values:
        raise ValueError("need at least one e-value")


def combine(evidence, weights: Sequence[float] | None = None) -> float:
    """Combine evidence on the same question from several sources into one e-value.

        combine([runner.e_value for runner in runners])     # or simply combine(runners)

    Takes the average. This is safe however the sources depend on each other,
    for example several runners sharing a noisy machine. The question is
    decided when the result reaches 1 / (1 - confidence), 20 at 95 %.

    Averaging protects against one lucky source; it never gives more evidence
    than the best single source. To gain from four machines, feed all their
    measurements into one test in the order they arrive. The result has no
    direction: give the tests a `direction` if two sources pointing opposite
    ways should not add up, and note that "no change" verdicts cannot be
    pooled this way.

    `weights` must be fixed before seeing the evidence and be non-negative. If
    they sum to less than 1, the rest counts as an e-value of 1 ("no evidence"),
    which damps the influence of the sources. Weighted averages of this form
    are the only combination rules that cannot be improved upon under unknown
    dependence.

    Vovk and Wang (2021), "E-values: calibration, combination and applications";
    Wang (2025), "The only admissible way of merging arbitrary e-values".
    """
    _, values = _read(evidence)
    _need_some(values)
    if weights is None:
        return sum(values) / len(values)
    w = [float(x) for x in weights]
    if len(w) != len(values):
        raise ValueError(f"{len(w)} weights for {len(values)} e-values")
    total = sum(w)
    if any(x < 0 for x in w) or total > 1.0 + 1e-9:
        raise ValueError("weights must be non-negative and sum to at most 1")
    return sum(x * e for x, e in zip(w, values, strict=True)) + max(0.0, 1.0 - total)


def combine_independent(evidence) -> float:
    """Combine evidence from independent sources into one e-value, by multiplying.

    Much more powerful than `combine`, and wrong if the sources are correlated:
    shared hardware, shared data or a shared model make the product overstate
    the evidence. Two CI jobs on the same host are not independent. It is also fragile: one source reporting 0 vetoes everything.
    Rule of thumb: multiply along time within one experiment, average across
    machines, teams or data sets. When in doubt, use `combine`.
    """
    _, values = _read(evidence)
    _need_some(values)
    return math.prod(values)


def _base_ebh(values: list[float], alpha: float) -> list[int]:
    k = len(values)
    order = sorted(range(k), key=lambda i: values[i], reverse=True)
    n_rejected = 0
    for rank in range(1, k + 1):
        if values[order[rank - 1]] >= k / (alpha * rank):
            n_rejected = rank
    return sorted(order[:n_rejected])


def flag(evidence, *, max_false_share: float = 0.05, closed: bool = True) -> list:
    """Which of many items to flag, keeping false flags a small share.

        flag({"parse": parse_test, "render": render_test, "io": io_test})   # -> ["render"]

    Give one e-value (or test) per item, each about its own question, for
    example one per benchmark. Returns the items to flag such that, on
    average, at most `max_false_share` of the flagged items are false alarms
    (statisticians: the false discovery rate), however the items depend on
    each other. Flagging ten items of which one may be wrong is usually an
    acceptable deal; if every flag triggers costly action, use `flag_strict`.

    With running tests the function may be called at any moment, provided all
    tests are fed in one common order of events. Pass the tests themselves or
    their current `e_value`, never the largest value each has ever reached.
    The promise is about the list you act on: if you call `flag` repeatedly,
    act on the latest list only.

    An item can be flagged while its own verdict is still "undecided" (the bar
    drops when many others clearly stand out). Values above 20 are strong;
    p-values do not belong here (convert them with `evalues.expert.p_to_e`).
    To list only slowdowns, give the tests `direction="increase"`.

    This is the closed e-BH procedure, which always flags at least as much as
    the original e-BH. `closed=False` gives the original, which is faster for
    many thousands of items.

    Wang and Ramdas (2022), "False discovery rate control with e-values";
    Xu, Solari, Fischer, de Heide, Ramdas and Goeman (2025), "Bringing closure
    to false discovery rate control"; Wang, Dandapanthula and Ramdas (2025)
    for the any-moment guarantee.
    """
    alpha = max_false_share
    if not 0.0 < alpha <= 0.5:
        hint = f"; did you mean {1 - alpha:g}?" if 0.5 < alpha < 1 else ""
        raise ValueError(
            f"max_false_share is the share of false flags you accept, usually 0.05; got {alpha}{hint}"
        )
    return _ebh(evidence, alpha, closed)


def _ebh(evidence, alpha: float, closed: bool) -> list:
    labels, values = _read(evidence)
    if not values:
        return []
    if not closed:
        return [labels[i] for i in _base_ebh(values, alpha)]

    # Flag the r largest e-values, for the largest r such that every subset S
    # of items satisfies  mean(e over S) >= |S and flagged| / (r * alpha).
    # The hardest subsets take the smallest flagged and the smallest
    # non-flagged e-values, so it is enough to check those.
    # Sums are built from the small end, so that one enormous e-value cannot
    # swallow the small ones through rounding.
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i], reverse=True)
    e = np.minimum(np.array([values[i] for i in order]), 1e300)
    for r in range(m, 0, -1):
        head = np.cumsum(e[:r][::-1])[:, None]                             # the j smallest flagged, j = 1..r
        tail = np.concatenate([[0.0], np.cumsum(e[r:][::-1])])[None, :]    # the k smallest others, k = 0..m-r
        j = np.arange(1, r + 1)[:, None]
        k = np.arange(0, m - r + 1)[None, :]
        need = (j + k) * j / (r * alpha)
        if np.all(head + tail >= need * (1.0 - 1e-12)):
            return [labels[i] for i in sorted(order[:r])]
    return []


def flag_strict(evidence, *, confidence: float = 0.95) -> list:
    """Which of many items to flag, with at most a small chance of even one false flag.

    Stricter than `flag`: with probability at least `confidence` not even one
    flagged item is a false alarm (statisticians: family-wise error rate),
    however the items depend on each other. Use it when every flag triggers
    costly action.

    Item i is flagged when, with bar = 1 / (1 - confidence),
        e_i >= bar + sum over other j with e_j < bar of (bar - e_j).
    This is closed testing with the average as the combination rule (e-Holm);
    it never flags less than the Bonferroni rule e_i >= K * bar. The bar rises
    with the number of items, so an item whose own verdict is already reached
    can be left out.

    Vovk and Wang (2021); Hartog and Lei (2025), "Family-wise error rate
    control with e-values".
    """
    return _holm(evidence, alpha_from(confidence))


def _holm(evidence, alpha: float) -> list:
    labels, values = _read(evidence)
    bar = 1.0 / alpha
    shortfall = sum(bar - e for e in values if e < bar)
    return [labels[i] for i, e in enumerate(values) if e >= bar and e >= bar + shortfall]


def e_to_p(e_value: float, u: float | None = None) -> float:
    """Turn an e-value into a p-value: min(1, 1 / e).

    Passing `u`, one uniform random number in (0, 1] drawn independently of the
    data, gives the smaller p-value min(1, u / e). It is still valid, at the
    price of a result that depends on the draw; record `u` with the result.
    """
    if math.isnan(e_value) or e_value < 0:
        raise ValueError(f"e-value must be non-negative, got {e_value}")
    if u is None:
        u = 1.0
    elif not 0.0 < u <= 1.0:
        raise ValueError(f"u must be in (0, 1], got {u}")
    return 1.0 if e_value == 0 else min(1.0, u / e_value)


def p_to_e(p_value: float, kappa: float | None = None) -> float:
    """Turn a p-value into an e-value.

    Lets results from a classical test join `combine`, `flag` or `flag_strict`.
    The conversion loses power (p = 0.01 becomes e = 4.5), and converting back
    and forth loses more each time, so stay with e-values once you have them.

    The default needs no tuning: (1 - p + p ln p) / (p (ln p)^2), the average
    of kappa * p^(kappa - 1) over all kappa. Pass `kappa` in (0, 1) for that
    single rule instead; small kappa rewards very small p-values, and kappa
    must be chosen before seeing the p-value.

    Vovk and Wang (2021); Shafer, Shen, Vereshchagin and Vovk (2011).
    """
    if not 0.0 < p_value <= 1.0:
        raise ValueError(f"p-value must be in (0, 1], got {p_value}")
    if kappa is None:
        if p_value == 1.0:
            return 0.5
        log_p = math.log(p_value)
        if log_p > -1e-4:                     # series around p = 1 to avoid 0 / 0
            return 0.5 - log_p / 6.0
        return (1.0 - p_value + p_value * log_p) / (p_value * log_p * log_p)
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must be between 0 and 1, got {kappa}")
    return kappa * p_value ** (kappa - 1.0)


# The names statisticians know these procedures by.
def merge(e_values, weights: Sequence[float] | None = None) -> float:
    """Alias of `combine`: arithmetic-mean merging of e-values."""
    return combine(e_values, weights)


def merge_independent(e_values) -> float:
    """Alias of `combine_independent`: product merging of e-values."""
    return combine_independent(e_values)


def ebh(e_values, alpha: float = 0.05, *, closed: bool = True) -> list:
    """Alias of `flag`: the (closed) e-BH procedure at FDR level `alpha`."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    return _ebh(e_values, alpha, closed)


def holm(e_values, alpha: float = 0.05) -> list:
    """Alias of `flag_strict`: e-Holm at family-wise error level `alpha`."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    return _holm(e_values, alpha)
