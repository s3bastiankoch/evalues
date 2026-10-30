"""The statistical building blocks under their usual names.

    ebh, holm              multiple testing with e-values (`flag`, `flag_strict`)
    merge, merge_independent   merging e-values (`combine`, `combine_independent`)
    p_to_e, e_to_p         calibrators between p-values and e-values
    rel_diff               the paired statistic used by `Comparison`
    grow_stake, regrow_stake   worst-case-optimal fixed stakes for a bounded mean
"""

from ._combine import e_to_p, ebh, holm, merge, merge_independent, p_to_e
from ._grow import grow_stake, regrow_stake
from ._mean import rel_diff

__all__ = [
    "e_to_p",
    "ebh",
    "grow_stake",
    "holm",
    "merge",
    "merge_independent",
    "p_to_e",
    "regrow_stake",
    "rel_diff",
]
