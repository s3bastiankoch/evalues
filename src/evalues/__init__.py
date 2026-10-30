"""evalues: statistical tests you can check after every observation.

Pick by the question you have:

    Comparison      is new larger or smaller than old, by enough to matter?
    MeanTest        is the average above or below a fixed baseline?
    MeanInterval    where is the average? (a range that is valid at every moment)
    DifferenceTest  is the average difference larger than X, or safely smaller?
    QuantileTest    is the median / p99 above or below a fixed threshold?
    SignTest        is new larger than old more often than it is smaller?
    DominanceTest   is new higher anywhere in the distribution, e.g. in the tail?
    ChangeDetector  watch a stream and raise an alarm when its average moves

    combine         several sources, one question: pool the evidence
    flag            many questions: which items stand out?
    flag_strict     the same, when a single false flag is costly

Every object has `update(...)`, `.done`, `.summary()` and `.result()`; all but
`MeanInterval` have `.verdict`. `evalues.expert` holds the statistical names.
"""

from . import check, expert
from ._combine import combine, combine_independent, flag, flag_strict
from ._detect import ChangeDetector
from ._dominance import DominanceTest
from ._mean import Comparison, DifferenceTest, MeanInterval, MeanTest
from ._order import QuantileTest, SignTest

__version__ = "0.1.0"

_KINDS = {
    cls.__name__: cls
    for cls in (Comparison, MeanTest, MeanInterval, DifferenceTest, QuantileTest, SignTest,
                DominanceTest, ChangeDetector)
}


def load(state: dict):
    """Rebuild any object from the dict its `to_dict()` produced.

        saved = json.dumps(test.to_dict())
        test = evalues.load(json.loads(saved))      # continues exactly where it stopped
    """
    if isinstance(state, str | bytes):
        raise ValueError(  # noqa: TRY004
            "load needs the dict, not the JSON text: evalues.load(json.loads(text))"
        )
    kind = state.get("kind") if isinstance(state, dict) else None
    if kind in _KINDS and "summary" in state:
        raise ValueError("this is a result(), not a snapshot; save test.to_dict() instead")
    if kind not in _KINDS:
        raise ValueError(f"not a snapshot made by to_dict(): kind={kind!r}")
    return _KINDS[kind].from_dict(state)


__all__ = [
    "ChangeDetector",
    "Comparison",
    "DifferenceTest",
    "DominanceTest",
    "MeanInterval",
    "MeanTest",
    "QuantileTest",
    "SignTest",
    "check",
    "combine",
    "combine_independent",
    "expert",
    "flag",
    "flag_strict",
    "load",
]
