"""Shared argument checks and wording."""

from __future__ import annotations

import math
from collections.abc import Sized
from numbers import Real

VERDICTS_FOUND = frozenset({"above", "below", "increase", "decrease", "higher_somewhere"})
VERDICTS_ALL = VERDICTS_FOUND | {"no_change", "no_increase", "no_decrease", "undecided"}


def number(name: str, value, hint: str = "") -> float:
    """A finite float, or an error that names the argument."""
    if isinstance(value, bool):
        return float(value)
    if value is None:
        raise ValueError(f"{name} is None; skip missing measurements instead of passing them")
    if not isinstance(value, Real):
        if isinstance(value, str):
            raise ValueError(f"{name} must be a number, got the text {value!r}{hint}")  # noqa: TRY004
        if isinstance(value, Sized) and not hasattr(value, "__float__"):
            raise ValueError(f"{name} must be one number; use update_many for a list or array")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be one number, got {type(value).__name__}; "
                             "use update_many for a list or array") from None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    return value


def alpha_from(confidence) -> float:
    """Allowed error rate, 1 - confidence."""
    confidence = number("confidence", confidence, "; for example 0.95, meaning at most 5 % wrong verdicts")
    if confidence >= 1.0:
        raise ValueError(f"confidence must be below 1, for example 0.95 for 95 %; got {confidence:g}")
    if confidence < 0.5:
        raise ValueError(
            f"confidence is usually 0.95, meaning at most 5 % of uses end in a wrong verdict. "
            f"Got {confidence:g}; "
            f"did you mean {1 - confidence:g}?"
        )
    return 1.0 - confidence


def pair(name: str, bounds) -> tuple[float, float]:
    try:
        lo, hi = bounds
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair (lowest, highest), got {bounds!r}") from None
    lo, hi = number(f"{name}[0]", lo), number(f"{name}[1]", hi)
    if not lo < hi:
        raise ValueError(f"{name} must be (lowest, highest) with lowest < highest, got ({lo:g}, {hi:g})")
    return lo, hi


def choice(name: str, value, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        options = ", ".join(repr(a) for a in allowed[:-1]) + f" or {allowed[-1]!r}"
        raise ValueError(f"{name} must be {options}, got {value!r}")
    return value


def whole(name: str, value, at_least: int = 1) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or int(value) != value or value < at_least:
        raise ValueError(f"{name} must be a whole number of at least {at_least}, got {value!r}")
    return int(value)


def error_rate(alpha: float) -> str:
    return f"error rate at most {100 * alpha:.3g} %"


def same_length(first, second, what: str = "new and old") -> None:
    if isinstance(first, Sized) and isinstance(second, Sized) and len(first) != len(second):
        raise ValueError(f"{what} must have the same length, got {len(first)} and {len(second)}")


class Reporting:
    """What every object offers besides `update`."""

    _estimate = "mean"      # name of the property holding the plain estimate

    def result(self) -> dict:
        """The current state as a small dict, for logs or dashboards."""
        out = {"kind": type(self).__name__, "n": self.n, "done": self.done}
        names = ("verdict", "decided_at", "direction", self._estimate, "low", "high", "e_value", "confidence")
        for name in names:
            if hasattr(self, name):
                value = getattr(self, name)
                out[name] = None if isinstance(value, float) and not math.isfinite(value) else value
        out["summary"] = self.summary()
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.summary()}>"
