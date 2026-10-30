"""The exponential-decay-optimal (EDO) stake of Clerico, Wegel, Azangulov and
Rebeschini (2026), "Time-sensitive anytime-valid testing", arXiv:2605.06521.

The usual stake maximises long-run growth of capital. EDO instead maximises
the expected value of exp(-tau / T), tau being the rejection time: rejecting
soon is worth more than rejecting eventually. For a coin with null bias p0 and
alternative bias p1 > p0 the paper's Appendix C.2 gives the solution below.
"""

from __future__ import annotations

import math


def _exponent_gap(eta: float, p0: float, p1: float, time_scale: float) -> float:
    q = 1.0 / (1.0 - eta)
    a = q * math.log(p1) - eta * q * math.log(p0)
    b = q * math.log1p(-p1) - eta * q * math.log1p(-p0) if p1 < 1.0 else -math.inf
    m = max(a, b)
    return (1.0 - eta) * (m + math.log(math.exp(a - m) + math.exp(b - m))) - 1.0 / time_scale


def edo_stake(p0: float, p1: float, time_scale: float) -> float:
    """Stake lambda for the game capital *= 1 + lambda * (x - p0), with p1 > p0.

    Returns the largest allowed stake, 1 / p0, when `time_scale` is so short
    that only an all-in bet has a chance.
    """
    if not 0.0 < p0 < p1 <= 1.0:
        raise ValueError("need 0 < p0 < p1 <= 1")
    if not time_scale > 0:
        raise ValueError("time_scale must be positive")
    lo, hi = 0.0, 1.0 - 1e-9
    if _exponent_gap(hi, p0, p1, time_scale) <= 0.0:
        return 1.0 / p0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _exponent_gap(mid, p0, p1, time_scale) < 0.0:
            lo = mid
        else:
            hi = mid
    q = 1.0 / (1.0 - 0.5 * (lo + hi))
    up = math.log(p0) + q * (math.log(p1) - math.log(p0))
    down = (math.log1p(-p0) + q * (math.log1p(-p1) - math.log1p(-p0))) if p1 < 1.0 else -math.inf
    p_star = 1.0 / (1.0 + math.exp(down - up))
    return (p_star - p0) / (p0 * (1.0 - p0))
