"""README figures: simulate, then render light and dark cards into docs/.

    uv run python scripts/figures/make.py          # about a minute; needs Chrome
"""
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cards import card, render  # noqa: E402

import evalues  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")
NOISE = 0.05          # 5 % lognormal noise per measurement
PAIRS = 1000
RUNS = 2000

ICON_EYE = ('<div class="icon"><svg width="15" height="15" viewBox="0 0 15 15"><path d="M1.5 7.5 C3.5 3.8 11.5 3.8 13.5 7.5 '
            'C11.5 11.2 3.5 11.2 1.5 7.5 Z" fill="none" stroke="var(--icontx)" stroke-width="1.5"/>'
            '<circle cx="7.5" cy="7.5" r="1.8" fill="var(--icontx)"/></svg></div>')
ICON_UP = ('<div class="icon"><svg width="15" height="15" viewBox="0 0 15 15"><path d="M2 12 L6 7.5 L8.5 9.5 L13 3.5 M9.5 3.5 H13 V7" '
           'stroke="var(--icontx)" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>')

W, H, L, R, T, B = 1100, 380, 120, 30, 30, 50     # chart box and margins, px


def chart(series, xmax, ymax, yticks, xticks, ylabel, xlabel, hline=None, ylog=False, ylo=None):
    """A small SVG line chart: series = [(css var, xs, ys, end_dot)]."""
    def fy(y):
        if ylog:
            a, b = math.log(ylo), math.log(ymax)
            return T + (H - T - B) * (1 - (math.log(max(y, ylo)) - a) / (b - a))
        return T + (H - T - B) * (1 - y / ymax)

    def fx(x):
        return L + (W - L - R) * x / xmax

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="margin-top:34px;display:block;font-family:Inter">']
    for v, text in yticks:
        y = fy(v)
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        s.append(f'<text x="{L - 12}" y="{y + 6:.1f}" text-anchor="end" font-size="17" fill="var(--muted)">{text}</text>')
    for v, text in xticks:
        x = fx(v)
        s.append(f'<text x="{x:.1f}" y="{H - B + 30}" text-anchor="middle" font-size="17" fill="var(--muted)">{text}</text>')
    s.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="var(--axis)" stroke-width="1.2"/>')
    if hline:
        v, text = hline
        y = fy(v)
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--fg)" stroke-width="1.4" '
                 'stroke-dasharray="6 6" opacity=".55"/>')
        s.append(f'<text x="{W - R}" y="{y - 10:.1f}" text-anchor="end" font-size="17" fill="var(--muted)">{text}</text>')
    for var, xs, ys, dot in series:
        pts = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
        s.append(f'<polyline points="{pts}" fill="none" stroke="var({var})" stroke-width="3" '
                 'stroke-linejoin="round" stroke-linecap="round"/>')
        if dot:
            s.append(f'<circle cx="{fx(xs[-1]):.1f}" cy="{fy(ys[-1]):.1f}" r="6.5" fill="var({var})"/>')
    s.append(f'<text x="{(L + W - R) / 2}" y="{H - 2}" text-anchor="middle" font-size="17" fill="var(--muted)">{xlabel}</text>')
    s.append(f'<text transform="translate(16 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" font-size="17" '
             f'fill="var(--muted)">{ylabel}</text>')
    s.append("</svg>")
    return "".join(s)


def peeking():
    """Identical builds; how often does each method ever call a difference?"""
    rng = np.random.default_rng(7)
    t_first = np.full(RUNS, np.inf)
    e_first = np.full(RUNS, np.inf)
    for run in range(RUNS):
        new = rng.lognormal(0, NOISE, PAIRS)
        old = rng.lognormal(0, NOISE, PAIRS)
        d = np.log(new / old)
        n = np.arange(1, PAIRS + 1)
        mean = np.cumsum(d) / n
        var = (np.cumsum(d * d) - n * mean ** 2) / np.maximum(n - 1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = mean / np.sqrt(var / n)
        p = 2 * stats.t.sf(np.abs(tstat), np.maximum(n - 1, 1))
        hits = np.nonzero((p < 0.05) & (n >= 10))[0]
        if hits.size:
            t_first[run] = hits[0] + 1
        test = evalues.Comparison(min_change=0.02)
        for i in range(PAIRS):
            test.update(new[i], old[i])
            if test.done:
                if test.verdict in ("increase", "decrease"):
                    e_first[run] = test.decided_at
                break
    grid = np.unique(np.r_[np.arange(10, 100, 2), np.arange(100, PAIRS + 1, 10)])
    t_rate = [(t_first <= g).mean() for g in grid]
    e_rate = [(e_first <= g).mean() for g in grid]
    return grid, t_rate, e_rate


def evidence(seed=3):
    """One Comparison per true change; its e-value after every pair."""
    out = []
    for change, var in ((0.06, "--c1"), (0.03, "--c2")):
        rng = np.random.default_rng(seed)
        test = evalues.Comparison(min_change=0.02)
        xs, ys = [0], [1.0]
        for _ in range(PAIRS):
            test.update(rng.lognormal(math.log1p(change), NOISE), rng.lognormal(0, NOISE))
            xs.append(test.n)
            ys.append(test.e_value)
            if test.done:
                break
        out.append((change, var, xs, ys, test))
    return out


def main():
    grid, t_rate, e_rate = peeking()
    t_end, e_end = t_rate[-1], e_rate[-1]
    print(f"peeking: t-test {t_end:.3f}, evalues {e_end:.4f} after {PAIRS} pairs, {RUNS} runs")
    render(card(
        ICON_EYE, "Checking after every pair",
        f"Identical builds, {RUNS:,} runs, wrong verdicts within {PAIRS:,} pairs",
        f"{100 * e_end:.0f} % instead of {100 * t_end:.0f} %",
        [("--c1", f"<b>evalues</b> Comparison: {round(e_end * RUNS)} of {RUNS:,} runs"),
         ("--c2", f"<b>t-test</b> checked after every pair: {100 * t_end:.0f} %")],
        chart([("--c2", grid, t_rate, False), ("--c1", grid, e_rate, False)], PAIRS, 0.5,
              [(0, "0 %"), (0.1, "10 %"), (0.2, "20 %"), (0.3, "30 %"), (0.4, "40 %"), (0.5, "50 %")],
              [(0, "0"), (250, "250"), (500, "500"), (750, "750"), (1000, "1,000")],
              "wrong verdicts so far", "pairs measured", hline=(0.05, "promised: at most 5 %")),
        "A t-test promises 5 % only if you look once. Look after every pair and stop at the first "
        "significant result, and almost half the runs on identical builds end in a false alarm."), OUT, "peeking")

    traces = evidence()
    (c1, v1, x1, y1, t1), (c2, v2, x2, y2, t2) = traces
    xmax = max(x1[-1], x2[-1])
    print(f"evidence: +{c1:.0%} {t1.verdict} at {t1.decided_at}; +{c2:.0%} {t2.verdict} at {t2.decided_at}")

    def leg(c, t):
        what = f"{t.verdict} at {t.decided_at:,}" if t.decided_at else f"undecided at {t.n:,}"
        return f"+{100 * c:.0f} % change: <b>{what}</b>"

    render(card(
        ICON_UP, "Evidence after every pair",
        "Comparison(min_change=0.02), 5 % noise per measurement",
        f"Decided after {t1.decided_at} pairs",
        [(v1, leg(c1, t1)), (v2, leg(c2, t2))],
        chart([(v2, x2, y2, t2.done), (v1, x1, y1, t1.done)], xmax, 40, [(1, "1"), (10, "10")],
              [(0, "0"), (round(xmax / 2, -1), f"{round(xmax / 2, -1):,.0f}"), (xmax, f"{xmax:,}")],
              "e-value (log scale)", "pairs measured", hline=(20, "decision at 1 / α = 20"),
              ylog=True, ylo=0.5),
        "A clear change is decided early. A change close to the 2 % threshold needs more pairs; "
        "the test stops by itself either way."), OUT, "evidence")


if __name__ == "__main__":
    main()
