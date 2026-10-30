"""Shared look of the README figures: a rounded card with an icon and title, a
grey panel, a label, a big number, a dot legend, the chart and a one-sentence
takeaway. Inter, indigo and sky data colours, light and dark.

render() writes the HTML next to fonts/ (Inter, SIL OFL, see fonts/OFL.txt) and screenshots it with
headless Chrome (set CHROME to override the path) into <name>-light.png and
<name>-dark.png.
"""
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
@font-face{font-family:Inter;src:url(fonts/inter-latin-400-normal.woff2);font-weight:400}
@font-face{font-family:Inter;src:url(fonts/inter-latin-500-normal.woff2);font-weight:500}
@font-face{font-family:Inter;src:url(fonts/inter-latin-600-normal.woff2);font-weight:600}
:root{--card:#fff;--edge:rgba(0,0,0,.07);--panel:#f6f6f8;--tile:#fff;--fg:#111114;--muted:#6e6e76;--faint:#b9b9c0;
 --c1:#322eb1;--c2:#7ab9f1;--c3:#8a80ea;--c4:#9a9aa3;--up:#2fb35a;--icon:#1d1d20;--icontx:#fff;--axis:#8d8d95;--grid:rgba(0,0,0,.07)}
.dark{--card:#17171a;--edge:rgba(255,255,255,.08);--panel:#202024;--tile:#2a2a2f;--fg:#ededef;--muted:#9c9ca5;--faint:#55555d;
 --c1:#9d94f1;--c2:#4f8fcf;--c3:#c7c1fa;--c4:#77777f;--up:#3fcf6e;--icon:#ededef;--icontx:#17171a;--axis:#77777f;--grid:rgba(255,255,255,.08)}
*{margin:0;box-sizing:border-box}
html,body{background:transparent;width:1200px}
body{font-family:Inter;color:var(--fg);-webkit-font-smoothing:antialiased;padding:6px}
.card{background:var(--card);border:1px solid var(--edge);border-radius:26px;padding:6px}
.head{display:flex;align-items:center;gap:14px;padding:22px 30px 24px;font-size:25px;font-weight:500;letter-spacing:-.01em}
.icon{width:24px;height:24px;border-radius:6px;background:var(--icon);display:grid;place-items:center}
.panel{background:var(--panel);border-radius:21px;padding:34px 38px 34px}
.lab{font-size:20px;color:var(--muted)}
.big{font-feature-settings:"tnum";font-size:34px;font-weight:500;margin-top:10px;letter-spacing:-.01em}
.leg{display:flex;gap:34px;margin-top:22px;font-size:20px;color:var(--muted)}
.leg b{font-weight:500;color:var(--fg)}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:10px;vertical-align:1px}
.take{font-size:20px;color:var(--muted);margin-top:30px;line-height:1.45}
"""

ICON_BOLT = ('<div class="icon"><svg width="15" height="15" viewBox="0 0 15 15"><path d="M8.6 1.5 L3.6 8.4 H7.3 L6.4 13.5 '
             'L11.4 6.6 H7.7 Z" fill="var(--icontx)" stroke="var(--icontx)" stroke-width="1" stroke-linejoin="round"/></svg></div>')
ICON_RULER = ('<div class="icon"><svg width="15" height="15" viewBox="0 0 15 15"><path d="M2 11.5 H13 M3.5 11.5 V8.5 M6 11.5 V9.8 '
              'M8.5 11.5 V7.5 M11 11.5 V9.8" stroke="var(--icontx)" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg></div>')


def card(icon, title, label, big, legend, chart, take):
    leg = "".join(f'<span><i class="dot" style="background:var({c})"></i>{t}</span>' for c, t in legend)
    return (f'<div class="card"><div class="head">{icon}{title}</div><div class="panel">'
            f'<div class="lab">{label}</div><div class="big">{big}</div><div class="leg">{leg}</div>'
            f'{chart}<div class="take">{take}</div></div></div>')


def tiles(stats):
    """Stat tiles in a row: (number, what, context)."""
    return '<div style="display:flex;gap:16px;align-items:stretch">' + "".join(
        f'<div style="flex:1;background:var(--card);border:1px solid var(--edge);border-radius:22px;padding:6px">'
        f'<div style="background:var(--panel);border-radius:17px;height:100%;padding:26px 28px 28px">'
        f'<div style="font-size:44px;font-weight:500;letter-spacing:-.02em">{n}</div>'
        f'<div style="font-size:20px;font-weight:500;margin-top:8px">{a}</div>'
        f'<div style="font-size:17px;color:var(--muted);margin-top:6px;line-height:1.4">{b}</div></div></div>'
        for n, a, b in stats) + "</div>"


def page(body):
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>'
            f'<script>if(location.hash==="#dark")document.documentElement.className="dark"</script>{body}</body></html>')


def chrome():
    for c in (os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")):
        if c and os.path.exists(c):
            return c
    raise SystemExit("no Chrome found; set CHROME")


def render(body, out_dir, name):
    """Screenshot the card at 2x into <name>-light.png and <name>-dark.png."""
    html = os.path.join(HERE, f".{name}.html")
    open(html, "w").write(page(body))
    os.makedirs(out_dir, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as prof:
            h = height(prof, html)
            for theme in ("light", "dark"):
                png = os.path.join(out_dir, f"{name}-{theme}.png")
                subprocess.run([chrome(), "--headless=new", f"--user-data-dir={prof}", "--disable-gpu",
                                "--hide-scrollbars", "--force-device-scale-factor=2", "--default-background-color=00000000",
                                f"--window-size=1200,{h + 16}", "--virtual-time-budget=2000",
                                f"--screenshot={png}", f"file://{html}#{theme}"], check=True, capture_output=True)
    finally:
        os.remove(html)


def height(prof, html):
    js = html + ".h.html"
    open(js, "w").write(open(html).read().replace(
        "</body>", '<script>document.title=Math.ceil(document.body.getBoundingClientRect().height)</script></body>'))
    try:
        out = subprocess.run([chrome(), "--headless=new", f"--user-data-dir={prof}", "--disable-gpu",
                              "--window-size=1200,4000", "--virtual-time-budget=2000", "--dump-dom", f"file://{js}"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split("<title>")[1].split("</title>")[0])
    finally:
        os.remove(js)
