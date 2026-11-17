"""Draws the three figures used in the report as plain SVG (no plotting library needed).

    python docs/make_figures.py

pipeline.svg      block diagram of the method
segmentation.svg  the public 5-minute window: evidence signals, our labels, ground truth, baseline
match.svg         our labels over the whole match, both halves
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import pipeline as pp  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
COL = {"H": "#1b4f9c", "A": "#c8342b", "D": "#a7adb5", "X": "#ffffff"}
INK, MUTED, GRID = "#222831", "#6b7280", "#d9dce1"
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


def svg(w, h, body):
    return ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d' width='100%%' %s "
            "font-size='11' fill='%s'>%s</svg>" % (w, h, FONT, INK, body))


def text(x, y, s, size=11, anchor="start", color=INK, weight="normal"):
    return "<text x='%.1f' y='%.1f' font-size='%s' text-anchor='%s' fill='%s' font-weight='%s'>%s</text>" % (
        x, y, size, anchor, color, weight, s)


def segments(y):
    out, s = [], 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[s]:
            out.append((y[s], s, i))
            s = i
    return out


def strip(labels, x0, y0, w, h, f0, f1):
    n = f1 - f0
    body = ""
    for lab, a, b in segments(labels[f0:f1]):
        if lab == "X":
            continue
        body += "<rect x='%.2f' y='%.1f' width='%.2f' height='%d' fill='%s'/>" % (
            x0 + w * a / n, y0, max(w * (b - a) / n, 0.4), h, COL[lab])
    return "<rect x='%.1f' y='%.1f' width='%.1f' height='%d' fill='none' stroke='%s'/>" % (x0, y0, w, h, GRID) + body


def legend(x, y):
    b = ""
    for i, (k, name) in enumerate([("H", "home possession"), ("A", "away possession"), ("D", "dead ball")]):
        b += "<rect x='%d' y='%d' width='12' height='12' fill='%s'/>" % (x + i * 140, y, COL[k])
        b += text(x + i * 140 + 17, y + 10, name)
    return b


def line(vals, x0, y0, w, h, lo, hi, color):
    v = np.clip(vals, lo, hi)
    xs = x0 + w * np.arange(len(v)) / (len(v) - 1)
    ys = y0 + h * (1 - (v - lo) / (hi - lo))
    pts = " ".join("%.1f,%.1f" % p for p in zip(xs, ys))
    return "<polyline points='%s' fill='none' stroke='%s' stroke-width='1.3'/>" % (pts, color)


def box(x, y, w, h, title, sub, fill="#f3f5f8", stroke="#9aa3b0"):
    b = "<rect x='%d' y='%d' width='%d' height='%d' rx='6' fill='%s' stroke='%s'/>" % (x, y, w, h, fill, stroke)
    b += text(x + w / 2, y + 22, title, 12, "middle", INK, "bold")
    for i, t in enumerate(sub):
        b += text(x + w / 2, y + 40 + i * 14, t, 10.5, "middle", MUTED)
    return b


def arrow(x1, y1, x2, y2):
    return ("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#566070' stroke-width='1.4' marker-end='url(#a)'/>"
            % (x1, y1, x2, y2))


def fig_pipeline():
    defs = ("<defs><marker id='a' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'>"
            "<path d='M0,0 L8,4 L0,8 z' fill='#566070'/></marker></defs>")
    b = defs
    b += box(20, 20, 170, 62, "Tracking (ATD)", ["46 track slots + ball", "per frame, tactical video"])
    b += box(20, 100, 170, 50, "ATD metadata", ["team of each slot"])
    b += box(250, 8, 230, 62, "Team evidence from the ball",
             ["nearest home vs nearest away player,", "down-weighted if the ball is far"])
    b += box(250, 84, 230, 62, "Player-only cue",
             ["used only where the ball is lost;", "logistic regression on team shape"])
    b += box(250, 160, 230, 62, "Dead-ball evidence",
             ["players walking, ball out / at a line,", "long tracker loss, ball moving fast"])
    b += box(540, 70, 190, 90, "3-state Viterbi",
             ["H / A / D per half", "switching cost = minimum", "length of a possession"], "#e8eef8", "#6f8fc4")
    b += box(780, 70, 120, 90, "Labels", ["H  A  D", "X outside the", "graded halves"], "#eef6ee", "#7fae80")
    b += arrow(190, 50, 250, 39) + arrow(190, 60, 250, 112) + arrow(190, 70, 250, 185)
    b += arrow(190, 125, 250, 125)
    b += arrow(480, 39, 540, 100) + arrow(480, 115, 540, 115) + arrow(480, 191, 540, 130)
    b += arrow(730, 115, 780, 115)
    b += text(10, 242, "The player-only cue is trained on pseudo-labels from the ball rule on this same match. No human labels anywhere.",
              10, "start", MUTED)
    return svg(910, 252, b)


def fig_segmentation(debug, ours, gt, base):
    f0, f1 = 11185, 18685
    W, X0 = 780, 100
    b = text(X0, 16, "Public 5-minute window (frames 11185-18684)", 12, "start", INK, "bold")
    e = pd.Series(debug["team_evidence"][f0:f1]).rolling(25, center=True, min_periods=1).mean().values
    y1 = 34
    b += text(X0 - 8, y1 + 27, "team", 10, "end", MUTED) + text(X0 - 8, y1 + 40, "evidence", 10, "end", MUTED)
    b += "<rect x='%d' y='%d' width='%d' height='60' fill='none' stroke='%s'/>" % (X0, y1, W, GRID)
    b += "<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='%s'/>" % (X0, y1 + 30, X0 + W, y1 + 30, GRID)
    b += line(e[::5], X0, y1, W, 60, -2.5, 2.5, "#1b4f9c")
    b += text(X0 + W + 6, y1 + 12, "favours H", 10, "start", COL["H"]) + text(X0 + W + 6, y1 + 54, "favours A", 10, "start", COL["A"])
    d = debug["dead_evidence"][f0:f1]
    y2 = 112
    b += text(X0 - 8, y2 + 27, "dead-ball", 10, "end", MUTED) + text(X0 - 8, y2 + 40, "evidence", 10, "end", MUTED)
    b += "<rect x='%d' y='%d' width='%d' height='60' fill='none' stroke='%s'/>" % (X0, y2, W, GRID)
    b += "<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='%s'/>" % (X0, y2 + 30, X0 + W, y2 + 30, GRID)
    b += line(d[::5], X0, y2, W, 60, -3, 3, "#6b7280")
    b += text(X0 + W + 6, y2 + 12, "dead", 10, "start", MUTED) + text(X0 + W + 6, y2 + 54, "alive", 10, "start", MUTED)
    y3 = 192
    for i, (name, lab) in enumerate([("ground truth", gt), ("HackFCB", ours), ("baseline_smooth50", base)]):
        yy = y3 + i * 30
        b += text(X0 - 8, yy + 15, name, 10, "end", MUTED)
        b += strip(lab, X0, yy, W, 22, f0, f1)
    for s in range(0, 301, 60):
        x = X0 + W * s / 300.0
        b += "<line x1='%.1f' y1='282' x2='%.1f' y2='287' stroke='%s'/>" % (x, x, MUTED)
        b += text(x, 300, "%d s" % s, 10, "middle", MUTED)
    b += legend(X0, 316)
    return svg(930, 340, b)


def fig_match(ours):
    W, X0 = 780, 100
    b = text(X0, 16, "Our labels over the whole match (one bar per half)", 12, "start", INK, "bold")
    halves = [("1st half", 11185, 90652), ("2nd half", 110808, 190198)]
    for i, (name, a, c) in enumerate(halves):
        yy = 32 + i * 46
        b += text(X0 - 8, yy + 17, name, 10, "end", MUTED)
        b += strip(ours, X0, yy, W, 26, a, c)
        for m in range(0, 91, 15):
            x = X0 + W * (m * 60 * 25) / (c - a)
            if x <= X0 + W + 1:
                b += "<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='%s'/>" % (x, yy + 26, x, yy + 31, MUTED)
                b += text(x, yy + 42, "%d'" % m, 10, "middle", MUTED)
    b += legend(X0, 134)
    return svg(930, 156, b)


def main():
    os.makedirs(OUT, exist_ok=True)
    dbg = {}
    ours = pp.run(debug=dbg)
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
    base = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"))["label"].values
    for name, content in [("pipeline", fig_pipeline()), ("segmentation", fig_segmentation(dbg, ours, gt, base)),
                          ("match", fig_match(ours))]:
        with open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(content)
        print("wrote", name + ".svg")


if __name__ == "__main__":
    main()
