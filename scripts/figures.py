"""Appendix figures, regenerated from the episode logs (main-text figures: main_figures.py).

  python3 scripts/figures.py        # -> results/figures/appA{1,2,3}_*.{png,pdf}

Colour encodes topology throughout (fully connected = blue, ring = orange; honest
source = neutral grey). Palette: the dataviz reference categorical slots 1-2,
validated all-pairs for colour-vision deficiency. Every series is also
direct-labelled, so identity never depends on colour alone.
"""

import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import analyze  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8f8d87"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
OUT = "results/figures"
CELL_RE = re.compile(r"-(bc|ring)_(cheat|honest|sweep)_(rep|norep)-s(\d+)$")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 300, "legend.frameon": False,
})


def load_runs(root):
    rows = []
    for name in sorted(os.listdir(root)):
        m = CELL_RE.search(name)
        d = os.path.join(root, name)
        if m and os.path.exists(os.path.join(d, "summary.json")):
            r = analyze(d)
            r.update(topo=m.group(1), src=m.group(2), rep=m.group(3), seed=int(m.group(4)))
            rows.append(r)
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, n=4000, seed=1):
    import random
    rng = random.Random(seed)
    if len(xs) < 2:
        return (mean(xs), mean(xs))
    ms = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)


def strip(ax, x, vals, color, jitter=0.07, hollow=False):
    import random
    rng = random.Random(x)
    xs = [x + rng.uniform(-jitter, jitter) for _ in vals]
    if hollow:
        ax.scatter(xs, vals, s=16, facecolors="white", edgecolors=color, alpha=0.8, linewidths=1, zorder=3)
    else:
        ax.scatter(xs, vals, s=16, color=color, alpha=0.55, linewidths=0, zorder=3)
    m = mean(vals)
    lo, hi = boot_ci(vals)
    ax.plot([x, x], [lo, hi], color=color, lw=2, solid_capstyle="round", zorder=4)
    ax.scatter([x], [m], s=46, color="white" if hollow else color, edgecolors=color if hollow else "white",
               linewidths=1.6 if hollow else 1.2, zorder=5)
    return m


def fig1(main, s20):
    """Adoption by condition: per-episode dots, mean and 95% bootstrap CI."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, gridspec_kw={"width_ratios": [6, 5]})
    for ax, rows, title, honest_cells in ((axes[0], main, "8 agents (10 episodes per cell)", ("bc", "ring")),
                                          (axes[1], s20, "20 agents (6 episodes per cell)", ("bc",))):
        slots = [("cheat", "norep", "Cheating,\nreport off"),
                 ("cheat", "rep", "Cheating,\nreport on"),
                 ("honest", "rep", "Honest,\nreport on")]
        for i, (src, rep, label) in enumerate(slots):
            for j, topo in enumerate(("bc", "ring")):
                vals = [r["adoption_fraction_others"] for r in rows if r["src"] == src and r["rep"] == rep and r["topo"] == topo]
                if not vals:
                    continue
                color = BLUE if topo == "bc" else ORANGE
                m = strip(ax, i + (j - 0.5) * 0.36, vals, color, hollow=(src == "honest"))
                ax.annotate("%.0f%%" % (100 * m), (i + (j - 0.5) * 0.36, m),
                            xytext=(0, 9) if src == "honest" else (7, 0), ha="center" if src == "honest" else "left",
                            textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([s[2] for s in slots], fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of non-source agents that used\nthe exploit (per episode)")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Fully connected"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="Ring (two neighbours)"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
                          label="Hollow = honest-source control")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=8)
    save(fig, "appA1_adoption_by_condition")


def fig4(main, s20):
    """Reporting agents (share of non-source agents) by source and topology, report tool on."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    for ax, rows, title, n_real in ((axes[0], main, "8 agents", 7), (axes[1], s20, "20 agents", 19)):
        slots = [("cheat", "Cheating source"), ("honest", "Honest source")]
        for i, (src, label) in enumerate(slots):
            for j, topo in enumerate(("bc", "ring")):
                vals = [r["n_reporters"] / n_real for r in rows if r["src"] == src and r["rep"] == "rep" and r["topo"] == topo]
                if not vals:
                    continue
                color = BLUE if topo == "bc" else ORANGE
                m = strip(ax, i + (j - 0.5) * 0.36, vals, color, hollow=(src == "honest"))
                ax.annotate("%.0f%%" % (100 * m), (i + (j - 0.5) * 0.36, m), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(2))
        ax.set_xticklabels([s[1] for s in slots])
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of non-source agents that filed\nany report (incl. general feedback)")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Fully connected"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="Ring"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
                          label="Hollow = honest-source control")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    save(fig, "appA2_reporting_any")


def fig5(main, think, sweep):
    """Matched-seed follow-ups: reasoning on vs off (F2), sweeping vs regular cheater (F3)."""
    base = {(r["topo"], r["rep"], r["seed"]): r for r in main if r["src"] == "cheat"}
    pooled = {}
    for name in ("think", "sweep"):
        path = "results/compare_%s.json" % name
        if os.path.exists(path):
            with open(path) as fh:
                pooled[name] = json.load(fh)["pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"wspace": 0.55})
    panels = [(axes[0], think, "adoption_fraction_others", "F2: exploit use", ("Reasoning\noff", "Reasoning\non")),
              (axes[1], think, "closed_genuine", "F2: genuine solves", ("Reasoning\noff", "Reasoning\non")),
              (axes[2], sweep, "adoption_fraction_others", "F3: exploit use", ("Regular\ncheater", "Sweeping\ncheater"))]
    for ax, rows, key, title, labels in panels:
        for r in rows:
            b = base.get((r["topo"], r["rep"], r["seed"]))
            if not b:
                continue
            color = BLUE if r["topo"] == "bc" else ORANGE
            y0, y1 = (b[key] or 0), (r[key] or 0)
            ax.plot([0, 1], [y0, y1], color=color, lw=1.2, alpha=0.6, marker="o", ms=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.25, 1.25)
        ax.grid(axis="x", visible=False)
        src = pooled.get("think" if rows is think else "sweep", {}).get(key)
        sub = ""
        if src:
            unit = " pts" if key == "adoption_fraction_others" else ""
            d = src["paired_diff"] * (100 if unit else 1)
            ptxt = "p < 0.001" if src["p"] < 0.001 else "p = %.2f" % src["p"]
            sub = "\npaired change %+.1f%s, %s" % (d, unit, ptxt)
        ax.set_title(title + sub, fontsize=8.5, loc="left")
        if key == "closed_genuine":
            ax.set_ylabel("Genuine solves per episode")
        if key == "adoption_fraction_others":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_ylim(-0.04, 1.04)
    axes[0].set_ylabel("Real agents that used the exploit")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", label="Fully connected"),
               plt.Line2D([], [], color=ORANGE, marker="o", label="Ring")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13), fontsize=8)
    save(fig, "appA3_followups")


def main():
    runs = {k: load_runs("runs/" + k) for k in ("main", "scale20", "think", "sweep")}
    fig1(runs["main"], runs["scale20"])
    fig4(runs["main"], runs["scale20"])
    fig5(runs["main"], runs["think"], runs["sweep"])
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
