#!/usr/bin/env python3
"""Build the single-panel project-card figure for gbibbo.github.io (CPU, 0 cr).

Every number is READ from the frozen result artifacts, never typed by hand:
  configs/research/draft5_opsweep_result.json  (severity-2 duration sweep, n=192)

The card image is displayed by the site at aspect-ratio 16/10, object-fit: contain,
in a box of clamp(300px, 38%, 370px) width, so the figure is authored at 16:10 with
type sized to stay legible at ~370 px wide.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_website_card_fig.py
"""
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "configs", "research", "draft5_opsweep_result.json")
OUT_DIR = os.path.join(ROOT, "docs", "figs")
OUT = os.path.join(OUT_DIR, "audioldm_recovery_operating_point.png")
OUT_SVG = os.path.join(OUT_DIR, "audioldm_recovery_operating_point.svg")

# ---------------------------------------------------------------- data (frozen)
sweep = json.load(open(SRC, encoding="utf-8"))
by_dur = sweep["secondary"]["by_duration"]
DURS = ["3.84", "5.12", "7.68", "10.24"]
x = [float(d) for d in DURS]
dense = [by_dur[d]["levels"]["dense"] for d in DURS]
pft = [by_dur[d]["levels"]["PFT"] for d in DURS]
prn = [by_dur[d]["levels"]["P"] for d in DURS]
R = [by_dur[d]["R"]["point"] for d in DURS]
rho = [by_dur[d]["rho_dense"]["point"] for d in DURS]
n = by_dur["10.24"]["R"]["n"]

# ------------------------------------------------------------------- palette
# Two categorical hues (validated: all checks PASS, all-pairs, light surface)
# plus a neutral for the dense reference series, which is also dashed + labelled,
# so identity is never carried by colour alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
BLUE = "#2a78d6"   # slot 1 - pruned + recovery
ORANGE = "#eb6834"  # slot 2 - pruned, no recovery
NEUTRAL = "#8a8880"  # reference series (dense)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

fig, ax = plt.subplots(figsize=(7.8, 4.875))  # 16:10
fig.subplots_adjust(left=0.125, right=0.985, top=0.975, bottom=0.15)

# recovery gain: the band between the pruned model and the same model after recovery
ax.fill_between(x, prn, pft, color=BLUE, alpha=0.14, linewidth=0, zorder=1)

ax.plot(x, dense, color=NEUTRAL, lw=2.3, ls=(0, (5, 2.6)), marker="o", ms=9,
        mfc=SURFACE, mew=2.2, mec=NEUTRAL, zorder=3)
ax.plot(x, pft, color=BLUE, lw=2.5, marker="s", ms=9, mfc=BLUE, mec=SURFACE,
        mew=2.2, zorder=4)
ax.plot(x, prn, color=ORANGE, lw=2.5, marker="^", ms=10, mfc=ORANGE, mec=SURFACE,
        mew=2.2, zorder=4)

# selective direct labels: the two endpoints of the sweep only
for i, ha, dx in ((0, "left", 0.16), (3, "right", -0.16)):
    ax.annotate("", xy=(x[i], pft[i]), xytext=(x[i], prn[i]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.7,
                                shrinkA=2.5, shrinkB=2.5), zorder=5)
    ax.text(x[i] + dx, (pft[i] + prn[i]) / 2,
            f"+{R[i]:.3f}\n{100 * rho[i]:.0f}% of dense gap",
            ha=ha, va="center", fontsize=14.5, color=INK, linespacing=1.32,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.24", fc=SURFACE, ec="none", alpha=0.82))

handles = [
    Line2D([], [], color=NEUTRAL, lw=2.3, ls=(0, (5, 2.6)), marker="o", ms=9,
           mfc=SURFACE, mew=2.2, mec=NEUTRAL, label="Dense AudioLDM-M (416 M)"),
    Line2D([], [], color=BLUE, lw=2.5, ls="-", marker="s", ms=9, mfc=BLUE,
           mec=SURFACE, mew=2.2, label="Pruned + recovery (71 M)"),
    Line2D([], [], color=ORANGE, lw=2.5, ls="-", marker="^", ms=10, mfc=ORANGE,
           mec=SURFACE, mew=2.2, label="Pruned, no recovery (71 M)"),
    Patch(facecolor=BLUE, alpha=0.14, label="Gain added by recovery"),
]
leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.005),
                fontsize=13.5, frameon=True, framealpha=0.94, edgecolor=GRID,
                facecolor=SURFACE, borderpad=0.55, labelspacing=0.45,
                handlelength=2.3, handletextpad=0.7)
leg.get_frame().set_linewidth(0.8)
for t in leg.get_texts():
    t.set_color(INK)

ax.set_xlabel("Requested audio duration (s)", labelpad=7)
ax.set_ylabel("CLAP score (text–audio alignment)", labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(DURS)
ax.set_xlim(3.45, 10.72)
ax.set_ylim(0.0, 0.475)
ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(0.9)

ax.text(0.995, 0.975,
        f"AudioLDM-M, 83% structured pruning\n{n} paired prompts, same noise",
        transform=ax.transAxes, ha="right", va="top", fontsize=12, color=INK_2,
        linespacing=1.35)

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT, dpi=160)
matplotlib.rcParams["svg.fonttype"] = "path"  # no font dependency in the browser
fig.savefig(OUT_SVG)
plt.close(fig)

digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
print(f"wrote {os.path.relpath(OUT, ROOT)}  sha256={digest[:16]}")
print(f"wrote {os.path.relpath(OUT_SVG, ROOT)}  "
      f"sha256={hashlib.sha256(open(OUT_SVG, 'rb').read()).hexdigest()[:16]}")
print(f"  source artifact sha256={sweep['artifact_sha256'][:16]}")
for d, gp, gr in zip(DURS, R, rho):
    print(f"  {d:>5} s  R={gp:+.3f}  rho_dense={100 * gr:.0f}%")
