#!/usr/bin/env python3
"""Build the single-panel project-card figure for gbibbo.github.io (CPU, 0 cr).

Narrative card: ONE series, the paired recovery gain R = CLAP(P+FT) - CLAP(P) of the
83%-pruned checkpoint, over the requested-duration sweep. Absolute levels, the dense
and real anchors, the chance floors and the rho_dense percentages are deliberately
NOT drawn: they belong to the paper, and they cost the card reader too much work.

Every number is READ from the frozen result artifacts, never typed by hand:
  configs/research/draft5_opsweep_result.json   (severity-2 duration sweep, n=192)
  configs/research/r2_posthoc_pooled_anchors.json  (hip-hop battery, n=127)

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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "configs", "research", "draft5_opsweep_result.json")
SRC_HH = os.path.join(ROOT, "configs", "research", "r2_posthoc_pooled_anchors.json")
OUT_DIR = os.path.join(ROOT, "docs", "figs")
OUT = os.path.join(OUT_DIR, "audioldm_recovery_operating_point.png")
OUT_SVG = os.path.join(OUT_DIR, "audioldm_recovery_operating_point.svg")
OUT_B = os.path.join(OUT_DIR, "audioldm_recovery_operating_point_domain.png")
OUT_B_SVG = os.path.join(OUT_DIR, "audioldm_recovery_operating_point_domain.svg")

# ---------------------------------------------------------------- data (frozen)
sweep = json.load(open(SRC, encoding="utf-8"))
Rd = sweep["R_by_duration"]
DURS = ["3.84", "5.12", "7.68", "10.24"]
x = [float(d) for d in DURS]
R = [Rd[d]["point"] for d in DURS]
Rlo = [Rd[d]["lo"] for d in DURS]
Rhi = [Rd[d]["hi"] for d in DURS]
n = Rd["10.24"]["n"]
FT_DURATION = 10.24  # the released recovery was fine-tuned at the native duration

hh = json.load(open(SRC_HH, encoding="utf-8"))["hiphop_127"]
hh_R = [hh["3.84"]["R_pooled"]["point"], hh["10.24"]["R_pooled"]["point"]]
hh_n = hh["10.24"]["R_pooled"]["n"]
HH_X = [3.84, 10.24]  # the hip-hop battery has only the two endpoint durations

# ------------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
BLUE = "#2a78d6"  # categorical slot 1 - the single series

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



def build(out_png, out_svg, with_hiphop=False):
    """Render the card. with_hiphop adds the flat out-of-domain music series."""
    fig, ax = plt.subplots(figsize=(7.8, 4.875))  # 16:10
    fig.subplots_adjust(left=0.125, right=0.965, top=0.975, bottom=0.15)

    # fine-tuning duration guide, recessive and behind the series
    ax.axvline(FT_DURATION, color=INK_2, lw=1.1, ls=(0, (1.5, 2.5)), alpha=0.6,
               zorder=2)
    # in the domain variant the flat hip-hop series occupies the bottom strip,
    # so the guide label moves above the confidence band instead
    guide_y = 0.312 if with_hiphop else 0.018
    guide_va = "top" if with_hiphop else "bottom"
    ax.text(FT_DURATION - 0.12, guide_y, "fine-tuning duration", ha="right",
            va=guide_va, fontsize=12.5, color=INK_2, zorder=6)

    # 95% prompt-bootstrap interval: present but visually secondary
    ax.fill_between(x, Rlo, Rhi, color=BLUE, alpha=0.18, linewidth=0, zorder=3)
    ax.plot(x, R, color=BLUE, lw=3.0, marker="s", ms=10, mfc=BLUE, mec=SURFACE,
            mew=2.2, zorder=5)

    # selective direct labels: the two endpoints only
    ax.text(x[0] + 0.14, R[0] - 0.004, f"+{R[0]:.3f}", ha="left", va="top",
            fontsize=15, color=INK, zorder=6)
    ax.text(x[-1] + 0.14, R[-1], f"+{R[-1]:.3f}", ha="left", va="center",
            fontsize=15, color=INK, zorder=6)

    # headline, inside the plot, in the empty upper-left corner
    ax.text(0.012, 0.985, "Recovery depends on the operating point",
            transform=ax.transAxes, ha="left", va="top", fontsize=18.5, color=INK)
    ax.text(0.012, 0.895,
            f"AudioLDM-M, 83% structured pruning · {n} paired AudioCaps prompts",
            transform=ax.transAxes, ha="left", va="top", fontsize=12.5, color=INK_2)

    ax.set_xlabel("Requested audio duration (s)", labelpad=7)
    ax.set_ylabel("Gain from recovery fine-tuning (CLAP)", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(DURS)
    ax.set_xlim(3.42, 11.05)
    ax.set_ylim(0.0, 0.380)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.9)

    if with_hiphop:
        ax.plot(HH_X, hh_R, color=INK_2, lw=2.2, ls=(0, (5, 2.4)), marker="o",
                ms=8, mfc=SURFACE, mec=INK_2, mew=2.0, zorder=4)
        ax.text(HH_X[0] + 0.16, hh_R[0] + 0.012,
                f"hip-hop prompts (n = {hh_n})", ha="left", va="bottom",
                fontsize=12.5, color=INK_2, zorder=6)

    fig.savefig(out_png, dpi=160)
    matplotlib.rcParams["svg.fonttype"] = "path"  # no font dependency in the browser
    fig.savefig(out_svg)
    plt.close(fig)
    return hashlib.sha256(open(out_png, "rb").read()).hexdigest()


os.makedirs(OUT_DIR, exist_ok=True)
digest = build(OUT, OUT_SVG, with_hiphop=False)
digest_b = build(OUT_B, OUT_B_SVG, with_hiphop=True)
print(f"wrote {os.path.relpath(OUT, ROOT)}  sha256={digest[:16]}")
print(f"wrote {os.path.relpath(OUT_B, ROOT)}  sha256={digest_b[:16]}  (domain variant)")
print(f"  source artifact sha256={sweep['artifact_sha256'][:16]}")
for d, pt, lo, hi in zip(DURS, R, Rlo, Rhi):
    print(f"  {d:>5} s  R={pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
print(f"  hip-hop (n={hh_n}): {hh_R[0]:+.3f} at 3.84 s, {hh_R[1]:+.3f} at 10.24 s")
