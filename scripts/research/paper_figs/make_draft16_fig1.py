#!/usr/bin/env python3
"""Draft-16 Figure 1: the two panels specified in TeX comments in icassp/sections/draft16_2.tex.

CPU, 0 GPU, 0 credits. Every plotted value is read from a committed frozen result JSON; nothing is
synthesised, interpolated or smoothed. The panels are written as two separate vector PDFs because the
manuscript reserves the footprint with two side-by-side `minipage`s, so each file drops straight into
its own box and the `figure*` footprint (0.49\\textwidth x 4.80 cm per panel) is preserved exactly.

  figs/fig1a_duration.pdf      -- panel (a), operating-point response of the released recovery
  figs/fig1b_intervention.pdf  -- panel (b), symmetric training-duration intervention

Sources (all committed, all read-only here):
  configs/research/draft5_opsweep_result.json  -- R by duration, severity 2 released recovery, n=192
  configs/research/r2_E1c_result.json          -- matched n=96 subset at 10.24 s and the 15.36 s cell
  configs/research/r2_EXT2x2_result.json       -- symmetric 20k short/long fine-tunes, pruned arm
  configs/research/xsev_result.json            -- cross-check of the two primary endpoints

Guards (assert, not comment): the plotted points are re-derived from the frozen level means and the
frozen contrasts, and the two endpoints of panel (a) are cross-checked against the independent
xsev_result.json verdict. Any drift between artifacts aborts the build instead of drawing.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_draft16_fig1.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "icassp", "figs")
os.makedirs(OUT, exist_ok=True)


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


SWEEP = load("configs/research/draft5_opsweep_result.json")
E1C = load("configs/research/r2_E1c_result.json")
EXT = load("configs/research/r2_EXT2x2_result.json")
XSEV = load("configs/research/xsev_result.json")


def ci(c):
    return c["point"], c["lo"], c["hi"]


# ------------------------------------------------------------------ geometry (must match the .tex)
# The manuscript reserves `\fbox{\rule{0pt}{4.80cm}\rule{0.965\linewidth}{0pt}}` inside a
# `minipage[t]{0.49\textwidth}`; spconf sets \textwidth = 178 mm.
TEXTWIDTH_IN = 178.0 / 25.4
PANEL_W = 0.49 * 0.965 * TEXTWIDTH_IN   # 3.314 in
PANEL_H = 4.80 / 2.54                   # 1.890 in

# ------------------------------------------------------------------ house style (Draft 2-13 figures)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.0,
    "axes.linewidth": 0.7,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.4,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_POST = "#1f3b73"   # dark blue    -> released recovery / short-trained arm
C_PRUN = "#c0561f"   # burnt orange -> long-trained arm
C_MATCH = "#4d4d4d"  # dark grey    -> matched n=96 subset
GRID = "#dddddd"
ZERO = "#9a9a9a"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.4, width=0.6, pad=1.6)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ============================================================ panel (a): released duration response
def panel_a():
    durs = [3.84, 5.12, 7.68, 10.24]
    pts = [ci(SWEEP["R_by_duration"][str(d)]) for d in durs]
    assert all(SWEEP["R_by_duration"][str(d)]["n"] == 192 for d in durs)
    # cross-artifact guard: the two primary endpoints must equal the frozen xsev verdict
    assert abs(pts[0][0] - XSEV["PRIMARY_A"]["R_short"]["point"]) < 1e-12
    assert abs(pts[-1][0] - XSEV["PRIMARY_A"]["R_native"]["point"]) < 1e-12
    # guard: R is the difference of the frozen level means
    lv = SWEEP["levels"]
    for d, (p, _, _) in zip(durs, pts):
        assert abs((lv[f"recovered2@{d}"] - lv[f"pruned2_A@{d}"]) - p) < 1e-9, d

    m1024 = ci(E1C["cells"]["10.24_first96"]["R"])
    m1536 = ci(E1C["cells"]["15.36"]["R"])
    d4 = ci(E1C["D4"])
    assert E1C["cells"]["10.24_first96"]["R"]["n"] == 96 and E1C["cells"]["15.36"]["R"]["n"] == 96
    assert abs((m1536[0] - m1024[0]) - d4[0]) < 1e-9
    assert abs((E1C["cells"]["15.36"]["PFT"] - E1C["cells"]["15.36"]["P"]) - m1536[0]) < 1e-9

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(left=0.148, right=0.995, top=0.865, bottom=0.215)
    style_axes(ax)

    # native-duration guide (a cue, not a maximum)
    ax.axvline(10.24, color=GRID, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.annotate("native", (10.24, 0.012), xytext=(-2.2, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.0, color="#8a8a8a", rotation=90)

    # main n=192 series
    ax.errorbar(durs, [p for p, _, _ in pts],
                yerr=[[p - lo for p, lo, _ in pts], [hi - p for p, _, hi in pts]],
                color=C_POST, lw=1.1, marker="o", ms=3.8, mec="white", mew=0.5,
                elinewidth=0.8, capsize=1.9, capthick=0.8, zorder=4, clip_on=False)

    # matched n=96 extension: the 15.36 s point is paired ONLY with its own 10.24 s subset
    mx, my = [10.24, 15.36], [m1024[0], m1536[0]]
    ax.plot(mx, my, color=C_MATCH, lw=0.9, ls=(0, (3.5, 2.0)), zorder=3)
    ax.errorbar(mx, my,
                yerr=[[m1024[0] - m1024[1], m1536[0] - m1536[1]],
                      [m1024[2] - m1024[0], m1536[2] - m1536[0]]],
                fmt="o", ms=6.2, mfc="none", mec=C_MATCH, mew=0.9, ecolor=C_MATCH,
                elinewidth=0.8, capsize=1.9, capthick=0.8, zorder=5, clip_on=False)

    ax.annotate(r"$R(15.36)-R(10.24)$" "\n"
                r"$= +0.021\ [-0.023,\,+0.067]$" "\n"
                "no clear increase",
                (15.36, m1536[0]), xytext=(-1.0, -13.0), textcoords="offset points",
                ha="right", va="top", fontsize=6.1, color=C_MATCH, linespacing=1.25)

    ax.set_xlim(3.3, 16.1)
    ax.set_ylim(0.0, 0.395)
    ax.set_xticks([3.84, 5.12, 7.68, 10.24, 15.36])
    ax.set_xticklabels(["3.84", "5.12", "7.68", "10.24", "15.36"])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("requested duration (s)", labelpad=1.0)
    ax.set_ylabel(r"recovery gain $R$", labelpad=1.5)
    ax.set_title(r"(a) released recovery, severity 2", pad=2.5)

    ax.legend(handles=[
        Line2D([], [], color=C_POST, lw=1.1, marker="o", ms=3.8, mec="white", mew=0.5,
               label=r"$10^{6}$-step recovery ($n{=}192$)"),
        Line2D([], [], color=C_MATCH, lw=0.9, ls=(0, (3.5, 2.0)), marker="o", ms=5.0,
               mfc="none", mec=C_MATCH, mew=0.9, label=r"matched subset ($n{=}96$)"),
    ], loc="upper left", bbox_to_anchor=(-0.018, 1.035), handlelength=2.1, handletextpad=0.5,
        borderpad=0.0, labelspacing=0.22)

    fig.savefig(os.path.join(OUT, "fig1a_duration.pdf"))
    fig.savefig(os.path.join(OUT, "fig1a_duration.png"), dpi=400)
    plt.close(fig)


# ====================================================== panel (b): symmetric 20k duration intervention
def panel_b():
    P = EXT["pruned"]
    sf = [ci(P["R_sf"]["3.84"]), ci(P["R_sf"]["10.24"])]
    lf = [ci(P["R_lf"]["3.84"]), ci(P["R_lf"]["10.24"])]
    j_sf, j_lf, dj = ci(P["J_sf"]), ci(P["J_lf"]), ci(P["dJ"])
    lv = P["levels"]
    # guards: both arms are measured against the SAME severity-2 P baseline, and the annotated
    # slopes are exactly the long-minus-short differences of the plotted points.
    assert abs((lv["shortft_3.84"] - lv["P_3.84"]) - sf[0][0]) < 1e-9
    assert abs((lv["shortft_10.24"] - lv["P_10.24"]) - sf[1][0]) < 1e-9
    assert abs((lv["longft_3.84"] - lv["P_3.84"]) - lf[0][0]) < 1e-9
    assert abs((lv["longft_10.24"] - lv["P_10.24"]) - lf[1][0]) < 1e-9
    assert abs(lv["P_3.84"] - SWEEP["levels"]["pruned2_A@3.84"]) < 1e-12
    assert abs(lv["P_10.24"] - SWEEP["levels"]["pruned2_A@10.24"]) < 1e-12
    assert abs((sf[1][0] - sf[0][0]) - j_sf[0]) < 1e-9
    assert abs((lf[1][0] - lf[0][0]) - j_lf[0]) < 1e-9
    assert abs((j_lf[0] - j_sf[0]) - dj[0]) < 1e-9
    assert EXT["pruned"]["reading_dJ"] == "REVERSED"

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(left=0.148, right=0.995, top=0.865, bottom=0.215)
    style_axes(ax)
    ax.axhline(0.0, color=ZERO, lw=0.7, zorder=1)

    x = [0.0, 1.0]
    dx = 0.026   # symmetric dodge so the two 95% CI whiskers stay readable
    for series, colour, marker, ls, dodge in ((sf, C_POST, "o", "-", -dx), (lf, C_PRUN, "s", "--", dx)):
        xs = [xi + dodge for xi in x]
        ax.errorbar(xs, [p for p, _, _ in series],
                    yerr=[[p - lo for p, lo, _ in series], [hi - p for p, _, hi in series]],
                    color=colour, lw=1.1, ls=ls, marker=marker, ms=3.8, mec="white", mew=0.5,
                    elinewidth=0.8, capsize=1.9, capthick=0.8, zorder=4, clip_on=False)

    ax.annotate(r"$J_{\mathrm{short}}=+0.065$" "\n" r"$[+0.044,+0.087]$",
                (1.0 - dx, sf[1][0]), xytext=(-3.0, 1.0), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.1, color=C_POST, linespacing=1.2)
    ax.annotate(r"$J_{\mathrm{long}}=+0.031$" "\n" r"$[+0.010,+0.052]$",
                (1.0 + dx, lf[1][1]), xytext=(-1.0, -11.5), textcoords="offset points",
                ha="right", va="top", fontsize=6.1, color=C_PRUN, linespacing=1.2)
    ax.annotate(r"$\Delta J = J_{\mathrm{long}}-J_{\mathrm{short}} = -0.035\ [-0.064,-0.004]$",
                (0.5, 0.995), xycoords=("data", "axes fraction"), xytext=(0, 0.0),
                textcoords="offset points", ha="center", va="top", fontsize=6.1, color="#333333")

    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.042, 0.152)
    ax.set_xticks(x)
    ax.set_xticklabels(["3.84 s", "10.24 s"])
    ax.set_yticks([0.0, 0.04, 0.08, 0.12])
    ax.set_xlabel("evaluation duration", labelpad=1.0)
    ax.set_ylabel(r"20k-step gain over P", labelpad=1.5)
    ax.set_title(r"(b) symmetric training-duration test", pad=2.5)

    ax.legend(handles=[
        Line2D([], [], color=C_POST, lw=1.1, ls="-", marker="o", ms=3.8, mec="white", mew=0.5,
               label=r"trained at $3.84$ s"),
        Line2D([], [], color=C_PRUN, lw=1.1, ls="--", marker="s", ms=3.8, mec="white", mew=0.5,
               label=r"trained at $10.24$ s"),
    ], loc="upper left", bbox_to_anchor=(-0.018, 0.80), handlelength=2.1, handletextpad=0.5,
        borderpad=0.0, labelspacing=0.22)

    fig.savefig(os.path.join(OUT, "fig1b_intervention.pdf"))
    fig.savefig(os.path.join(OUT, "fig1b_intervention.png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    panel_a()
    panel_b()
    for f in ("fig1a_duration.pdf", "fig1b_intervention.pdf"):
        p = os.path.join(OUT, f)
        print(f"wrote {os.path.relpath(p, ROOT)}  ({os.path.getsize(p)} bytes)")
    print(f"panel box: {PANEL_W:.4f} in x {PANEL_H:.4f} in  "
          f"(= 0.49*0.965*textwidth x 4.80 cm, the reserved footprint)")
