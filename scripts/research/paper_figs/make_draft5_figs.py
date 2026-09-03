#!/usr/bin/env python3
"""Draft-5 ICASSP figures from DURABLE artifacts only (CPU, 0 cr). No fabricated values.

Reuses the Draft-2/3 modules (frozen sources, style, FineLAP provenance guard) and rebuilds
  fig1_interaction.pdf  -- system x duration interaction, both severities, CI whiskers on the paired gain,
                           matched dense control in (a), PLUS two zero-cost anchors from
                           configs/research/draft5_floor_ceiling_result.json: the shuffled-caption CHANCE
                           FLOOR of each (system, duration) cell (short grey ticks) and the REAL-AUDIO
                           CEILING of the same prompts at both durations (grey triangles).
  fig2_where.pdf        -- unchanged Draft-3 two-panel (FineLAP profile; generation length vs scoring window).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_draft5_figs.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_manuscript_figs as M  # noqa: E402
import make_draft3_figs as D3     # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = M.OUT
FC = M.load("configs/research/draft5_floor_ceiling_result.json")
cells = FC["cells"]
C_REAL = "#4d4d4d"
C_FLOOR = "#9a9a9a"


def cell(name):
    c = cells[name]
    return c["matched_mean"], c["floor"]["point"]


def figure1():
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.05), sharex=True, sharey=True)
    x = [0.0, 1.0]
    xlabels = ["3.84 s\n(short)", "10.24 s\n(native)"]
    # (system, duration) -> cell name in the Draft-5 result
    names = {
        "sev1": {"pruned": ("pruned_short_sev1__armd80", "dense10s__pruned_sev1"),
                 "post": ("postft_short_sev1__armd80", "dense10s__recovered_sev1"),
                 "dense": ("dense_short_sev1__armd80", "dense10s__dense"),
                 "real": ("real_crop__sev1_80", "real_full__sev1_80")},
        "sev2": {"pruned": ("pruned2_A__ac_short", "pruned2_A__ac_native"),
                 "post": ("recovered2__ac_short", "recovered2__ac_native"),
                 "real": ("real_crop__sev2_192", "real_full__sev2_192")},
    }
    panels = [(axes[0], M.s1, r"(a) severity 1  $(1,2,3,1)$", "sev1"),
              (axes[1], M.s2, r"(b) severity 2  $(1,2,1,1)$", "sev2")]
    for ax, s, title, sev in panels:
        nm = names[sev]
        # guards: the frozen means used by Draft 4 must equal the Draft-5 cell means
        for key, (sh, na) in [("pruned", nm["pruned"]), ("post", nm["post"])]:
            assert abs(cell(sh)[0] - s[f"{key}_short"]) < 1e-6 and abs(cell(na)[0] - s[f"{key}_native"]) < 1e-6, (sev, key)  # float32 re-embedding
        # real-audio ceiling
        r_sh, r_na = cell(nm["real"][0])[0], cell(nm["real"][1])[0]
        ax.plot(x, [r_sh, r_na], color=C_REAL, ls="-.", lw=0.9, marker="^", ms=5.0, mec="white", mew=0.4,
                zorder=2, clip_on=False)
        ax.annotate("real audio", (0.0, r_sh), xytext=(-4, 4), textcoords="offset points", ha="right",
                    va="bottom", fontsize=6.3, color=C_REAL)
        # dense (matched control, severity 1 only)
        if sev == "sev1":
            d_sh, d_na = cell(nm["dense"][0])[0], cell(nm["dense"][1])[0]
            ax.plot(x, [d_sh, d_na], color=M.C_DENSE, ls=":", lw=1.0, marker="*", ms=8, mec="white", mew=0.4,
                    zorder=2, clip_on=False)
            ax.annotate("dense", (0.0, d_sh), xytext=(-4, -5), textcoords="offset points", ha="right",
                        va="top", fontsize=6.4, color=M.C_DENSE)
        # P and P+FT with CI whiskers of the paired gain
        ax.plot(x, [s["pruned_short"], s["pruned_native"]], color=M.C_PRUN, ls="--", lw=1.3, marker="s",
                ms=5.5, mec="white", mew=0.6, zorder=3, clip_on=False)
        ax.plot(x, [s["post_short"], s["post_native"]], color=M.C_POST, ls="-", lw=1.3, marker="o",
                ms=5.5, mec="white", mew=0.6, zorder=4, clip_on=False)
        for xpos, pv, qv, R in [(0.0, s["pruned_short"], s["post_short"], s["R_short"]),
                                (1.0, s["pruned_native"], s["post_native"], s["R_native"])]:
            p, lo, hi = M.ci(R)
            yerr = np.array([[qv - (pv + lo)], [(pv + hi) - qv]])
            ax.errorbar([xpos], [qv], yerr=yerr, fmt="none", ecolor=M.C_POST, elinewidth=1.0, capsize=2.6,
                        capthick=1.0, zorder=5, clip_on=False)
        # chance floors: one short tick per (system, duration), drawn at the cell's own floor
        for key, col in [("pruned", M.C_PRUN), ("post", M.C_POST)]:
            for xpos, cname in zip(x, nm[key]):
                fl = cell(cname)[1]
                ax.plot([xpos - 0.09, xpos + 0.09], [fl, fl], color=col, lw=1.6, alpha=0.55, zorder=2,
                        solid_capstyle="butt")
        # floor label once per panel, to the right of the native tick (clear of the y-axis and the R labels)
        fl_nat = cell(nm["pruned"][1])[1]
        if sev == "sev1":   # label once (panel (b): P sits just above its floor; the legend carries the meaning)
            ax.annotate("chance floors", (1.0, fl_nat), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=5.8, color="0.4")
        # R annotations
        _mid = 0.5 * (s["pruned_short"] + s["post_short"])
        _below = (s["post_short"] - s["pruned_short"]) < 0.03
        ax.annotate(r"$R_{\mathrm{short}}\,%+.3f$" % M.ci(s["R_short"])[0], (0.0, _mid),
                    xytext=(9, -9 if _below else 4), textcoords="offset points", ha="left",
                    va="top" if _below else "bottom", fontsize=6.6, color="0.15")
        ax.annotate(r"$R_{\mathrm{nat}}\,%+.3f$" % M.ci(s["R_native"])[0],
                    (1.0, 0.5 * (s["pruned_native"] + s["post_native"])), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.6, color="0.15")
        p, lo, hi = M.ci(s["J"])
        ax.set_title(title + "\n" + r"$J=%+.3f$  [$%+.3f,\,%+.3f$]" % (p, lo, hi), fontsize=7.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(-0.34, 1.34)
        ax.set_xlabel("")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.grid(axis="y", color=M.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes:
        ax.set_ylabel("CLAP cosine", fontsize=7.5)
    ymax = max(cell(n)[0] for n in ("real_full__sev2_192", "real_full__sev1_80")) + 0.03
    axes[0].set_ylim(-0.03, max(0.40, ymax))
    axes[0].set_xlabel("")
    handles = [
        Line2D([0], [0], color=M.C_POST, ls="-", marker="o", ms=5.5, mec="white", label="P+FT (fine-tuned)"),
        Line2D([0], [0], color=M.C_PRUN, ls="--", marker="s", ms=5.5, mec="white", label="P (pruned)"),
        Line2D([0], [0], color=M.C_DENSE, ls=":", marker="*", ms=8, mec="white", label="dense (matched)"),
        Line2D([0], [0], color=C_REAL, ls="-.", marker="^", ms=5, mec="white", label="real audio"),
        Line2D([0], [0], color="0.5", ls="-", lw=1.6, alpha=0.6, label="chance floor"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, handlelength=1.5, columnspacing=0.8,
               handletextpad=0.5, borderaxespad=0.0, fontsize=6.0, frameon=False, bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.885, bottom=0.215, hspace=0.62)
    fig.savefig(os.path.join(OUT, "fig1_interaction.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_interaction.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    D3.figure2()
    print("FIG1 anchors (matched | floor):")
    for n in ["real_crop__sev1_80", "real_full__sev1_80", "real_crop__sev2_192", "real_full__sev2_192",
              "pruned2_A__ac_short", "pruned2_A__ac_native", "recovered2__ac_short", "recovered2__ac_native"]:
        print(f"  {n:26s} {cell(n)[0]:.3f} | {cell(n)[1]:.3f}")
    print("wrote", os.path.join(OUT, "fig1_interaction.pdf"), "and", os.path.join(OUT, "fig2_where.pdf"))
