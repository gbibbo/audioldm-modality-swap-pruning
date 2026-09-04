#!/usr/bin/env python3
"""Draft-3 ICASSP figures from DURABLE artifacts only (CPU, 0 cr). No fabricated values.

Reuses the Draft-2 module (frozen sources, style, provenance-guarded FineLAP curve) and adds the
Draft-3 panels:
  fig1_interaction.pdf   -- unchanged Draft-2 interaction figure (system x duration, CI whiskers)
  fig2_where.pdf         -- two-panel: (a) FineLAP Delta-grounding vs time (label collisions fixed),
                            (b) generation length vs scoring window: the recovery gain R at the
                            separately generated 3.84 s clip, at the first 3.84 s of the 10.24 s clip
                            (crop), and at the full 10.24 s clip, per severity, 95% CI.
Sources: configs/research/native_crop_analysis_result.json (post-hoc diagnostic) + the frozen results
already read by make_manuscript_figs.py.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_draft3_figs.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_manuscript_figs as M  # noqa: E402  (module-level: loads frozen JSONs, sets style, no drawing)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = M.OUT
crop = M.load("configs/research/native_crop_analysis_result.json")


def draw_finelap_fixed(ax, title=None):
    """Draft-2 FineLAP panel with the boundary label and legend moved so nothing overlaps."""
    M.draw_finelap(ax, title=None)
    # remove the Draft-2 boundary text and re-place it top-left of the boundary; legend -> lower right
    for t in list(ax.texts):
        if "early | late" in t.get_text():
            t.remove()
        elif "T_" in t.get_text() or "late" in t.get_text():
            # the "late-early T" annotation: re-place it top-LEFT so it clears the lower-right legend
            # at the reduced panel height
            txt, fs, col = t.get_text(), t.get_fontsize(), t.get_color()
            t.remove()
            ax.text(0.02, 0.06, txt.replace("; ", "\n"), transform=ax.transAxes, fontsize=fs, color=col,
                    ha="left", va="bottom", linespacing=1.0)   # bottom-left is the only free corner
    # A6: keep the early/late labels near the zero line; at the top they ran into the
    # "late-early T" annotation (which stays at the top right).
    ax.text(3.84 - 0.12, 0.028, "early", fontsize=6.2, color="0.35", va="bottom", ha="right")
    ax.text(3.84 + 0.12, 0.028, "late", fontsize=6.2, color="0.35", va="bottom", ha="left")
    ax.legend(loc="lower right", handlelength=1.9, frameon=True, framealpha=0.88, edgecolor="none")
    if title:
        ax.set_title(title, fontsize=8.0)


def draw_crop(ax, title=None):
    sev = crop["severities"]
    rows = [("severity 1 ($n{=}80$)", sev["sev1_armd80"], M.C_PRUN, "--", "s"),
            ("severity 2 ($n{=}192$)", sev["sev2_xsev192_pruned2_A"], M.C_POST, "-", "o")]
    x = np.arange(3)
    for i, (lab, s, col, ls, mk) in enumerate(rows):
        pts = [s["R_short_frozen"], s["R_crop"], s["R_native_frozen"]]
        y = np.array([p["point"] for p in pts])
        lo = y - np.array([p["lo"] for p in pts])
        hi = np.array([p["hi"] for p in pts]) - y
        off = (-0.06 if i == 0 else 0.06)
        ax.errorbar(x + off, y, yerr=[lo, hi], color=col, ls=ls, marker=mk, ms=4.2, lw=1.2,
                    capsize=2.0, elinewidth=0.8, label=lab, zorder=3)
    ax.axhline(0.0, color="0.6", lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["3.84 s clip\n(generated)", "first 3.84 s\nof 10.24 s clip", "10.24 s clip\n(generated)"])
    ax.set_ylabel("recovery gain $R$" + "\n" + r"(CLAP, P+FT $-$ P)")  # A6: 2 lines so it fits the shorter panel
    ax.set_xlim(-0.5, 2.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(color=M.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", handlelength=2.0, frameon=True, framealpha=0.88, edgecolor="none")
    if title:
        ax.set_title(title, fontsize=8.0)


def figure2():
    # A6: natural size = one ICASSP column (3.375 in), included at \columnwidth
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.35, 2.00), gridspec_kw=dict(hspace=1.25))
    draw_finelap_fixed(a, title="(a) FineLAP grounding gain vs. time in clip")
    draw_crop(b, title="(b) generation length vs. scoring window")
    fig.subplots_adjust(left=0.185, right=0.99, top=0.90, bottom=0.195)  # A6: no clipping at column width
    fig.savefig(os.path.join(OUT, "fig2_where.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_where.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    M.figure1()
    figure2()
    print("FIG2b source values (post-FT - pruned):")
    for k, s in crop["severities"].items():
        print(f"  {k}: R_short {s['R_short_frozen']['point']:+.3f}  R_crop {s['R_crop']['point']:+.3f} "
              f"[{s['R_crop']['lo']:+.3f},{s['R_crop']['hi']:+.3f}]  R_native {s['R_native_frozen']['point']:+.3f}")
    print("wrote", os.path.join(OUT, "fig1_interaction.pdf"), "and", os.path.join(OUT, "fig2_where.pdf"))
