#!/usr/bin/env python3
"""Generate the ICASSP manuscript figures from DURABLE artifacts only (no fabricated values).

CPU, 0 GPU, 0 credits. Every plotted number is read from a committed frozen result JSON, so the
figures are traceable to the same source of truth as the manuscript tables.

Outputs (vector PDF + PNG preview) under icassp/figs/:
  fig1_interaction.pdf   -- system x temporal-operating-point interaction, severity 1 vs 2 (CENTRAL)
  fig2_forest.pdf        -- post-FT minus pruned paired contrasts w/ 95% CI across context x severity

Run:
  OPENBLAS_CORETYPE=Haswell python scripts/research/paper_figs/make_manuscript_figs.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "icassp", "figs")
os.makedirs(OUT, exist_ok=True)


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


# ---------------------------------------------------------------- durable sources
xsev = load("configs/research/xsev_result.json")
opd = load("configs/research/op_duration_discriminator_1_result.json")
music1 = load("configs/research/reversal_v1_r_music_clap.json")

# severity 1 (Arm-D matched 80-ytid subset) -- absolute CLAP means + contrasts
s1m = opd["PRIMARY_clap"]["means"]
s1 = dict(
    pruned_short=s1m["pruned_ctrl"], post_short=s1m["recovered_ctrl"],
    pruned_native=s1m["pruned_alt"], post_native=s1m["recovered_alt"],
    R_short=opd["PRIMARY_clap"]["R_ctrl_80"],
    R_native=opd["PRIMARY_clap"]["R_alt"],
    J=opd["PRIMARY_clap"]["J"],
)
dense_native = xsev["DENSE_CONTROL"]["C_dense_10s"]  # sev-1 dense reference @10.24 s (secondary)

# severity 2 (1,2,1,1) primary A'
s2m = xsev["PRIMARY_A"]["means"]
s2 = dict(
    pruned_short=s2m["pruned_short"], post_short=s2m["rec_short"],
    pruned_native=s2m["pruned_native"], post_native=s2m["rec_native"],
    R_short=xsev["PRIMARY_A"]["R_short"],
    R_native=xsev["PRIMARY_A"]["R_native"],
    J=xsev["PRIMARY_A"]["J"],
    R_music=xsev["PRIMARY_A"]["R_music"],
    K=xsev["PRIMARY_A"]["K"],
)
# severity 1 held-out music contrast (from the pre-registered reversal experiment)
R_music_s1 = music1["R_music"]

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed TrueType so labels stay selectable/searchable
    "ps.fonttype": 42,
})

C_POST = "#1f4e79"   # dark blue  -> post-fine-tuning
C_PRUN = "#c0561f"   # burnt orange -> pruned
C_DENSE = "#555555"  # grey        -> dense reference
GRID = "#d9d9d9"


# =============================================================== FIGURE 1
def figure1():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.38), sharey=True)
    x = [0.0, 1.0]
    xlabels = ["3.84 s\n(short)", "10.24 s\n(native)"]

    panels = [
        (axes[0], s1, "(a) severity 1: $(1,2,3,1)$", True),
        (axes[1], s2, "(b) severity 2: $(1,2,1,1)$", False),
    ]
    for ax, s, title, show_dense in panels:
        # profile lines: pruned (dashed/square) and post-FT (solid/circle)
        ax.plot(x, [s["pruned_short"], s["pruned_native"]], color=C_PRUN, ls="--",
                lw=1.4, marker="s", ms=6, mec="white", mew=0.7, zorder=3, clip_on=False)
        ax.plot(x, [s["post_short"], s["post_native"]], color=C_POST, ls="-",
                lw=1.4, marker="o", ms=6, mec="white", mew=0.7, zorder=3, clip_on=False)
        if show_dense:
            ax.plot([1.0], [dense_native], color=C_DENSE, marker="*", ms=11,
                    mec="white", mew=0.6, ls="none", zorder=4, clip_on=False)
            ax.annotate("dense", (1.0, dense_native), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=7, color=C_DENSE)

        # vertical gap arrows for R_short and R_native (= post-FT minus pruned);
        # point value only (full CIs are in Table 1 and Fig. 2). Labels sit in the
        # empty side margins so they never collide with the profile lines.
        def gap(xpos, lo_val, hi_val, name, contrast, side):
            ax.annotate("", xy=(xpos, hi_val), xytext=(xpos, lo_val),
                        arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.9))
            ha = "left" if side > 0 else "right"
            ax.annotate("%s\n$%+.3f$" % (name, contrast["point"]),
                        (xpos, 0.5 * (lo_val + hi_val)), xytext=(5 * side, 0),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=6.9, color="0.15")

        gap(0.0, s["pruned_short"], s["post_short"], r"$R_{\mathrm{short}}$", s["R_short"], -1)
        gap(1.0, s["pruned_native"], s["post_native"], r"$R_{\mathrm{nat}}$", s["R_native"], +1)

        jp = s["J"]["point"]
        jlo = s["J"]["ci95"][0] if "ci95" in s["J"] else s["J"]["lo"]
        jhi = s["J"]["ci95"][1] if "ci95" in s["J"] else s["J"]["hi"]
        ax.set_title(title + "\n" + r"$J=%+.3f$ [%+.3f, %+.3f]" % (jp, jlo, jhi), fontsize=8.3)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(-0.30, 1.30)
        ax.set_xlabel("generated clip duration")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel("CLAP cosine (text--audio)")
    axes[0].set_ylim(-0.03, 0.40)

    handles = [
        Line2D([0], [0], color=C_POST, ls="-", marker="o", ms=6, mec="white", label="post-fine-tuning"),
        Line2D([0], [0], color=C_PRUN, ls="--", marker="s", ms=6, mec="white", label="pruned"),
    ]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0),
                   handlelength=1.8, borderaxespad=0.2)

    fig.subplots_adjust(left=0.075, right=0.965, top=0.78, bottom=0.22, wspace=0.14)
    fig.savefig(os.path.join(OUT, "fig1_interaction.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_interaction.png"), dpi=200)
    plt.close(fig)


# =============================================================== FIGURE 2
def figure2():
    # rows top->bottom; grouped by severity. label, contrast dict(point,lo,hi), n
    def ci(c):
        lo = c["lo"] if "lo" in c else c["ci95"][0]
        hi = c["hi"] if "hi" in c else c["ci95"][1]
        return c["point"], lo, hi

    rows = [
        ("sev2", "AudioCaps 10.24 s", ci(s2["R_native"]), 192),
        ("sev2", "AudioCaps 3.84 s", ci(s2["R_short"]), 192),
        ("sev2", "held-out music 3.84 s", ci(s2["R_music"]), 64),
        ("sev1", "AudioCaps 10.24 s", ci(s1["R_native"]), 80),
        ("sev1", "AudioCaps 3.84 s", ci(s1["R_short"]), 80),
        ("sev1", "held-out music 3.84 s", ci(R_music_s1), 64),
    ]
    n = len(rows)
    ys = list(range(n))[::-1]  # first row highest

    fig, ax = plt.subplots(figsize=(3.5, 2.62))
    ax.axvline(0.0, color="0.3", lw=0.9, zorder=1)

    for y, (sev, lab, (p, lo, hi), nn) in zip(ys, rows):
        excl0 = (lo > 0) or (hi < 0)
        col = C_POST if p >= 0 else C_PRUN
        ax.plot([lo, hi], [y, y], color=col, lw=1.4, zorder=2, solid_capstyle="round")
        ax.plot([lo, lo], [y - 0.13, y + 0.13], color=col, lw=1.1)
        ax.plot([hi, hi], [y - 0.13, y + 0.13], color=col, lw=1.1)
        ax.plot([p], [y], marker="o", ms=6, color=col,
                mfc=(col if excl0 else "white"), mec=col, mew=1.2, zorder=3)

    # severity group separator + labels
    ax.axhline(2.5, color="0.8", lw=0.7, ls=(0, (4, 3)))
    ax.text(0.455, 5.35, "severity 2  $(1,2,1,1)$", fontsize=7.3, style="italic",
            ha="right", color="0.25", transform=ax.get_yaxis_transform())
    ax.text(0.455, 2.35, "severity 1  $(1,2,3,1)$", fontsize=7.3, style="italic",
            ha="right", color="0.25", transform=ax.get_yaxis_transform())

    ax.set_yticks(ys)
    ax.set_yticklabels([lab for (_, lab, _, _) in rows], fontsize=7.6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-0.20, 0.35)
    ax.set_xlabel(r"$\Delta$ CLAP cosine (post-FT $-$ pruned)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.text(-0.19, -0.62, "favours pruned", fontsize=6.6, color="0.45", ha="left", va="center")
    ax.text(0.02, -0.62, "favours post-FT", fontsize=6.6, color="0.45", ha="left", va="center")

    fig.subplots_adjust(left=0.42, right=0.985, top=0.985, bottom=0.14)
    fig.savefig(os.path.join(OUT, "fig2_forest.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_forest.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    # echo the plotted values for the provenance audit
    print("FIG1/2 source values (post-FT - pruned):")
    print(f"  sev1 R_short  {s1['R_short']['point']:+.4f} {s1['R_short']['ci95']}")
    print(f"  sev1 R_native {s1['R_native']['point']:+.4f} {s1['R_native']['ci95']}")
    print(f"  sev1 J        {s1['J']['point']:+.4f} {s1['J']['ci95']}")
    print(f"  sev1 R_music  {R_music_s1['point']:+.4f} [{R_music_s1['lo']:.4f},{R_music_s1['hi']:.4f}]")
    print(f"  sev1 dense@10 {dense_native:.4f}")
    print(f"  sev2 R_short  {s2['R_short']['point']:+.4f} [{s2['R_short']['lo']:.4f},{s2['R_short']['hi']:.4f}]")
    print(f"  sev2 R_native {s2['R_native']['point']:+.4f} [{s2['R_native']['lo']:.4f},{s2['R_native']['hi']:.4f}]")
    print(f"  sev2 J        {s2['J']['point']:+.4f} [{s2['J']['lo']:.4f},{s2['J']['hi']:.4f}]")
    print(f"  sev2 R_music  {s2['R_music']['point']:+.4f} [{s2['R_music']['lo']:.4f},{s2['R_music']['hi']:.4f}]")
    print(f"  sev2 K        {s2['K']['point']:+.4f} [{s2['K']['lo']:.4f},{s2['K']['hi']:.4f}]")
    print("wrote fig1_interaction.pdf, fig2_forest.pdf ->", OUT)
