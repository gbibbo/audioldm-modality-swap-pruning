#!/usr/bin/env python3
"""Draft-6 ICASSP figures: the Draft-5 content re-laid-out for readability (CPU, 0 cr).

Nothing scientific changes here. Every plotted value is still read from the same committed frozen
artifacts as Draft 5 (same guards, same conventions); only the LAYOUT changes, plus one new figure
that visualises the domain result already reported in Sec. 4.2 and Tables 1-2.

  fig1_duration.pdf  -- FULL TEXT WIDTH (7.0 in, \\figure*): (a) severity 1 and (b) severity 2 duration
                        sweep side by side. Supersedes the stacked column-width fig1_interaction.pdf,
                        whose 0.7 in panels made the annotations collide with the axes.
  fig2_anatomy.pdf   -- FULL TEXT WIDTH, three panels: (a) NEW, the matched-duration DOMAIN result -- per
                        cell, the AudioCaps and held-out hip-hop rows show P and P+FT against their own
                        chance floors, with the 95 % CI of the paired gain R. Same visual language as
                        Fig. 1 (square = P, circle = P+FT, tick = that cell's chance floor).
  fig3_where.pdf     -- FULL TEXT WIDTH: (a) FineLAP grounding gain vs. time and (b) generation length
                        vs. scoring window, side by side (was the stacked fig2_where.pdf).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_draft6_figs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_manuscript_figs as M  # noqa: E402
import make_draft3_figs as D3     # noqa: E402
import make_draft5_figs as D5     # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = M.OUT
cells = D5.cells
cell = D5.cell
SWEEP = D5.SWEEP
DENSE192 = D5.DENSE192
DUR4 = D5.DUR4
XPOS = D5.XPOS
C_REAL = D5.C_REAL

TEXTWIDTH = 7.0      # ICASSP \textwidth = 178 mm
COLWIDTH = 3.35      # ICASSP \columnwidth = 86 mm

# durable sources for the new domain figure (same JSONs the tables are filled from)
MUS_NAT = M.load("configs/research/xsev_music_native_1_result.json")


# ================================================= duration panels (a) severity 1 and (b) severity 2
def draw_duration(axes):
    x = [0.0, 1.0]
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
        # guards: the frozen means used by Draft 4/5 must equal the Draft-5 cell means
        for key, (sh, na) in [("pruned", nm["pruned"]), ("post", nm["post"])]:
            assert abs(cell(sh)[0] - s[f"{key}_short"]) < 1e-6 and abs(cell(na)[0] - s[f"{key}_native"]) < 1e-6, (sev, key)
        r_sh, r_na = cell(nm["real"][0])[0], cell(nm["real"][1])[0]
        sw = SWEEP["secondary"]["by_duration"] if (SWEEP is not None and sev == "sev2") else None
        if sw is not None:
            assert abs(sw["3.84"]["levels"]["real"] - r_sh) < 1e-6 and abs(sw["10.24"]["levels"]["real"] - r_na) < 1e-6
            assert abs(sw["3.84"]["levels"]["P"] - s["pruned_short"]) < 1e-6 and abs(sw["10.24"]["levels"]["PFT"] - s["post_native"]) < 1e-6
            xs4 = [XPOS[d] for d in DUR4]
            lv = lambda k: [sw[str(d)]["levels"][k] for d in DUR4]
            ax.plot(xs4, lv("real"), color=C_REAL, ls="-.", lw=1.0, marker="^", ms=5.4, mec="white", mew=0.5,
                    zorder=2, clip_on=False)
        else:
            ax.plot(x, [r_sh, r_na], color=C_REAL, ls="-.", lw=1.0, marker="^", ms=5.4, mec="white", mew=0.5,
                    zorder=2, clip_on=False)
        ax.annotate("real audio", (1.0, r_na), xytext=(7, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=7.0, color=C_REAL)
        d_pair = None
        if sev == "sev1":
            d_pair = (cell(nm["dense"][0])[0], cell(nm["dense"][1])[0])
        elif DENSE192 is not None:
            d_pair = (DENSE192["means"]["dense_short"], DENSE192["means"]["dense_native"])
        if d_pair is not None:
            d_sh, d_na = d_pair
            if sw is not None:
                assert abs(sw["3.84"]["levels"]["dense"] - d_sh) < 1e-6 and abs(sw["10.24"]["levels"]["dense"] - d_na) < 1e-6
                ax.plot(xs4, lv("dense"), color=M.C_DENSE, ls=":", lw=1.1, marker="*", ms=8.5, mec="white", mew=0.5,
                        zorder=2, clip_on=False)
            else:
                ax.plot(x, [d_sh, d_na], color=M.C_DENSE, ls=":", lw=1.1, marker="*", ms=8.5, mec="white", mew=0.5,
                        zorder=2, clip_on=False)
            ax.annotate("dense", (1.0, d_na), xytext=(7, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=7.0, color=M.C_DENSE)
        if sw is not None:
            ax.plot(xs4, lv("P"), color=M.C_PRUN, ls="--", lw=1.4, marker="s", ms=5.6, mec="white", mew=0.6,
                    zorder=3, clip_on=False)
            ax.plot(xs4, lv("PFT"), color=M.C_POST, ls="-", lw=1.4, marker="o", ms=5.6, mec="white", mew=0.6,
                    zorder=4, clip_on=False)
            pairs = [(XPOS[d], sw[str(d)]["levels"]["P"], sw[str(d)]["levels"]["PFT"],
                      SWEEP["R_by_duration"][str(d)]) for d in DUR4]
        else:
            ax.plot(x, [s["pruned_short"], s["pruned_native"]], color=M.C_PRUN, ls="--", lw=1.4, marker="s",
                    ms=5.6, mec="white", mew=0.6, zorder=3, clip_on=False)
            ax.plot(x, [s["post_short"], s["post_native"]], color=M.C_POST, ls="-", lw=1.4, marker="o",
                    ms=5.6, mec="white", mew=0.6, zorder=4, clip_on=False)
            pairs = [(0.0, s["pruned_short"], s["post_short"], s["R_short"]),
                     (1.0, s["pruned_native"], s["post_native"], s["R_native"])]
        for xpos, pv, qv, R in pairs:
            p, lo, hi = M.ci(R)
            yerr = np.array([[qv - (pv + lo)], [(pv + hi) - qv]])
            ax.errorbar([xpos], [qv], yerr=yerr, fmt="none", ecolor=M.C_POST, elinewidth=1.0, capsize=2.8,
                        capthick=1.0, zorder=5, clip_on=False)
        # chance floors: one short tick per (system, duration), at the cell's own floor
        for key, col in [("pruned", M.C_PRUN), ("post", M.C_POST)]:
            if sw is not None:
                fk = "P" if key == "pruned" else "PFT"
                ticks = [(XPOS[d], sw[str(d)]["floors"][fk]) for d in DUR4]
                assert abs(ticks[0][1] - cell(nm[key][0])[1]) < 1e-6 and abs(ticks[-1][1] - cell(nm[key][1])[1]) < 1e-6
            else:
                ticks = [(xpos, cell(cname)[1]) for xpos, cname in zip(x, nm[key])]
            for xpos, fl in ticks:
                ax.plot([xpos - 0.055, xpos + 0.055], [fl, fl], color=col, lw=1.8, alpha=0.55, zorder=2,
                        solid_capstyle="butt")
        # R annotations: R_short above its pair, R_nat in the empty margin right of the native point
        ax.annotate(r"$R_{\mathrm{short}}\,%+.3f$" % M.ci(s["R_short"])[0],
                    (0.0, 0.5 * (s["pruned_short"] + s["post_short"])), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7.2, color="0.15")
        ax.annotate(r"$R_{\mathrm{nat}}\,%+.3f$" % M.ci(s["R_native"])[0],
                    (1.0, 0.5 * (s["pruned_native"] + s["post_native"])), xytext=(9, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=7.2, color="0.15")
        p, lo, hi = M.ci(s["J"])
        ax.set_title(title + r"   $J=%+.3f$ [$%+.3f,\,%+.3f$]" % (p, lo, hi), fontsize=7.8)
        if sw is not None:
            ax.set_xticks([XPOS[d] for d in DUR4])
            ax.set_xticklabels(["3.84 s", "5.12", "7.68", "10.24 s"])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(["3.84 s", "10.24 s"])
        ax.set_xlim(*((-0.75, 1.60) if sev == "sev1" else (-0.42, 1.42)))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.grid(axis="y", color=M.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("CLAP cosine", fontsize=8.0)
    ymax = max(cell(n)[0] for n in ("real_full__sev2_192", "real_full__sev1_80")) + 0.035
    axes[0].set_ylim(-0.04, max(0.40, ymax))
    return [
        Line2D([0], [0], color=M.C_POST, ls="-", marker="o", ms=5.6, mec="white", label="P+FT"),
        Line2D([0], [0], color=M.C_PRUN, ls="--", marker="s", ms=5.6, mec="white", label="P"),
        Line2D([0], [0], color=M.C_DENSE, ls=":", marker="*", ms=8.5, mec="white", label="dense"),
        Line2D([0], [0], color=C_REAL, ls="-.", marker="^", ms=5.4, mec="white", label="real audio"),
        Line2D([0], [0], color="0.5", ls="-", lw=1.8, alpha=0.6, label="chance floor"),
    ]


# ============================================ FIG. 2 (anatomy of the gain, full width, 3 panels)
DOMAIN_ROWS = [
    # (band header, domain label, P cell, P+FT cell, paired gain R)
    ("severity 1,  3.84 s", "AudioCaps", "pruned_short_sev1__armd80", "postft_short_sev1__armd80",
     "s1:R_short"),
    (None, "hip-hop", "p1_pruned_ema_reconstructed__off", "p1_recovered__off", "s1:R_music"),
    ("severity 2,  3.84 s", "AudioCaps", "pruned2_A__ac_short", "recovered2__ac_short", "s2:R_short"),
    (None, "hip-hop", "pruned2_A__music", "recovered2__music", "s2:R_music"),
    ("severity 2,  10.24 s", "AudioCaps", "pruned2_A__ac_native", "recovered2__ac_native", "s2:R_native"),
    (None, "hip-hop", "pruned2_A__music_native", "recovered2__music_native", "s2:R_music_native"),
]


def _domain_R(key):
    return {"s1:R_short": M.s1["R_short"], "s1:R_music": M.R_music_s1,
            "s2:R_short": M.s2["R_short"], "s2:R_music": M.s2["R_music"],
            "s2:R_native": M.s2["R_native"],
            "s2:R_music_native": MUS_NAT["PRIMARY_R_music_native"]}[key]


def draw_domain(ax, title=None):
    """In-domain AudioCaps vs held-out hip-hop, per (severity, duration) cell.

    Levels and floors come from draft5_floor_ceiling_result.json (the source of Table 2); the paired
    gain R and its 95 % CI come from the same frozen results as Table 1. No new statistic.
    """
    # guard: the plotted paired gains must equal the frozen table values
    assert abs(M.ci(_domain_R("s1:R_music"))[0] - (-0.094)) < 5e-4
    assert abs(M.ci(_domain_R("s2:R_music_native"))[0] - 0.005) < 5e-4
    ys = []
    for i, (band, dom, cp, cq, rk) in enumerate(DOMAIN_ROWS):
        y = -(i + 0.55 * (i // 2))          # a small gap between the three bands
        ys.append(y)
        pv, pf = cell(cp)
        qv, qf = cell(cq)
        ax.plot([pv, qv], [y, y], color="0.55", lw=0.8, zorder=2)
        ax.plot([pf, pf], [y, y + 0.30], color=M.C_PRUN, lw=1.6, alpha=0.55, zorder=2, solid_capstyle="butt")
        ax.plot([qf, qf], [y - 0.30, y], color=M.C_POST, lw=1.6, alpha=0.55, zorder=2, solid_capstyle="butt")
        ax.plot([pv], [y], marker="s", ms=5.2, color=M.C_PRUN, mec="white", mew=0.6, ls="none", zorder=4)
        p, lo, hi = M.ci(_domain_R(rk))
        ax.errorbar([qv], [y], xerr=np.array([[qv - (pv + lo)], [(pv + hi) - qv]]), fmt="o", ms=5.2,
                    color=M.C_POST, mec="white", mew=0.6, ecolor=M.C_POST, elinewidth=1.0, capsize=2.4,
                    capthick=1.0, zorder=5)
        if band is not None:
            ax.annotate(band, (1.0, y + 0.68), xycoords=("axes fraction", "data"), xytext=(-2, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=6.5, color="0.30")
    ax.axvline(0.0, color="0.75", lw=0.7, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in DOMAIN_ROWS], fontsize=7.0)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 1.15)
    ax.set_xlim(-0.04, 0.40)
    ax.set_xlabel("CLAP cosine", fontsize=8.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color=M.GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], color=M.C_PRUN, marker="s", ms=5.2, mec="white", ls="none", label="P"),
        Line2D([0], [0], color=M.C_POST, marker="o", ms=5.2, mec="white", ls="none", label="P+FT"),
        Line2D([0], [0], color="0.5", ls="-", lw=1.6, alpha=0.6, label="chance floor"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, handlelength=1.0, columnspacing=0.6,
              handletextpad=0.3, fontsize=6.6, frameon=False, borderaxespad=0.0,
              bbox_to_anchor=(1.03, 1.04))
    if title:
        ax.set_title(title, fontsize=7.8)


def figure_all():
    """One full-width, five-panel figure: the recovery gain across every operating point we measured.

    Row 1 -- the duration axis, one panel per severity (the Draft-5 Fig. 1, side by side instead of
    stacked, so the annotations no longer collide with the axes).
    Row 2 -- (c) the NEW domain panel, (d) the FineLAP time course and (e) the crop analysis (the
    Draft-5 Fig. 2). Two column-width floats become one text-width float: every panel roughly doubles
    in area while the page budget grows by less than one column-inch.
    """
    fig = plt.figure(figsize=(TEXTWIDTH, 3.05))
    gs1 = fig.add_gridspec(1, 2, left=0.104, right=0.988, top=0.945, bottom=0.632, wspace=0.10)
    gs2 = fig.add_gridspec(1, 3, left=0.104, right=0.988, top=0.448, bottom=0.132, wspace=0.34)
    a1 = fig.add_subplot(gs1[0, 0])
    a2 = fig.add_subplot(gs1[0, 1], sharey=a1)
    handles = draw_duration([a1, a2])
    a2.tick_params(labelleft=False)
    fig.legend(handles=handles, loc="center", ncol=5, handlelength=1.6, columnspacing=1.4,
               handletextpad=0.4, fontsize=6.9, frameon=False, bbox_to_anchor=(0.53, 0.527))

    b1 = fig.add_subplot(gs2[0, 0])
    b2 = fig.add_subplot(gs2[0, 1])
    b3 = fig.add_subplot(gs2[0, 2])
    draw_domain(b1, title="(c) prompt domain")
    b1.get_legend().remove()          # markers are already keyed by the legend of panel (a)
    D3.draw_finelap_fixed(b2, title="(d) grounding vs. time in clip")
    b2.set_ylim(-0.02, 0.50)
    b2.legend(loc="upper left", handlelength=1.6, fontsize=6.4, frameon=True, framealpha=0.9,
              edgecolor="none", borderpad=0.22, labelspacing=0.2)
    D3.draw_crop(b3)
    b3.set_title("(e) generated vs. cropped", fontsize=7.8)
    b3.set_xticklabels(["3.84 s\n(gen.)", "first 3.84 s\n(crop)", "10.24 s\n(gen.)"], fontsize=6.4)
    b3.set_ylabel("recovery gain $R$", fontsize=7.6)
    b3.set_ylim(-0.055, 0.375)
    b3.legend(loc="upper left", handlelength=1.5, fontsize=6.4, frameon=True, framealpha=0.9,
              edgecolor="none", borderpad=0.22, labelspacing=0.2)
    fig.savefig(os.path.join(OUT, "fig1_operating_points.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_operating_points.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figure_all()
    print("FIG. 1(c) domain: plotted levels / floors (above chance = level - floor):")
    for _, _, cp, cq, _ in DOMAIN_ROWS:
        for n_ in (cp, cq):
            v, f = cell(n_)
            print(f"  {n_:34s} mean {v:+.4f}  floor {f:+.4f}  above chance {v - f:+.4f}")
    print("wrote", os.path.join(OUT, "fig1_operating_points.pdf"))
