#!/usr/bin/env python3
"""Generate the ICASSP manuscript figures from DURABLE artifacts only (no fabricated values).

CPU, 0 GPU, 0 credits. Every plotted number is read from a committed frozen result JSON, so the
figures are traceable to the same source of truth as the manuscript tables. Draft 2 adds visible
uncertainty (95% bootstrap CI whiskers), a dense descriptive reference, and a FineLAP frame-level
time-course reconstructed from the frozen per-frame scores.

Outputs (vector PDF + PNG preview) under icassp/figs/:
  fig1_interaction.pdf  -- system x temporal-operating-point interaction, severity 1 vs 2, CI whiskers
  fig_summary.pdf       -- two-panel: (a) forest of paired contrasts, (b) FineLAP Delta-grounding
  fig2_forest.pdf       -- standalone forest (kept for a longer/journal version; not embedded)
  fig3_finelap.pdf      -- standalone FineLAP time-course (kept for a longer/journal version)

Provenance guard: the FineLAP panel reads artifacts/finelap_temporal/scores_sev{1,2}.json (gitignored
raw frames) but ASSERTS their scores_sha256 equals the value recorded in the committed frozen verdict
configs/research/finelap_temporal_result.json, and that the reconstructed window means reproduce the
frozen D_early / D_late to <1e-6. The plotted curve is therefore provably the same object as the
frozen post-result diagnostic (no new statistic).

Run:
  OPENBLAS_CORETYPE=Haswell python scripts/research/paper_figs/make_manuscript_figs.py
"""
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
rev1 = load("configs/research/reversal_v1_1_result.json")
finelap = load("configs/research/finelap_temporal_result.json")


def ci(c):
    """Return (point, lo, hi) from either {point,lo,hi} or {point,ci95:[lo,hi]}."""
    lo = c["lo"] if "lo" in c else c["ci95"][0]
    hi = c["hi"] if "hi" in c else c["ci95"][1]
    return c["point"], lo, hi


# severity 1 (Arm-D matched 80-ytid subset) -- absolute CLAP means + contrasts
s1m = opd["PRIMARY_clap"]["means"]
s1 = dict(
    pruned_short=s1m["pruned_ctrl"], post_short=s1m["recovered_ctrl"],
    pruned_native=s1m["pruned_alt"], post_native=s1m["recovered_alt"],
    R_short=opd["PRIMARY_clap"]["R_ctrl_80"],
    R_native=opd["PRIMARY_clap"]["R_alt"],
    J=opd["PRIMARY_clap"]["J"],
)
# dense descriptive reference: short 3.84 s from the pre-registered V1.1 (n=96), native 10.24 s from
# the Arm-D dense union (n=80). Descriptive anchors only (different n, not paired).
dense_short_s1 = rev1["PRIMARY"]["C_dense"]          # 0.2039 @ 3.84 s (V1.1, n=96; Draft-2/3 anchor)
dense_native_s1 = xsev["DENSE_CONTROL"]["C_dense_10s"]  # 0.3520 @ 10.24 s
# Draft 4: MATCHED dense duration control (same 80 prompts, same 80-item scoring convention at both
# durations; configs/research/draft4_dense_duration_control_result.json). When present it replaces the
# unmatched 3.84 s anchor so the grey line in Fig. 1(a) is prompt-paired and convention-matched.
_ddc_path = os.path.join(ROOT, "configs/research/draft4_dense_duration_control_result.json")
dense_control = load("configs/research/draft4_dense_duration_control_result.json") if os.path.exists(_ddc_path) else None
if dense_control is not None:
    dense_short_s1 = dense_control["means"]["dense_short"]      # 0.2023 @ 3.84 s, 80-item call
    assert abs(dense_control["means"]["dense_native"] - dense_native_s1) < 1e-9

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
R_music_s1 = music1["R_music"]  # severity 1 held-out music contrast (pre-registered reversal expt)

# ---------------------------------------------------------------- style (publication conventions)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.0,
    "axes.linewidth": 0.7,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,  # embed TrueType so labels stay selectable/searchable
    "ps.fonttype": 42,
})

C_POST = "#1f3b73"   # dark blue    -> post-fine-tuning (darkest in grayscale)
C_PRUN = "#c0561f"   # burnt orange -> pruned (mid grey in grayscale; distinct linestyle/marker)
C_DENSE = "#8a8a8a"  # grey         -> dense descriptive reference
GRID = "#dddddd"


# =============================================================== FIGURE 1 (interaction, CI whiskers)
def figure1():
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.66), sharex=True, sharey=True)
    x = [0.0, 1.0]
    xlabels = ["3.84 s\n(short)", "10.24 s\n(native)"]

    panels = [
        (axes[0], s1, r"(a) severity 1  $(1,2,3,1)$", True),
        (axes[1], s2, r"(b) severity 2  $(1,2,1,1)$", False),
    ]
    for ax, s, title, show_dense in panels:
        ax.plot(x, [s["pruned_short"], s["pruned_native"]], color=C_PRUN, ls="--",
                lw=1.3, marker="s", ms=5.5, mec="white", mew=0.6, zorder=3, clip_on=False)
        ax.plot(x, [s["post_short"], s["post_native"]], color=C_POST, ls="-",
                lw=1.3, marker="o", ms=5.5, mec="white", mew=0.6, zorder=4, clip_on=False)
        # 95% CI whiskers of the PAIRED contrast R, positioned about the post-FT mean (= pruned + R).
        for xpos, pruned_val, post_val, R in [
            (0.0, s["pruned_short"], s["post_short"], s["R_short"]),
            (1.0, s["pruned_native"], s["post_native"], s["R_native"]),
        ]:
            p, lo, hi = ci(R)
            yerr = np.array([[post_val - (pruned_val + lo)], [(pruned_val + hi) - post_val]])
            ax.errorbar([xpos], [post_val], yerr=yerr, fmt="none", ecolor=C_POST,
                        elinewidth=1.0, capsize=2.6, capthick=1.0, zorder=5, clip_on=False)
        # R_short label: below the pair when the two short points nearly coincide (sev-1), else in the
        # gap between the two lines (sev-2) -- keeps it off both curves, the markers and the y-axis.
        _mid = 0.5 * (s["pruned_short"] + s["post_short"])
        _below = (s["post_short"] - s["pruned_short"]) < 0.03
        ax.annotate(r"$R_{\mathrm{short}}\,%+.3f$" % ci(s["R_short"])[0],
                    (0.0, _mid), xytext=(9, -9 if _below else 4),
                    textcoords="offset points", ha="left", va="top" if _below else "bottom",
                    fontsize=6.6, color="0.15")
        ax.annotate(r"$R_{\mathrm{nat}}\,%+.3f$" % ci(s["R_native"])[0],
                    (1.0, 0.5 * (s["pruned_native"] + s["post_native"])), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.6, color="0.15")
        if show_dense:
            ax.plot(x, [dense_short_s1, dense_native_s1], color=C_DENSE, ls=":", lw=1.0,
                    marker="*", ms=8, mec="white", mew=0.4, zorder=2, clip_on=False)
            ax.annotate("dense (matched)" if dense_control is not None else "dense (ref.)",
                        (1.0, dense_native_s1), xytext=(-4, 7),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=6.6, color=C_DENSE)
            if dense_control is not None:   # duration responses s(.) of the three systems, same 80 prompts
                sl = dense_control["slopes"]
                ax.annotate(r"$s$: dense $%+.3f$, P $%+.3f$, P+FT $%+.3f$" % (
                    sl["dense"]["point"], sl["pruned"]["point"], sl["postft"]["point"]),
                    (-0.30, 0.398), xytext=(0, 0), textcoords="offset points", ha="left", va="top",
                    fontsize=6.0, color="0.25")
        p, lo, hi = ci(s["J"])
        ax.set_title(title + "\n" + r"$J=%+.3f$  [$%+.3f,\,%+.3f$]" % (p, lo, hi), fontsize=7.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(-0.34, 1.34)
        ax.set_xlabel("generated clip duration")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for ax in axes:
        ax.set_ylabel("CLAP cosine", fontsize=7.5)
    axes[0].set_ylim(-0.03, 0.40)
    axes[0].set_xlabel("")
    handles = [
        Line2D([0], [0], color=C_POST, ls="-", marker="o", ms=5.5, mec="white", label="P+FT (fine-tuned)"),
        Line2D([0], [0], color=C_PRUN, ls="--", marker="s", ms=5.5, mec="white", label="P (pruned)"),
        Line2D([0], [0], color=C_DENSE, ls=":", marker="*", ms=8, mec="white", label="dense"),
    ]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.0),
                   handlelength=1.7, borderaxespad=0.2, fontsize=6.5)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.905, bottom=0.10, hspace=0.60)
    fig.savefig(os.path.join(OUT, "fig1_interaction.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_interaction.png"), dpi=200)
    plt.close(fig)


# =============================================================== FOREST (draw into a provided ax)
def draw_forest(ax, title=None):
    rows = [
        ("sev2", r"$R_{\mathrm{nat}}$ 10.24 s", ci(s2["R_native"])),
        ("sev2", r"$R_{\mathrm{short}}$ 3.84 s", ci(s2["R_short"])),
        ("sev2", r"$R_{\mathrm{music}}$ 3.84 s", ci(s2["R_music"])),
        ("sev2", r"$J=R_{\mathrm{nat}}-R_{\mathrm{short}}$", ci(s2["J"])),
        ("sev1", r"$R_{\mathrm{nat}}$ 10.24 s", ci(s1["R_native"])),
        ("sev1", r"$R_{\mathrm{short}}$ 3.84 s", ci(s1["R_short"])),
        ("sev1", r"$R_{\mathrm{music}}$ 3.84 s", ci(R_music_s1)),
        ("sev1", r"$J=R_{\mathrm{nat}}-R_{\mathrm{short}}$", ci(s1["J"])),
    ]
    n = len(rows)
    ys = list(range(n))[::-1]
    ax.axvline(0.0, color="0.35", lw=0.8, zorder=1)
    for y, (sev, lab, (p, lo, hi)) in zip(ys, rows):
        excl0 = (lo > 0) or (hi < 0)
        col = C_POST if p >= 0 else C_PRUN
        ax.plot([lo, hi], [y, y], color=col, lw=1.3, zorder=2)
        for xb in (lo, hi):
            ax.plot([xb, xb], [y - 0.14, y + 0.14], color=col, lw=1.0)
        ax.plot([p], [y], marker="o", ms=5.5, color=col,
                mfc=(col if excl0 else "white"), mec=col, mew=1.1, zorder=3)
    ax.axhline(3.5, color="0.8", lw=0.7, ls=(0, (4, 3)))
    ax.text(0.352, 7.35, r"severity 2  $(1,2,1,1)$", fontsize=7.0, style="italic",
            ha="right", color="0.25", transform=ax.get_yaxis_transform())
    ax.text(0.352, 3.35, r"severity 1  $(1,2,3,1)$", fontsize=7.0, style="italic",
            ha="right", color="0.25", transform=ax.get_yaxis_transform())
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for (_, lab, _) in rows], fontsize=7.2)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(-0.20, 0.36)
    ax.set_xlabel(r"$\Delta$ CLAP cosine (post-FT $-$ pruned), 95% CI")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(-0.19, -0.72, "favours pruned", fontsize=6.4, color="0.45", ha="left", va="center")
    ax.text(0.02, -0.72, "favours post-FT", fontsize=6.4, color="0.45", ha="left", va="center")
    if title:
        ax.set_title(title, fontsize=8.0)


# =============================================================== FINELAP (draw into a provided ax)
def _finelap_curve(sev):
    """Per-frame Delta-grounding (post-FT - pruned_A), nested prompt/event mean, GUARDED vs verdict."""
    path = os.path.join(ROOT, "artifacts/finelap_temporal", f"scores_{sev}.json")
    d = json.load(open(path))
    sha = hashlib.sha256(
        json.dumps([p["scores"] for p in d["prompts"]], sort_keys=True).encode()).hexdigest()
    assert sha == finelap["severities"][sev]["scores_sha256"], f"{sev} scores_sha256 mismatch"
    per_prompt = []
    for p in d["prompts"]:
        rec, pru = p["scores"]["recovered"], p["scores"]["pruned_A"]
        mids = list(rec.keys())
        rec_a = np.array([rec[m] for m in mids])
        pru_a = np.array([pru[m] for m in mids])
        per_prompt.append((rec_a - pru_a).mean(axis=0))
    curve = np.array(per_prompt).mean(axis=0)
    contr = finelap["severities"][sev]["contrasts"]["pruned_A"]
    d_early, d_late = curve[0:24].mean(), curve[24:64].mean()
    assert abs(d_early - contr["D_early"]["point"]) < 1e-6, (sev, d_early)
    assert abs(d_late - contr["D_late"]["point"]) < 1e-6, (sev, d_late)
    return curve, d_early, d_late


def draw_finelap(ax, title=None):
    spf = finelap["windows"]["seconds_per_frame"]
    t = np.arange(64) * spf + spf / 2.0
    boundary_s = finelap["windows"]["LATE_frames"][0] * spf
    c2, e2, l2 = _finelap_curve("sev2")
    c1, e1, l1 = _finelap_curve("sev1")
    n2 = finelap["severities"]["sev2"]["n_eligible_prompts"]
    n1 = finelap["severities"]["sev1"]["n_eligible_prompts"]
    T2 = finelap["verdict"]["T2"]["point"]
    T1 = finelap["verdict"]["T1_directional"]["point"]
    ax.set_xlim(0, 10.24)
    ax.set_ylim(-0.02, 0.40)
    ax.axhline(0.0, color="0.6", lw=0.7, zorder=1)
    ax.axvline(boundary_s, color="0.45", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(boundary_s + 0.12, 0.015, "early | late\nboundary 3.84 s",
            fontsize=6.2, color="0.35", va="bottom", ha="left")
    ax.plot(t, c2, color=C_POST, lw=1.4, zorder=4, label=r"severity 2 ($n{=}%d$)" % n2)
    ax.plot(t, c1, color=C_PRUN, lw=1.2, ls="--", zorder=3, label=r"severity 1 ($n{=}%d$)" % n1)
    for (e, l, col) in [(e2, l2, C_POST), (e1, l1, C_PRUN)]:
        ax.plot([t[0], boundary_s], [e, e], color=col, lw=0.8, ls=":", alpha=0.9, zorder=2)
        ax.plot([boundary_s, t[-1]], [l, l], color=col, lw=0.8, ls=":", alpha=0.9, zorder=2)
    ax.set_xlabel("time in clip (s)")
    ax.set_ylabel(r"$\Delta$ grounding")  # A6: short; title and caption give the metric and contrast
    ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", handlelength=1.9)
    ax.text(0.99, 0.965, r"late$-$early $T_2=%+.3f$; $T_1=%+.3f$" % (T2, T1),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color="0.3")
    if title:
        ax.set_title(title, fontsize=8.0)


# =============================================================== FIGURE SUMMARY (two-panel, embedded)
def figure_summary():
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.4),
                             gridspec_kw=dict(height_ratios=[1.06, 1.0]))
    draw_forest(axes[0], title="(a) paired contrasts (post-FT $-$ pruned)")
    draw_finelap(axes[1], title="(b) FineLAP grounding vs. time")
    fig.subplots_adjust(left=0.35, right=0.95, top=0.928, bottom=0.115, hspace=0.55)
    fig.savefig(os.path.join(OUT, "fig_summary.pdf"))
    fig.savefig(os.path.join(OUT, "fig_summary.png"), dpi=200)
    plt.close(fig)


# =============================================================== standalone spares (journal version)
def figure2():
    fig, ax = plt.subplots(figsize=(3.42, 2.62))
    draw_forest(ax)
    fig.subplots_adjust(left=0.40, right=0.98, top=0.99, bottom=0.13)
    fig.savefig(os.path.join(OUT, "fig2_forest.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_forest.png"), dpi=200)
    plt.close(fig)


def figure3():
    fig, ax = plt.subplots(figsize=(3.42, 2.32))
    draw_finelap(ax)
    fig.subplots_adjust(left=0.185, right=0.975, top=0.975, bottom=0.155)
    fig.savefig(os.path.join(OUT, "fig3_finelap.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_finelap.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure_summary()
    figure2()
    figure3()
    print("FIG source values (post-FT - pruned):")
    for k, v in [("sev1 R_short", s1["R_short"]), ("sev1 R_native", s1["R_native"]),
                 ("sev1 J", s1["J"]), ("sev1 R_music", R_music_s1),
                 ("sev2 R_short", s2["R_short"]), ("sev2 R_native", s2["R_native"]),
                 ("sev2 J", s2["J"]), ("sev2 R_music", s2["R_music"]), ("sev2 K", s2["K"])]:
        print(f"  {k:14s} {ci(v)}")
    print(f"  dense@3.84 {dense_short_s1:.4f}  dense@10.24 {dense_native_s1:.4f}")
    print("FineLAP guard PASSED (scores_sha256 + D_early/D_late reproduced).")
    print("wrote fig1_interaction.pdf, fig_summary.pdf (+ standalone fig2/fig3) ->", OUT)
