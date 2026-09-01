#!/usr/bin/env python3
"""Build the two README figures from frozen result artifacts.

Presentation only: this script computes NO new statistic. Every point estimate,
every confidence interval and every group mean plotted here is read verbatim
from a committed, sha-stamped result JSON produced by the corresponding frozen
experiment. If an artifact is missing the script fails loudly rather than
substituting a value.

Sources (all tracked in Git):
  configs/research/reversal_v1_1_result.json            RECOVERY-REVERSAL-V1.1 primary (n=96)
  configs/research/reversal_v1_r_music_clap.json        frozen held-out music baseline (n=64)
  configs/research/op_duration_discriminator_1_result.json  Arm D duration contrast (n=80)
  configs/research/xsev_result.json                     severity-2 replication (n=192 / 64)

Fig. 1  Forest plot of the paired contrast Delta = CLAP(recovered) - CLAP(pruned-only),
        with 95 % prompt-clustered percentile-bootstrap CIs, by pruning severity and
        evaluation context.
Fig. 2  Interaction plot: mean CLAP cosine against clip duration, one line per system,
        one panel per pruning severity, with the paired contrast and the interaction
        contrast J annotated.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/build_readme_figures.py
Out: docs/figures/fig1_paired_contrasts.{png,pdf}
     docs/figures/fig2_duration_interaction.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "research"
OUT = ROOT / "docs" / "figures"

# ICASSP-style: small serif labels, thin rules, no chartjunk, greyscale-safe.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    }
)

C_REC = "#1f4e79"   # recovered
C_PRU = "#b34a00"   # pruned-only
C_DEN = "#555555"   # dense reference


def load(name: str) -> dict:
    path = CFG / name
    if not path.exists():
        raise SystemExit(f"missing frozen artifact: {path}")
    return json.loads(path.read_text())


def ci(d: dict) -> tuple[float, float, float]:
    """(point, lo, hi) from either the {point, lo, hi} or {point, ci95} schema."""
    point = float(d["point"])
    if "ci95" in d:
        lo, hi = float(d["ci95"][0]), float(d["ci95"][1])
    else:
        lo, hi = float(d["lo"]), float(d["hi"])
    return point, lo, hi


def main() -> int:
    v11 = load("reversal_v1_1_result.json")
    music = load("reversal_v1_r_music_clap.json")
    armd = load("op_duration_discriminator_1_result.json")
    xsev = load("xsev_result.json")

    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Fig. 1
    # Forest plot. Rows are ordered bottom-up so the strongest severity sits on top.
    rows = [
        # (label, n, (point, lo, hi), severity group)
        ("Held-out music, 3.84 s", 64, ci(music["R_music"]), 1),
        ("AudioCaps, 3.84 s", 96, ci(v11["PRIMARY"]["R_AC"]), 1),
        ("AudioCaps, 10.24 s (native)", 80, ci(armd["PRIMARY_clap"]["R_alt"]), 1),
        ("Held-out music, 3.84 s", 64, ci(xsev["PRIMARY_A"]["R_music"]), 2),
        ("AudioCaps, 3.84 s", 192, ci(xsev["PRIMARY_A"]["R_short"]), 2),
        ("AudioCaps, 10.24 s (native)", 192, ci(xsev["PRIMARY_A"]["R_native"]), 2),
    ]
    # Draw severity-2 above severity-1 with a gap between the blocks.
    ypos = [0, 1, 2, 3.8, 4.8, 5.8]

    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    ax.axvline(0.0, color="black", lw=0.8, zorder=1)

    for y, (label, n, (p, lo, hi), sev) in zip(ypos, rows):
        color = C_REC if lo > 0 else (C_PRU if hi < 0 else "#666666")
        marker = "o" if lo > 0 or hi < 0 else "o"
        fill = color if (lo > 0 or hi < 0) else "white"
        ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle="butt", zorder=3)
        ax.plot([lo, lo], [y - 0.13, y + 0.13], color=color, lw=1.0, zorder=3)
        ax.plot([hi, hi], [y - 0.13, y + 0.13], color=color, lw=1.0, zorder=3)
        ax.plot([p], [y], marker=marker, ms=4.6, color=color, mfc=fill, mew=1.1, zorder=4)
        ax.text(
            hi + 0.008,
            y,
            f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}",
            va="center",
            ha="left",
            fontsize=7,
            color="#222222",
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-1.05, 6.5)
    ax.set_xlim(-0.20, 0.46)
    ax.set_xlabel(r"$\Delta$ CLAP cosine  (recovered $-$ pruned-only), 95 % CI")
    ax.set_title(
        "Paired recovered-vs-pruned contrast by evaluation context and pruning severity",
        pad=6,
    )

    # Severity block labels + separating rule.
    ax.axhline(3.4, color="#bbbbbb", lw=0.7, ls=(0, (3, 3)))
    bbox = dict(facecolor="white", edgecolor="none", pad=1.0)
    ax.text(-0.195, 6.15, r"severity 2  —  $(1,2,1,1)$", fontsize=7.5, style="italic",
            color="#333333", bbox=bbox, zorder=5)
    ax.text(-0.195, 2.62, r"severity 1  —  $(1,2,3,1)$", fontsize=7.5, style="italic",
            color="#333333", bbox=bbox, zorder=5)

    ax.text(-0.005, -0.85, "favours pruned-only", ha="right", fontsize=7, color="#666666")
    ax.text(0.005, -0.85, "favours recovered", ha="left", fontsize=7, color="#666666")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_paired_contrasts.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ Fig. 2
    m1 = armd["PRIMARY_clap"]["means"]
    m2 = xsev["PRIMARY_A"]["means"]
    dense10 = float(xsev["DENSE_CONTROL"]["C_dense_10s"])

    panels = [
        (
            r"(a) severity 1 — $(1,2,3,1)$, $-65\,\%$ params",
            [m1["pruned_ctrl"], m1["pruned_alt"]],
            [m1["recovered_ctrl"], m1["recovered_alt"]],
            ci(armd["PRIMARY_clap"]["R_ctrl_80"]),
            ci(armd["PRIMARY_clap"]["R_alt"]),
            ci(armd["PRIMARY_clap"]["J"]),
            80,
            dense10,
        ),
        (
            r"(b) severity 2 — $(1,2,1,1)$",
            [m2["pruned_short"], m2["pruned_native"]],
            [m2["rec_short"], m2["rec_native"]],
            ci(xsev["PRIMARY_A"]["R_short"]),
            ci(xsev["PRIMARY_A"]["R_native"]),
            ci(xsev["PRIMARY_A"]["J"]),
            192,
            None,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = [0, 1]
    for ax, (title, pru, rec, r_short, r_nat, jj, n, dense) in zip(axes, panels):
        if dense is not None:
            ax.plot(
                [1], [dense], marker="*", ms=9, color=C_DEN, ls="none",
                label="dense (10.24 s)", zorder=4,
            )
        ax.plot(x, rec, marker="o", ms=4.6, color=C_REC, label="recovered", zorder=3)
        ax.plot(x, pru, marker="s", ms=4.2, color=C_PRU, label="pruned-only", zorder=3)

        # Paired contrast annotated at each operating point, outside the line pair.
        for xi, (p, lo, hi) in ((0, r_short), (1, r_nat)):
            side = -1 if xi == 0 else +1
            resolved = lo > 0 or hi < 0
            ax.annotate(
                "",
                xy=(xi + side * 0.13, rec[xi]),
                xytext=(xi + side * 0.13, pru[xi]),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#444444", shrinkA=0, shrinkB=0),
            )
            ax.text(
                xi + side * 0.20,
                (rec[xi] + pru[xi]) / 2,
                f"$\\Delta$ {p:+.3f}\n[{lo:+.3f}, {hi:+.3f}]" + ("" if resolved else "\n(CI spans 0)"),
                ha="right" if xi == 0 else "left",
                va="center",
                fontsize=6.6,
                color="#333333",
            )

        jp, jlo, jhi = jj
        ax.set_xticks(x)
        ax.set_xticklabels(["3.84 s", "10.24 s\n(training scale)"])
        ax.set_xlim(-1.15, 2.05)
        ax.set_title(title, pad=14)
        ax.text(
            0.5, 1.015,
            f"interaction $J$ = {jp:+.3f} [{jlo:+.3f}, {jhi:+.3f}]   (n={n})",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="#333333",
        )
        ax.grid(axis="y", color="#e8e8e8", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("CLAP cosine (text–audio alignment)")
    axes[0].set_ylim(0.0, 0.40)
    for ax in axes:
        ax.set_xlabel("generated clip duration")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    axes[1].legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    fig.suptitle(
        "Recovery advantage is temporal-scale-conditional (in-domain AudioCaps, DDIM 50, CFG 2.5)",
        fontsize=9, y=1.03,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_duration_interaction.{ext}", bbox_inches="tight")
    plt.close(fig)

    print("wrote:")
    for f in sorted(OUT.iterdir()):
        print("  ", f.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
