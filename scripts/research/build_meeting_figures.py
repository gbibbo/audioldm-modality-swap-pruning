#!/usr/bin/env python3
"""Build the meeting slide: one standard two-panel scientific figure + a status line.

Presentation only, no new science. Sources (all frozen artifacts):
  artifacts/m3_pilot/m3b_saliency.pt              per-channel saliencies (M3B-SALIENCY-RUN)
  artifacts/m3_pilot/m3b_saliency_result.json     Gate B kept-set overlaps
  artifacts/m3_pilot/gateb_null_overlaps.json     1000 calibration half-split overlaps
                                                  (regenerated from gateb_perslot.pt; reproduces
                                                   gateb_null_distribution.json exactly)

Panel (a): channel importance under the audio probe vs the text probe, with the
           criterion control (L1 magnitude) on the same normalised scale.
Panel (b): the Gate-B statistic against its calibration-resampling null.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/build_meeting_figures.py
Out: artifacts/meeting/fig_modality.{pdf,png} and artifacts/meeting/modality_result.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SAL_PT = ROOT / "artifacts/m3_pilot/m3b_saliency.pt"
SAL_JSON = ROOT / "artifacts/m3_pilot/m3b_saliency_result.json"
NULL_OVERLAPS = ROOT / "artifacts/m3_pilot/gateb_null_overlaps.json"
OUTDIR = ROOT / "artifacts/meeting"

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#5b6472"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def build_figure():
    sal = torch.load(SAL_PT, map_location="cpu")["saliency"]
    gb = json.loads(SAL_JSON.read_text())["gate_b"]
    driven = list(gb["per_layer"].keys())
    null = np.asarray(json.loads(NULL_OVERLAPS.read_text())["overlaps"], dtype=float)
    observed = float(gb["weighted_overlap"])

    audio = np.concatenate([sal["S_audio_norm"][n].numpy() for n in driven])
    text = np.concatenate([sal["S_text_norm"][n].numpy() for n in driven])
    l1 = np.concatenate([sal["P0_L1"][n].numpy() for n in driven])

    rho_at = np.mean([spearmanr(sal["S_audio_norm"][n].numpy(),
                                sal["S_text_norm"][n].numpy()).correlation for n in driven])
    rho_al = np.mean([spearmanr(sal["S_audio_norm"][n].numpy(),
                                sal["P0_L1"][n].numpy()).correlation for n in driven])
    p_val = float((null <= observed).mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.05))

    # ---- (a) importance of the same channel under two probes ----
    h_l1 = ax1.scatter(l1 * 1e3, audio * 1e3, s=2.2, c=ORANGE, alpha=.16, lw=0,
                       rasterized=True,
                       label=rf"L1 magnitude, control  ($\bar\rho$ = {rho_al:.2f})")
    h_tx = ax1.scatter(text * 1e3, audio * 1e3, s=2.2, c=BLUE, alpha=.16, lw=0,
                       rasterized=True, label=rf"text probe  ($\bar\rho$ = {rho_at:.2f})")
    lims = [0.22, 12.0]
    ax1.plot(lims, lims, ls="--", lw=0.9, c="0.35", zorder=3)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"channel importance, other criterion ($\times 10^{-3}$)")
    ax1.set_ylabel(r"audio probe ($\times 10^{-3}$)")
    ax1.set_title("(a) Per-channel importance", loc="left", fontweight="bold")
    ax1.grid(alpha=.25, lw=.5)
    ax1.set_xticks([0.3, 1, 3, 10]); ax1.set_yticks([0.3, 1, 3, 10])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    leg = ax1.legend(handles=[h_tx, h_l1], loc="upper left", handletextpad=.2,
                     borderpad=.1, labelspacing=.25, markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1)

    # ---- (b) Gate-B statistic vs its resampling null ----
    ax2.hist(null, bins=26, color="0.82", edgecolor="0.45", lw=.5,
             label="same probe, resampled\ncalibration set (n = 1000)")
    ax2.axvline(observed, color=ORANGE, lw=1.8, zorder=4,
                label=f"audio vs text\n{observed:.3f}  (p = {p_val:.2f})")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.20)
    ax2.set_xlabel("kept-set overlap")
    ax2.set_ylabel("count")
    ax2.set_title("(b) Overlap vs resampling null", loc="left", fontweight="bold")
    ax2.grid(alpha=.25, lw=.5, axis="y")
    ax2.legend(loc="upper left", handlelength=1.1, handletextpad=.5, labelspacing=.7)
    ax2.annotate("hypothesis needed\noverlap $\\leq$ 0.80", xy=(0.985, 0.99),
                 xycoords="axes fraction", fontsize=8, color=GREY,
                 va="top", ha="right")

    fig.tight_layout(w_pad=2.0)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "fig_modality.pdf")
    fig.savefig(OUTDIR / "fig_modality.png")
    plt.close(fig)
    return dict(rho_at=rho_at, rho_al=rho_al, observed=observed, p_val=p_val,
                n_points=audio.size, n_layers=len(driven),
                null_med=float(np.median(null)))


def main() -> int:
    st = build_figure()
    png = (OUTDIR / "fig_modality.png").read_bytes()
    uri = "data:image/png;base64," + base64.b64encode(png).decode()

    page = TEMPLATE.format(
        img=uri,
        rho_at=f"{st['rho_at']:.2f}",
        observed=f"{st['observed']:.3f}",
        null_med=f"{st['null_med']:.3f}",
        p_val=f"{st['p_val']:.2f}",
        n_points=f"{st['n_points']:,}",
        n_layers=st["n_layers"],
    )
    (OUTDIR / "modality_result.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUTDIR}/fig_modality.{{pdf,png}} and modality_result.html")
    print(f"  rho(audio,text) {st['rho_at']:.3f} · rho(audio,L1) {st['rho_al']:.3f}")
    print(f"  overlap {st['observed']:.4f} · null median {st['null_med']:.4f} · p = {st['p_val']:.3f}")
    print(f"  page {(OUTDIR / 'modality_result.html').stat().st_size / 1024:.0f} KB")
    return 0


TEMPLATE = r"""<title>Same Channels, Either Way</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --plane:#f2f4f7; --ink:#0f1219; --ink-2:#454c5b; --ink-muted:#7a8293;
  --rule:rgba(15,18,25,.10); --accent:#2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane:#0a0c10; --ink:#f2f4f8; --ink-2:#b6bdca; --ink-muted:#868e9d;
    --rule:rgba(255,255,255,.12); --accent:#5c9df0;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#0a0c10; --ink:#f2f4f8; --ink-2:#b6bdca; --ink-muted:#868e9d;
  --rule:rgba(255,255,255,.12); --accent:#5c9df0;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  line-height:1.45; -webkit-font-smoothing:antialiased;
}}
.slide {{
  min-height:100vh; min-height:100svh; max-width:1120px; margin:0 auto;
  padding:clamp(28px,4vh,56px) clamp(20px,3.5vw,56px);
  display:flex; flex-direction:column; justify-content:center; gap:clamp(16px,2.6vh,28px);
}}
.eyebrow {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--ink-muted); margin:0;
}}
h1 {{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; margin:0;
  font-size:clamp(28px,4.4vw,48px); line-height:1.06; letter-spacing:-.025em;
  text-wrap:balance; max-width:20ch;
}}
.q {{ margin:0; font-size:clamp(15.5px,1.7vw,19px); color:var(--ink-2); max-width:74ch; }}
.q b {{ color:var(--ink); font-weight:600; }}
.figure {{ background:#fff; border:1px solid var(--rule); border-radius:10px; padding:14px 16px 10px; }}
.figure img {{ display:block; width:100%; height:auto; }}
.cap {{
  margin:12px 2px 0; font-size:12.5px; line-height:1.5; color:#3a4150;
  max-width:none;
}}
.cap b {{ color:#0f1219; font-weight:600; }}
.status {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
  gap:clamp(12px,2vw,26px); border-top:1px solid var(--rule); padding-top:clamp(14px,2.2vh,22px);
}}
.status div {{ display:flex; flex-direction:column; gap:4px; }}
.status .k {{
  font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-muted);
}}
.status .v {{ font-size:clamp(14px,1.4vw,16px); color:var(--ink-2); }}
.status .v b {{ color:var(--ink); font-weight:600; }}
</style>

<section class="slide">
  <p class="eyebrow">AudioLDM &middot; structured pruning &middot; modality-swap hypothesis</p>
  <h1>Audio and text select the same channels to prune</h1>
  <p class="q">Structured pruning ranks channels by importance under a probe. The hypothesis:
  probing with <b>audio</b> instead of <b>text</b> should expose modality-specific structure and
  yield a different, better pruning mask. Measured on the dense AudioLDM-M-Full U-Net at the
  published 65&nbsp;% budget.</p>

  <div class="figure">
    <img src="{img}" alt="Panel a: per-channel importance under the audio probe against the text probe and against L1 magnitude, on log axes with the identity line. Panel b: histogram of the calibration-resampling null for kept-set overlap with the observed audio-versus-text value marked.">
    <p class="cap"><b>Fig. 1.</b> (a) Per-channel importance under the audio probe vs the text probe
    (blue) and vs the L1-magnitude criterion (orange); {n_points} channels over {n_layers} pruned
    layers, identity line dashed. Changing the <em>criterion</em> moves the estimate; changing the
    <em>modality</em> does not. (b) Kept-set overlap between the audio- and text-derived masks
    ({observed}) against the null obtained by resampling the calibration set under a single probe
    (median {null_med}): <b>p&nbsp;=&nbsp;{p_val}</b>.</p>
  </div>

  <div class="status">
    <div>
      <span class="k">Tested</span>
      <span class="v">Two pre-registered gates &mdash; mask ranking <b>and</b> per-modality damage</span>
    </div>
    <div>
      <span class="k">Result</span>
      <span class="v">Hypothesis <b>rejected</b>; both gates fail to separate the modalities</span>
    </div>
    <div>
      <span class="k">Next</span>
      <span class="v">Does a <b>legacy LoRA adapter</b> survive pruning? &mdash; ICASSP 2027</span>
    </div>
  </div>
</section>
"""

if __name__ == "__main__":
    sys.exit(main())
