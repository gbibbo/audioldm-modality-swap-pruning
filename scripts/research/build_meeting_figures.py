#!/usr/bin/env python3
"""Build the one-slide status figure for a group meeting, from the frozen artifacts.

Presentation only: no new science. Re-reads `artifacts/m3_pilot/m3b_saliency_result.json`
(Gate B kept-set overlaps) and `gateb_null_distribution.json` (the calibration-noise null),
and renders a single slide: the question, one picture, the answer, and where the work stands.

Run: .venv/bin/python scripts/research/build_meeting_figures.py
Out: artifacts/meeting/modality_result.html  (gitignored)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAL_JSON = ROOT / "artifacts/m3_pilot/m3b_saliency_result.json"
NULL_JSON = ROOT / "artifacts/m3_pilot/gateb_null_distribution.json"
OUT = ROOT / "artifacts/meeting/modality_result.html"

COLS, ROWS = 16, 12          # 16 x 12 = 192 kept channels in a pruned layer
PITCH, R = 20, 5.5


def dot_grid(n_diff: int, seed: int) -> str:
    """192 dots = the channels a pruned layer keeps; `n_diff` of them are the ones
    the two criteria disagree about. Positions are cosmetic and deterministic."""
    total = COLS * ROWS
    rng = np.random.default_rng(seed)
    diff = set(rng.choice(total, size=n_diff, replace=False).tolist())
    w, h = COLS * PITCH, ROWS * PITCH
    p = [f'<svg viewBox="0 0 {w} {h}" class="grid" role="img" aria-label='
         f'"{total} kept channels, of which {n_diff} differ between the two criteria.">']
    for i in range(total):
        cx = (i % COLS) * PITCH + PITCH / 2
        cy = (i // COLS) * PITCH + PITCH / 2
        cls = "d on" if i in diff else "d off"
        p.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R}" class="{cls}"/>')
    p.append("</svg>")
    return "\n".join(p)


def main() -> int:
    gb = json.loads(SAL_JSON.read_text())["gate_b"]
    null = json.loads(NULL_JSON.read_text())

    per_layer = gb["per_layer"]
    k = int(np.median([v["k"] for v in per_layer.values()]))
    n = int(np.median([v["n"] for v in per_layer.values()]))
    assert k == COLS * ROWS, f"grid is drawn for k={COLS * ROWS}, artifact says k={k}"

    observed = float(gb["weighted_overlap"])
    null_med = float(null["null_median"])
    diff_obs = int(round(k * (1 - observed)))
    diff_null = int(round(k * (1 - null_med)))

    page = TEMPLATE.format(
        grid_a=dot_grid(diff_obs, 20260818),
        grid_b=dot_grid(diff_null, 20260820),
        k=k, n=n,
        diff_obs=diff_obs,
        diff_null=diff_null,
        overlap_pct=f"{observed * 100:.1f}",
        null_pct=f"{null_med * 100:.1f}",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  kept {k} of {n} channels/layer - differs: modality {diff_obs}, noise {diff_null}")
    print(f"  overlap {observed:.4f} - calibration-noise median {null_med:.4f}")
    return 0


TEMPLATE = r"""<title>Same Channels, Either Way</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --plane:#f2f4f7; --surface:#fcfdfe;
  --ink:#0f1219; --ink-2:#454c5b; --ink-muted:#7a8293;
  --rule:rgba(15,18,25,.10);
  --on:#2a78d6; --off:#ccd2dc; --accent:#2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane:#0a0c10; --surface:#14161c; --ink:#f2f4f8; --ink-2:#b6bdca;
    --ink-muted:#868e9d; --rule:rgba(255,255,255,.12);
    --on:#4a92ea; --off:#333944; --accent:#5c9df0;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#0a0c10; --surface:#14161c; --ink:#f2f4f8; --ink-2:#b6bdca;
  --ink-muted:#868e9d; --rule:rgba(255,255,255,.12);
  --on:#4a92ea; --off:#333944; --accent:#5c9df0;
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  line-height:1.45; -webkit-font-smoothing:antialiased;
}}
.slide {{
  min-height:100vh; min-height:100svh; max-width:1200px; margin:0 auto;
  padding:clamp(32px,4vh,64px) clamp(24px,4vw,64px);
  display:flex; flex-direction:column; justify-content:center; gap:clamp(20px,3.2vh,36px);
}}
.eyebrow {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--ink-muted); margin:0;
}}
h1 {{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; margin:0;
  font-size:clamp(32px,5.4vw,60px); line-height:1.05; letter-spacing:-.025em;
  text-wrap:balance; max-width:18ch;
}}
.q {{ margin:0; font-size:clamp(17px,1.9vw,22px); color:var(--ink-2); max-width:62ch; }}
.q b {{ color:var(--ink); font-weight:600; }}

.panels {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:clamp(16px,2.4vw,32px); align-items:start;
}}
.panel {{
  background:var(--surface); border:1px solid var(--rule); border-radius:14px;
  padding:clamp(18px,2.4vw,26px); display:flex; flex-direction:column; gap:14px;
}}
.panel h2 {{
  margin:0; font-size:clamp(15px,1.5vw,17px); font-weight:600; letter-spacing:-.005em;
}}
.panel .count {{ margin:0; font-size:clamp(15px,1.6vw,18px); color:var(--ink-2); }}
.panel .count b {{ color:var(--on); font-weight:600; font-size:1.35em; letter-spacing:-.02em; }}
svg.grid {{ display:block; width:100%; max-width:420px; height:auto; }}
.d.off {{ fill:var(--off); }}
.d.on {{ fill:var(--on); }}

.answer {{ border-top:1px solid var(--rule); padding-top:clamp(18px,2.6vh,28px); }}
.answer p {{ margin:0; font-size:clamp(18px,2.1vw,25px); color:var(--ink); max-width:58ch; line-height:1.3; }}
.answer p b {{ font-weight:600; }}

.status {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:clamp(14px,2vw,28px); }}
.status div {{ display:flex; flex-direction:column; gap:5px; }}
.status .k {{
  font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-muted);
}}
.status .v {{ font-size:clamp(14.5px,1.5vw,16.5px); color:var(--ink-2); }}
.status .v b {{ color:var(--ink); font-weight:600; }}
</style>

<section class="slide">
  <div style="display:flex;flex-direction:column;gap:14px">
    <p class="eyebrow">AudioLDM &middot; structured pruning &middot; modality-swap hypothesis</p>
    <h1>Audio or text, the model wants the same channels cut</h1>
    <p class="q">To shrink a text-to-audio diffusion model you delete channels. The hypothesis:
    <b>which</b> channels to delete depends on whether you probe the model with audio or with
    text. We measured it.</p>
  </div>

  <div class="panels">
    <div class="panel">
      <h2>Probe with audio vs probe with text</h2>
      {grid_a}
      <p class="count"><b>{diff_obs}</b> of {k} kept channels differ</p>
    </div>
    <div class="panel">
      <h2>Same probe, different calibration sample</h2>
      {grid_b}
      <p class="count"><b>{diff_null}</b> of {k} kept channels differ</p>
    </div>
  </div>

  <div class="answer">
    <p>Switching modality changes the pruning decision <b>no more than resampling the data
    does</b> &mdash; {overlap_pct}% agreement, against {null_pct}% from noise alone.</p>
  </div>

  <div class="status">
    <div>
      <span class="k">Tested</span>
      <span class="v">Two pre-registered gates on the dense model &mdash; ranking <b>and</b> damage</span>
    </div>
    <div>
      <span class="k">Result</span>
      <span class="v">Hypothesis <b>rejected</b>; neither gate separates the modalities</span>
    </div>
    <div>
      <span class="k">Next</span>
      <span class="v">Can a <b>legacy LoRA adapter</b> survive pruning? &mdash; ICASSP 2027</span>
    </div>
  </div>
</section>
"""

if __name__ == "__main__":
    sys.exit(main())
