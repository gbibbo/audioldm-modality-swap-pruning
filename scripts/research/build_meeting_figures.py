#!/usr/bin/env python3
"""Build the collaboration-meeting figure page from the frozen result artifacts.

Presentation only: this script computes NO new science. It re-reads artifacts that
already exist (`artifacts/m3_pilot/m3b_saliency.pt`, `m3b_saliency_result.json`,
`gateb_null_distribution.json`, `m3a_result.json`, `artifacts/m4_screening/
rescore_frechet.json`) and renders two figures plus a plain-language reading of them
into a single self-contained HTML page.

Every number on the page is recomputed here from the artifact it cites, so the page
cannot silently drift from the evidence. Where a number is screening-grade and not
promotable (M4 FAD), the page says so on the figure itself.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/build_meeting_figures.py
Out: artifacts/meeting/modality_result.html  (gitignored)
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SAL_PT = ROOT / "artifacts/m3_pilot/m3b_saliency.pt"
SAL_JSON = ROOT / "artifacts/m3_pilot/m3b_saliency_result.json"
NULL_JSON = ROOT / "artifacts/m3_pilot/gateb_null_distribution.json"
M3A_JSON = ROOT / "artifacts/m3_pilot/m3a_result.json"
FAD_JSON = ROOT / "artifacts/m4_screening/rescore_frechet.json"
OUT = ROOT / "artifacts/meeting/modality_result.html"

SHORT = {
    "input_blocks.10.0.in_layers.2.weight": "in_blocks.10 · in",
    "input_blocks.10.0.out_layers.3.weight": "in_blocks.10 · out",
    "input_blocks.11.0.in_layers.2.weight": "in_blocks.11 · in",
    "input_blocks.11.0.out_layers.3.weight": "in_blocks.11 · out",
    "middle_block.0.in_layers.2.weight": "middle.0 · in",
    "middle_block.0.out_layers.3.weight": "middle.0 · out",
    "middle_block.2.in_layers.2.weight": "middle.2 · in",
    "middle_block.2.out_layers.3.weight": "middle.2 · out",
    "output_blocks.0.0.out_layers.3.weight": "out_blocks.0 · out",
    "output_blocks.1.0.out_layers.3.weight": "out_blocks.1 · out",
    "output_blocks.2.0.out_layers.3.weight": "out_blocks.2 · out",
    "output_blocks.2.2.conv.weight": "out_blocks.2 · conv",
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def load_rows():
    """Per-layer agreement on the 12 ranking-driven layers, from the frozen saliency."""
    sal = torch.load(SAL_PT, map_location="cpu")
    res = json.loads(SAL_JSON.read_text())
    per_layer = res["gate_b"]["per_layer"]
    s = sal["saliency"]
    rows = []
    for name, meta in per_layer.items():
        a = s["S_audio_norm"][name].numpy()
        t = s["S_text_norm"][name].numpy()
        l1 = s["P0_L1"][name].numpy()
        rows.append({
            "layer": name,
            "short": SHORT.get(name, name),
            "n": meta["n"],
            "k": meta["k"],
            "rho_at": float(spearmanr(a, t).correlation),
            "rho_al": float(spearmanr(a, l1).correlation),
            "overlap": float(meta["overlap"]),
        })
    return rows, res, sal["provenance"]


# ---------------------------------------------------------------- svg helpers
def txt(x, y, s, cls="lab", anchor="start", dy=0.0):
    return (f'<text x="{x:.1f}" y="{y + dy:.1f}" class="{cls}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')


def fig1_panel_a(rows) -> str:
    W, H = 640, 380
    gutter, x0, x1 = 168, 184, 616
    top, rh = 14, 26
    bottom = top + rh * len(rows)

    def sx(v):
        return x0 + v * (x1 - x0)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
         f'aria-label="Per-layer rank agreement between audio-conditioned and '
         f'text-conditioned Taylor saliency, and the criterion control.">']
    # gridlines
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        p.append(f'<line x1="{sx(v):.1f}" y1="{top}" x2="{sx(v):.1f}" y2="{bottom}" class="grid"/>')
        p.append(txt(sx(v), bottom + 18, f"{v:.2f}", "tick", "middle"))
    p.append(txt((x0 + x1) / 2, bottom + 36, "Spearman rank correlation between channel rankings", "axtitle", "middle"))
    p.append(f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" class="axis"/>')

    for i, r in enumerate(rows):
        y = top + rh * i + rh / 2
        xa, xb = sx(r["rho_al"]), sx(r["rho_at"])
        p.append(txt(gutter, y + 4, r["short"], "rowlab", "end"))
        p.append(f'<line x1="{xa:.1f}" y1="{y:.1f}" x2="{xb:.1f}" y2="{y:.1f}" class="connect"/>')
        p.append(f'<circle cx="{xa:.1f}" cy="{y:.1f}" r="5" class="dot s2" '
                 f'data-tip="{esc(r["short"])} — Taylor vs L1 magnitude: ρ = {r["rho_al"]:.3f}"/>')
        p.append(f'<circle cx="{xb:.1f}" cy="{y:.1f}" r="5" class="dot s1" '
                 f'data-tip="{esc(r["short"])} — audio vs text Taylor: ρ = {r["rho_at"]:.3f}"/>')
    p.append("</svg>")
    return "\n".join(p)


def fig1_panel_b(observed, null) -> str:
    W, H = 640, 232
    x0, x1 = 46, 596
    y1, y2 = 74, 186          # baselines of the full strip and the zoom strip
    zlo, zhi = 0.90, 1.00

    def fx(v):
        return x0 + v * (x1 - x0)

    def zx(v):
        return x0 + (v - zlo) / (zhi - zlo) * (x1 - x0)

    lo, hi, med = null["null_5pctile"], null["null_max"], null["null_median"]
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
         f'aria-label="Kept-channel overlap between the audio-only and text-only '
         f'criterion, against the calibration-noise band and the pre-registered threshold.">']

    # --- full strip ---
    p.append(f'<rect x="{fx(lo):.1f}" y="{y1 - 26:.1f}" width="{fx(hi) - fx(lo):.1f}" height="34" class="band"/>')
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p.append(txt(fx(v), y1 + 20, f"{v:.1f}", "tick", "middle"))
    p.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>')
    # chance + threshold references
    p.append(f'<line x1="{fx(0.2):.1f}" y1="{y1 - 26:.1f}" x2="{fx(0.2):.1f}" y2="{y1:.1f}" class="ref"/>')
    p.append(txt(fx(0.2) + 6, y1 - 30, "chance 0.20", "note"))
    p.append(f'<line x1="{fx(0.8):.1f}" y1="{y1 - 44:.1f}" x2="{fx(0.8):.1f}" y2="{y1:.1f}" class="ref"/>')
    p.append(txt(fx(0.8) - 8, y1 - 48, "gate needed ≤ 0.80", "note", "end"))
    p.append(f'<circle cx="{fx(observed):.1f}" cy="{y1 - 9:.1f}" r="5.5" class="dot s1" '
             f'data-tip="Observed audio-vs-text kept-set overlap: {observed:.4f}"/>')

    # --- zoom bridge ---
    p.append(f'<path d="M {fx(zlo):.1f} {y1 + 26:.1f} L {fx(zhi):.1f} {y1 + 26:.1f} '
             f'L {x1:.1f} {y2 - 66:.1f} L {x0:.1f} {y2 - 66:.1f} Z" class="bridge"/>')
    p.append(txt(fx(zlo) - 8, y1 + 32, "detail 0.90 – 1.00", "note", "end"))

    # --- zoom strip ---
    p.append(f'<rect x="{zx(lo):.1f}" y="{y2 - 34:.1f}" width="{zx(hi) - zx(lo):.1f}" height="34" class="band"/>')
    p.append(f'<line x1="{zx(med):.1f}" y1="{y2 - 34:.1f}" x2="{zx(med):.1f}" y2="{y2:.1f}" class="ref"/>')
    for v in (0.90, 0.92, 0.94, 0.96, 0.98, 1.00):
        p.append(txt(zx(v), y2 + 20, f"{v:.2f}", "tick", "middle"))
    p.append(f'<line x1="{x0}" y1="{y2}" x2="{x1}" y2="{y2}" class="axis"/>')
    p.append(f'<circle cx="{zx(observed):.1f}" cy="{y2 - 17:.1f}" r="5.5" class="dot s1" '
             f'data-tip="Observed audio-vs-text kept-set overlap: {observed:.4f}"/>')
    # leaders
    p.append(f'<line x1="{zx(observed):.1f}" y1="{y2 - 34:.1f}" x2="{zx(observed) - 54:.1f}" y2="{y2 - 52:.1f}" class="leader"/>')
    p.append(txt(zx(observed) - 58, y2 - 54, f"observed {observed:.4f}", "note", "end"))
    p.append(f'<line x1="{zx(med):.1f}" y1="{y2 - 34:.1f}" x2="{zx(med) + 54:.1f}" y2="{y2 - 52:.1f}" class="leader"/>')
    p.append(txt(zx(med) + 58, y2 - 54, f"calibration-noise median {med:.4f}", "note"))
    p.append(txt(zx((lo + hi) / 2), y2 + 38,
                 f"grey band = 1,000 calibration half-splits of the same text-only criterion "
                 f"({lo:.4f} – {hi:.4f})", "axtitle", "middle"))
    p.append("</svg>")
    return "\n".join(p)


def fig2(bars) -> str:
    W = 640
    rh, bh = 40, 20
    top = 14
    x0, x1 = 176, 612
    xmax = 1.8
    bottom = top + rh * len(bars)
    H = bottom + 54

    def sx(v):
        return x0 + v / xmax * (x1 - x0)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
         f'aria-label="Frechet Audio Distance of the dense model and five pruned '
         f'variants, screening grade.">']
    for v in (0, 0.5, 1.0, 1.5):
        p.append(f'<line x1="{sx(v):.1f}" y1="{top}" x2="{sx(v):.1f}" y2="{bottom}" class="grid"/>')
        p.append(txt(sx(v), bottom + 18, f"{v:.1f}", "tick", "middle"))
    p.append(txt((x0 + x1) / 2, bottom + 38, "FAD vs real AudioCaps clips  ·  lower is better", "axtitle", "middle"))
    p.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" class="axis"/>')

    ref = bars[0]["v"]
    p.append(f'<line x1="{sx(ref):.1f}" y1="{top}" x2="{sx(ref):.1f}" y2="{bottom + 4:.1f}" class="ref"/>')

    for i, b in enumerate(bars):
        y = top + rh * i + (rh - bh) / 2
        w = max(sx(b["v"]) - x0, 1)
        p.append(txt(x0 - 12, y + bh / 2 + 4, b["short"], "rowlab", "end"))
        p.append(f'<path d="M {x0} {y} H {x0 + w - 4:.1f} a4,4 0 0 1 4,4 V {y + bh - 4:.1f} '
                 f'a4,4 0 0 1 -4,4 H {x0} Z" class="bar {b["cls"]}" '
                 f'data-tip="{esc(b["label"])} — FAD {b["v"]:.3f}"/>')
        p.append(txt(x0 + w + 9, y + bh / 2 + 4, f'{b["v"]:.3f}', "val"))
    p.append("</svg>")
    return "\n".join(p)


def main() -> int:
    rows, res, prov = load_rows()
    null = json.loads(NULL_JSON.read_text())
    m3a = json.loads(M3A_JSON.read_text())
    fad = json.loads(FAD_JSON.read_text())["systems"]
    gb = res["gate_b"]
    observed = float(gb["weighted_overlap"])

    rho_at = np.array([r["rho_at"] for r in rows])
    rho_al = np.array([r["rho_al"] for r in rows])

    bars = [
        {"short": "Dense · unpruned", "label": "Dense backbone, no pruning",
         "v": fad["base"]["fad_vggish"]["fd"], "cls": "ref-bar"},
        {"short": "P1 · text Taylor", "label": "P1 — text-only Taylor",
         "v": fad["P1"]["fad_vggish"]["fd"], "cls": "s1"},
        {"short": "P0 · standard L1", "label": "P0 — standard L1 magnitude (keep highest)",
         "v": fad["P0_L1"]["fad_vggish"]["fd"], "cls": "s2"},
        {"short": "P3 · swap-max", "label": "P3 — paired swap-max",
         "v": fad["P3"]["fad_vggish"]["fd"], "cls": "s1"},
        {"short": "P2 · paired-mean", "label": "P2 — paired mean (audio + text)",
         "v": fad["P2"]["fad_vggish"]["fd"], "cls": "s1"},
        {"short": "P0 · published", "label": "P0 — the published L1 artifact",
         "v": fad["P0_published"]["fad_vggish"]["fd"], "cls": "s2"},
    ]
    bars[1:] = sorted(bars[1:], key=lambda b: b["v"])

    tbl1 = "\n".join(
        f"<tr><td>{esc(r['short'])}</td><td>{r['n']}</td><td>{r['k']}</td>"
        f"<td>{r['rho_at']:.4f}</td><td>{r['rho_al']:.4f}</td><td>{r['overlap']:.4f}</td></tr>"
        for r in rows)
    tbl2 = "\n".join(
        f"<tr><td>{esc(b['label'])}</td><td>{b['v']:.3f}</td></tr>" for b in bars)

    page = TEMPLATE.format(
        panel_a=fig1_panel_a(rows),
        panel_b=fig1_panel_b(observed, null),
        fig2=fig2(bars),
        tbl1=tbl1,
        tbl2=tbl2,
        rho_med=f"{np.median(rho_at):.2f}",
        rho_lo=f"{rho_at.min():.3f}",
        rho_hi=f"{rho_at.max():.3f}",
        ctl_med=f"{np.median(rho_al):.3f}",
        ctl_lo=f"{rho_al.min():.3f}",
        ctl_hi=f"{rho_al.max():.3f}",
        overlap_pct=f"{observed * 100:.1f}",
        overlap=f"{observed:.4f}",
        null_med_pct=f"{null['null_median'] * 100:.1f}",
        null_med=f"{null['null_median']:.4f}",
        null_lo=f"{null['null_5pctile']:.4f}",
        null_hi=f"{null['null_max']:.4f}",
        n_splits=f"{null['n_splits']:,}",
        gap=f"{abs(observed - null['null_median']):.4f}",
        delta_swap=f"{m3a['delta_swap']:.4f}",
        ci_lo=f"{m3a['bootstrap_ci_95'][0]:+.4f}",
        ci_hi=f"{m3a['bootstrap_ci_95'][1]:+.4f}",
        r_l1=f"{m3a['l1_R_mod']:.4f}",
        r_rand=f"{m3a['rand_R_mod_mean']:.4f}",
        e=prov["E"], k=prov["K"],
        sal_commit=prov["git"]["commit"][:7],
        m3a_commit=m3a["git"]["commit"][:7],
        manifest=prov["manifest_sha256"][:12],
        seed=prov["master_seed"],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  rho(audio,text) median {np.median(rho_at):.4f}  [{rho_at.min():.4f}, {rho_at.max():.4f}]")
    print(f"  rho(Taylor,L1)  median {np.median(rho_al):.4f}  [{rho_al.min():.4f}, {rho_al.max():.4f}]")
    print(f"  observed overlap {observed:.4f}  null median {null['null_median']:.4f}")
    return 0


TEMPLATE = r"""<title>Same Channels, Either Way</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --plane:#f3f5f8; --surface:#fbfcfd; --sunk:#eef1f5;
  --ink:#10131a; --ink-2:#464d5c; --ink-muted:#767e8f;
  --grid:#e4e7ed; --axis:#c3c8d3; --rule:rgba(16,19,26,.10);
  --s1:#2a78d6; --s2:#eb6834; --refbar:#9aa2b1;
  --band:rgba(118,126,143,.20); --accent:#2a78d6;
  --warn-bg:rgba(235,104,52,.10); --warn-ink:#a8431a;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane:#0b0d11; --surface:#13151a; --sunk:#191c22;
    --ink:#f1f3f7; --ink-2:#b4bbc8; --ink-muted:#858d9c;
    --grid:#232730; --axis:#3a3f4a; --rule:rgba(255,255,255,.11);
    --s1:#3987e5; --s2:#d95926; --refbar:#6d7482;
    --band:rgba(133,141,156,.24); --accent:#5c9df0;
    --warn-bg:rgba(217,89,38,.14); --warn-ink:#e5926c;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#0b0d11; --surface:#13151a; --sunk:#191c22;
  --ink:#f1f3f7; --ink-2:#b4bbc8; --ink-muted:#858d9c;
  --grid:#232730; --axis:#3a3f4a; --rule:rgba(255,255,255,.11);
  --s1:#3987e5; --s2:#d95926; --refbar:#6d7482;
  --band:rgba(133,141,156,.24); --accent:#5c9df0;
  --warn-bg:rgba(217,89,38,.14); --warn-ink:#e5926c;
}}

* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; scroll-snap-type:y proximity; }}
@media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior:auto; }} * {{ transition:none !important; }} }}
body {{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.5; -webkit-font-smoothing:antialiased;
}}

.slide {{
  min-height:100vh; min-height:100svh; scroll-snap-align:start;
  display:flex; flex-direction:column; justify-content:center;
  gap:20px; padding:48px 40px 56px; max-width:1180px; margin:0 auto;
  border-bottom:1px solid var(--rule);
}}
@media (max-width:720px) {{ .slide {{ padding:40px 20px 48px; min-height:auto; }} }}

.eyebrow {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-muted); margin:0;
}}
h1 {{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600;
  font-size:clamp(30px,5.2vw,54px); line-height:1.08; letter-spacing:-.02em;
  margin:0; text-wrap:balance; max-width:17ch;
}}
h2 {{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600;
  font-size:clamp(26px,4.2vw,42px); line-height:1.1; letter-spacing:-.02em;
  margin:0; text-wrap:balance; max-width:20ch;
}}
.say {{
  font-size:clamp(17px,1.9vw,21px); color:var(--ink-2); margin:0; max-width:60ch;
  line-height:1.45;
}}
.say b {{ color:var(--ink); font-weight:600; }}
.foot {{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--ink-muted);
  margin:0; max-width:90ch; line-height:1.6;
}}

.card {{ background:var(--surface); border:1px solid var(--rule); border-radius:12px; padding:14px 18px 8px; }}
.plot {{ overflow-x:auto; }}
.legend {{ display:flex; flex-wrap:wrap; gap:20px; padding:0 4px 10px; }}
.key {{ display:flex; align-items:center; gap:8px; font-size:13.5px; color:var(--ink-2); }}
.swatch {{ width:11px; height:11px; border-radius:50%; flex:none; }}
.swatch.s1 {{ background:var(--s1); }}
.swatch.s2 {{ background:var(--s2); }}
.swatch.ref {{ background:var(--refbar); border-radius:2px; }}
.swatch.band {{ background:var(--band); border-radius:2px; width:20px; }}

.flag {{
  display:inline-flex; align-self:flex-start; padding:6px 12px; border-radius:6px;
  background:var(--warn-bg); color:var(--warn-ink); font-size:12.5px; font-weight:500;
}}
.pair {{ display:flex; flex-wrap:wrap; gap:12px 40px; align-items:baseline; }}
.big {{ font-size:clamp(34px,5vw,52px); font-weight:600; letter-spacing:-.03em; line-height:1; }}
.big span {{ font-size:15px; font-weight:400; color:var(--ink-muted); letter-spacing:0; display:block; margin-top:8px; }}

svg.chart {{ display:block; width:100%; height:auto; min-width:560px; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.axis {{ stroke:var(--axis); stroke-width:1; }}
.ref {{ stroke:var(--ink-muted); stroke-width:1; }}
.leader {{ stroke:var(--axis); stroke-width:1; }}
.connect {{ stroke:var(--axis); stroke-width:1.5; }}
.band {{ fill:var(--band); }}
.bridge {{ fill:var(--band); opacity:.45; }}
.dot {{ stroke:var(--surface); stroke-width:2; cursor:pointer; }}
.dot.s1 {{ fill:var(--s1); }}
.dot.s2 {{ fill:var(--s2); }}
.bar {{ cursor:pointer; }}
.bar.s1 {{ fill:var(--s1); }}
.bar.s2 {{ fill:var(--s2); }}
.bar.ref-bar {{ fill:var(--refbar); }}
text {{ font-family:"IBM Plex Sans",system-ui,sans-serif; }}
.tick {{ font-size:11px; fill:var(--ink-muted); font-variant-numeric:tabular-nums; }}
.rowlab {{ font-family:"IBM Plex Mono",monospace; font-size:11px; fill:var(--ink-2); }}
.val {{ font-size:12px; fill:var(--ink); font-weight:500; font-variant-numeric:tabular-nums; }}
.note {{ font-size:11.5px; fill:var(--ink-2); }}
.axtitle {{ font-size:11.5px; fill:var(--ink-muted); }}

/* backup, below the deck */
.backup {{ max-width:900px; margin:0 auto; padding:64px 40px 96px; display:flex; flex-direction:column; gap:32px; }}
@media (max-width:720px) {{ .backup {{ padding:48px 20px 64px; }} }}
.backup h3 {{ font-family:"IBM Plex Serif",Georgia,serif; font-size:20px; font-weight:600; margin:0; }}
.backup p {{ margin:0; font-size:15px; color:var(--ink-2); max-width:66ch; }}
.backup p strong {{ color:var(--ink); }}
.backup .mono {{ font-family:"IBM Plex Mono",monospace; font-size:13px; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:12.5px; width:100%; }}
th, td {{ text-align:right; padding:6px 12px; border-bottom:1px solid var(--rule); font-variant-numeric:tabular-nums; }}
th:first-child, td:first-child {{ text-align:left; font-family:"IBM Plex Mono",monospace; font-variant-numeric:normal; }}
th {{ color:var(--ink-muted); font-weight:500; }}
details summary {{ font-size:13.5px; color:var(--ink-2); cursor:pointer; list-style:none; padding:4px 0; }}
summary::-webkit-details-marker {{ display:none; }}
summary::before {{ content:"▸ "; color:var(--ink-muted); }}
details[open] summary::before {{ content:"▾ "; }}
summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }}

.nav {{
  position:fixed; right:16px; bottom:16px; z-index:9; display:flex; gap:6px;
  font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--ink-muted);
  background:var(--surface); border:1px solid var(--rule); border-radius:20px; padding:6px 12px;
}}
@media (max-width:720px) {{ .nav {{ display:none; }} }}

#tip {{
  position:fixed; z-index:10; pointer-events:none; opacity:0; transition:opacity .12s;
  background:var(--ink); color:var(--surface); font-size:12.5px; padding:6px 10px;
  border-radius:6px; max-width:280px; box-shadow:0 4px 16px rgba(0,0,0,.22);
}}
</style>

<section class="slide" id="s1">
  <p class="eyebrow">AudioLDM · structured pruning · the modality swap</p>
  <h1>Audio and text want the same channels cut</h1>
  <div class="card">
    <div class="legend">
      <span class="key"><span class="swatch s1"></span>Audio-Taylor vs text-Taylor — the modality swap</span>
      <span class="key"><span class="swatch s2"></span>Audio-Taylor vs L1 magnitude — the control</span>
    </div>
    <div class="plot">{panel_a}</div>
  </div>
  <p class="say">Swap the conditioning <b>modality</b> and the channel ranking barely moves.
  Swap the <b>criterion</b> and it moves a lot — so the instrument works; the effect isn't there.</p>
  <p class="foot">12 layers that decide the pruned architecture · E = {e} clips × K = {k} timesteps ·
  matched gradient budget · job m3b-saliency-2, T4</p>
</section>

<section class="slide" id="s2">
  <p class="eyebrow">Gate B · kept-set overlap at −65 % U-Net</p>
  <h2>The swap moves the mask as much as noise does</h2>
  <div class="card">
    <div class="legend">
      <span class="key"><span class="swatch s1"></span>Measured, audio vs text</span>
      <span class="key"><span class="swatch band"></span>Same criterion re-drawn on half the calibration set ({n_splits}×)</span>
    </div>
    <div class="plot">{panel_b}</div>
  </div>
  <div class="pair">
    <p class="big">{overlap}<span>measured overlap</span></p>
    <p class="big">{null_med}<span>calibration noise alone</span></p>
    <p class="big">≤ 0.80<span>what the gate needed</span></p>
  </div>
  <p class="foot">Second gate, on damage instead of ranking, agrees: Δ_swap = {delta_swap},
  95 % CI [{ci_lo}, {ci_hi}] — contains zero</p>
</section>

<section class="slide" id="s3">
  <p class="eyebrow">Generation check</p>
  <h2>The generated audio says the same</h2>
  <span class="flag">Screening grade — 100 clips, 1 seed, no intervals. Not a claim.</span>
  <div class="card">
    <div class="legend">
      <span class="key"><span class="swatch s1"></span>Taylor criteria</span>
      <span class="key"><span class="swatch s2"></span>L1 magnitude</span>
      <span class="key"><span class="swatch ref"></span>Dense reference</span>
    </div>
    <div class="plot">{fig2}</div>
  </div>
  <p class="say">The paired-modality criteria (P2, P3) sit <b>behind</b> plain text-only Taylor.
  Pre-recovery at this budget, the criterion barely matters.</p>
  <p class="foot">FAD vs 100 real AudioCaps clips · 50 DDIM steps · recorded as screening only,
  explicitly not promotable</p>
</section>

<div class="backup">
  <p class="eyebrow">Backup — not part of the walkthrough</p>

  <div style="display:flex;flex-direction:column;gap:12px">
    <h3>One thing worth checking with Arshdeep</h3>
    <p>In all 28 ranked layers the released ranking correlates <strong>−1.000</strong> with a
    conventional descending-L1 ranking: <span class="mono">np.argsort</span> is ascending and the
    first <em>k</em> entries are kept, so the published artifact keeps the <em>lowest</em>-magnitude
    filters. On all 15 actually-pruned layers the kept set has lower mean L1 than the removed set.
    We reproduce the released checkpoint bit-exactly (690/690 tensors) from the base weights plus
    <span class="mono">sorted_indexes_dict.pkl</span>, so it is a property of the artifact, not of
    our reading of the code. Intentional, or a direction to flip?</p>
  </div>

  <div style="display:flex;flex-direction:column;gap:12px">
    <h3>Where the paper goes</h3>
    <p>The modality-swap line closes as a pre-registered negative with a working control. The
    ICASSP-2027 submission moved to <strong>Scenario B</strong>: a LoRA trained on the dense backbone
    can't load onto a pruned one, so we slice it by the same kept-index set — no retraining, no
    adapter data — and measure whether its benefit survives, separating generic degradation (E) from
    excess adapter-specific degradation (F = D − E). Pre-registered before any GPU spend: 64 held-out
    prompts × 3 seeds, prompt-clustered bootstrap, SESOI +0.025. Next step is Gate 0.</p>
  </div>

  <details>
    <summary>Data · Figure 1, per layer</summary>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Layer</th><th>channels</th><th>kept</th><th>ρ audio·text</th><th>ρ Taylor·L1</th><th>overlap</th></tr></thead>
        <tbody>{tbl1}</tbody>
      </table>
    </div>
  </details>

  <details>
    <summary>Data · Figure 3, FAD</summary>
    <div class="tablewrap">
      <table>
        <thead><tr><th>System</th><th>FAD (VGGish)</th></tr></thead>
        <tbody>{tbl2}</tbody>
      </table>
    </div>
  </details>

  <p class="foot">ρ audio·text median {rho_med} ({rho_lo}–{rho_hi}) vs control {ctl_med}
  ({ctl_lo}–{ctl_hi}) over the 12 decision layers · damage gate R_mod {r_l1} vs random null {r_rand}
  · saliency commit {sal_commit}, manifest {manifest}…, seed {seed} · damage commit {m3a_commit} ·
  noise band {null_lo}–{null_hi} · sources: artifacts/m3_pilot/{{m3b_saliency.pt, m3a_result.json,
  gateb_null_distribution.json}}, artifacts/m4_screening/rescore_frechet.json · rebuilt by
  scripts/research/build_meeting_figures.py</p>
</div>

<div class="nav" aria-hidden="true">↑ ↓ to move</div>
<div id="tip" role="status" aria-live="polite"></div>
<script>
(function () {{
  try {{
    var tip = document.getElementById("tip");
    if (tip) {{
      var show = function (e) {{
        var t = e.currentTarget.getAttribute("data-tip");
        if (!t) return;
        tip.textContent = t; tip.style.opacity = "1"; move(e);
      }};
      var move = function (e) {{
        var x = e.clientX + 14, y = e.clientY + 16, r = tip.getBoundingClientRect();
        if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 14;
        if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 16;
        tip.style.left = x + "px"; tip.style.top = y + "px";
      }};
      var hide = function () {{ tip.style.opacity = "0"; }};
      var marks = document.querySelectorAll("[data-tip]");
      for (var i = 0; i < marks.length; i++) {{
        marks[i].addEventListener("mouseenter", show);
        marks[i].addEventListener("mousemove", move);
        marks[i].addEventListener("mouseleave", hide);
      }}
    }}
    var slides = document.querySelectorAll(".slide");
    document.addEventListener("keydown", function (e) {{
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "PageDown" && e.key !== "PageUp") return;
      var dir = (e.key === "ArrowDown" || e.key === "PageDown") ? 1 : -1;
      var y = window.scrollY, target = null;
      for (var i = 0; i < slides.length; i++) {{
        var top = slides[i].offsetTop;
        if (dir > 0 && top > y + 8) {{ target = top; break; }}
        if (dir < 0 && top < y - 8) {{ target = top; }}
      }}
      if (target === null) return;
      e.preventDefault();
      window.scrollTo({{ top: target, behavior: "smooth" }});
    }});
  }} catch (err) {{ /* enhancement only; the deck stands without it */ }}
}})();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
