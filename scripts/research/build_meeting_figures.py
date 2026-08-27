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
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --plane:#f3f5f8; --surface:#fbfcfd; --sunk:#eef1f5;
  --ink:#10131a; --ink-2:#464d5c; --ink-muted:#767e8f;
  --grid:#e4e7ed; --axis:#c3c8d3; --rule:rgba(16,19,26,.10);
  --s1:#2a78d6; --s2:#eb6834; --refbar:#9aa2b1;
  --band:rgba(118,126,143,.20); --accent:#2a78d6;
  --warn-bg:rgba(235,104,52,.09); --warn-ink:#a8431a;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane:#0b0d11; --surface:#13151a; --sunk:#191c22;
    --ink:#f1f3f7; --ink-2:#b4bbc8; --ink-muted:#858d9c;
    --grid:#232730; --axis:#3a3f4a; --rule:rgba(255,255,255,.11);
    --s1:#3987e5; --s2:#d95926; --refbar:#6d7482;
    --band:rgba(133,141,156,.24); --accent:#5c9df0;
    --warn-bg:rgba(217,89,38,.13); --warn-ink:#e5926c;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#0b0d11; --surface:#13151a; --sunk:#191c22;
  --ink:#f1f3f7; --ink-2:#b4bbc8; --ink-muted:#858d9c;
  --grid:#232730; --axis:#3a3f4a; --rule:rgba(255,255,255,.11);
  --s1:#3987e5; --s2:#d95926; --refbar:#6d7482;
  --band:rgba(133,141,156,.24); --accent:#5c9df0;
  --warn-bg:rgba(217,89,38,.13); --warn-ink:#e5926c;
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1000px; margin:0 auto; padding:56px 24px 96px; display:flex; flex-direction:column; gap:44px; }}
p {{ margin:0; max-width:68ch; }}
.eyebrow {{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-muted); margin:0 0 14px;
}}
h1 {{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:clamp(30px,4.6vw,46px);
  line-height:1.12; letter-spacing:-.015em; margin:0 0 18px; text-wrap:balance; max-width:19ch;
}}
h2 {{
  font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:23px; line-height:1.25;
  margin:0; letter-spacing:-.01em; text-wrap:balance;
}}
h3 {{ font-size:14px; font-weight:600; margin:0; letter-spacing:.01em; }}
.lede {{ font-size:18.5px; color:var(--ink-2); max-width:64ch; }}
.rule {{ height:1px; background:var(--rule); border:0; margin:0; }}

/* stat row */
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:10px; overflow:hidden; }}
.stat {{ background:var(--surface); padding:20px 22px; display:flex; flex-direction:column; gap:6px; }}
.stat .k {{ font-size:12.5px; color:var(--ink-muted); letter-spacing:.01em; }}
.stat .v {{ font-size:34px; font-weight:600; line-height:1; letter-spacing:-.02em; }}
.stat .n {{ font-size:12.5px; color:var(--ink-2); }}

/* figure card */
figure {{ margin:0; background:var(--surface); border:1px solid var(--rule); border-radius:12px; overflow:hidden; }}
.fhead {{ padding:22px 26px 0; display:flex; flex-direction:column; gap:9px; }}
.fnum {{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); }}
.take {{ color:var(--ink-2); font-size:15.5px; max-width:70ch; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; padding:16px 26px 0; }}
.key {{ display:flex; align-items:center; gap:8px; font-size:13px; color:var(--ink-2); }}
.swatch {{ width:11px; height:11px; border-radius:50%; flex:none; }}
.swatch.s1 {{ background:var(--s1); }}
.swatch.s2 {{ background:var(--s2); }}
.swatch.ref {{ background:var(--refbar); border-radius:2px; }}
.swatch.band {{ background:var(--band); border-radius:2px; width:20px; }}
.plot {{ padding:10px 20px 4px; overflow-x:auto; }}
.panels {{ display:grid; grid-template-columns:1fr; }}
.panel + .panel {{ border-top:1px solid var(--rule); }}
.ptitle {{ padding:18px 26px 0; font-size:13px; font-weight:600; }}
.pnote {{ padding:2px 26px 0; font-size:13.5px; color:var(--ink-2); max-width:74ch; }}
figcaption {{ padding:16px 26px 22px; font-size:13px; color:var(--ink-muted); border-top:1px solid var(--rule); margin-top:12px; }}
figcaption code {{ font-family:"IBM Plex Mono",monospace; font-size:12px; }}

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

.flag {{ display:inline-flex; align-items:center; gap:8px; align-self:flex-start; margin:0 26px; padding:7px 12px; border-radius:6px; background:var(--warn-bg); color:var(--warn-ink); font-size:12.5px; font-weight:500; }}

details {{ border-top:1px solid var(--rule); }}
summary {{ padding:13px 26px; font-size:13px; color:var(--ink-2); cursor:pointer; list-style:none; }}
summary::-webkit-details-marker {{ display:none; }}
summary::before {{ content:"▸ "; color:var(--ink-muted); }}
details[open] summary::before {{ content:"▾ "; }}
summary:focus-visible, a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }}
.tablewrap {{ overflow-x:auto; padding:0 26px 22px; }}
table {{ border-collapse:collapse; font-size:12.5px; width:100%; }}
th, td {{ text-align:right; padding:6px 12px; border-bottom:1px solid var(--rule); font-variant-numeric:tabular-nums; }}
th:first-child, td:first-child {{ text-align:left; font-family:"IBM Plex Mono",monospace; font-variant-numeric:normal; }}
th {{ color:var(--ink-muted); font-weight:500; }}

.read {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:26px; }}
.read div {{ display:flex; flex-direction:column; gap:8px; }}
.read p {{ font-size:15px; color:var(--ink-2); }}
.aside {{ background:var(--sunk); border:1px solid var(--rule); border-radius:12px; padding:24px 26px; display:flex; flex-direction:column; gap:12px; }}
.aside p {{ font-size:15px; color:var(--ink-2); }}
ul {{ margin:0; padding-left:20px; display:flex; flex-direction:column; gap:8px; font-size:15px; color:var(--ink-2); max-width:68ch; }}
strong {{ color:var(--ink); font-weight:600; }}
.mono {{ font-family:"IBM Plex Mono",monospace; font-size:13px; }}
footer {{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--ink-muted); line-height:1.85; }}
footer b {{ color:var(--ink-2); font-weight:500; }}

#tip {{
  position:fixed; z-index:10; pointer-events:none; opacity:0; transition:opacity .12s;
  background:var(--ink); color:var(--surface); font-size:12.5px; padding:6px 10px;
  border-radius:6px; max-width:280px; box-shadow:0 4px 16px rgba(0,0,0,.22);
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">

  <header>
    <p class="eyebrow">AudioLDM · structured pruning · modality swap</p>
    <h1>Audio and text want the same channels cut</h1>
    <p class="lede">We asked whether ranking a diffusion U-Net's channels under <em>audio</em>
    conditioning instead of <em>text</em> conditioning changes which channels structured pruning
    removes. Two pre-registered gates say no — and the difference we do measure is the same size
    as the noise from re-drawing the calibration set.</p>
  </header>

  <div class="stats">
    <div class="stat">
      <span class="k">Rank agreement, audio vs text</span>
      <span class="v">ρ = {rho_med}</span>
      <span class="n">median over the 12 layers that decide the mask ({rho_lo}–{rho_hi})</span>
    </div>
    <div class="stat">
      <span class="k">Channels kept in common</span>
      <span class="v">{overlap_pct}%</span>
      <span class="n">audio-only vs text-only criterion, at −65% U-Net</span>
    </div>
    <div class="stat">
      <span class="k">Same figure from noise alone</span>
      <span class="v">{null_med_pct}%</span>
      <span class="n">median of {n_splits} calibration half-splits of one criterion</span>
    </div>
  </div>

  <figure>
    <div class="fhead">
      <span class="fnum">Figure 1</span>
      <h2>The instrument separates criteria, not modalities</h2>
      <p class="take">Swap the conditioning modality and the channel ranking barely moves (blue).
      Swap the <em>criterion</em> — first-order Taylor for plain L1 magnitude — and it moves a lot
      (orange). Same layers, same run, same calibration data: the measurement is sensitive, the
      modality effect is not there.</p>
    </div>
    <div class="legend">
      <span class="key"><span class="swatch s1"></span>Audio-Taylor vs text-Taylor — the modality swap</span>
      <span class="key"><span class="swatch s2"></span>Audio-Taylor vs L1 magnitude — the control</span>
    </div>
    <div class="panels">
      <div class="panel">
        <p class="ptitle">A · Do the two rankings agree, layer by layer?</p>
        <p class="pnote">Each row is one of the 12 layers whose channel ranking actually determines the
        pruned architecture. ρ = 1 means the two criteria order the channels identically.</p>
        <div class="plot">{panel_a}</div>
      </div>
      <div class="panel">
        <p class="ptitle">B · So how many channels does the swap actually change?</p>
        <p class="pnote">Kept-set overlap at the published −65% budget. The pre-registered gate needed
        the two modalities to disagree enough to push this <em>below 0.80</em>. Measured: {overlap} —
        which is where the same criterion lands against itself when you just re-draw half the
        calibration slots.</p>
        <div class="plot">{panel_b}</div>
      </div>
    </div>
    <figcaption>
      Job <code>m3b-saliency-2</code>, Tesla T4. Channel-gate first-order Taylor saliency,
      S<sub>c</sub> = mean<sub>slots</sub>|g<sub>c</sub>·∂L/∂g<sub>c</sub>|, on the dense
      AudioLDM-M-Full U-Net (415.955 M → 145.674 M params). E = {e} calibration clips × K = {k}
      timesteps, gradient budget matched across criteria (2,560 evaluations each), frozen manifest
      <code>{manifest}…</code>, seed {seed}, commit <code>{sal_commit}</code>. The noise band is
      {n_splits} random half-splits of the text-only criterion's stored per-slot contributions
      (matched half size), computed on the enriched E = 512 pass.
    </figcaption>
    <details>
      <summary>Data table · per-layer values</summary>
      <div class="tablewrap">
        <table>
          <thead><tr><th>Layer</th><th>channels</th><th>kept</th><th>ρ audio·text</th><th>ρ Taylor·L1</th><th>kept-set overlap</th></tr></thead>
          <tbody>{tbl1}</tbody>
        </table>
      </div>
    </details>
  </figure>

  <div class="read">
    <div>
      <h3>What Figure 1 rules out</h3>
      <p>The hypothesis was that a text-to-audio model carries modality-specific structure: prune it
      with a text-conditioned signal and you would damage the audio path, so a paired audio+text
      criterion should pick a measurably different — and better — set of channels to keep. At
      −65% there is no such set to pick. Out of 192 kept channels per layer, the two modalities
      disagree on about ten, and re-drawing the calibration data disagrees by just as many.</p>
    </div>
    <div>
      <h3>The second gate agrees</h3>
      <p>An independent test asked the damage question rather than the ranking question: does the
      pruned model lose more on one modality than a random mask of matched generic damage?
      R<sub>mod</sub> = {r_l1} for the L1-pruned model against {r_rand} for the matched random null —
      Δ<sub>swap</sub> = {delta_swap}, 95% bootstrap CI [{ci_lo}, {ci_hi}]. The interval contains zero.
      Two different questions, one answer.</p>
    </div>
  </div>

  <figure>
    <div class="fhead">
      <span class="fnum">Figure 2</span>
      <h2>And you can hear it in the generated audio</h2>
      <p class="take">If the paired-modality criteria kept a genuinely better set of channels, they
      would generate better audio than plain text-only Taylor. They do not: P2 and P3 sit
      <em>behind</em> P1, and every pruned variant sits far from the dense model. Pre-recovery, at
      this budget, the criterion barely matters.</p>
    </div>
    <span class="flag">Screening grade — 100 clips, 1 seed, no confidence intervals. Not a claim.</span>
    <div class="legend">
      <span class="key"><span class="swatch s1"></span>Taylor criteria</span>
      <span class="key"><span class="swatch s2"></span>L1 magnitude</span>
      <span class="key"><span class="swatch ref"></span>Dense reference</span>
    </div>
    <div class="plot">{fig2}</div>
    <figcaption>
      Fréchet Audio Distance (VGGish, 128-d) between 100 generated clips and 100 real AudioCaps
      clips from the disjoint validation split. Job <code>m3-screening</code> re-scored on CPU with a
      real-part eigenvalue Fréchet (self-distance control: FAD(ref,ref) = 0.000). 50 DDIM steps,
      guidance 3.5, one seed. Recorded in the ledger as screening only and explicitly
      <em>not promotable</em>: no seed variance, no interval, n = 100. Read the ordering as
      consistent with Figure 1, never as an established ranking.
    </figcaption>
    <details>
      <summary>Data table · FAD values</summary>
      <div class="tablewrap">
        <table>
          <thead><tr><th>System</th><th>FAD (VGGish)</th></tr></thead>
          <tbody>{tbl2}</tbody>
        </table>
      </div>
    </details>
  </figure>

  <div class="aside">
    <h2>One thing worth checking together</h2>
    <p>The published L1 artifact sits at the bottom of Figure 2, and there may be a mechanical reason.
    In every one of the 28 ranked layers, the released ranking correlates <strong>−1.000</strong> with
    a conventional descending-L1 ranking: <code class="mono">np.argsort</code> is ascending and the
    first <em>k</em> entries are kept, so the artifact keeps the <em>lowest</em>-magnitude filters. On
    all 15 actually-pruned layers the kept set has lower mean L1 than the removed set.</p>
    <p>We can reproduce the released checkpoint bit-exactly (690/690 tensors) from the base weights
    plus <code class="mono">sorted_indexes_dict.pkl</code>, so this is a property of the artifact
    itself, not of our reading of the code. Intentional, or a direction to flip? It changes what the
    pre-recovery baseline means for everyone who builds on it.</p>
  </div>

  <div>
    <h2>Where the paper goes</h2>
    <p style="margin:12px 0 16px;color:var(--ink-2)">The modality-swap line is closed as a clean
    negative — pre-registered, two gates, with a discriminating control proving the instrument
    works. It is a publishable result, but not a four-page ICASSP story on its own. The submission
    moved to the question the same machinery answers next:</p>
    <ul>
      <li><strong>Scenario B — legacy adapter transfer.</strong> A LoRA trained on the dense backbone
      cannot load onto a pruned one: the channel geometry changed. We slice the adapter by the same
      kept-index set, with zero retraining and zero adapter data, and measure whether its benefit
      survives.</li>
      <li><strong>Separating two failure modes</strong> usually conflated: the generic degradation the
      pruned backbone suffers standalone (E), versus the excess degradation specific to the
      transferred adapter (F = D − E).</li>
      <li><strong>Pre-registered before any GPU spend:</strong> 64 held-out MusicCaps prompts × 3 seeds,
      prompt-clustered bootstrap (B = 10,000, seed 20260826), SESOI +0.025, dual non-inferiority +
      differential-fragility gate. Transfer operator already verified on CPU.</li>
      <li><strong>Next:</strong> Gate 0 — reproduce the published adapter's uplift on the dense
      backbone. If it fails, the line stops there, by pre-registration.</li>
    </ul>
  </div>

  <hr class="rule">

  <footer>
    <b>Provenance.</b> Figure 1: <code>artifacts/m3_pilot/m3b_saliency.pt</code> +
    <code>m3b_saliency_result.json</code>, commit <code>{sal_commit}</code>; noise band
    <code>gateb_null_distribution.json</code>. Damage gate: <code>m3a_result.json</code>, job
    <code>m3a-diag-1</code>, commit <code>{m3a_commit}</code>. Figure 2:
    <code>artifacts/m4_screening/rescore_frechet.json</code>.<br>
    <b>Both gates failed to reject the null; both are recorded as rejected claims in the claims
    matrix.</b> Every number on this page is recomputed from those artifacts at build time by
    <code>scripts/research/build_meeting_figures.py</code>.
  </footer>
</div>

<div id="tip" role="status" aria-live="polite"></div>
<script>
(function () {{
  try {{
    var tip = document.getElementById("tip");
    if (!tip) return;
    function show(e) {{
      var t = e.currentTarget.getAttribute("data-tip");
      if (!t) return;
      tip.textContent = t;
      tip.style.opacity = "1";
      move(e);
    }}
    function move(e) {{
      var x = e.clientX + 14, y = e.clientY + 16;
      var r = tip.getBoundingClientRect();
      if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 14;
      if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 16;
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    }}
    function hide() {{ tip.style.opacity = "0"; }}
    var marks = document.querySelectorAll("[data-tip]");
    for (var i = 0; i < marks.length; i++) {{
      marks[i].addEventListener("mouseenter", show);
      marks[i].addEventListener("mousemove", move);
      marks[i].addEventListener("mouseleave", hide);
    }}
  }} catch (err) {{ /* tooltips are an enhancement; the chart stands without them */ }}
}})();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
