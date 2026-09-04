#!/usr/bin/env python3
"""Third idempotent integration pass: the sweep shape seen by the corroborating scorers (Human-CLAP,
KL, PANNs capture; post-hoc). Reads configs/research/draft5_sweep_hc.json and
configs/research/draft5_sweep_secondary_metrics.json. Marker: %% sweep-corroboration-integrated.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_sweep_corroboration.py
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
HC = json.load(open(os.path.join(ROOT, "configs/research/draft5_sweep_hc.json")))
SM = json.load(open(os.path.join(ROOT, "configs/research/draft5_sweep_secondary_metrics.json")))
MARK = "%% sweep-corroboration-integrated"
kl, pn = SM["recovery_gain_KL"], SM["recovery_gain_PANN_top10_capture"]
# the prose below states: every step positive on all three, first two resolved, last unresolved
for blk, key in ((HC["steps"], None), (kl, "ci95"), (pn, "ci95")):
    for k in ("D1", "D2", "D3"):
        c = blk[k]; lo, hi = (c["lo"], c["hi"]) if key is None else tuple(c["ci95"])
        assert c["point"] > 0, (k, c)
        assert (lo > 0) == (k != "D3"), (k, c)
s = open(TEX, encoding="utf-8").read()
if MARK in s:
    sys.exit("already integrated")
def rep(a, b):
    global s
    pat = re.sub(r"\\\s+", r"\\s+", re.escape(a)); m = list(re.finditer(pat, s)); assert len(m) == 1, (len(m), a[:80]); s = s[:m[0].start()] + b + s[m[0].end():]
d3h, d3k, d3p = HC["steps"]["D3"], kl["D3"], pn["D3"]
rep("""(pre-specified gate; the frozen recipe on the same prompts gives $+0.168$, difference $+0.016$~\\ci{-0.040}{+0.072}).""",
    f"""(pre-specified gate; the frozen recipe on the same prompts gives $+0.168$, difference $+0.016$~\\ci{{-0.040}}{{+0.072}}).
Off the primary scorer (post-hoc), Human-CLAP, KL and PANNs capture all show $R$ rising at every step of the
sweep, the first two steps resolved and the last ($7.68\\!\\to\\!10.24$\\,s) not: Human-CLAP is flat there
(${d3h['point']:+.3f}$~\\ci{{{d3h['lo']:+.3f}}}{{{d3h['hi']:+.3f}}}), KL ${d3k['point']:+.2f}$~\\ci{{{d3k['ci95'][0]:+.2f}}}{{{d3k['ci95'][1]:+.2f}}}, PANNs ${d3p['point']:+.2f}$~\\ci{{{d3p['ci95'][0]:+.2f}}}{{{d3p['ci95'][1]:+.2f}}}.""")
# page budget: small trims elsewhere
rep("""FAD agrees ($6.92$ vs.\\ $27.4$, descriptive). Rescored""", """Rescored""")
rep("""\\item \\textbf{Where the gain lives.} A frame-level grounding model shows the native-duration gain spread uniformly over the clip; a crop analysis shows the short-duration deficit arises from \\emph{generating} short clips, not from scoring short excerpts.""",
    """\\item \\textbf{Where the gain lives.} Frame-level grounding shows the native gain spread uniformly over
      the clip; a crop analysis shows the short-duration deficit comes from \\emph{generating} short clips,
      not from scoring short excerpts.""")
rep("""Whether that registers as a ``penalty'' depends only on whether the pruned checkpoint still had music alignment to lose:""",
    """Whether that registers as a ``penalty'' depends on whether the pruned checkpoint had music alignment to lose:""")
s = s.replace("\\documentclass{article}", MARK + "\n\\documentclass{article}", 1)
open(TEX, "w", encoding="utf-8").write(s); print("integrated sweep corroboration: D3", d3h["point"], d3k["point"], d3p["point"])
