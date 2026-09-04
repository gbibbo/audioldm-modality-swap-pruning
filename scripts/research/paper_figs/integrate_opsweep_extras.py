#!/usr/bin/env python3
"""Second, idempotent integration pass for the sweep (CPU, 0 cr): Table 1 rows at 5.12 / 7.68 s
(levels, R with CI, W), the Holm family extended to 19 contrasts (draft5_holm_extension.json) and the
rho_real clause. Reads committed artifacts only. Marker: %% opsweep-extras-integrated.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_opsweep_extras.py
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
SW = json.load(open(os.path.join(ROOT, "configs/research/draft5_opsweep_result.json")))
HM = json.load(open(os.path.join(ROOT, "configs/research/draft5_holm_extension.json")))
MARK = "%% opsweep-extras-integrated"
R, W, B = SW["R_by_duration"], SW["W_by_duration"], SW["secondary"]["by_duration"]
assert HM["existing_conclusions_changed_by_extension"] == [] and all(HM["new_contrasts_survive_holm_all"].values())
d3 = [t for t in HM["family_all"] if t["contrast"].startswith("sev2 sweep D3")][0]
s = open(TEX, encoding="utf-8").read()
if MARK in s:
    sys.exit("already integrated")


def rep(a, b):
    global s
    pat = re.sub(r"\\\s+", r"\\s+", re.escape(a))
    m = list(re.finditer(pat, s)); assert len(m) == 1, ("MISSING/AMBIG", len(m), a[:80])
    s = s[:m[0].start()] + b + s[m[0].end():]


def row(d):
    r = R[str(d)]; lv = B[str(d)]["levels"]
    return (f"AudioCaps ${d}$\\,s & ${lv['P']:.3f}$ & ${lv['PFT']:.3f}$ & ${r['point']:+.3f}^{{\\dagger}}$~"
            f"\\ci{{{r['lo']:+.3f}}}{{{r['hi']:+.3f}}} & ${W[str(d)]:.2f}$\\\\")


# Table 1: two sweep rows between the severity-2 3.84 s and 10.24 s rows
rep("""AudioCaps $3.84$\\,s & $0.015$ & $0.100$ & $+0.085^{\\dagger}$~\\ci{+0.066}{+0.105} & $0.72$\\\\
AudioCaps $10.24$\\,s & $0.055$ & $0.299$""",
    f"""AudioCaps $3.84$\\,s & $0.015$ & $0.100$ & $+0.085^{{\\dagger}}$~\\ci{{+0.066}}{{+0.105}} & $0.72$\\\\
{row(5.12)}
{row(7.68)}
AudioCaps $10.24$\\,s & $0.055$ & $0.299$""")
# Holm family 13 -> 19
rep("""A Holm correction over all $13$ contrasts leaves every severity-2 conclusion unchanged ($p<10^{-4}$);""",
    f"""A Holm correction over all ${HM['family_size']}$ contrasts, sweep steps and published-recipe check included,
leaves every severity-2 conclusion unchanged ($p<10^{{-4}}$; last sweep step $p={d3['asl_two_sided']:.3f}$);""")
# rho_real clause after the rho_dense climb
rep("""$\\rho_{\\mathrm{dense}}$ climbs $44\\%$, $63\\%$, $78\\%$, $82\\%$ (Table~\\ref{tab:anchors}).""",
    f"""$\\rho_{{\\mathrm{{dense}}}}$ climbs $44\\%$, $63\\%$, $78\\%$, $82\\%$ (Table~\\ref{{tab:anchors}}); $\\rho_{{\\mathrm{{real}}}}$
dips at $10.24$\\,s only because the real clip itself gains {SW['secondary']['s_real_steps']['7.68_to_10.24']['point']:+.3f} once it fills the scorer's
$10$\\,s window without repeat-padding.""")
s = s.replace("\\documentclass{article}", MARK + "\n\\documentclass{article}", 1)
open(TEX, "w", encoding="utf-8").write(s)
print("integrated extras: rows", row(5.12), row(7.68), "| Holm", HM["family_size"], "| D3 p", d3["asl_two_sided"])
