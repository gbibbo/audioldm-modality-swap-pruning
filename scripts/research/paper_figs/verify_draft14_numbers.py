#!/usr/bin/env python3
"""Verify that the experimental numbers printed by Draft 14 (icassp/sections/draft14_*.tex) are reproduced from the
committed result artifacts (CPU, 0 cr). Every expected string is FORMATTED FROM THE ARTIFACT, never typed by hand.
All checks are exact-string checks on whitespace-normalised LaTeX (Draft 14 has no pgfplots figure: the figure is a
placeholder with a production specification, so there are no coordinate checks).
Exit 1 on any MISSING. Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/verify_draft14_numbers.py
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SECTIONS = sorted(glob.glob(os.path.join(ROOT, "icassp", "sections", "draft14_*.tex")))
raw = "".join(open(f, encoding="utf-8").read() for f in SECTIONS)
if "%% draft14-second-review" not in raw:
    sys.exit("verify_draft14_numbers.py: the manuscript sections do not carry the `%% draft14-second-review` marker")
tex = re.sub(r"[\s~$]|\\,|\;|\\!", "", raw)


def J(rel): return json.load(open(os.path.join(ROOT, rel)))
def tci(c, nd=3): return f"{c['point']:+.{nd}f}\\ci{{{c['lo']:+.{nd}f}}}{{{c['hi']:+.{nd}f}}}"
def tci2(c, nd=3):                       # artifacts that store ci95 as a 2-list
    lo, hi = c["ci95"]; return f"{c['point']:+.{nd}f}\\ci{{{lo:+.{nd}f}}}{{{hi:+.{nd}f}}}"
def pt(c, nd=3): return f"{c['point']:+.{nd}f}"
def tpc(c): return f"{100*c['point']:.0f}\\%"
def _t(v, nd):
    s = f"{v:.{nd}f}"; return s.replace("0.", ".", 1) if v >= 0 else s.replace("-0.", "-.", 1)
def tab(c, nd=3):                        # table form: .085[.066,.105]
    return f"{_t(c['point'], nd)}[{_t(c['lo'], nd)},{_t(c['hi'], nd)}]"
def tab2(c, nd=2):                       # table form from ci95 2-list, 2 decimals: .66[.43,.92]
    lo, hi = c["ci95"]; return f"{_t(c['point'], nd)}[{_t(lo, nd)},{_t(hi, nd)}]"


xsev = J("configs/research/xsev_result.json"); PA = xsev["PRIMARY_A"]
D = J("configs/research/xsev_dense_192_control_result.json"); DP = D["PRIMARY"]; DFC = D["FLOOR_CORRECTED"]
FC = J("configs/research/draft5_floor_ceiling_result.json")["sev2_xsev192"]
SW = J("configs/research/draft5_opsweep_result.json"); Rd = SW["R_by_duration"]
HC = J("configs/research/draft5_sweep_hc.json"); SM = J("configs/research/draft5_sweep_secondary_metrics.json")
PUB = J("configs/research/draft5_pubrecipe_result.json")
C2 = J("configs/research/native_crop_analysis_result.json")["severities"]["sev2_xsev192_pruned2_A"]
X = J("configs/research/r2_EXT2x2_result.json"); XP = X["pruned"]; XD = X["dense"]
E1c = J("configs/research/r2_E1c_result.json"); E5 = J("configs/research/r2_E5_result.json"); E8 = J("configs/research/r2_E8_result.json")
PH = J("configs/research/r2_posthoc_pooled_anchors.json"); HH = PH["hiphop_127"]; S1 = PH["sev1_heterogeneity"]
KLg = SM["recovery_gain_KL"]
PNk = [k for k in SM if k.startswith("recovery_gain") and "KL" not in k]
assert len(PNk) == 1, f"PANNs recovery block not identified uniquely: {PNk}"
PNg = SM[PNk[0]]

checks = [
    # abstract
    ("abs R short/native", "risesfrom" + pt(PA["R_short"]) + "at3.84sto" + pt(PA["R_native"]) + "at10.24s"),
    ("abs dJ pruned", "\\DeltaJ=" + pt(XP["dJ"]) + "\\ci{" + f"{XP['dJ']['lo']:+.3f}" + "}{" + f"{XP['dJ']['hi']:+.3f}" + "}"),
    # 4.1
    ("4.1 R_short", "R=" + tci(PA["R_short"]) + "at3.84s"), ("4.1 R_native", tci(PA["R_native"]) + "at10.24s"), ("4.1 J (frozen primary)", "J=" + tci(PA["J"])),
    ("4.1 rho_dense pair", "closes" + tpc(DP["rho_short"]) + "ofthegapfromPtodenseat3.84sand" + tpc(DP["rho_native"]) + "at10.24s"),
    ("4.1 J_pub", "J=" + tci(PUB["J_pub"])),
    ("4.1 D4", "changesby" + tci(E1c["D4"]) + "from10.24to15.36s"),
    # 4.2
    ("4.2 R_sf short", "gains" + tci(XP["R_sf"]["3.84"]) + "whenevaluatedat3.84s"), ("4.2 R_sf native", "and" + tci(XP["R_sf"]["10.24"]) + "at10.24s"),
    ("4.2 J_sf", "J_{3.84}=" + tci(XP["J_sf"])), ("4.2 J_lf", "J_{10.24}=" + tci(XP["J_lf"])),
    ("4.2 dJ pruned", "\\DeltaJ=" + tci(XP["dJ"])), ("4.2 dJ dense", "\\DeltaJ_{\\rmdense}=" + tci(XD["dJ"])),
    # Table 1 (duration block)
    ("T1 3.84 row", "3.84s&" + tab(PA["R_short"]) + "&" + tab(HC["R_HC_by_duration"]["3.84"]) + "&" + tab2(KLg["R@3.84"]) + "&" + tab2(PNg["R@3.84"])),
    ("T1 5.12 row", "5.12s&" + tab(Rd["5.12"]) + "&" + tab(HC["R_HC_by_duration"]["5.12"]) + "&" + tab2(KLg["R@5.12"]) + "&" + tab2(PNg["R@5.12"])),
    ("T1 7.68 row", "7.68s&" + tab(Rd["7.68"]) + "&" + tab(HC["R_HC_by_duration"]["7.68"]) + "&" + tab2(KLg["R@7.68"]) + "&" + tab2(PNg["R@7.68"])),
    ("T1 10.24 row", "10.24s&" + tab(PA["R_native"]) + "&" + tab(HC["R_HC_by_duration"]["10.24"]) + "&" + tab2(KLg["R@10.24"]) + "&" + tab2(PNg["R@10.24"])),
    ("T1 J row (frozen primary)", "J&" + tab(PA["J"]) + "&" + tab(HC["J_HC_native_minus_short"]) + "&" + tab2(KLg["J_native_minus_short"]) + "&" + tab2(PNg["J_native_minus_short"])),
    # Table 1 (domain block)
    ("T1 AudioCaps row", "AudioCaps&192&" + tab(PA["R_short"]) + "&" + tab(PA["R_native"]) + "&" + _t(DP["rho_native"]["point"], 2)),
    ("T1 Clotho row", "Clotho&96&" + tab(E5["cells"]["3.84"]["R"]) + "&" + tab(E5["cells"]["10.24"]["R"]) + "&" + _t(E5["cells"]["10.24"]["rho_dense"]["point"], 2)),
    ("T1 Hip-hop row (POSTHOC pooled)", "Hip-hop&127&" + tab(HH["3.84"]["R_pooled"]) + "&" + tab(HH["10.24"]["R_pooled"]) + "&" + tab(HH["10.24"]["rho_dense_pooled"])),
    # 4.3
    ("4.3 R_clo native + rho", "ClothogivesR=" + tci(E5["cells"]["10.24"]["R"]) + "andcloses" + tpc(E5["cells"]["10.24"]["rho_dense"])),
    ("4.3 D_clo", "byonly" + tci(E5["D_clo"]["10.24"])),
    ("4.3 hip-hop R pooled", "recoveryis" + tci(HH["3.84"]["R_pooled"]) + "at3.84sand" + tci(HH["10.24"]["R_pooled"]) + "at10.24s"),
    ("4.3 hip-hop rho pooled", "closesabout" + tpc(HH["3.84"]["rho_dense_pooled"]) + "ofthedensegapat3.84sand" + tpc(HH["10.24"]["rho_dense_pooled"]) + "at10.24s"),
    # 4.4
    ("4.4 sev1 J armd80", "80-promptdrawgivesJ=" + tci(S1["J_armd80"])), ("4.4 sev1 J new96", "gives" + tci(S1["J_new96"])),
    ("4.4 sev1 diff", "Theirunpaireddifferenceis" + tci(S1["J_new96_minus_armd80"])), ("4.4 sev1 pooled", "n=176estimate," + tci(E8["pooled176_J"])),
    ("4.4 dense s_c", "from3.84to10.24sis" + tci(DFC["s_c_dense"])), ("4.4 real crop response", "cropresponseof" + tci(FC["s_c_real"])),
    ("4.4 R_crop", "crop}=" + tci(C2["R_crop"])), ("4.4 crop minus short", "whichis" + tci(C2["R_crop_minus_R_short"]) + "above"),
]
bad = 0
for label, exp in checks:
    ok = exp in tex; bad += not ok; print(f"{'OK     ' if ok else 'MISSING'} {label}: {exp}")
print(f"\n{len(checks) - bad}/{len(checks)} OK")
sys.exit(1 if bad else 0)
