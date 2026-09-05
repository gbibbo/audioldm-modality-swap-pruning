#!/usr/bin/env python3
"""Verify that the experimental numbers printed by Draft 13 (icassp/sections/draft13_*.tex) are reproduced from the
committed result artifacts (CPU, 0 cr). Every expected string is FORMATTED FROM THE ARTIFACT, never typed by hand.
Prose/table numbers are exact-string checks on whitespace-normalised LaTeX; the pgfplots coordinates of Fig. 1 are
numeric checks (|diff| <= 5.5e-4: the figure rounds some points to 3 decimals).
Exit 1 on any MISSING. Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/verify_draft13_numbers.py
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SECTIONS = sorted(glob.glob(os.path.join(ROOT, "icassp", "sections", "draft13_*.tex")))
raw = "".join(open(f, encoding="utf-8").read() for f in SECTIONS)
if "%% draft13-reviewer-followup" not in raw:
    sys.exit("verify_draft13_numbers.py: the manuscript sections do not carry the `%% draft13-reviewer-followup` marker")
tex = re.sub(r"[\s~$]|\\,|\;|\\!", "", raw)


def J(rel): return json.load(open(os.path.join(ROOT, rel)))
def tci(c, nd=3): return f"{c['point']:+.{nd}f}\\ci{{{c['lo']:+.{nd}f}}}{{{c['hi']:+.{nd}f}}}"
def pt(c, nd=3): return f"{c['point']:+.{nd}f}"
def tpc(c): return f"{100*c['point']:.0f}\\%"
def tab(c):                              # table form: .085[.066,.105]
    f = lambda v: f"{v:.3f}".replace("0.", ".", 1) if v >= 0 else f"{v:.3f}".replace("-0.", "-.", 1)
    return f"{f(c['point'])}[{f(c['lo'])},{f(c['hi'])}]"


xsev = J("configs/research/xsev_result.json"); PA = xsev["PRIMARY_A"]
FC = J("configs/research/draft5_floor_ceiling_result.json"); s2 = FC["sev2_xsev192"]
D = J("configs/research/xsev_dense_192_control_result.json")["PRIMARY"]
SW = J("configs/research/draft5_opsweep_result.json"); Rd = SW["R_by_duration"]
fl = J("configs/research/finelap_temporal_result.json")["verdict"]
C2 = J("configs/research/native_crop_analysis_result.json")["severities"]["sev2_xsev192_pruned2_A"]
E3 = J("configs/research/r2_E3_result.json"); B_ = J("configs/research/r2_B_result.json"); E1c = J("configs/research/r2_E1c_result.json")
E5 = J("configs/research/r2_E5_result.json"); E6 = J("configs/research/r2_E6_result.json"); E7 = J("configs/research/r2_E7_result.json"); E8 = J("configs/research/r2_E8_result.json")

checks = [
    ("abstract R short/native", "risesfrom" + pt(PA["R_short"]) + "at3.84sto" + pt(PA["R_native"]) + "at10.24s"),
    ("4.1 R_short", "R=" + tci(PA["R_short"]) + "at3.84s"), ("4.1 R_native", tci(PA["R_native"]) + "at10.24s"), ("4.1 J", "J=" + tci(PA["J"])),
    ("4.1 rho_dense pair", "closes" + tpc(D["rho_short"]) + "ofthegapfromPtodenseat3.84sand" + tpc(D["rho_native"]) + "at10.24s"),
    ("4.1 rho_real pair", "fractionsare" + tpc(s2["rho_real_short"]) + "and" + tpc(s2["rho_real_native"])),
    ("4.1 R(15.36)", "At15.36s,recoveryis" + tci(E1c["cells"]["15.36"]["R"])), ("4.1 D4", "isonly" + tci(E1c["D4"])),
    ("4.1 pooled sev-1 J", "pooledJ=" + tci(E8["pooled176_J"])),
    ("4.2 R_sf short", "gainisonly" + tci(E3["R_sf"]["3.84"])), ("4.2 R_sf native", "gains" + tci(E3["R_sf"]["10.24"])), ("4.2 J_sf", "yieldingJ=" + tci(E3["J_sf"])),
    ("4.2 G_tf short", "changesby" + tci(B_["G_tf"]["3.84"])), ("4.2 G_tf native", "andby" + tci(B_["G_tf"]["10.24"]) + "at10.24s"), ("4.2 J_tf", "withJ=" + tci(B_["J_tf"])),
    ("4.3 R_clo native", "ClothogivesR=" + tci(E5["cells"]["10.24"]["R"]) + "andcloses" + tpc(E5["cells"]["10.24"]["rho_dense"])),
    ("4.3 D_clo", "isonly" + tci(E5["D_clo"]["10.24"])),
    ("4.3 hip-hop 127", tci(E7["cells"]["3.84"]["pooled127_R"]) + "at3.84sand" + tci(E7["cells"]["10.24"]["pooled127_R"]) + "at10.24s"),
    ("4.3 A_dense sev2 native", "denselies" + tci(E6["cells"]["sev2_10.24"]["A_dense"]) + "aboveshuffled-captionchance"),
    ("4.4 FineLAP T", "T=" + tci(fl["T2"])), ("4.4 R_crop", "crop}=" + tci(C2["R_crop"])), ("4.4 crop minus short", "whichis" + tci(C2["R_crop_minus_R_short"]) + "above"),
    ("T1 CLAP 3.84", "3.84s&" + tab(PA["R_short"])), ("T1 CLAP 5.12", "5.12s&" + tab(Rd["5.12"])), ("T1 CLAP 7.68", "7.68s&" + tab(Rd["7.68"])),
    ("T1 CLAP 10.24", "10.24s&" + tab(PA["R_native"])), ("T1 CLAP J", "J&" + tab(PA["J"])),
    ("T1 AudioCaps row", "AudioCaps&192&" + tab(PA["R_short"]) + "&" + tab(PA["R_native"]) + "&" + f"{D['rho_native']['point']:.2f}".lstrip("0")),
    ("T1 Clotho row", "Clotho&96&" + tab(E5["cells"]["3.84"]["R"]) + "&" + tab(E5["cells"]["10.24"]["R"]) + "&" + f"{E5['cells']['10.24']['rho_dense']['point']:.2f}".lstrip("0")),
    ("T1 Hip-hop row", "Hip-hop&127&" + tab(E7["cells"]["3.84"]["pooled127_R"]) + "&" + tab(E7["cells"]["10.24"]["pooled127_R"]) + "&" + f"{E6['cells']['sev2_10.24']['rho_dense']['point']:.2f}".lstrip("0")),
    ("T1 footnote A_dense", "floorby" + tab(E6["cells"]["sev2_10.24"]["A_dense"])),
]
bad = 0
for label, exp in checks:
    ok = exp in tex; bad += not ok; print(f"{'OK     ' if ok else 'MISSING'} {label}: {exp}")

# Fig. 1 coordinates (numeric): (x, point) +- (0, half-width)
fig = re.findall(r"\(([\d.]+),(-?[\d.]+)\)\+-\(0,([\d.]+)\)", tex)
fig = {(float(x), round(float(y), 4)): float(e) for x, y, e in fig}
def num(label, x, c):
    y = round(c["point"], 4); hw = (c["hi"] - c["lo"]) / 2
    hit = [(k, v) for k, v in fig.items() if k[0] == x and abs(k[1] - c["point"]) <= 5.5e-4 and abs(v - hw) <= 5.5e-4]
    global bad; bad += not hit; print(f"{'OK     ' if hit else 'MISSING'} fig {label}: ({x}, {c['point']:.4f}) +- {hw:.4f}")
num("sweep 3.84 (frozen primary)", 3.84, PA["R_short"]); num("sweep 10.24 (frozen primary)", 10.24, PA["R_native"])
for x in ("5.12", "7.68"):
    num(f"sweep {x}", float(x), Rd[x])
num("15.36 follow-up", 15.36, E1c["cells"]["15.36"]["R"])
num("shortft 3.84", 3.84, E3["R_sf"]["3.84"]); num("shortft 10.24", 10.24, E3["R_sf"]["10.24"])
num("textft 3.84", 3.84, B_["G_tf"]["3.84"]); num("textft 10.24", 10.24, B_["G_tf"]["10.24"])
print(f"\n{len(checks) + 9 - bad}/{len(checks) + 9} OK")
sys.exit(1 if bad else 0)
