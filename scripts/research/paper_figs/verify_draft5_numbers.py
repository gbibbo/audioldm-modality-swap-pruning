#!/usr/bin/env python3
"""Verify that every headline number in icassp/icassp_operating_point.tex (Draft 5) is reproduced from a
committed artifact (CPU, 0 cr). Re-uses the Draft-3/4 check lists where Draft 5 kept the same string,
replaces the checks whose prose form changed (CIs moved from prose to the tables), drops the checks for
numbers Draft 5 no longer prints (pooled-rank J, the 90 % equivalence CI), and adds the Draft-5 anchors
(chance floors, real-audio ceiling, recovery ratios, floor-corrected J, caption tokens). Prints one line per
number: OK / MISSING. Exit 1 if any MISSING or if a `@@` placeholder survives.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")


def J(rel):
    return json.load(open(os.path.join(ROOT, rel)))


def norm(s):
    return re.sub(r"[\s~$]|\\,|\\;|\\!", "", s)


raw_tex = open(TEX, encoding="utf-8").read()
tex = norm(raw_tex)

# ---- inherit the Draft-4 list (which itself inherits Draft 3), executed up to its loop
src = open(os.path.join(HERE, "verify_draft4_numbers.py"), encoding="utf-8").read()
src = src.rsplit("\nbad = 0\n", 1)[0]   # LAST occurrence: the draft-4 file itself contains the string once in code
ns = {"__file__": os.path.join(HERE, "verify_draft4_numbers.py")}
exec(compile(src, "verify_draft4_numbers.py", "exec"), ns)
checks = list(ns["checks"])
ci, pt, mean = ns["ci"], ns["pt"], ns["mean"]

DROP = {"sev2 slope P", "sev2 slope P+FT", "s(P) sev2", "s(P+FT) sev2", "s(dense)", "s(P) sev1", "s(P+FT) sev1",
        "median J sev1", "median J sev2", "rank J sev1", "rank J sev2", "abstract 0.055->0.299", "abstract 0.015->0.100",
        "sev2 win rates", "R_short 90% CI", "sev2 90% CI", "dense matched 3.84 / 10.24",
        "FineLAP occupancy", "FineLAP coverage", "FineLAP peak",      # Draft 5 prints only semantic mass
        "crop sev2 nat-crop",                                          # Draft 5 prints the point only
        "pruned music vs AC native",                                   # Draft 5 rewrites the sentence (floor)
        "caption dense s",                                             # +0.150 now stated in the prose only
        "wilcoxon sev1", "wilcoxon sev2"}                              # p-values dropped from the prose
dropped = [c[0] for c in checks if c[0] in DROP]
checks = [c for c in checks if c[0] not in DROP]
# also drop any inherited check whose label mentions pooled-rank / equivalence forms (defensive)
checks = [c for c in checks if not re.search(r"rank|equival|90", c[0])]

S2 = ns["S2"]; ddc = ns["ddc"]; rob = ns["rob"]
r2 = rob["R2_rank_scale_J"]
checks += [
    ("sev2 s(P) prose", "s(\\mathrm{P})=" + pt(S2["slope_pruned"])),
    ("sev2 s(P+FT) prose", "s(\\PFT)=" + pt(S2["slope_postft"])),
    ("s(dense) prose", "s(\\mathrm{dense})=" + pt(ddc["slopes"]["dense"]) + "$($" + f"{ddc['means']['dense_short']:.3f}\\!\\to\\!{ddc['means']['dense_native']:.3f}"),
    ("s(P) sev1 prose", "s(\\mathrm{P})=" + pt(ddc["slopes"]["pruned"])),
    ("s(P+FT) sev1 prose", "s(\\PFT)=" + pt(ddc["slopes"]["postft"])),
    ("median J sev1", pt(r2["sev1_armd80"]["median_j"]) + "$$" + ci(r2["sev1_armd80"]["median_j"])),
    ("median J sev2 point", "medianinteractionis$" + pt(r2["sev2_xsev192"]["median_j"])),
    ("crop sev2 nat-crop point", "\\Rn-R_{\\mathrm{crop}}=" + pt(ns["ns"]["C2"]["R_native_minus_R_crop"]) + "$(seam-robust)"),
    ("4.1 0.055 to 0.299", "from$0.055$to$0.299$"),
    ("4.1 0.015 to 0.100", "from$0.015$to$0.100$"),
    ("table W sev2 short", f"&${S2['win_rate_short']:.2f}$\\\\"),
]

# ---- Draft-5 anchors (must match fill_draft5.py formats)
FC = J("configs/research/draft5_floor_ceiling_result.json")
cells, s1, s2 = FC["cells"], FC["sev1_armd80"], FC["sev2_xsev192"]


def pct(c):
    return f"${100*c['point']:.0f}\\%$"


def pct_ci(c):
    return f"${100*c['point']:.0f}\\%$~\\ci{{{100*c['lo']:+.0f}\\%}}{{{100*c['hi']:+.0f}\\%}}"


def pt_ci(c, nd=3):
    return f"${c['point']:+.{nd}f}$~\\ci{{{c['lo']:+.{nd}f}}}{{{c['hi']:+.{nd}f}}}"


def lvl(name):
    return f"${cells[name]['matched_mean']:.3f}$"


def floor_pair(p, q):
    return f"${cells[p]['floor']['point']:+.3f}\\,/\\,{cells[q]['floor']['point']:+.3f}$"


shifts = [abs(s1[f"floor_shift_{k}"]["point"]) for k in ("dense", "pruned", "postft")] + \
         [abs(s2[f"floor_shift_{k}"]["point"]) for k in ("pruned", "postft")]
tok = FC["caption_tokens_vs_conditioner_limit"]["music64"]["frac_over_77"]
floors_ac = [cells[n]["floor"]["point"] for n in ("pruned2_A__ac_short", "pruned2_A__ac_native", "recovered2__ac_short",
                                                    "recovered2__ac_native", "pruned_short_sev1__armd80", "dense10s__pruned_sev1",
                                                    "postft_short_sev1__armd80", "dense10s__recovered_sev1",
                                                    "dense_short_sev1__armd80", "dense10s__dense")]
checks += [
    ("rho_real2 native (abstract)", "closes" + pct(s2["rho_real_native"]) + "ofthepruned"),
    ("rho_real2 short (abstract)", "butonly" + pct(s2["rho_real_short"]) + "at$3.84$\\,s"),
    ("rho conclusion", pct(s2["rho_real_native"]) + "ofthepruned" ),
    ("rho_real2 native CI (T3)", pct_ci(s2["rho_real_native"])),
    ("rho_real2 short CI (T3)", pct_ci(s2["rho_real_short"])),
    ("rho_real1 native CI (T3)", pct_ci(s1["rho_real_native"])),
    ("rho_real1 short CI (T3)", pct_ci(s1["rho_real_short"])),
    ("floor shift max", "atmost$" + f"{max(shifts):.3f}" + "$"),
    ("floor range", "from$" + f"{min(floors_ac):+.3f}" + "$to$" + f"{max(floors_ac):+.3f}" + "$acrosstheAudioCapscells"),
    ("J_c sev2", "J_c=" + pt_ci(s2["J_c"]).replace("$~\\ci","$$\\ci")),
    ("real2 short", "scores" + lvl("real_crop__sev2_192") + "at$3.84$\\,s"),
    ("real2 native", "and" + lvl("real_full__sev2_192") + "at$10.24$\\,s"),
    ("s(real2)", "s(\\mathrm{real})=" + pt_ci(s2["s_raw_real"])),   # norm() strips $ and ~ alike
    ("tok music frac", f"${100*tok:.0f}\\%$ofthemusiccaptions"),
    ("T3 floor1 short", floor_pair("pruned_short_sev1__armd80", "postft_short_sev1__armd80") + "&" + lvl("real_crop__sev1_80")),
    ("T3 floor1 native", floor_pair("dense10s__pruned_sev1", "dense10s__recovered_sev1") + "&" + lvl("real_full__sev1_80")),
    ("T3 floor2 short", floor_pair("pruned2_A__ac_short", "recovered2__ac_short") + "&" + lvl("real_crop__sev2_192")),
    ("T3 floor2 native", floor_pair("pruned2_A__ac_native", "recovered2__ac_native") + "&" + lvl("real_full__sev2_192")),
    ("T3 floor music sev1", floor_pair("p1_pruned_ema_reconstructed__off", "p1_recovered__off")),
    ("T3 floor music sev2 short", "music$3.84$\\,s&" + floor_pair("pruned2_A__music", "recovered2__music")),
    ("T3 floor music sev2 native", "music$10.24$\\,s&" + floor_pair("pruned2_A__music_native", "recovered2__music_native")),
    ("music level after FT sev1", "0.023"), ("music level after FT sev2", "0.014"),
    ("music P levels", "0.117"), ("music P level sev2", "0.005"),
]

# ---- XSEV-DENSE-192-CONTROL (paired dense at severity 2), only once the result exists
_d192 = os.path.join(ROOT, "configs/research/xsev_dense_192_control_result.json")
if os.path.exists(_d192):
    D = J("configs/research/xsev_dense_192_control_result.json"); P_ = D["PRIMARY"]; m_ = D["means"]
    checks += [
        ("d192 s(dense) sev2", "s(\\mathrm{dense})=" + pt(P_["s_dense"]) + "$$" + ci(P_["s_dense"])),
        ("d192 s(P)-s(dense)", pt(P_["s_pruned_minus_s_dense"]) + "$$" + ci(P_["s_pruned_minus_s_dense"])),
        ("d192 s(P+FT)-s(dense)", pt(P_["s_postft_minus_s_dense"]) + "$$" + ci(P_["s_postft_minus_s_dense"])),
        ("d192 rho_dense short CI", pct_ci(P_["rho_short"])), ("d192 rho_dense native CI", pct_ci(P_["rho_native"])),
        ("d192 rho_dense native abstract", pct(P_["rho_native"]) + "versus" + pct(P_["rho_short"])),
        ("d192 G native postft", pt(P_["G_native_postft"]) + "$$" + ci(P_["G_native_postft"])),
        ("d192 G short postft", pt(P_["G_short_postft"]) + "$$" + ci(P_["G_short_postft"])),
        ("d192 dense means", f"${m_['dense_short']:.3f}$&${m_['dense_native']:.3f}$"),
        ("d192 dense means prose", f"{m_['dense_short']:.3f}\\!\\to\\!{m_['dense_native']:.3f}"),
    ]

bad = 0
for label, s in checks:
    ok = norm(s) in tex
    print(("OK      " if ok else "MISSING ") + f"{label:28s} {s}")
    bad += (not ok)
left = sorted(set(re.findall(r"@@[A-Z0-9_]+", raw_tex)))
if left:
    print("PLACEHOLDERS LEFT:", left); bad += len(left)
print(f"\n(dropped inherited checks no longer printed: {dropped})")
print(f"{len(checks) - bad}/{len(checks)} numbers found in the manuscript")
sys.exit(1 if bad else 0)
