#!/usr/bin/env python3
"""Verify that every headline number in icassp/icassp_operating_point.tex (Draft 4) is reproduced from a
committed artifact (CPU, 0 cr). Extends verify_draft3_numbers.py with the Draft-4 additions (matched
dense duration control, multiplicity, rank-scale J, caption-length check). Prints one line per number:
OK / MISSING. Exit 1 if any MISSING.
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


tex = norm(open(TEX, encoding="utf-8").read())

# ---- reuse the Draft-3 check list (its module runs the checks at import time only under __main__ guard? no:
# it executes at import). Re-implement the list by exec-ing the file with the print/exit suppressed.
src = open(os.path.join(HERE, "verify_draft3_numbers.py"), encoding="utf-8").read()
src = src.split("bad = 0")[0]                      # keep definitions + `checks` list, drop the loop/exit
ns = {"__file__": os.path.join(HERE, "verify_draft3_numbers.py")}
exec(compile(src, "verify_draft3_numbers.py", "exec"), ns)
checks = list(ns["checks"])
ci, pt, mean = ns["ci"], ns["pt"], ns["mean"]

# ---- Draft-3 strings that Draft 4 intentionally changed (replace, do not drop)
DC = ns["DC"]
checks = [c for c in checks if c[0] not in ("dense 3.84 / 10.24", "sev2 slope P", "sev2 slope P+FT", "sev1 win rates", "sev2 win rates")]
S1 = ns["S1"]; S2 = ns["S2"]
checks += [("sev1 win rates", f"from${S1['win_rate_short']:.2f}$to${S1['win_rate_native']:.2f}$"),
           ("sev2 win rates", f"from${S2['win_rate_short']:.2f}$to${S2['win_rate_native']:.2f}$")]

ddc = J("configs/research/draft4_dense_duration_control_result.json")
rob = J("configs/research/draft4_robustness_result.json")
S2 = ns["S2"]


def pctf(c):
    return f"{100*c['point']:.0f}\\%$$\\ci{{{100*c['lo']:+.0f}\\%}}{{{100*c['hi']:+.0f}\\%}}"


fam = {t["contrast"]: t for t in rob["R1_multiplicity"]["family_all"]}
r2 = rob["R2_rank_scale_J"]
r3 = rob["R3_caption_length"]
checks += [
    # matched dense duration control (sev-1, n=80)
    ("dense matched 3.84 / 10.24", f"({ddc['means']['dense_short']:.3f}\\to{ddc['means']['dense_native']:.3f})"),
    ("s(dense)", "s(\\mathrm{dense})=" + pt(ddc["slopes"]["dense"]) + "$$" + ci(ddc["slopes"]["dense"])),
    ("s(P) sev1", "s(\\mathrm{P})=" + pt(ddc["slopes"]["pruned"]) + "$$" + ci(ddc["slopes"]["pruned"])),
    ("s(P+FT) sev1", "s(\\PFT)=" + pt(ddc["slopes"]["postft"]) + "$$" + ci(ddc["slopes"]["postft"])),
    ("dslope P-dense", pt(ddc["dslope_pruned_minus_dense"]) + "$$" + ci(ddc["dslope_pruned_minus_dense"])),
    ("dslope P+FT-dense", pt(ddc["dslope_postft_minus_dense"]) + "$$" + ci(ddc["dslope_postft_minus_dense"])),
    ("gap closed short", pctf(ddc["gap_closed_fraction"]["short"])),
    ("gap closed native", pctf(ddc["gap_closed_fraction"]["native"])),
    ("table s sev1", f"duration$s$;$J$&${ddc['slopes']['pruned']['point']:+.3f}$&${ddc['slopes']['postft']['point']:+.3f}$&${ns['OC']['J']['point']:+.3f}$"),
    ("caption dense s", f"matcheddenseresponseatseverity~1:${ddc['slopes']['dense']['point']:+.3f}$"),
    # sev-2 duration responses (frozen sensitivity artifact) in the new s(.) notation + table row
    ("s(P) sev2", "s(\\mathrm{P})=" + pt(S2["slope_pruned"]) + "$$" + ci(S2["slope_pruned"])),
    ("s(P+FT) sev2", "s(\\PFT)=" + pt(S2["slope_postft"]) + "$$" + ci(S2["slope_postft"])),
    ("table s sev2", f"duration$s$;$J$&${S2['slope_pruned']['point']:+.3f}$&${S2['slope_postft']['point']:+.3f}$&${ns['PA']['J']['point']:+.3f}^{{\\ddagger}}$"),
    # multiplicity
    ("holm sev1 R_nat p", f"\\Rn($p={fam['sev1 R_native (n=80)']['asl_two_sided']:.3f}$)"),
    ("holm sev1 J p", f"J($p={fam['sev1 J (n=80)']['asl_two_sided']:.3f}$)"),
    ("holm family size", f"{len(rob['R1_multiplicity']['family_all'])}$contrasts"),
    # rank-scale J
    ("median J sev1", pt(r2["sev1_armd80"]["median_j"]) + "$$" + ci(r2["sev1_armd80"]["median_j"])),
    ("median J sev2", pt(r2["sev2_xsev192"]["median_j"]) + "$$" + ci(r2["sev2_xsev192"]["median_j"])),
    ("wilcoxon sev1", f"p={r2['sev1_armd80']['wilcoxon_signed_rank']['p_two_sided']:.2f}"),
    ("rank J sev1", pt(r2["rank_transformed_J"]["sev1_armd80"]) + "$$" + ci(r2["rank_transformed_J"]["sev1_armd80"])),
    ("rank J sev2", pt(r2["rank_transformed_J"]["sev2_xsev192"]) + "$$" + ci(r2["rank_transformed_J"]["sev2_xsev192"])),
    # caption length + floor
    ("caption words", f"median${r3['caption_words']['music_sev2_median']:g}$vs.\\${r3['caption_words']['audiocaps_sev2_median']:.0f}$words"),
    ("spearman", "\\rho=" + f"{r3['sev2_native_gain_vs_caption_length']['rho']:+.2f}" + "$$" + f"\\ci{{{r3['sev2_native_gain_vs_caption_length']['lo']:+.2f}}}{{{r3['sev2_native_gain_vs_caption_length']['hi']:+.2f}}}"),
    ("pruned music vs AC native", f"onmusic(${r3['pruned_absolute_music_vs_indomain_native_sev2']['pruned2_A_music_10p24s_mean']:.3f}$)thanin-domain(${r3['pruned_absolute_music_vs_indomain_native_sev2']['pruned2_A_audiocaps_10p24s_mean']:.3f}$)"),
    # music duration interaction
    ("J_music", pt(J("configs/research/xsev_music_native_1_result.json")["secondary_J_music"]) + "$$" + ci(J("configs/research/xsev_music_native_1_result.json")["secondary_J_music"])),
    # FineLAP n
    ("FineLAP n", f"{J('configs/research/finelap_temporal_result.json')['severities']['sev2']['n_eligible_prompts']}$and${J('configs/research/finelap_temporal_result.json')['severities']['sev1']['n_eligible_prompts']}$eligible"),
]
# wilcoxon sev2 p<1e-17 check
assert r2["sev2_xsev192"]["wilcoxon_signed_rank"]["p_two_sided"] < 1e-17
# Holm claims: every sev-2 dagger survives; at sev-1 only the music rows
assert all(t["holm_pass_005"] for t in rob["R1_multiplicity"]["family_all"] if t["contrast"].startswith("sev2") and t["lo"] * t["hi"] > 0)
s1pass = {t["contrast"]: t["holm_pass_005"] for t in rob["R1_multiplicity"]["family_all"] if t["contrast"].startswith("sev1")}
assert s1pass["sev1 R_music (n=64)"] and s1pass["sev1 domain 3.84 s = R_AC - R_music"] and not s1pass["sev1 R_native (n=80)"] and not s1pass["sev1 J (n=80)"]
# batch-stability claim (<= 0.002 on means)
bd = ddc["batch_composition_diagnostic"]
assert max(abs(v[0] - v[1]) for k, v in bd.items() if k.endswith("_mean_192call_vs_80call")) <= 0.002

bad = 0
for label, s in checks:
    ok = norm(s) in tex
    print(("OK      " if ok else "MISSING ") + f"{label:28s} {s}")
    bad += (not ok)
print(f"\n{len(checks) - bad}/{len(checks)} numbers found in the manuscript")
sys.exit(1 if bad else 0)
