#!/usr/bin/env python3
"""Verify that every headline number in icassp/icassp_operating_point.tex is reproduced from a
committed artifact (CPU, 0 cr). Prints one line per number: OK / MISSING. Exit 1 if any MISSING.

Each check = (label, value formatted exactly as the manuscript prints it). The tex is searched for the
formatted string (whitespace-insensitive, '~' and '\\,' ignored). Numbers the tex rounds differently
must be listed with the manuscript's rounding, so the table below doubles as a provenance map.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")


def J(rel):
    return json.load(open(os.path.join(ROOT, rel)))


def norm(s):
    return re.sub(r"[\s~$]|\\,|\\;|\\!", "", s)


tex = norm(open(TEX, encoding="utf-8").read())
xsev = J("configs/research/xsev_result.json"); PA = xsev["PRIMARY_A"]; PB = xsev["SENSITIVITY_B"]; DC = xsev["DENSE_CONTROL"]
opd = J("configs/research/op_duration_discriminator_1_result.json"); OC = opd["PRIMARY_clap"]; HC1 = opd["SECONDARY_humanclap"]
rev = J("configs/research/reversal_v1_1_result.json")["PRIMARY"]
mus1 = J("configs/research/reversal_v1_r_music_clap.json")["R_music"]
hc2 = xsev["secondaries"]["human_clap"]  # frozen HC values as recorded in the primary result artifact
sec2 = J("configs/research/xsev_secondary_metrics.json")
fl = J("configs/research/finelap_temporal_result.json")["severities"]["sev2"]
sens = J("configs/research/draft3_sensitivity_result.json"); S1 = sens["B1_scale_free_interaction"]["sev1_armd80"]; S2 = sens["B1_scale_free_interaction"]["sev2_xsev192"]
crop = J("configs/research/native_crop_analysis_result.json")["severities"]; C1 = crop["sev1_armd80"]; C2 = crop["sev2_xsev192_pruned2_A"]


def ci(c, nd=3):
    lo = c["lo"] if "lo" in c else c["ci95"][0]; hi = c["hi"] if "hi" in c else c["ci95"][1]
    return f"\\ci{{{lo:+.{nd}f}}}{{{hi:+.{nd}f}}}"


def pt(c, nd=3):
    return f"{c['point']:+.{nd}f}"


def mean(v, nd=3):
    return f"{v:.{nd}f}"


checks = [
    # severity 2 primary
    ("sev2 R_native", pt(PA["R_native"]) + "^{\\dagger}" + ci(PA["R_native"])),
    ("sev2 R_short", pt(PA["R_short"]) + "^{\\dagger}" + ci(PA["R_short"])),
    ("sev2 R_music", pt(PA["R_music"]) + ci(PA["R_music"])),
    ("sev2 J", pt(PA["J"]) + "^{\\ddagger}" + ci(PA["J"])),
    ("sev2 J text", "J=" + pt(PA["J"]) + "$$" + ci(PA["J"])),
    ("sev2 J B'", "J=" + pt(PB["J"])),
    ("sev2 equiv 90CI", ci(PA["equiv_90ci"])),
    ("sev2 means native", mean(PA["means"]["pruned_native"]) + "$&$" + mean(PA["means"]["rec_native"])),
    ("sev2 means short", mean(PA["means"]["pruned_short"]) + "$&$" + mean(PA["means"]["rec_short"])),
    ("sev2 means music", mean(PA["means"]["pruned_music"]) + "$&$" + mean(PA["means"]["rec_music"])),
    ("abstract 0.055->0.299", "0.055\\to0.299"),
    ("abstract 0.015->0.100", "0.015\\to0.100"),
    # severity 1 (Arm-D)
    ("sev1 R_short", pt(OC["R_ctrl_80"]) + ci(OC["R_ctrl_80"])),
    ("sev1 R_native", pt(OC["R_alt"]) + "^{\\dagger}" + ci(OC["R_alt"])),
    ("sev1 J", pt(OC["J"]) + ci(OC["J"])),
    ("sev1 means short", mean(OC["means"]["pruned_ctrl"]) + "$&$" + mean(OC["means"]["recovered_ctrl"])),
    ("sev1 means native", mean(OC["means"]["pruned_alt"]) + "$&$" + mean(OC["means"]["recovered_alt"])),
    ("sev1 HC J", "J_{\\mathrm{HC}}=" + pt(HC1["J"]) + "$$" + ci(HC1["J"])),
    # severity 1 domain test + music
    ("sev1 R_AC", "R_{\\mathrm{AC}}=" + pt(rev["R_AC"]) + "$$" + ci(rev["R_AC"])),
    ("sev1 I domain", pt(rev["I"]) + "^{\\dagger}" + ci(rev["I"])),
    ("sev1 R_music", pt(mus1) + "^{\\dagger}" + ci(mus1)),
    ("dense 3.84 / 10.24", f"{rev['C_dense']:.3f}\\to{DC['C_dense_10s']:.3f}"),
    ("dense gaps", pt(DC["G_pruned_dense_minus_pruned"]) + "$$" + ci(DC["G_pruned_dense_minus_pruned"])),
    ("dense gap postft", pt(DC["G_recovered_dense_minus_recovered"]) + "$$" + ci(DC["G_recovered_dense_minus_recovered"])),
    # HC sev2, KL, PANN, FAD
    ("HC2 R_native", pt(hc2["R_native_A"]) + "$$" + ci(hc2["R_native_A"])),
    ("HC2 J", "J=" + pt(hc2["J_A"])),
    ("HC2 music", pt(hc2["R_music_A"]) + "$$" + ci(hc2["R_music_A"])),
    ("KL means", f"{sec2['means']['KL']['recovered2']:.2f}$vs.\\${sec2['means']['KL']['pruned2_A']:.2f}"),
    ("KL delta", "\\Delta=" + f"{sec2['contrasts_recovered_vs_prunedA']['R_KL']['point']:+.2f}" + "$$" + ci(sec2['contrasts_recovered_vs_prunedA']['R_KL'], 2)),
    ("PANN means", f"{sec2['means']['PANN_top10_capture']['recovered2']:.2f}$vs.\\${sec2['means']['PANN_top10_capture']['pruned2_A']:.2f}"),
    ("PANN delta", "\\Delta=" + f"{sec2['contrasts_recovered_vs_prunedA']['R_cap']['point']:+.2f}" + "$$" + ci(sec2['contrasts_recovered_vs_prunedA']['R_cap'], 2)),
    ("FAD", f"{sec2['FAD_vggish']['recovered2']:.2f}$vs.\\${sec2['FAD_vggish']['pruned2_A']:.1f}"),
    # FineLAP
    ("FineLAP D_early", "D_{\\mathrm{early}}=" + pt(fl["contrasts"]["pruned_A"]["D_early"])),
    ("FineLAP D_late", "D_{\\mathrm{late}}=" + pt(fl["contrasts"]["pruned_A"]["D_late"])),
    ("FineLAP T", "T=" + pt(fl["contrasts"]["pruned_A"]["T"]) + "$$" + ci(fl["contrasts"]["pruned_A"]["T"])),
    ("FineLAP mass", pt(fl["secondaries"]["recovered_minus_prunedA"]["semantic_mass"], 2) + "$$" + ci(fl["secondaries"]["recovered_minus_prunedA"]["semantic_mass"], 2)),
    ("FineLAP occupancy", "occupancy$" + pt(fl["secondaries"]["recovered_minus_prunedA"]["occupancy"], 2)),
    ("FineLAP coverage", "quartercoverage$" + pt(fl["secondaries"]["recovered_minus_prunedA"]["quarter_coverage"], 2)),
    ("FineLAP peak", "peak$" + pt(fl["secondaries"]["recovered_minus_prunedA"]["peak"], 2)),
    # sensitivity B1/B2
    ("sev1 win rates", f"{S1['win_rate_short']:.2f}\\to{S1['win_rate_native']:.2f}"),
    ("sev1 dW", "\\DeltaW=" + pt(S1["dW_native_minus_short"], 2) + "$$" + ci(S1["dW_native_minus_short"], 2)),
    ("sev2 win rates", f"{S2['win_rate_short']:.2f}\\to{S2['win_rate_native']:.2f}"),
    ("sev2 dW", pt(S2["dW_native_minus_short"], 2) + "$$" + ci(S2["dW_native_minus_short"], 2)),
    ("sev2 slope P", pt(S2["slope_pruned"]) + "$$" + ci(S2["slope_pruned"])),
    ("sev2 slope P+FT", pt(S2["slope_postft"]) + "$$" + ci(S2["slope_postft"])),
    ("sev2 domain gap 3.84", pt(sens["B2_matched_duration_domain_contrast"]["sev2_R_short_minus_R_music_at_3p84s"]) + "^{\\dagger}" + ci(sens["B2_matched_duration_domain_contrast"]["sev2_R_short_minus_R_music_at_3p84s"])),
    ("table W sev1", f"{S1['win_rate_short']:.2f}$\\\\" ), ("table W sev2 native", f"&${S2['win_rate_native']:.2f}$\\\\"),
    ("music W sev2", f"{sens['B2_matched_duration_domain_contrast']['sev2_music_win_rate']:.2f}"),
    # crop
    ("crop sev1 R_crop", "R_{\\mathrm{crop}}=" + pt(C1["R_crop"]) + "$$" + ci(C1["R_crop"])),
    ("crop sev1 nat-crop", pt(C1["R_native_minus_R_crop"]) + "$$" + ci(C1["R_native_minus_R_crop"])),
    ("crop sev1 crop-short", pt(C1["R_crop_minus_R_short"]) + "$$" + ci(C1["R_crop_minus_R_short"])),
    ("crop sev2 R_crop", "R_{\\mathrm{crop}}=" + pt(C2["R_crop"]) + "$$" + ci(C2["R_crop"])),
    ("crop sev2 crop-short", pt(C2["R_crop_minus_R_short"]) + "$$" + ci(C2["R_crop_minus_R_short"])),
    ("crop sev2 nat-crop", pt(C2["R_native_minus_R_crop"]) + "$$" + ci(C2["R_native_minus_R_crop"])),
]
# music @10.24 s (only once the result exists)
mn_path = os.path.join(ROOT, "configs/research/xsev_music_native_1_result.json")
if os.path.exists(mn_path):
    MN = json.load(open(mn_path))
    checks += [
        ("MN R_music_native", pt(MN["PRIMARY_R_music_native"]) + "$$" + ci(MN["PRIMARY_R_music_native"])),
        ("MN means", mean(MN["means"]["pruned_music_native"]) + "$&$" + mean(MN["means"]["rec_music_native"])),
        ("MN W", f"{MN['win_rate_music_native']:.2f}"),
        ("MN D_nat", pt(MN["secondary_D_native_domain_contrast_AC_minus_music"]) ),
    ]

bad = 0
for label, s in checks:
    ok = norm(s) in tex
    print(("OK      " if ok else "MISSING ") + f"{label:28s} {s}")
    bad += (not ok)
print(f"\n{len(checks) - bad}/{len(checks)} numbers found in the manuscript")
sys.exit(1 if bad else 0)
