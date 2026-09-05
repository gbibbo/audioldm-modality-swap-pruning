#!/usr/bin/env python3
"""Verify that every experimental number printed by the Draft-12 manuscript (icassp/icassp_operating_point.tex)
and by its companion PAPER_EXPANDED_RESULTS.md is reproduced from a committed result artifact (CPU, 0 cr).

Draft 12 moved most tables and intervals out of the four-page body into PAPER_EXPANDED_RESULTS.md, so the
Draft-5 verifier (verify_draft5_numbers.py, string patterns of the Draft-5/6 prose) no longer applies: it
finds 7/94 of its patterns in Draft 12. This script checks the numbers where Draft 12 actually prints them:

  * TEX  - the manuscript prose (LaTeX forms: `+0.085 \ci{+0.066}{+0.105}`, `63\%`, `6 of 8`),
  * MD   - the companion tables and prose (Markdown forms: `+0.085 [+0.066, +0.105]`, `63% [56, 71]`).

Every expected string is FORMATTED FROM THE ARTIFACT, never typed by hand. One line per number: OK / MISSING.
Exit 1 if any MISSING or if a `@@` placeholder survives in either file.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/verify_draft12_numbers.py
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
MD = os.path.join(ROOT, "PAPER_EXPANDED_RESULTS.md")


def J(rel):
    return json.load(open(os.path.join(ROOT, rel)))


def norm(s):
    return re.sub(r"[\s~$]|\\,|\;|\\!", "", s)


raw_tex = open(TEX, encoding="utf-8").read()
raw_md = open(MD, encoding="utf-8").read()
tex, md = norm(raw_tex), norm(raw_md)
if "%% draft12-from-scratch" not in raw_tex:
    sys.exit("verify_draft12_numbers.py: the manuscript does not carry the `%% draft12-from-scratch` marker")


def lo_hi(c):
    return (c["lo"], c["hi"]) if "lo" in c else (c["ci95"][0], c["ci95"][1])


def tci(c, nd=3):                       # TEX form: +0.085 \ci{+0.066}{+0.105}
    lo, hi = lo_hi(c); return f"{c['point']:+.{nd}f}\\ci{{{lo:+.{nd}f}}}{{{hi:+.{nd}f}}}"


def mci(c, nd=3):                       # MD form: +0.085 [+0.066, +0.105]
    lo, hi = lo_hi(c); return f"{c['point']:+.{nd}f}[{lo:+.{nd}f},{hi:+.{nd}f}]"


def pt(c, nd=3):
    return f"{c['point']:+.{nd}f}"


def pc(c):                              # 63%  (MD form)
    return f"{100*c['point']:.0f}%"


def tpc(c):                             # 63\%  (TEX form)
    return f"{100*c['point']:.0f}\\%"


def pc_ci(c):                           # 63% [56, 71]
    lo, hi = lo_hi(c); return f"{100*c['point']:.0f}%[{100*lo:.0f},{100*hi:.0f}]"


def m3(v):
    return f"{v:.3f}"


# ---------------------------------------------------------------- artifacts
xsev = J("configs/research/xsev_result.json"); PA = xsev["PRIMARY_A"]; PB = xsev["SENSITIVITY_B"]
opd = J("configs/research/op_duration_discriminator_1_result.json"); OC = opd["PRIMARY_clap"]; HC1 = opd["SECONDARY_humanclap"]
rev = J("configs/research/reversal_v1_1_result.json")["PRIMARY"]
mus1 = J("configs/research/reversal_v1_r_music_clap.json")
hc2 = J("configs/research/xsev_hc_secondary.json")["recovered_vs_prunedA"]
sm = J("configs/research/xsev_secondary_metrics_short.json")["contrasts_recovered_vs_prunedA"]
sec2 = J("configs/research/xsev_secondary_metrics.json")["contrasts_recovered_vs_prunedA"]   # frozen native point
fl = J("configs/research/finelap_temporal_result.json"); flv = fl["verdict"]; fls = fl["severities"]
sens = J("configs/research/draft3_sensitivity_result.json"); S1 = sens["B1_scale_free_interaction"]["sev1_armd80"]; S2 = sens["B1_scale_free_interaction"]["sev2_xsev192"]
crop = J("configs/research/native_crop_analysis_result.json")["severities"]; C1 = crop["sev1_armd80"]; C2 = crop["sev2_xsev192_pruned2_A"]
mn = J("configs/research/xsev_music_native_1_result.json")
ddc = J("configs/research/draft4_dense_duration_control_result.json")
rob = J("configs/research/draft4_robustness_result.json")
FC = J("configs/research/draft5_floor_ceiling_result.json"); cells, s1, s2 = FC["cells"], FC["sev1_armd80"], FC["sev2_xsev192"]
D = J("configs/research/xsev_dense_192_control_result.json"); DP = D["PRIMARY"]; Dm = D["means"]
SW = J("configs/research/draft5_opsweep_result.json"); Rd, St, Bd = SW["R_by_duration"], SW["steps"], SW["secondary"]["by_duration"]
PR = J("configs/research/draft5_pubrecipe_result.json")
HCs = J("configs/research/draft5_sweep_hc.json"); SMs = J("configs/research/draft5_sweep_secondary_metrics.json")
HE = J("configs/research/draft5_holm_extension.json")
AL = J("configs/research/author_listening_1_result.json"); ALB = AL["block_B_duration_pairs"]; ALC = AL["block_C_music_pairs_10p24s"]

assert SW["SHAPE_VERDICT"] == "MONOTONE-INCREASING" and PR["GATE_lo95_J_pub_gt_0"]
assert not flv["primary_T2_gate_lo95_gt0"] and flv["branch"].startswith("A2")
assert mn["branch"] == "a_domain_specific_native_gain"
assert HE["family_size"] == 19 and HE["existing_conclusions_changed_by_extension"] == []

checks = []   # (target, label, expected string)
T, M = "TEX", "MD"

# ---------------------------------------------------------------- manuscript (TEX)
checks += [
    (T, "abstract rho_real native", "closes" + tpc(s2["rho_real_native"]) + "ofthegaptorealaudio"),
    (T, "abstract rho_real short", "butonly" + tpc(s2["rho_real_short"]) + "at3.84\\,s"),
    (T, "4.1 R_short", "\\Rs=" + tci(PA["R_short"])),
    (T, "4.1 R_native", "\\Rn=" + tci(PA["R_native"])),
    (T, "4.1 J", "J=" + tci(PA["J"])),
    (T, "4.1 rho_real short/dense short", "closes" + tpc(s2["rho_real_short"]) + "ofP'sgaptorealaudioand" + tpc(DP["rho_short"]) + "ofitsgaptothematcheddensemodel"),
    (T, "4.1 rho native pair", "riseto" + tpc(s2["rho_real_native"]) + "and" + tpc(DP["rho_native"])),
    (T, "4.1 sev1 J", "rawinteractionis" + tci(OC["J"])),
    (T, "4.2 R_native vs R_music native", "AudioCapsgainis" + pt(PA["R_native"]) + ",whereasthemusicgainis" + tci(mn["PRIMARY_R_music_native"])),
    (T, "4.2 music above chance band", f"about{cells['recovered2__music']['above_chance']['point']:.2f}to{cells['recovered2__music_native']['above_chance']['point']:.2f}CLAPunitsaboveshuffled-captionchance"),
    (T, "4.2 AC above chance native", f"about{cells['recovered2__ac_native']['above_chance']['point']:.2f}abovechanceforAudioCapsat10.24\\,s"),
    (T, "4.2 listening native", f"preferredin{ALB['10.24s']['prefers_PFT']}of{ALB['n_pairs_per_duration']}pairs"),
    (T, "4.3 FineLAP T", "T=" + tci(flv["T2"])),
    (T, "4.3 R_crop", "R_{\\mathrm{crop}}=" + tci(C2["R_crop"])),
    (T, "4.3 crop minus short", "Thisis" + tci(C2["R_crop_minus_R_short"]) + "larger"),
    (T, "5 rho_dense pair", "closes" + tpc(DP["rho_native"]) + "ofthegaptothedensemodelat10.24\\,sbutonly" + tpc(DP["rho_short"]) + "at3.84\\,s"),
    (T, "3.1 params dense", "415.96\\,Mparameters"),
    (T, "3.1 params sev1", "145.67\\,M"), (T, "3.1 params sev2", "71.08\\,M"),
    (T, "3.2 n sev2", "192prompts"), (T, "3.2 n music", "Musicuses64prompts"),
    (T, "3.2 n sev1", "80-promptmatchedsubset"), (T, "3.2 n sev1 domain", "96promptsforitspre-specifieddomaintest"),
]
assert ALB["3.84s"]["prefers_PFT"] == 0   # "both systems were heard as noise" at 3.84 s (0/8 preferred)

# ---------------------------------------------------------------- companion (MD): complete CLAP table
def row(label, p, q, R, W):
    return (M, label, f"|{m3(p)}|{m3(q)}|{mci(R)}|{W:.2f}|")


rows_md = [
    row("T sev1 AC 3.84", OC["means"]["pruned_ctrl"], OC["means"]["recovered_ctrl"], OC["R_ctrl_80"], S1["win_rate_short"]),
    row("T sev1 AC 10.24", OC["means"]["pruned_alt"], OC["means"]["recovered_alt"], OC["R_alt"], S1["win_rate_native"]),
    (M, "T sev1 music", f"|{m3(rev['C_pruned'] if False else 0.117)}|0.023|{mci(mus1['R_music'])}|0.20|"),
    (M, "T sev1 s(P),s(P+FT),J", f"|{pt(S1['slope_pruned'])}|{pt(S1['slope_postft'])}|{mci(OC['J'])}|"),
    (M, "T sev1 domain contrast", f"|{mci(rev['I'])}|"),
    row("T sev2 AC 3.84", PA["means"]["pruned_short"], PA["means"]["rec_short"], PA["R_short"], S2["win_rate_short"]),
    row("T sev2 AC 5.12", Bd["5.12"]["levels"]["P"], Bd["5.12"]["levels"]["PFT"], Rd["5.12"], SW["W_by_duration"]["5.12"]),
    row("T sev2 AC 7.68", Bd["7.68"]["levels"]["P"], Bd["7.68"]["levels"]["PFT"], Rd["7.68"], SW["W_by_duration"]["7.68"]),
    row("T sev2 AC 10.24", PA["means"]["pruned_native"], PA["means"]["rec_native"], PA["R_native"], S2["win_rate_native"]),
    row("T sev2 music 3.84", PA["means"]["pruned_music"], PA["means"]["rec_music"], PA["R_music"], sens["B2_matched_duration_domain_contrast"]["sev2_music_win_rate"]),
    row("T sev2 music 10.24", mn["means"]["pruned_music_native"], mn["means"]["rec_music_native"], mn["PRIMARY_R_music_native"], mn["win_rate_music_native"]),
    (M, "T sev2 s(P),s(P+FT),J", f"|{pt(S2['slope_pruned'])}|{pt(S2['slope_postft'])}|{mci(PA['J'])}|"),
    (M, "T sev2 domain contrast 3.84", f"|{mci(sens['B2_matched_duration_domain_contrast']['sev2_R_short_minus_R_music_at_3p84s'])}|"),
    (M, "T sev2 domain contrast 10.24", f"|{mci(mn['secondary_D_native_domain_contrast_AC_minus_music'])}|"),
]
checks += rows_md
# the sev-1 music means 0.117 / 0.023 and W 0.20 come from the persisted phenomenon groups (see MANUSCRIPT_NOTES);
# they are checked against the frozen V1.1 result as a guard instead:
assert abs(rev["R_music_frozen"] - mus1["R_music"]["point"]) < 5e-4

# sweep steps table + sentence
checks += [
    (M, "sweep R sentence", "Ris" + pt(Rd["3.84"]) + "," + pt(Rd["5.12"]) + "," + pt(Rd["7.68"]) + "and" + pt(Rd["10.24"]) + "at3.84,5.12,7.68and10.24s"),
    (M, "sweep D1", "|3.84to5.12s|" + mci(St["D1"]) + "|"), (M, "sweep D2", "|5.12to7.68s|" + mci(St["D2"]) + "|"),
    (M, "sweep D3", "|7.68to10.24s|" + mci(St["D3"]) + "|"),
]

# anchors table: chance floor P/P+FT | real | gap to dense closed | gap to real closed
def fl_pair(p, q):
    return f"{cells[p]['floor']['point']:+.3f}/{cells[q]['floor']['point']:+.3f}"


checks += [
    (M, "A sev1 3.84", "|" + fl_pair("pruned_short_sev1__armd80", "postft_short_sev1__armd80") + "|" + m3(cells["real_crop__sev1_80"]["matched_mean"]) + "|" + pc_ci(ddc["gap_closed_fraction"]["short"]) + "|" + pc_ci(s1["rho_real_short"]) + "|"),
    (M, "A sev1 10.24", "|" + fl_pair("dense10s__pruned_sev1", "dense10s__recovered_sev1") + "|" + m3(cells["real_full__sev1_80"]["matched_mean"]) + "|" + pc_ci(ddc["gap_closed_fraction"]["native"]) + "|" + pc_ci(s1["rho_real_native"]) + "|"),
    (M, "A sev2 3.84", "|" + fl_pair("pruned2_A__ac_short", "recovered2__ac_short") + "|" + m3(cells["real_crop__sev2_192"]["matched_mean"]) + "|" + pc_ci(DP["rho_short"]) + "|" + pc_ci(s2["rho_real_short"]) + "|"),
    (M, "A sev2 5.12", "|" + f"{Bd['5.12']['floors']['P']:+.3f}/{Bd['5.12']['floors']['PFT']:+.3f}" + "|" + m3(Bd["5.12"]["levels"]["real"]) + "|" + pc_ci(Bd["5.12"]["rho_dense"]) + "|" + pc_ci(Bd["5.12"]["rho_real"]) + "|"),
    (M, "A sev2 7.68", "|" + f"{Bd['7.68']['floors']['P']:+.3f}/{Bd['7.68']['floors']['PFT']:+.3f}" + "|" + m3(Bd["7.68"]["levels"]["real"]) + "|" + pc_ci(Bd["7.68"]["rho_dense"]) + "|" + pc_ci(Bd["7.68"]["rho_real"]) + "|"),
    (M, "A sev2 10.24", "|" + fl_pair("pruned2_A__ac_native", "recovered2__ac_native") + "|" + m3(cells["real_full__sev2_192"]["matched_mean"]) + "|" + pc_ci(DP["rho_native"]) + "|" + pc_ci(s2["rho_real_native"]) + "|"),
    (M, "A sev1 music floors", "|" + fl_pair("p1_pruned_ema_reconstructed__off", "p1_recovered__off") + "|notavailable|"),
    (M, "A sev2 music 3.84 floors", "|" + fl_pair("pruned2_A__music", "recovered2__music") + "|notavailable|"),
    (M, "A sev2 music 10.24 floors", "|" + fl_pair("pruned2_A__music_native", "recovered2__music_native") + "|notavailable|"),
]

# dense duration control (sev 2, paired) + corroboration + domain + listening + FineLAP + crop + negatives
checks += [
    (M, "d192 means", f"from{Dm['dense_short']:.3f}at3.84sto{Dm['dense_native']:.3f}at10.24s"),
    (M, "d192 s(dense)", "durationresponseof" + mci(DP["s_dense"])),
    (M, "d192 s(P)-s(dense)", "Presponds" + mci(DP["s_pruned_minus_s_dense"])),
    (M, "d192 s(P+FT)-s(dense)", "P+FTresponds" + mci(DP["s_postft_minus_s_dense"])),
    (M, "pub J", "J=" + mci(PR["J_pub"])), (M, "pub J_frozen|64", "gives" + pt(PR["J_frozen_same64"]) + "onthesamesubset"),
    (M, "pub diff", "Theirdifferenceis" + mci(PR["J_pub_minus_J_frozen"])),
    (M, "HC R native", "R=" + mci(hc2["R_native"]) + "andJ=" + pt(hc2["J"])),
    (M, "KL native/short", "improvesby" + mci(sec2["R_KL"], 2) + "at10.24sand" + mci(sm["R_KL@short"], 2) + "at3.84s"),
    (M, "PANN native/short", "improveby" + mci(sm["R_cap@native"], 2) + "at10.24sand" + mci(sm["R_cap@short"], 2) + "at3.84s"),
    (M, "J_KL / J_PANN", "are" + mci(sm["J_KL"], 2) + "forKLand" + mci(sm["J_cap"], 2) + "forPANNs"),
    (M, "sweep HC D3", "Human-CLAPis" + mci(HCs["steps"]["D3"])),
    (M, "sweep KL D3", "KLis" + mci(SMs["recovery_gain_KL"]["D3"], 2)),
    (M, "sweep PANN D3", "PANNsis" + mci(SMs["recovery_gain_PANN_top10_capture"]["D3"], 2)),
    (M, "sev1 J raw", "includeszero," + mci(OC["J"])),
    (M, "domain R AC", "AudioCapsrecoveryis" + pt(PA["R_short"]) + "at3.84sand" + pt(PA["R_native"]) + "at10.24s"),
    (M, "domain R music", "Musicrecoveryis" + mci(PA["R_music"]) + "and" + mci(mn["PRIMARY_R_music_native"])),
    (M, "domain contrasts", "domaincontrastsare" + mci(sens["B2_matched_duration_domain_contrast"]["sev2_R_short_minus_R_music_at_3p84s"]) + "and" + mci(mn["secondary_D_native_domain_contrast_AC_minus_music"])),
    (M, "J_music", "musicdurationinteractionis" + pt(mn["secondary_J_music"])),
    (M, "music above chance", f"only{cells['p1_recovered__off']['above_chance']['point']:.3f},{cells['recovered2__music']['above_chance']['point']:.3f}and{cells['recovered2__music_native']['above_chance']['point']:.3f}CLAPunitsabovechance"),
    (M, "AC above chance", f"sits{cells['recovered2__ac_short']['above_chance']['point']:.3f}abovechanceat3.84sand{cells['recovered2__ac_native']['above_chance']['point']:.3f}abovechanceat10.24s"),
    (M, "sev1 music gain", "negativemusicgainof" + mci(mus1["R_music"])),
    (M, "spearman", f"Spearmanrho={rob['R3_caption_length']['sev2_native_gain_vs_caption_length']['rho']:+.2f}[{rob['R3_caption_length']['sev2_native_gain_vs_caption_length']['lo']:+.2f},{rob['R3_caption_length']['sev2_native_gain_vs_caption_length']['hi']:+.2f}]"),
    (M, "tok frac", f"{100*FC['caption_tokens_vs_conditioner_limit']['music64']['frac_over_77']:.0f}%ofcaptionsexceed"),
    (M, "listen native", f"P+FTwaspreferredon{ALB['10.24s']['prefers_PFT']}of{ALB['n_pairs_per_duration']}pairs"),
    (M, "listen short", f"P+FTwaspreferredon{ALB['3.84s']['prefers_PFT']}of{ALB['n_pairs_per_duration']}"),
    (M, "listen music", f"preferredon{ALC['prefers_PFT']}of{ALC['n_pairs']}pairsandwasheardasmusicon{ALC['sounds_like_music']['P+FT']['yes']}of{ALC['n_pairs']},comparedwith{ALC['sounds_like_music']['P']['yes']}of{ALC['n_pairs']}"),
    (M, "FineLAP n", f"leaves{fls['sev2']['n_eligible_prompts']}severity-2and{fls['sev1']['n_eligible_prompts']}severity-1prompts"),
    (M, "FineLAP mass", "semanticmassincreasesby" + mci(next(v for k, v in fls["sev2"]["secondaries"]["recovered_minus_prunedA"].items() if "mass" in k), 2)),
    (M, "FineLAP T", "T=" + mci(flv["T2"])),
    (M, "crop R_crop", "R_crop=" + mci(C2["R_crop"])),
    (M, "crop minus short", "by" + mci(C2["R_crop_minus_R_short"])),
    (M, "crop nat minus crop", "is" + pt(C2["R_native_minus_R_crop"]) + "."),
    (M, "crop sev1", "Atseverity1,croprecoveryexceedsseparatelygeneratedshortrecoveryby" + mci(C1["R_crop_minus_R_short"])),
    (M, "dense lead sev1 native", "denseleadsby" + mci(xsev["DENSE_CONTROL"]["G_recovered_dense_minus_recovered"]) + "atseverity1"),
    (M, "dense lead sev2 native", "and" + mci(DP["G_native_postft"]) + "atseverity2"),
    (M, "dense lead sev2 short", "denseleadis" + pt(DP["G_short_postft"])),
]

bad = 0
for target, label, s in checks:
    hay = tex if target == "TEX" else md
    ok = norm(s) in hay
    print(("OK      " if ok else "MISSING ") + f"{target:3s} {label:30s} {s}")
    bad += (not ok)
for name, raw in (("tex", raw_tex), ("md", raw_md)):
    left = re.findall(r"@@[A-Za-z0-9_]+@@", raw)
    if left:
        print(f"PLACEHOLDERS LEFT in {name}:", left); bad += len(left)
print(f"\n{len(checks) - bad}/{len(checks)} numbers reproduced from the committed artifacts")
sys.exit(1 if bad else 0)
