#!/usr/bin/env python3
"""Third review (Draft 14), items 3 and 4 — POST-HOC descriptive re-analyses of existing per-prompt scores (CPU only,
0 cr; no generation). NOT pre-specified in any frozen protocol; changes no frozen verdict.

  * item 3 — is the severity-2 duration interaction J as sensitive to prompt selection as severity 1's? The 192-prompt
    manifest (configs/research/xsev_audiocaps_manifest.json) is one outcome-blind seeded-hash draw whose prompt_index
    follows the ascending selection hash, so its first and second halves are exchangeable outcome-blind sub-samples:
    J on prompt_index 0-95 vs 96-191 and their unpaired difference. Also the same-prompt cross-severity comparison:
    the severity-1 "new 96" (E8) ARE prompt_index 0-95 of this manifest, so J_sev1 - J_sev2 is paired by prompt.
  * item 4 — hip-hop shows no duration interaction (E7 J_music_127, pre-specified) while AudioCaps does: the unpaired
    AudioCaps-minus-hip-hop difference of J, and (if the Clotho per-prompt scores are on disk) AudioCaps-minus-Clotho.

Writes configs/research/r2_posthoc_review3.json. Bootstrap conventions = r2_verdict.py (B = 10 000, percentile,
unit = prompt), seed namespace suffixed "|POSTHOC3" so no frozen draw is re-used.
Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/r2_posthoc_review3.py
"""
import hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
import r2_verdict as V

NS = V.NS + "|POSTHOC3"
OUT = "configs/research/r2_posthoc_review3.json"
MANIFEST = "configs/research/xsev_audiocaps_manifest.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cell(P3, Q3, P10, Q10, ns, n):
    bt = V.Boot(ns, n)
    return {"n": n, "R_3.84": bt.ci(Q3 - P3), "R_10.24": bt.ci(Q10 - P10), "J": bt.ci((Q10 - P10) - (Q3 - P3)),
            "levels": {"P_3.84": float(P3.mean()), "PFT_3.84": float(Q3.mean()), "P_10.24": float(P10.mean()), "PFT_10.24": float(Q10.mean())},
            "sd_J": float(((Q10 - P10) - (Q3 - P3)).std(ddof=1))}


def main():
    out = {"artifact": "r2_posthoc_review3",
           "class": "POST-HOC descriptive re-analysis of existing per-prompt scores (third review, items 3 and 4); "
                    "not pre-specified in any frozen protocol; changes no frozen verdict",
           "bootstrap": {"B": V.B, "seed_namespace": NS, "unit": "prompt", "ci": "percentile 95%"}, "SESOI": V.SESOI, "inputs": {}}
    # ---- severity-2 frozen cells, prompt_index order (0..191)
    P3 = V.frozen_pp("pruned2_A__ac_short"); Q3 = V.frozen_pp("recovered2__ac_short")
    P10 = V.frozen_pp("pruned2_A__ac_native"); Q10 = V.frozen_pp("recovered2__ac_native")
    assert len(P3) == len(Q3) == len(P10) == len(Q10) == 192
    man = json.load(open(MANIFEST)); keys = [p["selection_key"] for p in man["prompts"]]
    assert keys == sorted(keys), "prompt_index must follow the ascending selection hash for the halves to be exchangeable"
    J_all = (Q10 - P10) - (Q3 - P3)
    a, b = slice(0, 96), slice(96, 192)
    first = cell(P3[a], Q3[a], P10[a], Q10[a], NS + "|sev2split|first96", 96)
    last = cell(P3[b], Q3[b], P10[b], Q10[b], NS + "|sev2split|last96", 96)
    out["sev2_split_half"] = {
        "design": "prompt_index 0-95 vs 96-191 of the frozen severity-2 192 manifest (one outcome-blind seeded-hash draw; "
                  "prompt_index = ascending sha256(selection_salt|YTID|ytid), so the halves are exchangeable sub-samples). "
                  "Same checkpoints (pruned2_A, recovered2), same jobs, same x_T seeds, same scorer calls as the primary result.",
        "J_all192_recomputed": V.Boot(NS + "|sev2split|all192", 192).ci(J_all),
        "first96": first, "last96": last,
        "J_last96_minus_first96": V.two_sample(NS + "|sev2split|het", J_all[b], J_all[a]),
        "abs_half_difference_vs_sev1_between_set_difference": {
            "sev2_halves_abs_diff": float(abs(J_all[b].mean() - J_all[a].mean())),
            "sev1_new96_minus_armd80_point": None}}
    # ---- same-prompt cross-severity: severity-1 new96 (E8; r2-gen-c/c2) = prompt_index 0..95 of this manifest
    C = V.load_results(V.gout("c"))
    S3, _, o3 = V.per_prompt(C["p1_pruned__ac_short"]); T3, _, _ = V.per_prompt(C["p1_recovered__ac_short"])
    S10, _, o10 = V.per_prompt(C["p1_pruned__ac_native"]); T10, _, _ = V.per_prompt(C["p1_recovered__ac_native"])
    assert list(o3) == list(o10) == list(range(96))
    J1 = (T10 - S10) - (T3 - S3); J2 = J_all[a]
    bt = V.Boot(NS + "|xsev96|paired", 96)
    out["same_prompts_sev1_vs_sev2_first96"] = {
        "design": "prompt_index 0-95: severity-1 (p1_pruned -> p1_recovered, E8 jobs r2-gen-c/c2) vs severity-2 (pruned2_A -> recovered2, "
                  "frozen xsev jobs) on the SAME 96 prompts; paired by prompt (x_T seeds differ between the two generation campaigns).",
        "J_sev1_new96": bt.ci(J1), "J_sev2_first96": bt.ci(J2), "J_sev1_minus_sev2_paired": bt.ci(J1 - J2),
        "pearson_r_perprompt_J": float(np.corrcoef(J1, J2)[0, 1]),
        "spearman_rho_perprompt_J": float(np.corrcoef(np.argsort(np.argsort(J1)), np.argsort(np.argsort(J2)))[0, 1])}
    # ---- item 4: AudioCaps vs hip-hop duration interaction (unpaired; different prompts)
    A = V.load_results(V.gout("a"))
    P3e, _, _ = V.per_prompt(A["pruned2_A__music_ext"]); Q3e, _, _ = V.per_prompt(A["recovered2__music_ext"])
    P10e, _, _ = V.per_prompt(A["pruned2_A__music_ext_native"]); Q10e, _, _ = V.per_prompt(A["recovered2__music_ext_native"])
    J_hh = np.concatenate([V.frozen_pp("recovered2__music_native") - V.frozen_pp("pruned2_A__music_native")
                           - (V.frozen_pp("recovered2__music") - V.frozen_pp("pruned2_A__music")), (Q10e - P10e) - (Q3e - P3e)])
    assert len(J_hh) == 127
    E7 = json.load(open("configs/research/r2_E7_result.json"))
    out["audiocaps_vs_hiphop_J"] = {
        "J_audiocaps_192": V.Boot(NS + "|dom|ac", 192).ci(J_all),
        "J_hiphop_127_recomputed": V.Boot(NS + "|dom|hh", 127).ci(J_hh),
        "J_hiphop_127_prespecified_E7": E7["J_music_127"],
        "J_audiocaps_minus_hiphop": V.two_sample(NS + "|dom|ac-hh", J_all, J_hh),
        "rho_dense_hiphop_pooled_3.84_10.24": "see configs/research/r2_posthoc_pooled_anchors.json hiphop_127.*.rho_dense_pooled"}
    # ---- optional: AudioCaps (first 96) vs Clotho (96) J, if the Clotho per-prompt scores are on disk
    try:
        Bg = V.load_results(V.gout("b"))
        names = sorted(Bg.keys()); out["audiocaps_vs_clotho_J"] = {"groups_seen": names}
        cP3, _, _ = V.per_prompt(Bg["pruned2_A__clotho_short"]); cQ3, _, _ = V.per_prompt(Bg["recovered2__clotho_short"])
        cP10, _, _ = V.per_prompt(Bg["pruned2_A__clotho_native"]); cQ10, _, _ = V.per_prompt(Bg["recovered2__clotho_native"])
        J_clo = (cQ10 - cP10) - (cQ3 - cP3); assert len(J_clo) == 96
        out["audiocaps_vs_clotho_J"].update({"J_clotho_96": V.Boot(NS + "|dom|clo", 96).ci(J_clo),
                                             "J_audiocaps_first96": V.Boot(NS + "|dom|ac96", 96).ci(J_all[a]),
                                             "J_audiocaps192_minus_clotho": V.two_sample(NS + "|dom|ac-clo", J_all, J_clo)})
    except Exception as e:  # noqa: BLE001 - descriptive extra; record why it is absent
        out["audiocaps_vs_clotho_J"] = {"skipped": f"{type(e).__name__}: {e}"[:200]}
    # ---- sev-1 reference value for the comparison sentence
    PH = json.load(open("configs/research/r2_posthoc_pooled_anchors.json"))
    out["sev2_split_half"]["abs_half_difference_vs_sev1_between_set_difference"]["sev1_new96_minus_armd80_point"] = PH["sev1_heterogeneity"]["J_new96_minus_armd80"]["point"]
    for p in (V.FROZEN_OUT, V.MUSIC_NATIVE_OUT, V.gout("a"), V.gout("c"), MANIFEST, "configs/research/r2_E7_result.json",
              "configs/research/r2_posthoc_pooled_anchors.json"):
        if os.path.exists(p): out["inputs"][p] = sha(p)
    js = json.dumps(out, indent=1, sort_keys=True); out["artifact_sha256"] = hashlib.sha256(js.encode()).hexdigest()
    json.dump(out, open(OUT, "w"), indent=1)
    S = out["sev2_split_half"]; X = out["same_prompts_sev1_vs_sev2_first96"]; Dm = out["audiocaps_vs_hiphop_J"]
    f = lambda c: f"{c['point']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c.get('n', c.get('n_a'))})"
    print("SEV-2 J all 192 (recomputed):", f(S["J_all192_recomputed"]))
    print("SEV-2 J first 96:", f(S["first96"]["J"]), "| last 96:", f(S["last96"]["J"]), "| last-first:", f(S["J_last96_minus_first96"]))
    print("SEV-1 new96 J:", f(X["J_sev1_new96"]), "| SEV-2 same 96 J:", f(X["J_sev2_first96"]), "| paired sev1-sev2:", f(X["J_sev1_minus_sev2_paired"]),
          "| r =", round(X["pearson_r_perprompt_J"], 3))
    print("J AudioCaps 192:", f(Dm["J_audiocaps_192"]), "| J hip-hop 127:", f(Dm["J_hiphop_127_recomputed"]), "| AC-HH:", f(Dm["J_audiocaps_minus_hiphop"]))
    print("Clotho block:", json.dumps(out["audiocaps_vs_clotho_J"])[:400])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
