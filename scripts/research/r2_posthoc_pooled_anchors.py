#!/usr/bin/env python3
"""Round-2 review, items 5 and 6 — POST-HOC descriptive pooling of existing scores (CPU only, 0 cr; no generation).

NOT part of the frozen protocol docs/reviewer2_followup.md (which pre-specified the frozen-64 hip-hop anchors and
the pooled n=176 severity-1 interaction). This script only re-uses the already-scored per-prompt cosines:

  * item 6 — dense anchors on ALL 127 hip-hop prompts: the E7 job generated dense clips for the 63 extension
    prompts too (docs/reviewer2_followup.md §4), so rho_dense and the above-chance margin A_dense can be pooled
    over 64 + 63 prompts instead of the 64-prompt footnote in Draft 13's Table 1;
  * item 5 — severity-1 between-subset heterogeneity: the unpaired difference of the duration interaction J
    between the 96 new prompts (E8) and the frozen Arm-D 80, with the selection provenance of both sets.

Writes configs/research/r2_posthoc_pooled_anchors.json. Bootstrap conventions = r2_verdict.py (B = 10 000,
percentile, unit = prompt), seed namespace suffixed "|POSTHOC" so no frozen draw is re-used.
Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/r2_posthoc_pooled_anchors.py
"""
import hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
import r2_verdict as V

NS = V.NS + "|POSTHOC"
OUT = "configs/research/r2_posthoc_pooled_anchors.json"
ARMD_RESULT = "configs/research/op_duration_discriminator_1_result.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    A = V.load_results(V.gout("a"))
    out = {"artifact": "r2_posthoc_pooled_anchors", "class": "POST-HOC descriptive pooling of existing scores (round-2 review items 5, 6); "
           "not pre-specified in docs/reviewer2_followup.md; changes no frozen verdict",
           "bootstrap": {"B": V.B, "seed_namespace": NS, "unit": "prompt", "ci": "percentile 95%"}, "SESOI": V.SESOI, "inputs": {}}
    # ---- item 6: hip-hop dense anchors pooled over 127 prompts
    hip = {}
    for d, ctx, fctx in (("3.84", "music_ext", "music"), ("10.24", "music_ext_native", "music_native")):
        Pf = V.frozen_pp(f"pruned2_A__{fctx}"); Qf = V.frozen_pp(f"recovered2__{fctx}")
        Df, fDf, _ = V.per_prompt(A[f"dense__{fctx}"])                      # E6 dense on the frozen 64 (3 reps at 3.84 s, 1 at 10.24 s)
        P, fP, _ = V.per_prompt(A[f"pruned2_A__{ctx}"]); Q, fQ, _ = V.per_prompt(A[f"recovered2__{ctx}"]); D, fD, _ = V.per_prompt(A[f"dense__{ctx}"])
        assert len(Pf) == len(Qf) == len(Df) == 64 and len(P) == len(Q) == len(D) == 63
        R = np.concatenate([Qf - Pf, Q - P]); G = np.concatenate([Df - Pf, D - P]); M = np.concatenate([Df - fDf, D - fD])
        DQ = np.concatenate([Df - Qf, D - Q])
        bt = V.Boot(NS + f"|hiphop127|{d}", 127)
        hip[d] = {"n": 127, "levels": {"P": float(np.concatenate([Pf, P]).mean()), "PFT": float(np.concatenate([Qf, Q]).mean()),
                                        "dense": float(np.concatenate([Df, D]).mean()), "dense_floor": float(np.concatenate([fDf, fD]).mean())},
                  "R_pooled": bt.ci(R), "A_dense_pooled": bt.ci(M), "rho_dense_pooled": bt.ratio(R, G), "dense_minus_PFT_pooled": bt.ci(DQ),
                  "rho_dense_frozen64": V.Boot(NS + f"|hiphop64|{d}", 64).ratio(Qf - Pf, Df - Pf),
                  "rho_dense_ext63": V.Boot(NS + f"|hiphop63|{d}", 63).ratio(Q - P, D - P),
                  "battery_discriminates_for_dense_pooled": None}
        hip[d]["battery_discriminates_for_dense_pooled"] = bool(hip[d]["A_dense_pooled"]["lo"] > V.SESOI)
    out["hiphop_127"] = hip
    # ---- item 5: severity-1 between-subset heterogeneity
    ad = json.load(open(ARMD_RESULT))["raw_cosines"]
    pc, rc, pa, ra = [np.asarray(ad[k], float) for k in ("pruned_ctrl", "recovered_ctrl", "pruned_alt", "recovered_alt")]
    J80 = (ra - pa) - (rc - pc)
    C = V.load_results(V.gout("c"))
    P3, _, _ = V.per_prompt(C["p1_pruned__ac_short"]); Q3, _, _ = V.per_prompt(C["p1_recovered__ac_short"])
    P10, _, _ = V.per_prompt(C["p1_pruned__ac_native"]); Q10, _, _ = V.per_prompt(C["p1_recovered__ac_native"])
    J96 = (Q10 - P10) - (Q3 - P3)
    assert len(J80) == 80 and len(J96) == 96
    out["sev1_heterogeneity"] = {
        "J_armd80": V.Boot(NS + "|sev1|armd80", 80).ci(J80), "J_new96": V.Boot(NS + "|sev1|new96", 96).ci(J96),
        "J_new96_minus_armd80": V.two_sample(NS + "|sev1|het", J96, J80),
        "components": {"armd80": {"P_3.84": float(pc.mean()), "PFT_3.84": float(rc.mean()), "P_10.24": float(pa.mean()), "PFT_10.24": float(ra.mean()),
                                  "R_3.84": float((rc - pc).mean()), "R_10.24": float((ra - pa).mean())},
                       "new96": {"P_3.84": float(P3.mean()), "PFT_3.84": float(Q3.mean()), "P_10.24": float(P10.mean()), "PFT_10.24": float(Q10.mean()),
                                 "R_3.84": float((Q3 - P3).mean()), "R_10.24": float((Q10 - P10).mean())}},
        "sd_J": {"armd80": float(J80.std(ddof=1)), "new96": float(J96.std(ddof=1))},
        "selection_provenance": {
            "armd80": "80 of the 96 V1.1 AudioCaps-test ytids (V1.1: Convention-B seeded-hash draw over the eligible test set, "
                      "docs/recovery_reversal_v1_1.md), chosen by sha256('OP-DURATION-DISCRIMINATOR-1|SUBSET|2026-08-30|YTID|'+ytid) ascending, "
                      "first 80 (docs/op_duration_discriminator_1.md §3; outcome-blind). 3.84-s clips = V1.1 replicate 0 (job reversal-v11-gen-1), "
                      "10.24-s clips = job reversal-armd-gen-1; both scored as 80-item seed-once calls.",
            "new96": "first 96 prompt_index entries of the frozen severity-2 manifest configs/research/xsev_audiocaps_manifest.json "
                     "(seeded-hash draw over AudioCaps test EXCLUDING the V1.1-96, train, music-64 and Kim-44 ids, 5-caption rows only; "
                     "docs/reviewer2_followup.md §1, §5). Both durations replicate 0, jobs r2-gen-c (P) / r2-gen-c2 (P+FT), 96-item seed-once calls.",
            "same_between_sets": "checkpoints (p1_pruned = L1 selection on the dense EMA at (1,2,3,1); p1_recovered = public recovered EMA), "
                                 "sampler (DDIM 50 / 2.5 / eta 0 / fp32 / single), scorer (fused CLAP rev 365dea6e, seed-once), T4 hardware class, "
                                 "CRN pairing within duration. Neither draw used any outcome.",
            "differs_between_sets": "prompt pool (disjoint by construction), selection salt, the 5-caption-row eligibility rule of the 192 manifest, "
                                    "generation job/date, x_T seeds (V1.1 seeds vs frozen GEN_SALT)."}}
    for p in (V.gout("a"), V.gout("c"), ARMD_RESULT, V.FROZEN_OUT, V.MUSIC_NATIVE_OUT):
        out["inputs"][p] = sha(p)
    s = json.dumps(out, indent=1, sort_keys=False); out["artifact_sha256"] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(out, open(OUT, "w"), indent=1)
    for d in ("3.84", "10.24"):
        h = hip[d]; print(f"hip-hop {d} s n=127: R {h['R_pooled']['point']:+.3f} [{h['R_pooled']['lo']:+.3f},{h['R_pooled']['hi']:+.3f}]  "
                          f"A_dense {h['A_dense_pooled']['point']:+.3f} [{h['A_dense_pooled']['lo']:+.3f},{h['A_dense_pooled']['hi']:+.3f}]  "
                          f"rho_dense {h['rho_dense_pooled']['point']:.3f} [{h['rho_dense_pooled']['lo']:.3f},{h['rho_dense_pooled']['hi']:.3f}] "
                          f"(frozen64 {h['rho_dense_frozen64']['point']:.3f}, ext63 {h['rho_dense_ext63']['point']:.3f})")
    het = out["sev1_heterogeneity"]
    print(f"sev-1 J: armd80 {het['J_armd80']['point']:+.3f} [{het['J_armd80']['lo']:+.3f},{het['J_armd80']['hi']:+.3f}]  new96 {het['J_new96']['point']:+.3f} "
          f"[{het['J_new96']['lo']:+.3f},{het['J_new96']['hi']:+.3f}]  diff {het['J_new96_minus_armd80']['point']:+.3f} "
          f"[{het['J_new96_minus_armd80']['lo']:+.3f},{het['J_new96_minus_armd80']['hi']:+.3f}]")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
