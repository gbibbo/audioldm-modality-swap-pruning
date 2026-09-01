#!/usr/bin/env python3
"""Build configs/research/final_claim_registry.json — the structured, self-hashed claim
registry for the future manuscript. Reads ONLY committed frozen artifacts (no recompute).
Pairs with docs/final_scientific_story.md. CPU, read-only.
"""
import json, os, hashlib
os.chdir("/teamspace/studios/this_studio/audioldm-modality-swap-pruning")
def g(p): return json.load(open("configs/research/" + p))

od = g("op_duration_discriminator_1_result.json")
rm = g("reversal_v1_r_music_clap.json")
xs = g("xsev_result.json")
hc = g("xsev_hc_secondary.json")
sm = g("xsev_secondary_metrics.json")
fr = g("finelap_temporal_result.json")["verdict"]
v11 = g("reversal_v1_1_result.json")["PRIMARY"]

def ci(o, lo="lo", hi="hi"):
    return {"point": round(o["point"], 4),
            "ci95": [round(o.get(lo, o.get("ci95", [None, None])[0]) if lo in o else o["ci95"][0], 4),
                     round(o.get(hi, o.get("ci95", [None, None])[1]) if hi in o else o["ci95"][1], 4)]}

reg = {
  "artifact": "final_claim_registry",
  "status": "FROZEN pre-manuscript source of truth; not manuscript prose; no result changed",
  "framing": "evaluation of post-pruning fine-tuning; mechanistic attribution BLOCKED (no matched dense-FT control)",
  "clap_scorer": {"model": "laion/clap-htsat-fused", "revision": "365dea6ef167def6676140ed93bbc43f84dabb28"},
  "terminology_safe": ["pruned checkpoint", "post-fine-tuning checkpoint", "post-pruning fine-tuning",
    "native-duration semantic advantage", "temporal operating-point interaction", "context dependence",
    "frame-level grounding evidence", "sample-level heterogeneity"],

  "primary_findings": [
    {"id": "sev2_CASE_C", "class": "prospectively_frozen_primary",
     "provenance": "configs/research/xsev_result.json (02e3bd11)",
     "R_native": ci(xs["PRIMARY_A"]["R_native"]), "R_short": ci(xs["PRIMARY_A"]["R_short"]),
     "R_music": ci(xs["PRIMARY_A"]["R_music"]), "K": ci(xs["PRIMARY_A"]["K"]), "J": ci(xs["PRIMARY_A"]["J"]),
     "gates": xs["PRIMARY_A"]["gates"], "case": xs.get("REPLICATION_CASE"),
     "statement": "Severity-2 context (K) and duration (J) dependence replicate with a large native advantage; music-negative sign pattern does NOT replicate (CASE C)."},
    {"id": "sev1_reversal_negative", "class": "prospectively_frozen_primary_negative",
     "provenance": "configs/research/reversal_v1_1_result.json (cad7c454)",
     "R_AC": ci(v11["R_AC"]), "I": ci(v11["I"]), "PASS": False,
     "statement": "The severity-1 recovery-reversal / domain-specialization hypothesis is a pre-registered NEGATIVE at the controlled 3.84 s operating point."},
  ],

  "prospective_replications": [
    {"id": "sev1_duration_interaction_directional", "class": "prospectively_specified_followup",
     "provenance": "configs/research/op_duration_discriminator_1_result.json (9c92552a)",
     "R_short": ci(od["PRIMARY_clap"]["R_ctrl_80"]), "R_native": ci(od["PRIMARY_clap"]["R_alt"]),
     "J": ci(od["PRIMARY_clap"]["J"]),
     "note": "Severity-1 J is DIRECTIONAL: 95% CI narrowly includes 0 (gate not passed); severity-2 J resolves it."},
    {"id": "sev2_seam_robustness", "class": "prospectively_frozen_primary",
     "provenance": "xsev_result.json:seam_robustness",
     "K_seam_robust": xs["seam_robustness"]["K"]["seam_robust"],
     "J_seam_robust": xs["seam_robustness"]["J"]["seam_robust"],
     "sign_pattern_seam_robust": xs["seam_robustness"]["sign_pattern"]["seam_robust"]},
  ],

  "corroborative_secondaries": [
    {"id": "sev1_music_clap", "class": "prospectively_frozen_secondary",
     "provenance": "configs/research/reversal_v1_r_music_clap.json",
     "R_music": {"point": round(rm["R_music"]["point"], 4), "ci95": [round(rm["R_music"]["lo"], 4), round(rm["R_music"]["hi"], 4)]},
     "note": "music-negative at severity 1"},
    {"id": "human_clap_sev2", "class": "prospectively_frozen_secondary_no_gate",
     "provenance": "configs/research/xsev_hc_secondary.json",
     "R_native": ci(hc["recovered_vs_prunedA"]["R_native"]), "R_music": ci(hc["recovered_vs_prunedA"]["R_music"]),
     "note": "HC shows a small music-negative that the PRIMARY CLAP does not; HC has no gate role."},
    {"id": "kl_pann_sev2", "class": "pre_specified_secondary_implemented_after_primary_result",
     "provenance": "configs/research/xsev_secondary_metrics.json",
     "R_KL": {"point": round(sm["contrasts_recovered_vs_prunedA"]["R_KL"]["point"], 3), "ci95": sm["contrasts_recovered_vs_prunedA"]["R_KL"]["ci95"], "inference": "paired ytid bootstrap, CI excludes 0"},
     "R_cap_captured_label_count": {"point": round(sm["contrasts_recovered_vs_prunedA"]["R_cap"]["point"], 3), "ci95": sm["contrasts_recovered_vs_prunedA"]["R_cap"]["ci95"], "inference": "paired, CI excludes 0; a COUNT not a rate"}},
    {"id": "fd_fad_sev2", "class": "descriptive_distribution_level_no_CI",
     "provenance": "configs/research/xsev_secondary_metrics.json",
     "FD": {"post_ft": round(sm["FD_pann2048"]["recovered2"], 1), "pruned": round(sm["FD_pann2048"]["pruned2_A"], 1)},
     "FAD": {"post_ft": round(sm["FAD_vggish"]["recovered2"], 2), "pruned": round(sm["FAD_vggish"]["pruned2_A"], 2)},
     "note": "DESCRIPTIVE ONLY — single scalar per system, NO paired CI"},
  ],

  "post_result_diagnostics": [
    {"id": "finelap_temporal", "class": "prospectively_frozen_POST_RESULT_diagnostic",
     "provenance": "configs/research/finelap_temporal_result.json (ce5519c8)",
     "T_2": ci(fr["T2"]), "D_early2": ci(fr["D_early2"]), "D_late2": ci(fr["D_late2"]),
     "T_1": ci(fr["T1_directional"]), "seam_T2_B": ci(fr["seam_sev2_prunedB"]),
     "branch": fr["branch"], "late_gain_wording_allowed": fr["late_gain_wording_allowed"],
     "interpretation_boundary": "frame-level grounding evidence; NOT calibrated probability, NOT causal activation, NOT perceptual quality; NOT an independent preregistered confirmation",
     "statement": "Large temporally-broad frame-level grounding gain after fine-tuning (mass/occupancy/coverage/peak CIs exclude 0), but NO preferential late redistribution (T_2 gate FAILS)."},
  ],

  "exploratory_posthoc": [
    {"id": "cross_severity_magnitude", "class": "exploratory_posthoc_confounded",
     "note": "Severity is entangled with two experiments, prompt sets, n, and checkpoints; never promotable to preregistered evidence."}],

  "negative_results": [
    "severity-1 recovery-reversal/domain-specialization FAILED (R_AC~0, PASS=FALSE)",
    "severity-1 primary temporal J narrowly missed gate (CI includes 0)",
    "severity-2 native-positive/music-negative sign pattern FAILED (H_music FALSE)",
    "severity-2 short-duration equivalence FAILED (R_short lo95>0)",
    "FineLAP late-redistribution FAILED (T_2~0, seam-robust; T_1 not>0)",
    "no demonstrated restoration to dense (G_post_ft CI includes 0; no TOST)",
    "no dense fine-tuned control available (deleted)",
    "no human listening study (cancelled pre-launch; 0 participants, 0 data)",
    "sample-level post-fine-tuning failures exist (heterogeneity; e.g. public Example 4 sev1 10.24s: dCLAP -0.275, FineLAP pruned>post-FT) — illustration only, not evidence",
  ],

  "limitations": [
    "mechanistic attribution BLOCKED (no matched dense-FT control; audioldm-m-text-ft is NOT equivalent) -> evaluation claim only, not falsification",
    "single primary scorer family (CLAP); no human evaluation; FineLAP is a different family + post-result diagnostic",
    "off-recipe operating point vs Singh (DDIM50, 3.84/10.24s, guidance 2.5, single-gen) -> external validity untested",
    "cross-severity comparison exploratory/confounded",
    "sample-level heterogeneity: population advantages not monotonic per prompt",
    "public examples illustrative, deterministically selected, not human evidence; sev1 music + full dense reference not retained",
  ],

  "forbidden_claims": [
    "fine-tuning restores the pruned model", "post-fine-tuning consistently improves samples",
    "post-fine-tuning restores dense performance", "recovery causes specialization",
    "pruning causes the domain dependence", "fine-tuning causes OOD degradation",
    "the post-fine-tuning advantage disappears at 3.84 s", "the improvement occurs because later events are generated",
    "severity 2 is universally better/worse", "FineLAP measures local causal activations",
    "FineLAP measures perceptual audio quality", "human listeners confirmed the results",
    "recovered/post-fine-tuning is better (for an individual sample)",
  ],

  "human_listening": {"status": "CANCELLED_PRE_LAUNCH", "participants": 0, "human_data": 0, "for_inference": "VOID",
    "public_examples": "illustrative companion only; not human evidence; https://gbibbo.github.io/audioldm-modality-swap-pruning/"},

  "contribution_one_sentence": "Post-pruning fine-tuning of a text-to-audio diffusion model cannot be characterized by a single aggregate evaluation point: its advantage over the pruned checkpoint is context- and temporal-operating-point-dependent and prompt-heterogeneous, established by a prospectively frozen cross-severity evaluation with independent frame-level corroboration and explicit negatives.",
  "readiness": "READY_FOR_MANUSCRIPT (drafting only after explicit Gabriel GO; do not reopen experiments)",
}
payload = json.dumps(reg, indent=2, sort_keys=True)
reg["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
json.dump(reg, open("configs/research/final_claim_registry.json", "w"), indent=2, sort_keys=True)
print("wrote final_claim_registry.json self_sha256", reg["self_sha256"][:16])
print("primary:", [f["id"] for f in reg["primary_findings"]], "| negatives:", len(reg["negative_results"]),
      "| forbidden:", len(reg["forbidden_claims"]))
