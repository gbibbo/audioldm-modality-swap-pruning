#!/usr/bin/env python3
"""DRAFT4-DENSE-DURATION-CONTROL — matched-prompt dense duration control (CPU, 0 cr).

POST-HOC CONTROL (Gabriel review request 2026-09-02, manuscript Draft-4 pass). Changes no frozen verdict.

Why. The manuscript states that "every system scores higher at 10.24 s (dense 0.204 -> 0.352)", but those
two dense means come from DIFFERENT prompt sets (V1.1 96 prompts x 2 reps at 3.84 s; Arm-D 80 prompts at
10.24 s) and different scorer batches. The severity-1 duration interaction J is borderline on the raw
scale, and a reviewer's first objection is "everything gains from the longer clip". The clean answer is a
PAIRED dense duration slope on the SAME 80 prompts, scored under the SAME frozen convention as the four
Arm-D groups (one seed-once 80-item fused-CLAP call, subset order, rev 365dea6e), so that the three
systems' duration responses (dense, P, P+FT) are directly comparable and prompt-paired.

What exists already (frozen):
  * P and P+FT at 3.84 s and 10.24 s on the 80 Arm-D prompts: op_duration_discriminator_1_result.json
    (raw_cosines: pruned_ctrl / recovered_ctrl / pruned_alt / recovered_alt), each one 80-item call.
  * dense at 10.24 s on the same 80 (dense10s__dense) in the persisted scorer output
    artifacts/icassp_gate0/_score_tmp/xsev_dense_groups_out.json (one 80-item call; reproduces the frozen
    DENSE_CONTROL of xsev_result.json).
What is new: dense at 3.84 s on the same 80 prompts = the existing V1.1 dense r0 WAVs
  (dense_noadapter_p{v1_1_prompt_index}_r0.wav), re-scored as ONE 80-item seed-once group in subset order
  (exactly how the Arm-D short groups were produced from the same V1.1 WAVs). No generation.

Quantities (unit = prompt, n = 80, percentile bootstrap B = 10000, 95%):
  slope(sys)      = mean_i [ CLAP(sys, 10.24 s)_i - CLAP(sys, 3.84 s)_i ]            for dense, P, P+FT
  dslope_postft   = slope(P+FT) - slope(dense)   paired per prompt   (excess duration response of P+FT)
  dslope_pruned   = slope(P)    - slope(dense)   paired per prompt
  G_short(sys)    = mean_i [ CLAP(dense, 3.84)_i - CLAP(sys, 3.84)_i ]   (dense gap at the short point)
  G_native(sys)   = frozen DENSE_CONTROL values, reproduced here from the persisted scores
  gap_closed(op)  = mean_i R_op_i / mean_i G_op(P)_i = R_op / G_op(P)      (ratio of means, bootstrap CI)
  Consistency guards: R_short / R_native / J reproduce the frozen Arm-D points; G_native reproduces the
  frozen DENSE_CONTROL points (max |diff| must be < 1e-9 or the script aborts).
  Diagnostic: batch-composition sensitivity of the frozen fused-CLAP convention — the SAME V1.1 r0 WAVs
  scored inside the 192-item V1.1 call vs the 80-item Arm-D call (descriptive; explains why absolute
  means are only comparable within one scoring convention).

Seed namespace "DRAFT4-DENSE-DURATION-CONTROL|BOOTSTRAP|2026-09-02" -> PCG64(int(sha256(ns)[:8],16) % 2**31).

Run:
  # 1) emit the one new scorer group (dense 3.84 s, 80 items, subset order)
  OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft4_dense_duration_control.py --emit
  # 2) score it with the frozen scorer (one seed-once call per group; CPU)
  OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/gate0_clap_scorer.py \
      --score-groups artifacts/icassp_gate0/_score_tmp/draft4_dense_short_groups_in.json \
                     artifacts/icassp_gate0/_score_tmp/draft4_dense_short_groups_out.json
  # 3) verdict
  OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft4_dense_duration_control.py --verdict
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np

NS = "DRAFT4-DENSE-DURATION-CONTROL|BOOTSTRAP|2026-09-02"
SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000
SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
OPD = "configs/research/op_duration_discriminator_1_result.json"
XSEV = "configs/research/xsev_result.json"
REV11 = "configs/research/reversal_v1_1_result.json"
DENSE_NATIVE_OUT = "artifacts/icassp_gate0/_score_tmp/xsev_dense_groups_out.json"
V11_GEN = "/teamspace/jobs/reversal-v11-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_v1_1_gen"
TMP = "artifacts/icassp_gate0/_score_tmp"
GROUPS_IN = f"{TMP}/draft4_dense_short_groups_in.json"
GROUPS_OUT = f"{TMP}/draft4_dense_short_groups_out.json"
OUT = "configs/research/draft4_dense_duration_control_result.json"
GROUP = "dense_short_sev1__armd80"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def subset():
    return sorted(json.load(open(SUBSET))["prompts"], key=lambda p: p["subset_prompt_index"])


def emit():
    prompts = subset()
    items = []
    for p in prompts:
        w = f"{V11_GEN}/dense_noadapter_p{p['v1_1_prompt_index']}_r0.wav"
        if not os.path.exists(w):
            raise SystemExit(f"missing dense V1.1 r0 WAV: {w}")
        items.append({"caption": p["caption"], "wav": w, "ytid": p["ytid"],
                      "v1_1_prompt_index": p["v1_1_prompt_index"], "src_sha256": sha(w)})
    os.makedirs(TMP, exist_ok=True)
    json.dump({"groups": [{"name": GROUP, "items": items}],
               "convention": "ONE seed-once 80-item fused-CLAP call, subset_prompt_index order 0..79 — identical "
                             "to the Arm-D short/native groups and to the xsev dense-native group",
               "source": "existing V1.1 dense r0 3.84 s WAVs (job reversal-v11-gen-1); no new generation",
               "subset_sha256": json.load(open(SUBSET)).get("subset_sha256")},
              open(GROUPS_IN, "w"), indent=1)
    print(f"emitted {len(items)} items -> {GROUPS_IN}")


def pct(vals, stat, rng):
    n = len(next(iter(vals.values())))
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boots[i] = stat({k: v[idx] for k, v in vals.items()})
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(stat(vals)), "lo": float(lo), "hi": float(hi), "n": int(n),
            "boot_frac_le0": float(np.mean(boots <= 0))}


def verdict():
    prompts = subset()
    opd = json.load(open(OPD))
    rc = {k: np.asarray(v, float) for k, v in opd["raw_cosines"].items()}
    dn_out = json.load(open(DENSE_NATIVE_OUT))
    dn = {r["name"]: np.asarray(r["cosines"], float) for r in dn_out["results"]}
    ds_out = json.load(open(GROUPS_OUT))
    ds = {r["name"]: np.asarray(r["cosines"], float) for r in ds_out["results"]}
    assert ds_out["revision"] == dn_out["revision"] == opd["scorer"]["revision"], "scorer revision mismatch"
    v = {
        "dense_short": ds[GROUP],
        "dense_native": dn["dense10s__dense"],
        "pruned_short": rc["pruned_ctrl"], "postft_short": rc["recovered_ctrl"],
        "pruned_native": rc["pruned_alt"], "postft_native": rc["recovered_alt"],
    }
    for k, a in v.items():
        assert a.shape == (80,), (k, a.shape)
    # ---- consistency guards against frozen artifacts (points must match to 1e-9)
    fr = opd["PRIMARY_clap"]
    guards = {
        "R_short": (float((v["postft_short"] - v["pruned_short"]).mean()), fr["R_ctrl_80"]["point"]),
        "R_native": (float((v["postft_native"] - v["pruned_native"]).mean()), fr["R_alt"]["point"]),
        "J": (float(((v["postft_native"] - v["pruned_native"]) - (v["postft_short"] - v["pruned_short"])).mean()), fr["J"]["point"]),
        "pruned_native_mean_vs_xsev_dense_group": (float(dn["dense10s__pruned_sev1"].mean()), fr["means"]["pruned_alt"]),
        "postft_native_mean_vs_xsev_dense_group": (float(dn["dense10s__recovered_sev1"].mean()), fr["means"]["recovered_alt"]),
    }
    DC = json.load(open(XSEV))["DENSE_CONTROL"]
    guards["G_native_pruned"] = (float((v["dense_native"] - v["pruned_native"]).mean()), DC["G_pruned_dense_minus_pruned"]["point"])
    guards["G_native_postft"] = (float((v["dense_native"] - v["postft_native"]).mean()), DC["G_recovered_dense_minus_recovered"]["point"])
    guards["C_dense_10s"] = (float(v["dense_native"].mean()), DC["C_dense_10s"])
    worst = max(abs(a - b) for a, b in guards.values())
    if worst > 1e-9:
        raise SystemExit(f"consistency guard FAILED (max |diff| {worst:.3e}): {guards}")

    rng = np.random.default_rng(np.random.PCG64(SEED))
    res = {
        "artifact": "draft4_dense_duration_control_result",
        "class": "POST-HOC CONTROL (Draft-4 review pass, 2026-09-02); no new generation; changes no frozen verdict",
        "design": "80 Arm-D prompts (subset of the pre-specified V1.1 96), r0, three systems x two durations, "
                  "each cell one seed-once 80-item fused-CLAP call in subset order (frozen convention)",
        "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": SEED, "unit": "prompt", "ci": "percentile 95%"},
        "inputs": {p: sha(p) for p in (SUBSET, OPD, XSEV, DENSE_NATIVE_OUT, GROUPS_OUT, GROUPS_IN)},
        "scorer": {"model": ds_out["model"], "revision": ds_out["revision"], "lib_versions": ds_out["scorer_provenance"]["lib_versions"]},
        "consistency_guards_max_abs_diff": float(worst),
        "means": {k: float(a.mean()) for k, a in v.items()},
    }
    sl = lambda s: (lambda x: (x[f"{s}_native"] - x[f"{s}_short"]).mean())
    res["slopes"] = {s: pct(v, sl(s), rng) for s in ("dense", "pruned", "postft")}
    res["dslope_postft_minus_dense"] = pct(v, lambda x: ((x["postft_native"] - x["postft_short"]) - (x["dense_native"] - x["dense_short"])).mean(), rng)
    res["dslope_pruned_minus_dense"] = pct(v, lambda x: ((x["pruned_native"] - x["pruned_short"]) - (x["dense_native"] - x["dense_short"])).mean(), rng)
    res["dense_gap"] = {
        "short_pruned": pct(v, lambda x: (x["dense_short"] - x["pruned_short"]).mean(), rng),
        "short_postft": pct(v, lambda x: (x["dense_short"] - x["postft_short"]).mean(), rng),
        "native_pruned": pct(v, lambda x: (x["dense_native"] - x["pruned_native"]).mean(), rng),
        "native_postft": pct(v, lambda x: (x["dense_native"] - x["postft_native"]).mean(), rng),
    }
    res["gap_closed_fraction"] = {
        "short": pct(v, lambda x: (x["postft_short"] - x["pruned_short"]).mean() / (x["dense_short"] - x["pruned_short"]).mean(), rng),
        "native": pct(v, lambda x: (x["postft_native"] - x["pruned_native"]).mean() / (x["dense_native"] - x["pruned_native"]).mean(), rng),
    }
    res["gap_closed_difference_native_minus_short"] = pct(
        v, lambda x: (x["postft_native"] - x["pruned_native"]).mean() / (x["dense_native"] - x["pruned_native"]).mean()
        - (x["postft_short"] - x["pruned_short"]).mean() / (x["dense_short"] - x["pruned_short"]).mean(), rng)
    # ---- diagnostic: batch-composition sensitivity of the frozen convention (same WAVs, two batch sizes)
    rev = json.load(open(REV11))["raw_clap_scores"]
    idx = [p["v1_1_prompt_index"] for p in prompts]
    r0 = lambda name: np.asarray(rev[name], float).reshape(96, 2)[:, 0][idx]   # order (prompt, replicate)
    res["batch_composition_diagnostic"] = {
        "note": "identical WAVs (V1.1 r0, 80 prompts) scored inside the 192-item V1.1 call vs a matched 80-item call; "
                "fused-CLAP's batch-level is_longer draw makes absolute means batch-dependent, which is why every "
                "contrast in the paper is taken within one scoring convention",
        "dense_short_mean_192call_vs_80call": [float(r0("dense_ema").mean()), float(v["dense_short"].mean())],
        "pruned_short_mean_192call_vs_80call": [float(r0("p1_pruned_ema_reconstructed").mean()), float(v["pruned_short"].mean())],
        "postft_short_mean_192call_vs_80call": [float(r0("p1_recovered").mean()), float(v["postft_short"].mean())],
        "R_short_192call_vs_80call": [float((r0("p1_recovered") - r0("p1_pruned_ema_reconstructed")).mean()),
                                      float((v["postft_short"] - v["pruned_short"]).mean())],
        "max_abs_per_item_diff_dense": float(np.max(np.abs(r0("dense_ema") - v["dense_short"]))),
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("means", "slopes", "dslope_postft_minus_dense", "dslope_pruned_minus_dense",
                                          "dense_gap", "gap_closed_fraction", "gap_closed_difference_native_minus_short",
                                          "batch_composition_diagnostic")}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.emit:
        emit()
    if a.verdict:
        verdict()
    if not (a.emit or a.verdict):
        ap.print_help()
