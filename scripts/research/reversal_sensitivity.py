#!/usr/bin/env python3
"""CPU-only statistical preflight for the RECOVERY-REVERSAL-V1 96x2 AudioCaps design.

Uses ONLY the frozen historical music data (64 prompts x 3 paired replicates) to estimate a
random-effects variance decomposition, then projects the *future* AudioCaps arm
(96 prompts x 2 paired replicates) under hypothesised true effects R_AC and conservative
variance-inflation scenarios. NO AudioCaps outcome, generated audio, or model output enters
this calculation. GPU is never touched. This is SENSITIVITY, not post-data N optimisation.

For each (R_AC effect x variance scenario) it reports expected point-estimate variability,
expected CI half-width, P(lower_CI95(R_AC) > 0), P(point >= SESOI AND lower_CI95(R_AC) > 0),
and the interaction I = R_AC - R_music with the historical music uncertainty retained (joint
two-sample bootstrap), plus P(all three PASS conditions).

Randomness: PCG64(20260827) via SeedSequence spawning -> exactly reproducible, order-invariant.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/reversal_sensitivity.py \
        --out artifacts/icassp_gate0/reversal_sensitivity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import (  # noqa: E402
    BOOTSTRAP_SEED_V1, N_PROMPTS_V1, N_REPLICATES_V1, SESOI,
    decompose_variance, reconstruct_music_grids, simulate_design)

A = "artifacts/icassp_gate0"
GROUPS_IN = f"{A}/_phenom_groups_in.json"
GROUPS_OUT = f"{A}/_phenom_groups_out.json"
EFFECTS = [0.025, 0.040, 0.050, 0.075]
# primary interpretation: inflate BOTH variance components together (total-variance stress);
# transparency alternatives isolate each component at the strongest inflation.
PRIMARY_SCENARIOS = [("both", 1.0, 1.0), ("both", 1.5, 1.5), ("both", 2.0, 2.0)]
ALT_SCENARIOS = [("between_only", 2.0, 1.0), ("within_only", 1.0, 2.0)]


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run(out_path: str, n_sim: int, b_boot: int) -> dict:
    recon = reconstruct_music_grids(json.load(open(GROUPS_IN)), json.load(open(GROUPS_OUT)))
    vc = decompose_variance(recon.paired_diff)
    music_prompt_diff = recon.prompt_mean_diff

    # child seed streams: one per (scenario, effect) cell, deterministic + order-invariant.
    scenarios = PRIMARY_SCENARIOS + ALT_SCENARIOS
    root = np.random.SeedSequence(BOOTSTRAP_SEED_V1)
    children = root.spawn(len(scenarios) * len(EFFECTS))

    cells = []
    ci = 0
    for label, bs, ws in scenarios:
        for eff in EFFECTS:
            res = simulate_design(vc, music_prompt_diff, r_ac=eff,
                                  n_prompts=N_PROMPTS_V1, n_reps=N_REPLICATES_V1,
                                  between_scale=bs, within_scale=ws,
                                  n_sim=n_sim, b_boot=b_boot, sesoi=SESOI,
                                  seed_seq=children[ci])
            res["scenario"] = label
            cells.append(res)
            ci += 1

    payload = {
        "artifact": "reversal_sensitivity",
        "status": "DRAFT preflight for RECOVERY-REVERSAL-V1 — NOT FROZEN; NO AUDIOCAPS DATA",
        "design_projected": {"n_prompts": N_PROMPTS_V1, "n_replicates": N_REPLICATES_V1,
                             "sesoi_point": SESOI,
                             "pass_conditions": ["R_AC point >= 0.025",
                                                 "lower_CI95(R_AC) > 0",
                                                 "lower_CI95(I=R_AC-R_music) > 0"]},
        "variance_source": {"from": "historical music paired diff (64 prompts x 3 reps)",
                            **vc.as_dict()},
        "music_R_point": float(music_prompt_diff.mean()),
        "inflation_interpretation": {
            "primary": "BOTH components scaled together (total-variance stress) at 1.0/1.5/2.0x",
            "alternatives": "between-only 2.0x and within-only 2.0x, reported for transparency",
            "justification": "cross-domain generalisation can plausibly widen both prompt-level "
                             "heterogeneity of the reversal effect (between) and generation noise "
                             "(within); the primary stresses both, the alternatives localise it."},
        "bootstrap": {"seed": BOOTSTRAP_SEED_V1, "n_sim": n_sim, "b_boot": b_boot,
                      "generator": "PCG64 via SeedSequence.spawn per cell"},
        "cells": cells,
    }
    payload["artifact_sha256"] = _sha256_obj(payload)
    json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
    open(out_path + ".sha256", "w").write(
        hashlib.sha256(open(out_path, "rb").read()).hexdigest() + "  "
        + os.path.basename(out_path) + "\n")
    return payload


def _print_table(payload: dict) -> None:
    print(f"\nvariance: sigma_between={payload['variance_source']['sigma_between']:.4f} "
          f"sigma_within={payload['variance_source']['sigma_within']:.4f} "
          f"(music R={payload['music_R_point']:.4f})")
    hdr = ("scenario", "R_AC", "pt_sd", "CI_hw", "P(loRac>0)", "P(Rac_req)", "P(loI>0)", "P(all3)")
    print("{:<13}{:>6}{:>8}{:>8}{:>12}{:>11}{:>10}{:>9}".format(*hdr))
    for c in payload["cells"]:
        print("{:<13}{:>6.3f}{:>8.4f}{:>8.4f}{:>12.3f}{:>11.3f}{:>10.3f}{:>9.3f}".format(
            c["scenario"], c["r_ac_true"], c["point_sd"], c["mean_ci_halfwidth"],
            c["P_lowerCI_Rac_gt0"], c["P_Rac_requirement"], c["P_lowerCI_I_gt0"],
            c["P_pass_all_three"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{A}/reversal_sensitivity.json")
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--b-boot", type=int, default=2000)
    ap.add_argument("--fast", action="store_true", help="quick wiring check (small n_sim/b_boot)")
    args = ap.parse_args()
    n_sim, b_boot = (300, 500) if args.fast else (args.n_sim, args.b_boot)
    payload = run(args.out, n_sim, b_boot)
    _print_table(payload)
    print("\nREVERSAL sensitivity written to", args.out, "sha256", payload["artifact_sha256"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
