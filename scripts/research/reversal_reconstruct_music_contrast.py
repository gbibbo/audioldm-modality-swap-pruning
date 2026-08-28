#!/usr/bin/env python3
"""Reconstruct + persist the FROZEN historical music contrast for RECOVERY-REVERSAL-V1.

Rebuilds the 64 prompts x 3 paired-replicate music contrast
    d[p,r] = C_recovered_off[p,r] - C_pruned_ema_reconstructed_off[p,r]
from the persisted phenomenon artifacts (`_phenom_groups_in.json`, `_phenom_groups_out.json`)
WITHOUT rescoring any WAV, joins prompt_index -> ytid via the frozen Gate-0 battery, and
cross-checks the ytid via musiccaps-public.csv. Writes a self-hashing provenance artifact and
HARD-VERIFIES that R_music reproduces the frozen ledger value -0.0941, CI95 [-0.1241, -0.0646]
(prompt-cluster percentile bootstrap, historical seed 20260826).

Purpose is PROVENANCE, not re-analysis. No AudioCaps data. No GPU. CPU only.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python \
        scripts/research/reversal_reconstruct_music_contrast.py \
        --out artifacts/icassp_gate0/reversal_music_contrast.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import (  # noqa: E402
    R_MUSIC_TARGET_HI, R_MUSIC_TARGET_LO, R_MUSIC_TARGET_POINT,
    reconstruct_music_grids, r_music)

A = "artifacts/icassp_gate0"
GROUPS_IN = f"{A}/_phenom_groups_in.json"
GROUPS_OUT = f"{A}/_phenom_groups_out.json"
VERDICT = f"{A}/phenomenon_verdict.json"
BATTERY = "configs/research/icassp_gate0_battery.json"
MUSICCAPS = f"{A}/musiccaps-public.csv"
HIST_SEED = 20260826
TOL_POINT = 1e-4
TOL_CI = 5e-4


def _sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _md5_file(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def build(out_path: str) -> dict:
    groups_in = json.load(open(GROUPS_IN))
    groups_out = json.load(open(GROUPS_OUT))
    battery = json.load(open(BATTERY))
    verdict = json.load(open(VERDICT))

    recon = reconstruct_music_grids(groups_in, groups_out)

    # prompt_index -> ytid from the frozen battery; verify caption agreement
    bat_ytids = [p["ytid"] for p in battery["prompts"]]
    bat_caps = [p["caption"] for p in battery["prompts"]]
    if bat_caps != recon.captions:
        raise SystemExit("battery captions disagree with reconstructed phenom captions")

    # independent ytid cross-check via musiccaps caption->ytid
    def norm(s: str) -> str:
        return " ".join(s.split()).strip()
    mc = {norm(r["caption"]): r["ytid"] for r in csv.DictReader(open(MUSICCAPS))}
    mc_hits = sum(1 for c in recon.captions if norm(c) in mc)
    mc_agree = sum(1 for yt, c in zip(bat_ytids, recon.captions)
                   if mc.get(norm(c)) == yt)
    if mc_hits != len(recon.captions) or mc_agree != len(recon.captions):
        raise SystemExit(f"musiccaps ytid cross-check failed: hits {mc_hits}, "
                         f"agree {mc_agree} / {len(recon.captions)}")

    ci = r_music(recon, seed=HIST_SEED)

    # HARD regression checks against the frozen ledger value
    if abs(ci.point - R_MUSIC_TARGET_POINT) > TOL_POINT:
        raise SystemExit(f"R_music point {ci.point:.6f} != frozen {R_MUSIC_TARGET_POINT}")
    if (abs(ci.lo - R_MUSIC_TARGET_LO) > TOL_CI or abs(ci.hi - R_MUSIC_TARGET_HI) > TOL_CI):
        raise SystemExit(f"R_music CI [{ci.lo:.6f},{ci.hi:.6f}] != frozen "
                         f"[{R_MUSIC_TARGET_LO},{R_MUSIC_TARGET_HI}]")

    prompts = []
    for p in range(recon.paired_diff.shape[0]):
        prompts.append({
            "prompt_index": p,
            "ytid": bat_ytids[p],
            "caption": recon.captions[p],
            "replicate_indices": [0, 1, 2],
            "recovered_off_clap": [float(x) for x in recon.recovered[p]],
            "pruned_off_clap": [float(x) for x in recon.pruned[p]],
            "paired_diff": [float(x) for x in recon.paired_diff[p]],
            "prompt_mean_diff": float(recon.prompt_mean_diff[p]),
        })

    sp = verdict.get("scorer_provenance", {})
    rp = verdict.get("provenance", {}).get("recovered_provenance", {})
    payload = {
        "artifact": "reversal_music_contrast",
        "status": "FROZEN-HISTORICAL provenance (music arm); no AudioCaps data",
        "description": "Post-hoc music-domain contrast C_recovered_off - C_pruned_off that "
                       "MOTIVATED the Recovery-Reversal hypothesis. Reconstructed without "
                       "rescoring WAVs. R_music is frozen; the prospective validation arm is "
                       "the future AudioCaps-test experiment (see docs/recovery_reversal_v1.md).",
        "design": {"n_prompts": 64, "n_replicates": 3, "pairing": "by generation seed r "
                   "across backbones (derive_paired_seed(salt, ytid, r))",
                   "off_arms": ["p1_recovered__off", "p1_pruned_ema_reconstructed__off"]},
        "R_music": {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "n": ci.n, "b": ci.b,
                    "bootstrap_seed": HIST_SEED,
                    "definition": "mean_p mean_r (C_recovered_off - C_pruned_off); "
                                  "prompt-cluster percentile bootstrap (frozen convention)"},
        "regression_target": {"point": R_MUSIC_TARGET_POINT,
                              "lo": R_MUSIC_TARGET_LO, "hi": R_MUSIC_TARGET_HI,
                              "reproduced": True},
        "prompt_sign_fraction_neg": float((recon.prompt_mean_diff < 0).mean()),
        "sources": {
            "groups_in": {"path": GROUPS_IN, "sha256": _sha256_file(GROUPS_IN)},
            "groups_out": {"path": GROUPS_OUT, "sha256": _sha256_file(GROUPS_OUT)},
            "battery": {"path": BATTERY, "sha256": _sha256_file(BATTERY),
                        "prompts_sha256": battery.get("prompts_sha256")},
            "musiccaps_csv": {"path": MUSICCAPS, "sha256": _sha256_file(MUSICCAPS)},
            "phenomenon_verdict": {"path": VERDICT, "md5": _md5_file(VERDICT),
                                   "expected_md5": "326eb63990d39e9dd2b80fbe26fe3025"},
        },
        "scorer_provenance": {
            "model": sp.get("model"), "hf_revision": sp.get("revision") or sp.get("hf_revision"),
            "lib_versions": sp.get("lib_versions"),
            "scoring_git_sha": sp.get("scoring_git_sha"),
            "convention": sp.get("convention"),
        },
        "generation_provenance": {
            "git_sha": rp.get("git_sha"),
            "recovered_source_checkpoint_sha256": rp.get("source_checkpoint_sha256"),
            "checkpoint_convention": rp.get("checkpoint_convention"),
        },
        "prompts": prompts,
    }
    payload["artifact_sha256"] = _sha256_obj(payload)
    json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
    # sidecar checksum of the exact bytes written
    open(out_path + ".sha256", "w").write(_sha256_file(out_path) + "  " +
                                          os.path.basename(out_path) + "\n")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{A}/reversal_music_contrast.json")
    args = ap.parse_args()
    payload = build(args.out)
    print(json.dumps({"R_music": payload["R_music"],
                      "regression_reproduced": payload["regression_target"]["reproduced"],
                      "sign_frac_neg": payload["prompt_sign_fraction_neg"],
                      "artifact_sha256": payload["artifact_sha256"],
                      "verdict_md5_ok": payload["sources"]["phenomenon_verdict"]["md5"]
                      == payload["sources"]["phenomenon_verdict"]["expected_md5"]}, indent=2))
    print("REVERSAL music contrast written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
