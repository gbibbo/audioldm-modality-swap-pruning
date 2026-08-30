#!/usr/bin/env python3
"""V1.1 SECONDARY Human-CLAP scoring + verdict on the AudioCaps WAVs (corroborative; NO PASS role).

Scores the three V1.1 systems' AudioCaps WAVs with the frozen Human-CLAP implementation (same
convention as the historical preflight), then computes the preregistered secondary contrast:

    R_AC_HC = HC_recovered,AC - HC_pruned,AC          (prompt-cluster percentile bootstrap)
    I_HC    = R_AC_HC - R_music_HC                     (joint two-sample bootstrap vs frozen HC music)

CORROBORATIVE, CLAP-family, NOT human evaluation. No SESOI. Cannot change primary PASS. Reported
regardless of direction. Pinned model provenance (docs/recovery_reversal_v1.md §7a):
sarulab-speech/human-clap-wsce-mae rev 06788887..., safetensors sha256 09357f50..., SR 48000,
truncation=fusion, np.random.seed(20260826) once per 192-item system. Run in .venv-metrics.

Run: OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/reversal_v1_1_humanclap.py \
        --wav-root <job reversal_v1_1_gen dir> --out artifacts/icassp_gate0/reversal_v1_1_humanclap.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/research")
import numpy as np  # noqa: E402
from reversal_humanclap import HumanClapScorer  # reuse the frozen scorer  # noqa: E402
from research_pruning.eval.reversal import (  # noqa: E402
    BOOTSTRAP_SEED_V1, N_PROMPTS_V1, N_REPLICATES_V1, secondary_hc_verdict)

MANIFEST = "configs/research/reversal_v1_1_audiocaps_manifest.json"
R_MUSIC_HC = "configs/research/reversal_v1_1_r_music_humanclap.json"  # fallback below
R_MUSIC_HC_ALT = "configs/research/reversal_v1_r_music_humanclap.json"
PREFIX = {"dense_ema": "dense_noadapter",
          "p1_pruned_ema_reconstructed": "p1_pruned_ema_reconstructed_noadapter",
          "p1_recovered": "p1_recovered_noadapter"}
N = N_PROMPTS_V1 * N_REPLICATES_V1


def _grid(cos):
    return np.asarray(cos, dtype=np.float64).reshape(N_PROMPTS_V1, N_REPLICATES_V1)


def _sha256_obj(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run(wav_root: str, out_path: str) -> dict:
    man = json.load(open(MANIFEST))
    prompts = sorted(man["prompts"], key=lambda p: p["prompt_index"])
    hcm_path = R_MUSIC_HC if os.path.exists(R_MUSIC_HC) else R_MUSIC_HC_ALT
    hcm = json.load(open(hcm_path))
    music_hc = np.array([p["prompt_mean_diff"] for p in sorted(hcm["prompts"], key=lambda x: x["prompt_index"])],
                        dtype=np.float64)

    sc = HumanClapScorer()
    sys_cos = {}
    for sysname, prefix in PREFIX.items():
        caps, wavs = [], []
        for p in prompts:
            for r in range(N_REPLICATES_V1):
                caps.append(p["caption"]); wavs.append(os.path.join(wav_root, f"{prefix}_p{p['prompt_index']}_r{r}.wav"))
        for fp in wavs:
            if not os.path.exists(fp):
                raise SystemExit(f"missing WAV: {fp}")
        cos = sc.cosine(caps, wavs)
        if cos.size != N:
            raise SystemExit(f"{sysname}: {cos.size} scores != {N}")
        sys_cos[sysname] = cos

    rec, pru = _grid(sys_cos["p1_recovered"]), _grid(sys_cos["p1_pruned_ema_reconstructed"])
    dense = _grid(sys_cos["dense_ema"])
    verdict = secondary_hc_verdict(rec, pru, music_hc, seed=BOOTSTRAP_SEED_V1)
    import transformers
    payload = {
        "artifact": "reversal_v1_1_humanclap",
        "status": "SECONDARY / CORROBORATIVE — CLAP-family, NOT human eval; cannot change primary PASS",
        "model": "sarulab-speech/human-clap-wsce-mae",
        "revision": "06788887d254df15db5c0ca9d54da39d46188063",
        "safetensors_sha256": "09357f504d52900cb1bc3bf2fe1f3173dd1702781ef0bdedb122a6e47d4c5c61",
        "lib_versions": {"transformers": transformers.__version__},
        "R_music_HC_baseline": {"path": hcm_path, "point": float(music_hc.mean()),
                                "sha256": hcm.get("artifact_sha256")},
        "HC_dense_mean": float(dense.mean()), "HC_pruned_mean": float(pru.mean()),
        "HC_recovered_mean": float(rec.mean()),
        "verdict": verdict,
        "raw_scores": {s: [float(x) for x in sys_cos[s]] for s in PREFIX},
    }
    payload["artifact_sha256"] = _sha256_obj({k: v for k, v in payload.items() if k != "raw_scores"})
    json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-root", required=True)
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_v1_1_humanclap.json")
    args = ap.parse_args()
    p = run(args.wav_root, args.out)
    v = p["verdict"]
    print(json.dumps({"HC_dense": p["HC_dense_mean"], "HC_pruned": p["HC_pruned_mean"],
                      "HC_recovered": p["HC_recovered_mean"], "R_AC_HC": v["R_AC_HC"],
                      "I_HC": v.get("I_HC"), "frac_rec_gt_pru": v["prompt_sign_fraction_pos"]}, indent=2))
    print("written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
