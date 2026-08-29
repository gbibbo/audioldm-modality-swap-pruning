#!/usr/bin/env python3
"""Persist the DURABLE frozen historical music baselines for RECOVERY-REVERSAL-V1.

The prospective V1 interaction bootstrap (I = R_AC - R_music, and the secondary
I_HC = R_AC_HC - R_music_HC) must resample the frozen 64 per-prompt music contrasts. The
underlying artifacts live under artifacts/ (gitignored), so this writes small TRACKED config
files carrying the 64 per-prompt contrasts + CIs + provenance, self-hashed, so the future
verdict never depends on a gitignored artifact.

  configs/research/reversal_v1_r_music_clap.json       (primary laion CLAP, seed 20260826)
  configs/research/reversal_v1_r_music_humanclap.json  (secondary Human-CLAP)

Primary is recomputed from the persisted phenom groups (deterministic, no rescoring). Human-CLAP
is distilled from the validated artifacts/icassp_gate0/reversal_humanclap.json (deterministic given
the pinned model revision + safetensors sha), regenerable via scripts/research/reversal_humanclap.py.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/reversal_v1_freeze_r_music.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from research_pruning.eval.reversal import reconstruct_music_grids, r_music  # noqa: E402

A = "configs/research"
GROUPS_IN = "artifacts/icassp_gate0/_phenom_groups_in.json"
GROUPS_OUT = "artifacts/icassp_gate0/_phenom_groups_out.json"
BATTERY = "configs/research/icassp_gate0_battery.json"
HC_ARTIFACT = "artifacts/icassp_gate0/reversal_humanclap.json"
HC_MODEL = "sarulab-speech/human-clap-wsce-mae"
HC_REVISION = "06788887d254df15db5c0ca9d54da39d46188063"
HC_SAFETENSORS_SHA256 = "09357f504d52900cb1bc3bf2fe1f3173dd1702781ef0bdedb122a6e47d4c5c61"


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _sha256_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write_primary() -> dict:
    recon = reconstruct_music_grids(json.load(open(GROUPS_IN)), json.load(open(GROUPS_OUT)))
    bat = [p["ytid"] for p in json.load(open(BATTERY))["prompts"]]
    ci = r_music(recon, seed=20260826)
    payload = {
        "artifact": "reversal_v1_r_music_clap",
        "scorer": "laion/clap-htsat-fused rev 365dea6e (frozen primary Option-B convention)",
        "seed": 20260826, "n": 64,
        "R_music": {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "b": ci.b},
        "definition": "per prompt: mean_r (C_recovered_off - C_pruned_off); prompt-cluster percentile bootstrap",
        "sources": {"groups_in_sha256": _sha256_file(GROUPS_IN),
                    "groups_out_sha256": _sha256_file(GROUPS_OUT),
                    "battery_prompts_sha256": json.load(open(BATTERY)).get("prompts_sha256")},
        "prompts": [{"prompt_index": p, "ytid": bat[p],
                     "prompt_mean_diff": float(recon.prompt_mean_diff[p])} for p in range(64)],
    }
    payload["artifact_sha256"] = _sha256_obj(payload)
    out = f"{A}/reversal_v1_r_music_clap.json"
    json.dump(payload, open(out, "w"), indent=1, ensure_ascii=False)
    return payload


def write_humanclap() -> dict:
    hc = json.load(open(HC_ARTIFACT))
    diffs = [(pr["prompt_index"], pr["ytid"], float(pr["prompt_mean_diff"])) for pr in hc["prompts"]]
    diffs.sort(key=lambda x: x[0])
    vec = np.array([d[2] for d in diffs], dtype=np.float64)
    # regression check vs the artifact's own R_music_HC point (mean of per-prompt diffs)
    assert abs(vec.mean() - hc["R_music_humanclap"]["point"]) < 1e-6, "HC per-prompt mean drift"
    payload = {
        "artifact": "reversal_v1_r_music_humanclap",
        "scorer": "Human-CLAP (secondary, corroborative, CLAP-family; NOT human eval)",
        "model": HC_MODEL, "revision": HC_REVISION, "safetensors_sha256": HC_SAFETENSORS_SHA256,
        "processor": "laion/clap-htsat-fused", "sampling_rate": 48000,
        "convention": "SR48k, truncation=fusion, get_*_features cosine, np.random.seed(20260826) once/192-system",
        "seed": 20260826, "n": 64,
        "R_music_HC": {k: hc["R_music_humanclap"][k] for k in ("point", "lo", "hi", "b")},
        "prompt_sign_fraction_neg": hc["prompt_sign_fraction_neg"],
        "regenerate_with": "scripts/research/reversal_humanclap.py (.venv-metrics)",
        "prompts": [{"prompt_index": pi, "ytid": yt, "prompt_mean_diff": d} for pi, yt, d in diffs],
    }
    payload["artifact_sha256"] = _sha256_obj(payload)
    out = f"{A}/reversal_v1_r_music_humanclap.json"
    json.dump(payload, open(out, "w"), indent=1, ensure_ascii=False)
    return payload


def main() -> int:
    p = write_primary()
    h = write_humanclap()
    print(json.dumps({
        "primary_R_music": p["R_music"], "primary_sha": p["artifact_sha256"][:12],
        "humanclap_R_music_HC": h["R_music_HC"], "humanclap_sha": h["artifact_sha256"][:12],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
