#!/usr/bin/env python3
"""D1 physical-waveform panel for the historical music arm (Recovery-Reversal robustness).

DIAGNOSTIC / SUPPORTING ONLY. Does NOT redefine R_music and does NOT replace the frozen
primary CLAP result. Asks a single question: does recovered's poor held-out music behaviour
have an objective signal-level signature (loudness / near-clipping / spectral tilt), or is it
purely a scorer artefact? Uses ONLY the already-persisted historical OFF (no-adapter) WAVs:
    dense, p1_pruned_ema_reconstructed, p1_recovered  (64 prompts x 3 replicates each).
No generation, no scoring, no GPU. Run in .venv-metrics (librosa/soundfile).

Per-clip deterministic statistics (all standard, defined here explicitly):
    rms            = sqrt(mean(x^2))
    peak           = max(|x|)
    near_clip_frac = mean(|x| >= 0.99)
    crest_db       = 20*log10(peak / rms)
    spectral_centroid_hz = mean over frames of librosa spectral centroid
Reports per-SYSTEM distributions (median/IQR/quantiles) and per-PROMPT paired summaries,
not only global means.

Run: OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python \
        scripts/research/reversal_waveform_panel.py \
        --out artifacts/icassp_gate0/reversal_waveform_panel.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np  # noqa: E402

DENSE_ROOT = ("/teamspace/jobs/gate0-gen-1/artifacts/audioldm-modality-swap-pruning/"
              "artifacts/icassp_gate0/gen_gate0")
DOWN_ROOT = ("/teamspace/jobs/gate0-phenom-1/artifacts/audioldm-modality-swap-pruning/"
             "artifacts/icassp_gate0/gen_phenomenon")
SYSTEMS = {
    "dense": (DENSE_ROOT, "dense_noadapter"),
    "p1_pruned_ema_reconstructed": (DOWN_ROOT, "p1_pruned_ema_reconstructed_noadapter"),
    "p1_recovered": (DOWN_ROOT, "p1_recovered_noadapter"),
}
N_PROMPTS, N_REPS = 64, 3
CLIP_THRESH = 0.99


def clip_stats(x: np.ndarray, sr: int) -> dict:
    import librosa
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    near_clip = float(np.mean(np.abs(x) >= CLIP_THRESH))
    crest_db = float(20.0 * np.log10(peak / rms)) if rms > 0 and peak > 0 else float("nan")
    cent = librosa.feature.spectral_centroid(y=x.astype(np.float32), sr=sr)
    centroid = float(np.mean(cent))
    return {"rms": rms, "peak": peak, "near_clip_frac": near_clip,
            "crest_db": crest_db, "spectral_centroid_hz": centroid}


def _dist(vals: np.ndarray) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "q10": float(np.percentile(v, 10)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "q90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max())}


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run(out_path: str, systems: dict | None = None, n_prompts: int = N_PROMPTS,
        n_reps: int = N_REPS) -> dict:
    import librosa
    systems = systems if systems is not None else SYSTEMS
    metrics = ["rms", "peak", "near_clip_frac", "crest_db", "spectral_centroid_hz"]
    # per_clip[system][metric] shape (n_prompts, n_reps)
    per_clip = {s: {m: np.full((n_prompts, n_reps), np.nan) for m in metrics} for s in systems}
    sr_seen = set()
    for s, (root, prefix) in systems.items():
        for p in range(n_prompts):
            for r in range(n_reps):
                fp = os.path.join(root, f"{prefix}_p{p}_r{r}.wav")
                if not os.path.exists(fp):
                    raise SystemExit(f"missing WAV: {fp}")
                x, sr = librosa.load(fp, sr=None, mono=True)
                sr_seen.add(int(sr))
                st = clip_stats(x, int(sr))
                for m in metrics:
                    per_clip[s][m][p, r] = st[m]

    systems_out = {}
    for s in systems:
        systems_out[s] = {}
        for m in metrics:
            arr = per_clip[s][m]
            prompt_mean = arr.mean(axis=1)
            systems_out[s][m] = {"clip_dist": _dist(arr.ravel()),
                                 "prompt_mean_dist": _dist(prompt_mean)}

    # per-prompt paired contrasts (descriptive; NOT a gate) — only for the standard 3-system panel
    contrasts = {}
    std = {"p1_recovered", "p1_pruned_ema_reconstructed", "dense"}
    if std.issubset(systems):
        def paired(a_sys, b_sys, m):
            a = per_clip[a_sys][m].mean(axis=1); b = per_clip[b_sys][m].mean(axis=1)
            diff = a - b
            return {"prompt_mean_diff_median": float(np.median(diff)),
                    "prompt_mean_diff_mean": float(diff.mean()),
                    "frac_prompts_a_gt_b": float((diff > 0).mean())}
        for m in metrics:
            contrasts[m] = {
                "recovered_minus_dense": paired("p1_recovered", "dense", m),
                "recovered_minus_pruned": paired("p1_recovered", "p1_pruned_ema_reconstructed", m),
                "pruned_minus_dense": paired("p1_pruned_ema_reconstructed", "dense", m),
            }

    payload = {
        "artifact": "reversal_waveform_panel",
        "status": "DIAGNOSTIC / SUPPORTING — not a gate; does not redefine R_music",
        "systems": {s: {"root": systems[s][0], "prefix": systems[s][1],
                        "n_clips": n_prompts * n_reps} for s in systems},
        "sample_rate_hz": sorted(sr_seen),
        "clip_threshold": CLIP_THRESH,
        "metrics": metrics,
        "per_system": systems_out,
        "paired_prompt_contrasts": contrasts,
    }
    payload["artifact_sha256"] = _sha256_obj(payload)
    json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
    open(out_path + ".sha256", "w").write(
        hashlib.sha256(open(out_path, "rb").read()).hexdigest() + "  "
        + os.path.basename(out_path) + "\n")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_waveform_panel.json")
    args = ap.parse_args()
    payload = run(args.out)
    print(f"sample_rate_hz {payload['sample_rate_hz']}")
    for s in SYSTEMS:
        ps = payload["per_system"][s]
        print(f"{s:32} rms med {ps['rms']['clip_dist']['median']:.3f}  "
              f"peak med {ps['peak']['clip_dist']['median']:.3f}  "
              f"nearclip med {ps['near_clip_frac']['clip_dist']['median']:.3f}  "
              f"centroid med {ps['spectral_centroid_hz']['clip_dist']['median']:.0f}Hz")
    rc = payload["paired_prompt_contrasts"]
    print("recovered-vs-dense  rms frac>  ", rc["rms"]["recovered_minus_dense"]["frac_prompts_a_gt_b"])
    print("recovered-vs-pruned rms frac>  ", rc["rms"]["recovered_minus_pruned"]["frac_prompts_a_gt_b"])
    print("written to", args.out, "sha256", payload["artifact_sha256"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
