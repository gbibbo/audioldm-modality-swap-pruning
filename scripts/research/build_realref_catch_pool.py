#!/usr/bin/env python3
"""Build a pool of REAL AudioCaps reference clips for the matched-vs-unrelated
attention catch (v1.1 amendment). CPU only. Outcome-independent: uses only real
audio + ground-truth AudioSet labels + human captions; no generated-study outcome.

A real AudioCaps clip genuinely realizes its human caption, so a matched real clip
is unambiguously more relevant than an unrelated real clip -> a robust attention
control that may enter the gross-failure criterion (unlike a generated clip, which
can fail to realize its own caption).

Selects a deterministic pool (new salt), excludes every ytid used in the study,
resamples 32 kHz->16 kHz, pads to the native 163872-sample length, verifies -36 LUFS
feasibility, and stages copies under artifacts/listening_study/real_refs/ (gitignored).

Emits configs/research/listening_study_realref_pool.json (committed metadata:
ytid, caption, labels, sha, lufs, peak, feasible).

Run: .venv-loudness/bin/python scripts/research/build_realref_catch_pool.py
"""
import json, os, hashlib
import numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy.signal import resample_poly

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
LABELS = os.path.join(ROOT, "data/dataset/metadata/audiocaps/datafiles/audiocaps_test_label.json")
AUDIO_BASE = os.path.join(ROOT, "data/dataset/audioset")
STAGE = os.path.join(ROOT, "artifacts/listening_study/real_refs")
INV = os.path.join(ROOT, "configs/research/listening_study_inventory.json")
OUT = os.path.join(ROOT, "configs/research/listening_study_realref_pool.json")
SALT = "LISTENING-STUDY|REALREF-CATCH|2026-08-31"
NATIVE_N = 163872
TARGET, CEIL = -36.0, -1.0
POOL_SIZE = 48


def main():
    os.chdir(ROOT)
    os.makedirs(STAGE, exist_ok=True)
    lab = json.load(open(LABELS))["data"]
    inv = json.load(open(INV))
    exclude = set()
    for sev in ("sev1", "sev2"):
        for p in inv["prompts"][sev]:
            exclude.add(p["ytid"])

    cand = []
    for r in lab:
        base = os.path.basename(r["wav"])
        if not base.startswith("Y") or not base.endswith(".wav"):
            continue
        ytid = base[1:-4]
        if ytid in exclude:
            continue
        cap = (r.get("caption") or "").strip()
        labs = [m for m in (r.get("labels") or "").split(",") if m]
        if not cap or not labs:
            continue
        path = os.path.join(AUDIO_BASE, r["wav"])
        if not os.path.exists(path):
            continue
        cand.append((ytid, cap, labs, path))
    # deterministic diverse selection by hash
    cand.sort(key=lambda t: hashlib.sha256((SALT + "|" + t[0]).encode()).hexdigest())

    pool, meter = {}, None
    for ytid, cap, labs, path in cand:
        if len(pool) >= POOL_SIZE:
            break
        x, sr = sf.read(path)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = np.asarray(x, dtype=np.float64)
        if sr != 16000:
            g = np.gcd(sr, 16000)
            x = resample_poly(x, 16000 // g, sr // g)
        # pad/trim to native length
        if len(x) < NATIVE_N:
            x = np.concatenate([x, np.zeros(NATIVE_N - len(x))])
        else:
            x = x[:NATIVE_N]
        meter = pyln.Meter(16000)
        lufs = float(meter.integrated_loudness(x))
        peak_db = 20 * np.log10(np.max(np.abs(x)) + 1e-12)
        if not np.isfinite(lufs):
            continue
        post_peak = peak_db + (TARGET - lufs)
        if post_peak > CEIL:   # keep pool peak-safe at the frozen target
            continue
        out = os.path.join(STAGE, f"rr_{ytid}.wav")
        sf.write(out, x.astype(np.float32), 16000, subtype="PCM_16")
        sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
        pool[ytid] = {"ytid": ytid, "caption": cap, "labels": labs,
                      "staged_path": out, "sha256": sha,
                      "integrated_lufs": round(lufs, 3), "sample_peak_dbfs": round(peak_db, 3),
                      "post_gain_peak_dbfs": round(post_peak, 3)}
    o = {"artifact": "listening_study_realref_pool", "salt": SALT,
         "native_samples": NATIVE_N, "target_lufs": TARGET, "peak_ceiling_dbfs": CEIL,
         "n_pool": len(pool), "excluded_study_ytids": len(exclude), "pool": pool}
    payload = json.dumps(o, indent=2, sort_keys=True)
    o["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json.dump(o, open(OUT, "w"), indent=2, sort_keys=True)
    print(f"pool size {len(pool)} (target {POOL_SIZE}); staged in {STAGE}")
    print("self_sha256", o["self_sha256"][:16])


if __name__ == "__main__":
    main()
