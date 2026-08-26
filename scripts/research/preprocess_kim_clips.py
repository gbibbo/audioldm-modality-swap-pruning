#!/usr/bin/env python3
"""Deterministically materialize Kim's 193 hip-hop clips as 3.84-s / 16 kHz wavs for Gate-0 training.

Realizes the FROZEN crop policy (DECISION-V4-10): resample each clip to 16 kHz, then deterministic
CENTER crop to exactly 61440 samples (3.84 s); clips shorter than 3.84 s -> symmetric zero-pad. No
random offset. This removes the AudioLDM loader's `random_segment_wav` non-determinism at the source:
the emitted wavs are exactly 3.84 s, so the trainer's loader crops nothing.

RUN IN AN ISOLATED ENV (do NOT touch the frozen .venv):
    uv run --no-project --with pyarrow==17.0.0 --with soundfile --with librosa \
        python scripts/research/preprocess_kim_clips.py

Inputs : artifacts/icassp_gate0/kim_hiphop_193.parquet  (audio @ 44.1 kHz + caption)
Outputs: artifacts/icassp_gate0/kim193_wav_3p84s/*.wav  (gitignored)
         artifacts/icassp_gate0/kim193_train_manifest.json  ({wav, caption} + provenance + sha256)
"""
import hashlib, io, json, os
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import librosa

SR = 16000
CLIP_S = 3.84
N_SAMPLES = int(round(SR * CLIP_S))  # 61440
SRC = "artifacts/icassp_gate0/kim_hiphop_193.parquet"
OUT_DIR = "artifacts/icassp_gate0/kim193_wav_3p84s"
OUT_MANIFEST = "artifacts/icassp_gate0/kim193_train_manifest.json"


def center_crop_pad(w, n):
    if len(w) == n:
        return w
    if len(w) > n:
        s = (len(w) - n) // 2          # deterministic center crop
        return w[s:s + n]
    pad = n - len(w)                    # symmetric zero-pad
    l = pad // 2
    return np.concatenate([np.zeros(l, np.float32), w, np.zeros(pad - l, np.float32)])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t = pq.read_table(SRC)
    audio = t.column("audio").to_pylist()
    caps = t.column("caption").to_pylist()
    durations = []
    entries = []
    for i, (a, cap) in enumerate(zip(audio, caps)):
        # HF audio feature = {"bytes": <encoded>, "path": ...} or already-decoded {"array","sampling_rate"}
        if isinstance(a, dict) and a.get("bytes") is not None:
            w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        elif isinstance(a, dict) and a.get("array") is not None:
            w, sr = np.asarray(a["array"], np.float32), a["sampling_rate"]
        else:
            raise SystemExit(f"clip {i}: unrecognized audio struct keys {list(a) if isinstance(a,dict) else type(a)}")
        if w.ndim > 1:
            w = w.mean(axis=1)          # to mono
        durations.append(len(w) / sr)
        if sr != SR:
            w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=SR)
        w = center_crop_pad(w.astype(np.float32), N_SAMPLES)
        fn = f"kim_{i:03d}.wav"
        sf.write(os.path.join(OUT_DIR, fn), w, SR, subtype="PCM_16")
        entries.append({"wav": os.path.join(OUT_DIR, fn), "caption": cap})
    durations = np.array(durations)
    manifest = {
        "name": "kim193_gate0_train",
        "n": len(entries),
        "clip_seconds": CLIP_S, "sample_rate": SR, "n_samples": N_SAMPLES,
        "crop_policy": "center (DECISION-V4-10)",
        "source_parquet_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
        "src_duration_s": {"min": float(durations.min()), "max": float(durations.max()),
                           "mean": float(durations.mean()),
                           "n_shorter_than_clip": int((durations < CLIP_S).sum())},
        "data": entries,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest["data"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    with open(OUT_MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in manifest.items() if k != "data"}, indent=2))
    print("WROTE", OUT_MANIFEST, "and", len(entries), "wavs to", OUT_DIR)


if __name__ == "__main__":
    main()
