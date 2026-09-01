#!/usr/bin/env python3
"""Build the deployed loudness-normalized listening-copy audio bundle from the
FROZEN private assignment key. CPU only. Creates listening_study/audio/*.wav.

Listening copies ONLY: originals are never modified. Each copy is the source WAV
scaled by a single fixed gain to the frozen BS.1770 target (-36 LUFS), verified to
keep sample peak <= -1 dBFS. No limiting, no compression. Records original and
listening-copy SHA256.

Run AFTER the freeze commit (freeze order step 9):
  .venv-loudness/bin/python scripts/research/build_listening_bundle.py
"""
import json, os, hashlib
import numpy as np, soundfile as sf, pyloudnorm as pyln

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
PRIV = os.path.join(ROOT, "configs/research/listening_study_assignments_private.json")
INV = os.path.join(ROOT, "configs/research/listening_study_inventory.json")
AUDIO_DIR = os.path.join(ROOT, "listening_study/audio")
BUNDLE_MANIFEST = os.path.join(ROOT, "configs/research/listening_study_bundle_manifest.json")
TARGET = -36.0
CEIL = -1.0


def main():
    os.chdir(ROOT)
    priv = json.load(open(PRIV))
    render = priv["audio_render_map"]  # hash_name -> {stim_id, src_path, src_sha256}
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # cache normalized arrays per source path (many hash copies share a source)
    cache = {}
    problems = []
    bundle = {}
    for hn, meta in sorted(render.items()):
        sid = meta["stim_id"]; src = meta["src_path"]; src_sha = meta["src_sha256"]
        if sid not in cache:
            x, sr = sf.read(src)
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = np.asarray(x, dtype=np.float64)
            meter = pyln.Meter(sr)
            lufs = float(meter.integrated_loudness(x))
            gain_db = TARGET - lufs
            y = x * (10.0 ** (gain_db / 20.0))
            peak_db = 20.0 * np.log10(np.max(np.abs(y)) + 1e-12)
            if peak_db > CEIL + 1e-6:
                problems.append(f"PEAK {sid}: {peak_db:.3f} dBFS > {CEIL}")
            # verify achieved loudness
            lufs_after = float(meter.integrated_loudness(y))
            cache[sid] = (y.astype(np.float32), sr, lufs, gain_db, peak_db, lufs_after)
        y, sr, lufs, gain_db, peak_db, lufs_after = cache[sid]
        out = os.path.join(AUDIO_DIR, hn)
        sf.write(out, y, sr, subtype="PCM_16")
        copy_sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
        bundle[hn] = {
            "stim_id": sid, "src_sha256": src_sha, "copy_sha256": copy_sha,
            "src_lufs": round(lufs, 3), "gain_db": round(gain_db, 3),
            "copy_peak_dbfs": round(peak_db, 3), "copy_lufs": round(lufs_after, 3),
            "sr": sr, "n_samples": int(len(y)),
        }
    out = {
        "artifact": "listening_study_bundle_manifest",
        "target_lufs": TARGET, "peak_ceiling_dbfs": CEIL,
        "n_files": len(bundle), "n_unique_sources": len(cache),
        "public_bundle_sha256": priv.get("public_bundle_sha256"),
        "problems": problems, "files": bundle,
    }
    payload = json.dumps(out, indent=2, sort_keys=True)
    out["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with open(BUNDLE_MANIFEST, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"WROTE {len(bundle)} listening copies to {AUDIO_DIR}")
    print(f"unique sources normalized: {len(cache)}; problems: {len(problems)}")
    for p in problems[:20]:
        print("  ", p)
    lufs_after_all = [b["copy_lufs"] for b in bundle.values()]
    peaks = [b["copy_peak_dbfs"] for b in bundle.values()]
    print(f"copy LUFS range: {min(lufs_after_all):.2f}..{max(lufs_after_all):.2f} (target {TARGET})")
    print(f"copy peak max: {max(peaks):.2f} dBFS (ceiling {CEIL})")
    print("bundle_manifest self_sha256:", out["self_sha256"])


if __name__ == "__main__":
    main()
