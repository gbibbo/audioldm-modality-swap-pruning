#!/usr/bin/env python3
"""Re-score the existing M4 screening audio with a finite, real-part Frechet (Q1).

F-eval-3: `audioldm_eval` returns NaN for both `frechet_audio_distance` (VGGish) and
`frechet_distance` (Cnn14) because its sqrtm guard raises on the imaginary component.
The audio and the extracted features are fine — only the sqrtm post-processing is
broken. This script recomputes, from the feature caches the screening eval already
wrote (no GPU, no re-generation, no audioldm_eval Frechet), using the tracked
`research_pruning.eval.frechet` real-part implementation:

  * FAD  = Frechet on cached VGGish embeddings   (`*_fad_feature_cache.npy`, 128-dim)
  * FD   = Frechet on cached Cnn14 2048 features  (`*classifier_logits_feature_cache.pkl`)

and re-derives IS and KL from the cached PANNs logits as a provenance cross-check
against the numbers recorded in M4-SCREEN-FOUND (proves the caches are the screening
audio). Self-distance FAD(ref,ref)/FD(ref,ref) ~ 0 is the correctness control.

    .venv/bin/python scripts/research/rescore_m4_screening.py \
        --out artifacts/m4_screening/rescore_frechet.json

Values are SCREENING-ONLY (1 seed, 100 clips, Cnn14-FD covariance rank <= 99 << 2048).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

from research_pruning.eval.frechet import frechet_distance

SCREEN_ROOT = "artifacts/m4_screening"
SYSTEMS = ["base", "P0_published", "P0_L1", "P1", "P2", "P3"]


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def gen_dir_for(system: str):
    hits = sorted(glob.glob(os.path.join(SCREEN_ROOT, system, "gen", "infer_*")))
    hits = [h for h in hits if os.path.isdir(h)]
    return hits[-1] if hits else None


def vggish_npy_for(system: str):
    d = gen_dir_for(system)
    if d is None:
        return None
    hits = sorted(glob.glob(d + "*_fad_feature_cache.npy"))
    return hits[-1] if hits else None


def cnn14_pkl_for(system: str):
    d = gen_dir_for(system)
    if d is None:
        return None
    hits = sorted(glob.glob(d + "classifier_logits_feature_cache.pkl"))
    return hits[-1] if hits else None


def cross_check_is(gen_pkl_path):
    """Re-derive IS from cached logits to confirm the cache is the screening audio.

    IS depends only on the generated logits (no pairing), so it must reproduce the
    M4-SCREEN-FOUND values almost exactly if the cache is the same audio. (KL is not
    recomputed here: it needs caption pairing that the plain cache API returns a -1
    sentinel for; the validly-paired KL is already recorded in the ledger.)
    """
    out = {}
    try:
        from audioldm_eval import calculate_isc
    except Exception as e:  # pragma: no cover
        out["error"] = f"audioldm_eval import failed: {e}"
        return out
    gen = load_pkl(gen_pkl_path)
    try:
        isc = calculate_isc(gen, feat_layer_name="logits", rng_seed=2020,
                            samples_shuffle=True, splits=10)
        out.update({k: float(v) for k, v in dict(isc).items()})
    except Exception as e:
        out["isc_error"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SCREEN_ROOT, "rescore_frechet.json"))
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--no-cross-check", action="store_true")
    args = ap.parse_args()

    systems = [s for s in args.systems.split(",") if s]

    ref_vgg_path = os.path.join(SCREEN_ROOT, "_reference_fad_feature_cache.npy")
    ref_pkl_path = os.path.join(SCREEN_ROOT, "_referenceclassifier_logits_feature_cache.pkl")
    for p in (ref_vgg_path, ref_pkl_path):
        if not os.path.exists(p):
            print(f"FATAL: reference cache missing: {p}", file=sys.stderr)
            return 2

    ref_vgg = np.load(ref_vgg_path, allow_pickle=True).astype(np.float64)
    ref_pkl = load_pkl(ref_pkl_path)
    ref_cnn = _to_np(ref_pkl["2048"]).astype(np.float64)
    print(f"reference: VGGish {ref_vgg.shape}, Cnn14-2048 {ref_cnn.shape}")

    # Correctness control: self-distance must be ~0.
    ctrl_fad = frechet_distance(ref_vgg, ref_vgg)
    ctrl_fd = frechet_distance(ref_cnn, ref_cnn)
    print(f"self-distance control: FAD(ref,ref)={ctrl_fad.fd:.3e}  FD(ref,ref)={ctrl_fd.fd:.3e}")

    result = {
        "source": "cached screening features (no re-generation, no GPU)",
        "note": "SCREENING ONLY: 1 seed, 100 gen clips vs 100 ref clips; "
                "Cnn14-FD covariance is rank-deficient (rank<=99 << 2048).",
        "reference": {"vggish_frames": int(ref_vgg.shape[0]),
                      "cnn14_clips": int(ref_cnn.shape[0])},
        "self_distance_control": {"fad_ref_ref": ctrl_fad.fd, "fd_ref_ref": ctrl_fd.fd},
        "systems": {},
    }

    header = f"{'system':<14}{'FAD(VGGish)':>13}{'FD(Cnn14)':>12}{'IS':>8}"
    print("\n" + header)
    print("-" * len(header))
    for s in systems:
        vgg_path = vggish_npy_for(s)
        pkl_path = cnn14_pkl_for(s)
        entry = {"vggish_cache": vgg_path, "cnn14_cache": pkl_path}
        if vgg_path is None or pkl_path is None:
            entry["error"] = "cache(s) missing"
            result["systems"][s] = entry
            print(f"{s:<14}  MISSING CACHE")
            continue

        gen_vgg = np.load(vgg_path, allow_pickle=True).astype(np.float64)
        gen_cnn = _to_np(load_pkl(pkl_path)["2048"]).astype(np.float64)

        fad = frechet_distance(gen_vgg, ref_vgg)
        fd = frechet_distance(gen_cnn, ref_cnn)
        entry["fad_vggish"] = fad.as_dict()
        entry["fd_cnn14"] = fd.as_dict()

        cc = {} if args.no_cross_check else cross_check_is(pkl_path)
        if cc:
            entry["cross_check"] = cc

        result["systems"][s] = entry
        is_mean = cc.get("inception_score_mean", float("nan"))
        print(f"{s:<14}{fad.fd:>13.4f}{fd.fd:>12.3f}{is_mean:>8.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {args.out}")

    all_finite = all(
        np.isfinite(e.get("fad_vggish", {}).get("fd", np.nan))
        and np.isfinite(e.get("fd_cnn14", {}).get("fd", np.nan))
        for e in result["systems"].values() if "error" not in e
    )
    print(f"all FAD/FD finite: {all_finite}")
    return 0 if all_finite else 1


if __name__ == "__main__":
    sys.exit(main())
