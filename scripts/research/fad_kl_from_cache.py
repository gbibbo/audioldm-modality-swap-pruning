#!/usr/bin/env python3
"""Compute KL / IS from cached PANNs classifier features, bypassing audioldm_eval's
broken Frechet path (F-eval-3), and demonstrate the standard real-part Frechet fix.

`audioldm_eval.EvaluationHelper.main` aborts on the FAD/FID `sqrtm` imaginary
component before returning KL/IS. This script loads the two `..classifier_logits_
feature_cache.pkl` files that `main` wrote and calls `calculate_kl` / `calculate_isc`
directly (neither uses sqrtm), then computes a Frechet distance on the cached 2048
features using the standard `covmean.real` fix so the metric is finite.

    .venv/bin/python scripts/research/fad_kl_from_cache.py \
        --gen-cache .../genclassifier_logits_feature_cache.pkl \
        --gt-cache  .../gtclassifier_logits_feature_cache.pkl \
        --out .../fad_kl_smoke_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import scipy.linalg


def frechet_realpart(x1: np.ndarray, x2: np.ndarray) -> float:
    """Standard Frechet distance with the real-part fix audioldm_eval omits."""
    mu1, mu2 = x1.mean(0), x2.mean(0)
    s1, s2 = np.cov(x1, rowvar=False), np.cov(x2, rowvar=False)
    covmean, _ = scipy.linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # <-- the fix: take the real part instead of asserting
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2.0 * covmean))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-cache", required=True)
    ap.add_argument("--gt-cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from audioldm_eval.audio.tools import load_pickle
    from audioldm_eval import calculate_kl, calculate_isc

    gen = load_pickle(args.gen_cache)
    gt = load_pickle(args.gt_cache)
    print("loaded caches; logits:", tuple(gen["logits"].shape), tuple(gt["logits"].shape))

    metrics = {}

    kl_out = calculate_kl(gen, gt, "logits", same_name=False)
    metric_kl = kl_out[0] if isinstance(kl_out, tuple) else kl_out
    metrics.update({k: float(v) for k, v in dict(metric_kl).items()})

    isc = calculate_isc(gen, feat_layer_name="logits", rng_seed=2020,
                        samples_shuffle=True, splits=10)
    metrics.update({k: float(v) for k, v in dict(isc).items()})

    # Frechet on the 2048 features with the standard real-part fix (finite, but noisy
    # at N=256 vs 2048 dims — a smoke, not a scientific value).
    g2 = gen["2048"].detach().cpu().numpy() if hasattr(gen["2048"], "detach") else np.asarray(gen["2048"])
    t2 = gt["2048"].detach().cpu().numpy() if hasattr(gt["2048"], "detach") else np.asarray(gt["2048"])
    metrics["frechet_distance_2048_realpart"] = frechet_realpart(g2, t2)

    print("\n==== METRICS (from cached features) ====")
    for k, v in metrics.items():
        print(f"  {k:<36} {v}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"source": "cached_classifier_features", "metrics": metrics}, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
