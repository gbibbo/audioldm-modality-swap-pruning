#!/usr/bin/env python3
"""M0 closure: run the audioldm_eval FAD/KL pipeline end-to-end on a real folder pair.

This exercises `audioldm_eval.EvaluationHelper.main` (previously only proven to
import) and records the exact invocation and outputs. It is a PIPELINE smoke, not
a scientific evaluation: the two folders are arbitrary disjoint AudioCaps subsets,
so the metric values are not meaningful — only "the pipeline runs and returns a
metrics dict" is being established.

Known caveat surfaced here (recorded in the eval protocol): in audioldm_eval 0.0.5
the Cnn14 backbone used for FD/KL/IS is constructed WITHOUT pretrained weights
(no checkpoint load exists anywhere in the package), so FD/KL/IS from it are not
scientifically valid until a pretrained Cnn14 is supplied. FAD uses VGGish fetched
via torch.hub and is the only trustworthy metric out of the box.

    .venv/bin/python scripts/research/fad_kl_smoke.py --gen DIR --gt DIR --out JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="folder of 'generated' wavs")
    ap.add_argument("--gt", required=True, help="folder of 'target' wavs")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="clear audioldm_eval's path-keyed feature caches first (they are "
                         "keyed on the folder path, so re-symlinking a folder silently reuses "
                         "stale features otherwise)")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    if args.fresh:
        for base in (args.gen, args.gt):
            for suffix in ("_fad_feature_cache.npy", "classifier_logits_feature_cache.pkl"):
                p = base.rstrip("/") + suffix
                if os.path.exists(p):
                    os.remove(p)
                    print(f"removed stale cache {p}")

    import torch
    from audioldm_eval import EvaluationHelper

    device = torch.device("cpu")
    print(f"Constructing EvaluationHelper(sr={args.sr}, device=cpu) — downloads VGGish via torch.hub on first run")
    helper = EvaluationHelper(args.sr, device)

    # FAD via VGGish uses scipy sqrtm; on some data the matrix square root has a
    # non-negligible imaginary component that exceeds audioldm_eval 0.0.5's
    # tolerance, so `frechet.score` returns an int sentinel and the library's own
    # loop then crashes on `out.update(int)`. Wrap the instance method so a failed
    # FAD yields NaN and the KL / IS / FID metrics (computed afterwards) still run.
    _orig_score = helper.frechet.score
    def _safe_score(*a, **k):
        r = _orig_score(*a, **k)
        return r if isinstance(r, dict) else {"frechet_distance": float("nan")}
    helper.frechet.score = _safe_score

    # The Cnn14-2048 FID (calculate_fid) uses the same scipy sqrtm and ALSO raises an
    # AssertionError on a non-negligible imaginary component (a 2048-dim covariance is
    # rank-deficient for small N). Left unhandled it aborts main() *after* KL and IS are
    # computed, discarding them. Wrap it so a failed FID yields NaN and KL/IS survive.
    import audioldm_eval.eval as _evalmod
    _orig_fid = _evalmod.calculate_fid
    def _safe_fid(*a, **k):
        try:
            return _orig_fid(*a, **k)
        except Exception:  # noqa: BLE001
            return {"frechet_inception_distance": float("nan")}
    _evalmod.calculate_fid = _safe_fid

    print(f"Running main(gen={args.gen}, gt={args.gt}, limit_num={args.limit})")
    try:
        metrics = helper.main(args.gen, args.gt, limit_num=args.limit)
    except Exception:
        traceback.print_exc()
        return 2

    # metrics may contain numpy/torch scalars; coerce to plain floats for JSON.
    clean = {}
    for k, v in dict(metrics).items():
        try:
            clean[k] = float(v)
        except Exception:
            clean[k] = str(v)
    print("\n==== METRICS ====")
    for k, v in clean.items():
        print(f"  {k:<24} {v}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"sr": args.sr, "gen": args.gen, "gt": args.gt,
                       "limit": args.limit, "metrics": clean}, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
