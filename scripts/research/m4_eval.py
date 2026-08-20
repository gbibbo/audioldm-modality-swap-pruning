#!/usr/bin/env python3
"""M4 evaluation driver — FAD/KL/FD + PANNs top-k per criterion (CPU, no GPU cost).

Runs on the free CPU Studio over the audio the screening job (m4_screening.py) generated,
so the RQ2 verdict costs no credits. For each criterion folder it computes:
  * Frechet distance / KL / IS from `audioldm_eval` (Cnn14 16 kHz backbone),
  * PANNs top-k semantic labels,
against a REFERENCE folder of real AudioCaps validation audio (the disjoint val split).

Known eval limitations carried from M0-006 (see docs/.../eval_pipeline_closure.md):
  * F-eval-3: audioldm_eval's VGGish FAD is numerically unusable (sqrtm imaginary); it is
    worked around to NaN so KL/IS/FD survive. A publication FAD needs a real-part FAD; the
    screening comparison therefore leans on Cnn14-FD + KL + PANNs semantics.
  * F-eval-1/2: CPU-sanitised Cnn14 checkpoint + `--fresh` cache discipline.

This driver only ORCHESTRATES the already-validated M0-006 scripts (fad_kl_smoke.py,
panns_topk.py) as subprocesses and aggregates their JSON into one comparison table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

VAL_MANIFEST = "configs/research/val_split_disjoint.json"
METADATA_ROOT = "data/dataset/metadata/dataset_root.json"
SCREEN_ROOT = "artifacts/m4_screening"
VENV = ".venv/bin/python"
CRITERIA = ["base", "P0_published", "P0_L1", "P1", "P2", "P3"]


def build_reference(n, ref_dir):
    """Copy the first n real val-split wavs into ref_dir (the FAD/KL ground truth)."""
    os.makedirs(ref_dir, exist_ok=True)
    root = json.load(open(METADATA_ROOT))["audiocaps"]
    items = json.load(open(VAL_MANIFEST))["items"][:n]
    copied = 0
    for it in items:
        src = os.path.join(root, it["wav"])
        if os.path.exists(src):
            shutil.copy(src, os.path.join(ref_dir, os.path.basename(it["wav"])))
            copied += 1
    return copied


def gen_dir_for(crit):
    """The screening job writes to artifacts/m4_screening/<crit>/gen/infer_.../*.wav."""
    hits = sorted(glob.glob(os.path.join(SCREEN_ROOT, crit, "gen", "infer_*")))
    return hits[-1] if hits else None


def run_json(cmd):
    """Run a subprocess that writes JSON to a temp --out; return the parsed dict."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-n", type=int, default=200, help="reference clips from the val split")
    ap.add_argument("--criteria", default=",".join(CRITERIA))
    ap.add_argument("--ref-dir", default="artifacts/m4_screening/_reference")
    ap.add_argument("--out", default="artifacts/m4_screening/eval_comparison.json")
    ap.add_argument("--panns-k", type=int, default=10)
    ap.add_argument("--skip-fad-kl", action="store_true", help="PANNs only (faster)")
    args = ap.parse_args()

    criteria = [c for c in args.criteria.split(",") if c]
    ncopied = build_reference(args.ref_n, args.ref_dir)
    print(f"reference: {ncopied} real val wavs -> {args.ref_dir}")

    comparison = {"reference_dir": args.ref_dir, "reference_clips": ncopied, "criteria": {}}
    for crit in criteria:
        gdir = gen_dir_for(crit)
        entry = {"gen_dir": gdir}
        if gdir is None:
            entry["error"] = "no generated audio found (run m4_screening.py first)"
            comparison["criteria"][crit] = entry
            print(f"  {crit:14s} MISSING generated audio")
            continue
        n_gen = len(glob.glob(os.path.join(gdir, "*.wav")))
        entry["n_generated"] = n_gen

        if not args.skip_fad_kl:
            out_fk = f"/tmp/m4_fadkl_{crit}.json"
            rc, so, se = run_json([VENV, "scripts/research/fad_kl_smoke.py",
                                   "--gen", gdir, "--gt", args.ref_dir, "--fresh", "--out", out_fk])
            if rc == 0 and os.path.exists(out_fk):
                entry["fad_kl"] = json.load(open(out_fk))
            else:
                entry["fad_kl_error"] = (se or so)[-500:]

        out_p = f"/tmp/m4_panns_{crit}.json"
        rc, so, se = run_json([VENV, "scripts/research/panns_topk.py",
                               "--dir", gdir, "--k", str(args.panns_k), "--out", out_p])
        if rc == 0 and os.path.exists(out_p):
            entry["panns_topk"] = json.load(open(out_p))
        else:
            entry["panns_error"] = (se or so)[-500:]

        comparison["criteria"][crit] = entry
        fk = entry.get("fad_kl", {})
        print(f"  {crit:14s} n={n_gen}  "
              f"FD={fk.get('frechet_distance', fk.get('fd', 'n/a'))}  "
              f"KL={fk.get('kullback_leibler', fk.get('kl', 'n/a'))}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(comparison, fh, indent=2)
    print(f"\ncomparison written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
