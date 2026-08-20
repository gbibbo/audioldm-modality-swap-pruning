#!/usr/bin/env python3
"""M4 screening — materialize the pruned models per criterion and GENERATE audio (S=50).

Consumes the M3B saliency artifact (`--saliency`, produced by m3b_saliency.py
`--save-saliency`) and, for each pruning criterion, builds a structurally-pruned
`(1,2,3,1)` U-Net whose kept channels follow that criterion's saliency ranking, then
generates audio on the disjoint validation split at DDIM S=50 (the screening budget
authorized by DECISION-CG-001). All criteria share the SAME materialization pipeline
(`research_pruning.diagnostics.random_masks.materialize` + `LAYER_MAP`), so the pruned
models differ ONLY where channel selection is ranking-driven — the criterion effect is
isolated. The base VAE, CLAP conditioner and HiFi-GAN vocoder are shared across all
models; only the diffusion U-Net is swapped.

Criteria generated: base (unpruned reference), P0_published (Arshdeep's inverted L1),
P0_L1 (keep-highest L1, secondary reference), P1 (text-only Taylor), P2 (paired mean),
P3 (swap-robust max). Generated audio is gitignored; evaluation (FAD/KL/PANNs) is a
separate step (scripts/research/fad_kl_smoke.py, panns_topk.py) over these folders.

Cost discipline mirrors measure_tgen.py: refuses to run without CUDA unless
--dry-run-cpu (which uses a tiny clip/step budget and the CPU-only load shims and
refuses --out).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

# Reuse the validated generation-stack build from measure_tgen.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_tgen import _cpu_torch_load, _nullctx, load_config, build_model, build_loader, git_provenance  # noqa: E402

from research_pruning.diagnostics.random_masks import load_l1_ranking, materialize  # noqa: E402

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
RANKING_PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
SCRATCH_LOG = "artifacts/m4_screening"
CRITERIA = ["base", "P0_published", "P0_L1", "P1", "P2", "P3"]


def ranking_from_saliency(sal_layer_dict, l1_ranking):
    """Build a per-layer channel ordering (descending saliency) keyed like the L1 ranking.

    keep_topk / materialize keep the FIRST k of each layer's order, so descending
    saliency => keep the highest-saliency channels. For P0_published the saved saliency
    is already -L1 (published inverted convention), so this keeps the lowest-L1 set.
    """
    ranking = {}
    for k in l1_ranking:
        v = sal_layer_dict[k].detach().cpu().numpy()
        ranking[k] = np.argsort(-v).astype(np.int64).tolist()
    return ranking


def base_unet_state(base_ckpt):
    sd = torch.load(base_ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    return {k[len("model.diffusion_model."):]: v
            for k, v in sd.items() if k.startswith("model.diffusion_model.")}


def preflight(args, result):
    prov = git_provenance()
    result["git"] = prov
    if args.expect_commit and prov["commit"] != args.expect_commit:
        raise SystemExit(f"PREFLIGHT FAIL: commit {prov['commit']} != {args.expect_commit}")
    if prov["dirty"] and not args.allow_dirty and not args.dry_run_cpu:
        raise SystemExit("PREFLIGHT FAIL: dirty tree (use --allow-dirty only for CPU dev)")
    if not args.dry_run_cpu:
        if not torch.cuda.is_available():
            raise SystemExit("PREFLIGHT FAIL: no CUDA and not --dry-run-cpu")
        name = torch.cuda.get_device_name(0)
        result["gpu_name"] = name
        if args.expect_gpu and args.expect_gpu.lower() not in name.lower():
            raise SystemExit(f"PREFLIGHT FAIL: GPU {name} != expected {args.expect_gpu}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saliency", default="artifacts/m3_pilot/m3b_saliency.pt",
                    help="the .pt written by m3b_saliency.py --save-saliency")
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--criteria", default=",".join(CRITERIA),
                    help="comma-separated subset of: " + ",".join(CRITERIA))
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu refuses --out")

    result = {"script": "m4_screening.py", "dry_run_cpu": args.dry_run_cpu}
    preflight(args, result)

    criteria = [c for c in args.criteria.split(",") if c]
    if args.dry_run_cpu:
        args.clips, args.batch, args.ddim_steps = 2, 2, 4
        criteria = criteria[:2] if len(criteria) > 1 else criteria

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    config = load_config()

    # saliency artifact (skip if only 'base' requested)
    need_sal = any(c != "base" for c in criteria)
    if need_sal:
        if not os.path.exists(args.saliency):
            raise SystemExit(f"PREFLIGHT FAIL: saliency artifact missing: {args.saliency}")
        sal_pt = torch.load(args.saliency, map_location="cpu")
        sal = sal_pt["saliency"]
        l1_ranking = load_l1_ranking(RANKING_PKL)
        base_unet_sd = base_unet_state(BASE_CKPT)

    _load_ctx = _cpu_torch_load() if args.dry_run_cpu else _nullctx()
    with _load_ctx:
        t0 = time.perf_counter()
        model, load_info = build_model(config, device)  # base LatentDiffusion
        base_diffusion_model = model.model.diffusion_model  # keep to restore for 'base'
        result["model_build_s"] = time.perf_counter() - t0
        result["load_info"] = load_info

        points = []
        for crit in criteria:
            if crit == "base":
                model.model.diffusion_model = base_diffusion_model
            else:
                if crit not in sal:
                    raise SystemExit(f"criterion {crit} not in saliency artifact keys {list(sal)}")
                ranking = ranking_from_saliency(sal[crit], l1_ranking)
                pruned = materialize(base_unet_sd, ranking, config).to(device)
                model.model.diffusion_model = pruned

            outdir = os.path.join(SCRATCH_LOG, crit)
            os.makedirs(outdir, exist_ok=True)
            model.set_log_dir(SCRATCH_LOG, crit, "gen")

            loader = build_loader(config, args.clips, args.batch)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            with torch.no_grad():
                model.generate_sample(
                    loader, ddim_steps=args.ddim_steps, n_gen=args.n_gen,
                    unconditional_guidance_scale=args.guidance, limit_num=args.clips,
                )
            wall = time.perf_counter() - t1
            peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            pt = {"criterion": crit, "clips": args.clips, "ddim_steps": args.ddim_steps,
                  "wall_s": wall, "tgen_per_clip_s": wall / args.clips, "peak_vram_gb": peak,
                  "outdir": outdir}
            points.append(pt)
            print(f"  {crit:14s} clips={args.clips} S={args.ddim_steps} "
                  f"Tgen={pt['tgen_per_clip_s']:.3f}s/clip peak={peak:.2f}GB -> {outdir}")

    result["points"] = points
    result["measured"] = not args.dry_run_cpu
    out = json.dumps(result, indent=2)
    print(out)
    if args.dry_run_cpu:
        print("\nDRY RUN — flow validated on a tiny subset. NO --out written.")
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"\nRESULT written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
