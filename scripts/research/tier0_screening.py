#!/usr/bin/env python3
"""Tier-0 event-level screening generation (plan v4 §7, third Tier-0 GPU job).

Generates audio at DDIM S=50 for {base, P0-std, P1-nat} conditioned on the Q2
heterogeneity-screen prompts (`configs/research/prompts_heterogeneity_screen.json`,
200 stratified prompts). The generated clips are then scored (CPU, `tier0_screen_eval.py`)
for PANNs top-10 recall per requested event — the v4 primary event metric — plus the
fixed real-part FAD/FD (Q1).

Systems (FINDING-P0-COLLAPSE fix): P0-std = the L1 pkl REVERSED per layer (keep-HIGHEST-L1,
the primary baseline, DECISION-V4-01); P1-nat = the P1 text-Taylor saliency (M3B artifact);
base = the unpruned model. P0-std is NOT taken from the collapsed saliency `P0_L1` entry.

Reuses the validated generation stack from `measure_tgen`/`m4_screening`. Generated wavs
are named by the source-prompt wav id, so the eval matches each clip to its requested
event(s). No FAD gates this (F-eval-3 fixed in Q1). Refuses without CUDA unless
--dry-run-cpu. ~1.4 credits at 3 systems × 200 clips × S=50 (compute_budget.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_tgen import _cpu_torch_load, _nullctx, load_config, build_model, git_provenance  # noqa: E402
from research_pruning.diagnostics.random_masks import load_l1_ranking, materialize  # noqa: E402

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
RANKING_PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
SALIENCY = "artifacts/m3_pilot/m3b_saliency.pt"
SCREEN_PROMPTS = "configs/research/prompts_heterogeneity_screen.json"
SCRATCH_LOG = "artifacts/tier0_screening"
SYSTEMS = ["base", "P0-std", "P1-nat"]


def ranking_from_saliency(sal_layer_dict, l1_ranking):
    return {k: np.argsort(-sal_layer_dict[k].detach().cpu().numpy()).astype(np.int64).tolist()
            for k in l1_ranking}


def base_unet_state(base_ckpt):
    sd = torch.load(base_ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    return {k[len("model.diffusion_model."):]: v
            for k, v in sd.items() if k.startswith("model.diffusion_model.")}


def build_loader_from_prompts(config, prompts, batch):
    """Loader over the screen prompts' captions (same pattern as measure_tgen.build_loader).
    Generated files are named by the source wav id."""
    from torch.utils.data import DataLoader
    from audioldm_train.utilities.data.dataset import AudioDataset
    data = [{"wav": p["wav"], "caption": p["caption"]} for p in prompts]
    ds = AudioDataset(config=config, split="test", waveform_only=False, dataset_json={"data": data})
    return DataLoader(ds, batch_size=batch, shuffle=False)


def preflight(args, result):
    prov = git_provenance()
    result["git"] = prov
    if args.expect_commit and prov["commit"] != args.expect_commit:
        raise SystemExit(f"PREFLIGHT FAIL: commit {prov['commit']} != {args.expect_commit}")
    if prov["dirty"] and not args.allow_dirty and not args.dry_run_cpu:
        raise SystemExit("PREFLIGHT FAIL: dirty tree")
    for p in (BASE_CKPT, RANKING_PKL, SALIENCY, SCREEN_PROMPTS):
        if not os.path.exists(p):
            raise SystemExit(f"PREFLIGHT FAIL: missing {p}")
    if not args.dry_run_cpu:
        if not torch.cuda.is_available():
            raise SystemExit("PREFLIGHT FAIL: no CUDA and not --dry-run-cpu")
        name = torch.cuda.get_device_name(0)
        result["gpu_name"] = name
        if args.expect_gpu and args.expect_gpu.lower() not in name.lower():
            raise SystemExit(f"PREFLIGHT FAIL: GPU {name} != expected {args.expect_gpu}")


def system_rankings(sal, l1_ranking):
    """P0-std = reversed pkl (keep-highest-L1); P1-nat = P1 saliency. (FINDING-P0-COLLAPSE.)"""
    return {
        "P0-std": {k: list(reversed(l1_ranking[k])) for k in l1_ranking},
        "P1-nat": ranking_from_saliency(sal["P1"], l1_ranking),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--saliency", default=SALIENCY)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu refuses --out")

    result = {"script": "tier0_screening.py", "dry_run_cpu": args.dry_run_cpu}
    preflight(args, result)

    systems = [s for s in args.systems.split(",") if s]
    if args.dry_run_cpu:
        args.clips, args.batch, args.ddim_steps = 2, 2, 4
        systems = systems[:2] if len(systems) > 1 else systems

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    config = load_config()

    prompts = json.load(open(SCREEN_PROMPTS))["prompts"][:args.clips]
    result["n_prompts"] = len(prompts)
    # wav_id -> requested events (for the eval)
    req_map = {os.path.splitext(os.path.basename(p["wav"]))[0]: p.get("requested_events", [])
               for p in prompts}

    need_pruned = any(s != "base" for s in systems)
    if need_pruned:
        sal = torch.load(args.saliency, map_location="cpu")["saliency"]
        l1_ranking = load_l1_ranking(RANKING_PKL)
        base_unet_sd = base_unet_state(BASE_CKPT)
        rankings = system_rankings(sal, l1_ranking)

    _load_ctx = _cpu_torch_load() if args.dry_run_cpu else _nullctx()
    with _load_ctx:
        t0 = time.perf_counter()
        model, load_info = build_model(config, device)
        base_diffusion_model = model.model.diffusion_model
        model.use_ema = False
        result["use_ema"] = False
        result["model_build_s"] = time.perf_counter() - t0
        result["load_info"] = load_info

        points = []
        for sysname in systems:
            if sysname == "base":
                model.model.diffusion_model = base_diffusion_model
            else:
                pruned = materialize(base_unet_sd, rankings[sysname], config).to(device)
                model.model.diffusion_model = pruned

            outdir = os.path.join(SCRATCH_LOG, sysname)
            os.makedirs(outdir, exist_ok=True)
            model.set_log_dir(SCRATCH_LOG, sysname, "gen")

            loader = build_loader_from_prompts(config, prompts, args.batch)
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
            pt = {"system": sysname, "clips": args.clips, "ddim_steps": args.ddim_steps,
                  "wall_s": wall, "tgen_per_clip_s": wall / max(args.clips, 1),
                  "peak_vram_gb": peak, "outdir": outdir}
            points.append(pt)
            print(f"  {sysname:8s} clips={args.clips} S={args.ddim_steps} "
                  f"Tgen={pt['tgen_per_clip_s']:.3f}s/clip peak={peak:.2f}GB -> {outdir}", flush=True)

    result["points"] = points
    result["requested_events_map"] = req_map
    result["measured"] = not args.dry_run_cpu

    out = json.dumps(result, indent=2)
    print(out)
    if args.dry_run_cpu:
        print("\nDRY RUN — NO RESULT WRITTEN. Generation flow validated on a tiny subset.")
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"\nRESULT written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
