#!/usr/bin/env python3
"""M5 convergence PROBE — PEFT recovery loss-vs-steps on the pruned model (real data).

Purpose: the single most impactful budget lever (compute_budget.md lever #1) is the
recovery step count. The master plan's 100k steps is not an optimised number; if
parameter-efficient recovery plateaus far sooner, M5 collapses from 46.46 to ~9 GPU-h
per model. This probe measures the early loss-vs-steps curve so that decision rests on
data, not on the plan's placeholder. Authorized by DECISION-CG-001 (~2k steps ≈ 0.8 cr).

It is a PROBE, not the recovery run: it trains the published L1-pruned `(1,2,3,1)` model
with PEFT for a few thousand steps on a STREAM of real AudioCaps TRAIN clips (built-in
`split="train"`, so paths resolve and no empty-waveform substitution occurs — unlike a
custom dataset_json), with the real diffusion objective under AUDIO CLAP conditioning
(the modality AudioLDM trains on). Loss is logged smoothed; the curve is saved.

Faithfulness: batches STREAM from the full 49 502-item train split (shuffled), so ~2k
steps × batch 8 ≈ 16k distinct clips (~0.3 epoch, no small-pool overfitting that would
make convergence look faster than the real recovery). Train is disjoint from val/test.

Cost discipline mirrors gpu_benchmark.py: refuses without CUDA unless --dry-run-cpu
(tiny steps, refuses --out). Reuses the M1 PEFT setup (`m1_gpu_acceptance.build`) whose
resume/state machinery is already regression-tested (F11/S4).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_gpu_acceptance as m1  # reuse build() (pruned + PEFT + opt + EMA)
gb = m1.gb

from research_pruning.diagnostics import (  # noqa: E402
    load_config, build_clap, build_vae, vae_encode, read_scale_factor, NoiseSchedule,
)
from research_pruning.diagnostics.conditioning import clap_embed  # noqa: E402

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"


def git_provenance():
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], text=True).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def build_train_loader(config, batch, num_workers):
    from torch.utils.data import DataLoader
    from audioldm_train.utilities.data.dataset import AudioDataset
    ds = AudioDataset(config=config, split="train", waveform_only=False)  # built-in path resolution
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=num_workers,
                      drop_last=True, persistent_workers=(num_workers > 0))


@torch.no_grad()
def batch_to_inputs(batch, clap, vae, scale_factor, schedule, device, gen):
    """Real (z_t, t, e_audio, noise) from a dataloader batch. AUDIO conditioning."""
    wav = batch["waveform"].float().to(device)          # [B,1,T]
    mel = batch["log_mel_spec"].unsqueeze(1).float().to(device)  # [B,1,1024,64]
    z0 = vae_encode(vae, mel, scale_factor)             # [B,C,H,W]
    e_audio = clap_embed(clap, wav, "audio").squeeze(1)  # [B,512]
    b = z0.shape[0]
    # CPU generator for reproducibility, then move to the model device. The schedule
    # buffers are moved to `device` once in main(), so q_sample stays on-device (a CPU
    # dry-run cannot catch a device mismatch here — both sides are CPU there).
    t = torch.randint(0, schedule.timesteps, (b,), generator=gen).to(device)
    noise = torch.randn(z0.shape, generator=gen).to(device)
    z_t = schedule.q_sample(z0, t, noise)
    return z_t, t, e_audio, noise


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
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu refuses --out")

    result = {"script": "m5_recovery_probe.py", "dry_run_cpu": args.dry_run_cpu}
    preflight(args, result)
    gb.DRY_RUN = args.dry_run_cpu  # makes gb._sync/_peak_gb no-op on CPU (as in m1_gpu_acceptance)

    if args.dry_run_cpu:
        args.steps, args.batch, args.log_every, args.num_workers = 4, 2, 1, 0

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    config = load_config()

    t0 = time.perf_counter()
    holder, unet, opt, ema = m1.build(device)  # pruned L1 model + PEFT (real weights, guarded)
    clap = build_clap(config, unconditional_prob=0.0).to(device)
    vae = build_vae(config, BASE_CKPT).to(device)
    scale_factor = read_scale_factor(BASE_CKPT)
    schedule = NoiseSchedule(config)
    # Move the schedule buffers to the model device so q_sample stays on-device.
    schedule.sqrt_alphas_cumprod = schedule.sqrt_alphas_cumprod.to(device)
    schedule.sqrt_one_minus_alphas_cumprod = schedule.sqrt_one_minus_alphas_cumprod.to(device)
    result["build_s"] = time.perf_counter() - t0

    loader = build_train_loader(config, args.batch, args.num_workers)
    gen = torch.Generator().manual_seed(args.seed)

    losses, steps_log, timings = [], [], []
    step = 0
    t_train = time.perf_counter()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            z_t, t, e_audio, noise = batch_to_inputs(batch, clap, vae, scale_factor, schedule, device, gen)
            opt.zero_grad(set_to_none=True)
            gb._sync(); ts = time.perf_counter()
            pred = gb._call(unet, z_t, t, e_audio)
            loss = torch.nn.functional.mse_loss(pred, noise)
            loss.backward()
            opt.step()
            ema.update(holder)
            gb._sync(); dt = time.perf_counter() - ts
            v = loss.item()
            if not math.isfinite(v):
                raise RuntimeError(f"loss became {v} at step {step}")
            timings.append(dt)
            if step % args.log_every == 0:
                losses.append(v); steps_log.append(step)
                print(f"  step {step:5d}  loss {v:.5f}  {dt:.3f}s/step")
            step += 1
    result["train_s"] = time.perf_counter() - t_train
    result["steps"] = step
    result["batch"] = args.batch
    result["loss_curve"] = {"step": steps_log, "loss": losses}
    result["mean_sec_per_step"] = sum(timings[5:]) / max(1, len(timings[5:]))
    if torch.cuda.is_available():
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    result["first_loss"] = losses[0] if losses else None
    result["last_loss"] = losses[-1] if losses else None
    result["measured"] = not args.dry_run_cpu

    out = json.dumps(result, indent=2)
    print(out)
    if args.dry_run_cpu:
        print("\nDRY RUN — flow validated. NO --out written.")
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"\nRESULT written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
