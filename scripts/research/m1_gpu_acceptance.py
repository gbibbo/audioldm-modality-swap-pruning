#!/usr/bin/env python3
"""M1 GPU acceptance: several hundred real PEFT steps + an exact resume test, as a JOB.

The compute benchmark (`gpu_benchmark.py`) sizes the budget; it is explicitly NOT M1
acceptance. This script is. It runs a sustained parameter-efficient recovery workload on
the real pruned `(1,2,3,1)` U-Net with the real published weights and answers the three
questions the master plan asks of M1 on hardware:

    sustained   does a several-hundred-step PEFT run stay healthy — no OOM, no NaN, no
                drift in sec/step — and does the loss actually move?
    memory      what is the steady-state and peak VRAM at the chosen batch?
    resume      is training state EXACTLY resumable (F7)? Run A trains 0..N snapshotting
                at K. Run B rebuilds the model from scratch, loads the snapshot, and
                replays K..N over the identical batch sequence. The two final parameter
                sets must agree. This is the only test that proves
                `training_state_dict`/`load_training_state_dict` are complete: if the
                optimizer moments, the EMA shadows or the step counter were missing, the
                two runs would diverge.

Deterministic by construction: every batch is drawn from a per-step CPU generator seeded
`--seed + step`, so run B sees byte-identical inputs to run A's tail. Batches are
synthetic — M1 acceptance is about training MECHANICS (memory, throughput, resumability),
not about learning quality, and synthetic input keeps the comparison exact.

Real weights are mandatory for the same reason as in the benchmark: on a fresh-init model
`out.2` is `zero_module`-ed, so gradient reaches only 1 of 284 adapters and the run would
exercise almost nothing (finding R7c, ledger M1-009). `assert_real_weights` enforces it.

    lightning job run --name m1-acceptance --machine T4 \\
        --studio gabriel-allgd-deploy-model-devbox \\
        --teamspace general --org independentaudioresearch \\
        --command "cd audioldm-modality-swap-pruning && .venv/bin/python \\
                   scripts/research/m1_gpu_acceptance.py --steps 400 --snapshot-at 200 \\
                   --batch <MAX_STABLE_BATCH from the benchmark> --expect-gpu T4 \\
                   --expect-commit <SHA> --out artifacts/m1/gpu_acceptance.json"

`--dry-run-cpu` exercises the whole flow on the free CPU Studio (use a tiny --steps); it
forces DRY_RUN into the JSON and refuses --out, so it can never be mistaken for a result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import sys
import time

import torch

# `scripts/research` is not a package, so reuse the benchmark's verified model/guard
# helpers by explicit path import rather than duplicating them.
_GB = pathlib.Path(__file__).with_name("gpu_benchmark.py")
_spec = importlib.util.spec_from_file_location("gpu_benchmark", _GB)
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)

N_ADAPTERS = 284


def batch_for_step(step: int, batch: int, device, seed: int):
    """Byte-identical batches across runs: a fresh CPU generator per step, then .to()."""
    g = torch.Generator().manual_seed(seed + step)
    z_t = torch.randn(batch, 8, 256, 16, generator=g)
    t = torch.randint(0, 1000, (batch,), generator=g)
    y = torch.randn(batch, 512, generator=g)
    eps = torch.randn(batch, 8, 256, 16, generator=g)
    return z_t.to(device), t.to(device), y.to(device), eps.to(device)


def build(device):
    """Real weights -> guard -> move -> inject PEFT -> optimizer -> trainable-only EMA."""
    from audioldm_peft import (PeftConfig, setup_peft, build_peft_optimizer,
                               build_trainable_only_ema)
    torch.manual_seed(1234)                    # identical adapter init in both runs
    unet = gb.build_pruned_unet(real_weights=True)
    gb.assert_real_weights(unet)
    holder = gb.Holder(unet).to(device)
    setup_peft(holder, PeftConfig(root_path="model.diffusion_model", rank=8, alpha=16))
    opt, _ = build_peft_optimizer(holder, lora_lr=1e-4, auxiliary_lr=1e-4)
    ema = build_trainable_only_ema(holder, decay=0.999)
    return holder, holder.model.diffusion_model, opt, ema


def trainable_snapshot(unet):
    return {n: p.detach().float().cpu().clone()
            for n, p in unet.named_parameters() if p.requires_grad}


def run_steps(holder, opt, ema, device, start, end, batch, seed, timings=None):
    """Train [start, end). Returns the loss trace. Raises on NaN.

    Takes the HOLDER, not the inner U-Net: `build_trainable_only_ema` registered its
    shadow buffers against the holder's parameter names (`model.diffusion_model.*`), so
    `ema.update` must be handed the same module or every lookup misses.
    """
    unet = holder.model.diffusion_model
    losses = []
    for step in range(start, end):
        z_t, t, y, eps = batch_for_step(step, batch, device, seed)
        opt.zero_grad(set_to_none=True)
        gb._sync(); t0 = time.perf_counter()
        pred = gb._call(unet, z_t, t, y)
        loss = torch.nn.functional.mse_loss(pred, eps)
        loss.backward()
        opt.step()
        ema.update(holder)
        gb._sync(); dt = time.perf_counter() - t0
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError(f"loss became {value} at step {step}: the run is unhealthy")
        losses.append(value)
        if timings is not None:
            timings.append(dt)
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="total steps in run A")
    ap.add_argument("--snapshot-at", type=int, default=None,
                    help="step at which to snapshot training state (default: steps//2)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5, help="steps excluded from the timing")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--resume-tol", type=float, default=1e-6)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gb.DRY_RUN = args.dry_run_cpu
    if gb.DRY_RUN and args.out:
        raise SystemExit("--dry-run-cpu produces no result and refuses --out.")
    snap_at = args.snapshot_at if args.snapshot_at is not None else args.steps // 2
    if not 0 < snap_at < args.steps:
        raise SystemExit(f"--snapshot-at must be in (0, {args.steps})")

    result = {"schema": "m1_gpu_acceptance/1", "argv": sys.argv[1:],
              "DRY_RUN": gb.DRY_RUN, "torch": torch.__version__,
              "steps": args.steps, "snapshot_at": snap_at, "batch": args.batch,
              "seed": args.seed}

    # Same fail-fast preflight contract as the benchmark (commit, tree, ckpt, CUDA, R7a).
    gb.preflight(argparse.Namespace(expect_commit=args.expect_commit,
                                    expect_gpu=args.expect_gpu,
                                    allow_dirty=args.allow_dirty, r7="skip"), result)
    device = torch.device("cpu" if gb.DRY_RUN else "cuda")
    if not gb.DRY_RUN:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        result["cudnn_deterministic"] = True

    # ---------------------------------------------------------------- run A
    print(f"=== RUN A: 0..{args.steps}, snapshot at {snap_at} ===", flush=True)
    holder, unet, opt, ema = build(device)
    gb._reset_peak()
    timings: list = []
    losses = run_steps(holder, opt, ema, device, 0, snap_at, args.batch, args.seed, timings)

    from audioldm_peft.state import training_state_dict, load_training_state_dict
    snapshot = training_state_dict(holder, optimizer=opt, ema=ema, global_step=snap_at,
                                   extra={"seed": args.seed, "batch": args.batch})
    n_adapter_tensors = len(snapshot["adapter"])
    print(f"  snapshot at {snap_at}: {n_adapter_tensors} trainable tensors, "
          f"optimizer={snapshot['optimizer'] is not None}, ema={snapshot['ema'] is not None}")

    losses += run_steps(holder, opt, ema, device, snap_at, args.steps, args.batch,
                        args.seed, timings)
    final_a = trainable_snapshot(unet)
    peak_a = gb._peak_gb()
    steady = timings[args.warmup:] or timings
    sec_per_step = sum(steady) / len(steady)
    print(f"  loss {losses[0]:.6f} -> {losses[-1]:.6f}  "
          f"(min {min(losses):.6f}, all finite)")
    print(f"  sec/step {sec_per_step:.4f} (steady, excluding {args.warmup} warmup)  "
          f"peak {peak_a:.3f} GB", flush=True)

    result.update({
        "TRAIN_SEC_PER_STEP": round(sec_per_step, 6),
        "PEAK_TRAIN_VRAM_GB": None if gb.DRY_RUN else round(peak_a, 3),
        "loss_first": losses[0], "loss_last": losses[-1], "loss_min": min(losses),
        "loss_moved": losses[-1] != losses[0],
        "all_losses_finite": True,
        "trainable_tensors": n_adapter_tensors,
        "sec_per_step_first_decile": round(sum(steady[:max(1, len(steady)//10)])
                                           / max(1, len(steady)//10), 6),
        "sec_per_step_last_decile": round(sum(steady[-max(1, len(steady)//10):])
                                          / max(1, len(steady)//10), 6),
    })

    # ---------------------------------------------------------------- run B
    print(f"=== RUN B: rebuild, load snapshot, replay {snap_at}..{args.steps} ===",
          flush=True)
    del holder, unet, opt, ema
    gb._empty_cache()
    holder_b, unet_b, opt_b, ema_b = build(device)
    restored = load_training_state_dict(holder_b, snapshot, optimizer=opt_b, ema=ema_b,
                                        strict=True)
    print(f"  restored global_step = {restored}")
    run_steps(holder_b, opt_b, ema_b, device, snap_at, args.steps, args.batch, args.seed)
    final_b = trainable_snapshot(unet_b)

    keys = sorted(set(final_a) & set(final_b))
    missing = sorted(set(final_a) ^ set(final_b))
    max_delta = max((final_a[k] - final_b[k]).abs().max().item() for k in keys)
    moved = max(final_a[k].abs().max().item() for k in keys)
    ok = (restored == snap_at and not missing and len(keys) == len(final_a)
          and max_delta <= args.resume_tol)
    print(f"  compared {len(keys)} trainable tensors, {len(missing)} key mismatches")
    print(f"  RESUME max|Delta| = {max_delta:.3e}  (tolerance {args.resume_tol:.1e}) "
          f"-> {'EXACT' if max_delta == 0 else 'within tolerance' if ok else 'DIVERGED'}")

    result.update({"RESUME_MAX_DELTA": max_delta, "RESUME_TOL": args.resume_tol,
                   "RESUME_OK": bool(ok), "restored_global_step": restored,
                   "resume_key_mismatches": len(missing),
                   "max_abs_param": moved})

    blob = json.dumps(result, indent=2, sort_keys=True)
    print("=== RESULT JSON ===")
    print(blob, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(blob + "\n")
        print(f"\nwrote {args.out}")

    if not ok:
        print("\nM1 GPU ACCEPTANCE: FAIL — training state is not exactly resumable.",
              file=sys.stderr)
        return 1
    print(f"\nM1 GPU ACCEPTANCE: {'DRY RUN (no claim)' if gb.DRY_RUN else 'PASS'} — "
          f"{args.steps} steps, loss finite throughout, resume exact to {max_delta:.1e}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
