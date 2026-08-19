#!/usr/bin/env python3
"""GPU benchmark for the compute gate (master plan §7.2) — WRITE-ONLY until a GPU exists.

Measures every §7.2 variable on the real pruned `(1,2,3,1)` U-Net so that
`docs/compute_budget.md` can be populated with MEASURED values and Compute Gate CG
resolved. It refuses to run without CUDA and never fabricates a number: if no GPU
is attached it prints why and exits non-zero, leaving compute_budget.md untouched.

Measured variables:
    GPU_MODEL, VRAM_GB
    TRAIN_SEC_PER_STEP, PEAK_TRAIN_VRAM_GB          (PEFT recovery step: fwd+bwd+opt)
    SALIENCY_SEC_PER_GRAD_EVAL_OR_BATCH, PEAK_SALIENCY_VRAM_GB
                                                    (Taylor: fwd+bwd, weight grads)
    FORWARD_SEC_PER_DIAGNOSTIC_BATCH, PEAK_FORWARD_VRAM_GB   (D_gen/D_mod fwd only)
    GEN_SEC_PER_CLIP_OR_BATCH, GEN_BATCH_SIZE, PEAK_GENERATION_VRAM_GB
                                                    (only with --with-generation +
                                                     the full LatentDiffusion stack)

The train/saliency/forward paths reuse the tested M1 PEFT setup and the frozen
U-Net config; they build on CPU too (for a structural dry run) but timings are
meaningless without CUDA, hence the hard CUDA guard.

    # only on a GPU studio:
    .venv/bin/python scripts/research/gpu_benchmark.py --steps 30 --batch 8 --out docs/compute_budget_measured.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time

import torch
import yaml
from torch import nn

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"


def build_pruned_unet():
    from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
    with open(CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    params = copy.deepcopy(cfg["model"]["params"]["unet_config"]["params"])
    params["channel_mult"] = [1, 2, 3, 1]
    return UNetModel(**params)


class Holder(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = unet


def _sync():
    torch.cuda.synchronize()


def _peak_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _fake_batch(batch, device):
    z_t = torch.randn(batch, 8, 256, 16, device=device)
    t = torch.randint(0, 1000, (batch,), device=device)
    y = torch.randn(batch, 512, device=device)
    eps = torch.randn_like(z_t)
    return z_t, t, y, eps


def _call(unet, z_t, t, y):
    return unet(z_t, t, y=y, context_list=[], context_attn_mask_list=[])


def time_train_step(device, batch, steps, warmup):
    """PEFT recovery step: forward + backward + optimizer step."""
    from audioldm_peft import setup_peft, build_peft_optimizer, PeftConfig
    m = Holder(build_pruned_unet()).to(device)
    setup_peft(m, PeftConfig(root_path="model.diffusion_model", rank=8, alpha=16))
    opt, _ = build_peft_optimizer(m, lora_lr=1e-4, auxiliary_lr=1e-4)
    unet = m.model.diffusion_model
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(warmup + steps):
        z_t, t, y, eps = _fake_batch(batch, device)
        opt.zero_grad(set_to_none=True)
        _sync(); t0 = time.perf_counter()
        pred = _call(unet, z_t, t, y)
        loss = torch.nn.functional.mse_loss(pred, eps)
        loss.backward()
        opt.step()
        _sync(); dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return sum(times) / len(times), _peak_gb()


def time_saliency(device, batch, iters, warmup):
    """Taylor saliency: forward + backward populating weight grads (no opt step)."""
    m = build_pruned_unet().to(device)
    for p in m.parameters():
        p.requires_grad = True
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(warmup + iters):
        z_t, t, y, eps = _fake_batch(batch, device)
        m.zero_grad(set_to_none=True)
        _sync(); t0 = time.perf_counter()
        pred = _call(m, z_t, t, y)
        loss = torch.nn.functional.mse_loss(pred, eps)
        loss.backward()
        _sync(); dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return sum(times) / len(times), _peak_gb()


def time_forward(device, batch, iters, warmup):
    """Diagnostic forward only (D_gen/D_mod path), no grad."""
    m = build_pruned_unet().to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for i in range(warmup + iters):
            z_t, t, y, eps = _fake_batch(batch, device)
            _sync(); t0 = time.perf_counter()
            _call(m, z_t, t, y)
            _sync(); dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
    return sum(times) / len(times), _peak_gb()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--with-generation", action="store_true",
                    help="also time the full generation stack (requires LatentDiffusion + sampler)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("REFUSING TO BENCHMARK: no CUDA device is available.", file=sys.stderr)
        print("This script never fabricates GPU numbers. Attach a GPU and re-run; "
              "docs/compute_budget.md must stay TBD_MEASURED until then.", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    result = {
        "GPU_MODEL": torch.cuda.get_device_name(0),
        "VRAM_GB": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3),
        "batch": args.batch, "steps": args.steps, "iters": args.iters, "warmup": args.warmup,
    }

    t_train, peak_train = time_train_step(device, args.batch, args.steps, args.warmup)
    result["TRAIN_SEC_PER_STEP"] = round(t_train, 6)
    result["PEAK_TRAIN_VRAM_GB"] = round(peak_train, 3)

    t_sal, peak_sal = time_saliency(device, args.batch, args.iters, args.warmup)
    result["SALIENCY_SEC_PER_GRAD_EVAL_OR_BATCH"] = round(t_sal, 6)
    result["PEAK_SALIENCY_VRAM_GB"] = round(peak_sal, 3)

    t_fwd, peak_fwd = time_forward(device, args.batch, args.iters, args.warmup)
    result["FORWARD_SEC_PER_DIAGNOSTIC_BATCH"] = round(t_fwd, 6)
    result["PEAK_FORWARD_VRAM_GB"] = round(peak_fwd, 3)

    if args.with_generation:
        # The generation stack (LatentDiffusion + DDIM/PLMS sampler + VAE decode +
        # vocoder) is not wired here; a GPU session must add it and record
        # GEN_SEC_PER_CLIP_OR_BATCH / GEN_BATCH_SIZE / PEAK_GENERATION_VRAM_GB.
        result["GEN_SEC_PER_CLIP_OR_BATCH"] = None
        result["GEN_BATCH_SIZE"] = None
        result["PEAK_GENERATION_VRAM_GB"] = None
        result["_generation_note"] = "not measured: wire the full LatentDiffusion sampler and re-run"

    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
