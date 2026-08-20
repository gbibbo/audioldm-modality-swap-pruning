#!/usr/bin/env python3
"""Measure GEN_SEC_PER_CLIP (Tgen) end-to-end on the REAL generation stack.

`docs/compute_budget.md` currently carries `Tgen` as a DERIVED number
(`DDIM_steps × Tfwd/batch × 1.15`) and flags it as the weakest link in the whole
M4/M5 cost table. This script replaces it with a MEASURED value by running the
upstream generation path verbatim — `LatentDiffusion.generate_sample` (DDIM
sampling with classifier-free guidance + VAE decode + HiFi-GAN vocoder) — on the
real base `(1,2,3,5)` model with the real published weights.

It mirrors the cost discipline of `scripts/research/gpu_benchmark.py`:
  * refuses to write a measurement without CUDA (no invented numbers);
  * `--dry-run-cpu` validates the ENTIRE flow on the free CPU Studio and refuses
    `--out`, so a typo never costs a GPU job;
  * fail-fast preflight: expected git commit, clean tree, checkpoint present,
    CUDA + `--expect-gpu`;
  * records the exact git commit and prints the JSON to stdout as well as `--out`.

Timing model: one warm-up generation call (discarded) absorbs CUDA/cuDNN init,
then a timed call over `--clips` clips at the requested DDIM step count. Tgen is
wall / clips. Guidance stays at the config's 3.5 (two U-Net forwards per step,
the real M4 cost); `--n-gen 1` measures the single-candidate cost (the config's
`n_candidates_per_samples=3` is a pure multiplier, applied in the budget, not here).

Data is the disjoint validation split (`configs/research/val_split_disjoint.json`),
never the test set. Tgen is architecture timing and does not depend on which clips
are used, but using the disjoint split keeps the run clean of any test-set optics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import contextlib

import torch
import yaml


def _nullctx():
    return contextlib.nullcontext()

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
VAL_MANIFEST = "configs/research/val_split_disjoint.json"
SCRATCH_LOG = "artifacts/m3_pilot/tgen_gen"  # gitignored; waveforms land here


# --------------------------------------------------------------------------- utils
def git_provenance() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class _cpu_torch_load:
    """CPU-dry-run-only shims so the WHOLE generation flow validates on the free CPU
    Studio. Two upstream spots hardcode CUDA and would only be exercised on the GPU
    job otherwise:

      1. `get_vocoder` (`model_util.py:282`) does `torch.load(path)` with no
         map_location, and the vocoder checkpoint carries CUDA storage tags ->
         raises on a CPU-only machine. Force `map_location='cpu'`.
      2. `DDIMSampler.__init__` defaults `device=torch.device('cuda')` and
         `sample_log` instantiates it without a device (`ddpm.py:1800`), so the
         sampler's buffers go to CUDA regardless of the model's device. Force the
         sampler device to CPU.

    Both are active ONLY under --dry-run-cpu; upstream code is never modified. On the
    real GPU job CUDA is available and the native paths run. Mirrors the repo's
    torch_load() fallback pattern (HANDOFF)."""

    def __enter__(self):
        self._orig_load = torch.load

        def patched_load(*a, **k):
            k.setdefault("map_location", "cpu")
            return self._orig_load(*a, **k)

        torch.load = patched_load

        import audioldm_train.modules.latent_diffusion.ddim as _ddim
        self._ddim = _ddim
        self._orig_ddim_init = _ddim.DDIMSampler.__init__

        def patched_ddim_init(inner_self, model, schedule="linear", device=None, **kw):
            self._orig_ddim_init(inner_self, model, schedule=schedule,
                                 device=torch.device("cpu"), **kw)

        _ddim.DDIMSampler.__init__ = patched_ddim_init
        return self

    def __exit__(self, *exc):
        torch.load = self._orig_load
        self._ddim.DDIMSampler.__init__ = self._orig_ddim_init
        return False


def load_config() -> dict:
    return yaml.load(open(CONFIG), Loader=yaml.FullLoader)


def build_dataset_json(n: int) -> dict:
    manifest = json.load(open(VAL_MANIFEST))
    items = manifest["items"][:n]
    return {"data": [{"wav": it["wav"], "caption": it["caption"]} for it in items]}


def build_model(config: dict, device: torch.device):
    from audioldm_train.utilities.model_util import instantiate_from_config

    model = instantiate_from_config(config["model"])
    ckpt = torch.load(BASE_CKPT, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    # The checkpoint stores the conditioner as `cond_stage_model.*` (singular) but the
    # instantiated LatentDiffusion holds it in a ModuleList as `cond_stage_models.0.*`.
    # Remap so the real (frozen) CLAP weights load from the checkpoint rather than
    # falling back to CLAP's own pretrained init (identical weights, but this makes the
    # run unambiguous). The U-Net `model.diffusion_model.*` already loads 0-missing.
    remapped = {}
    for k, v in sd.items():
        if k.startswith("cond_stage_model.") and not k.startswith("cond_stage_models."):
            remapped["cond_stage_models.0." + k[len("cond_stage_model."):]] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    unet_missing = [k for k in missing if k.startswith("model.diffusion_model")]
    if unet_missing:
        raise SystemExit(f"BUILD FAIL: {len(unet_missing)} U-Net weights did not load from base ckpt")
    model.eval()
    model = model.to(device)
    model.set_log_dir(SCRATCH_LOG, "tgen", "measure")
    return model, {"missing": len(missing), "unexpected": len(unexpected)}


def build_loader(config: dict, n: int, batch: int):
    from torch.utils.data import DataLoader
    from audioldm_train.utilities.data.dataset import AudioDataset

    ds = AudioDataset(
        config=config, split="test", waveform_only=False,
        dataset_json=build_dataset_json(n),
    )
    return DataLoader(ds, batch_size=batch, shuffle=False)


def time_generation(model, loader, device, steps, n_gen, guidance, clips):
    _reset_peak()
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate_sample(
            loader,
            ddim_steps=steps,
            n_gen=n_gen,
            unconditional_guidance_scale=guidance,
            limit_num=clips,
        )
    _sync(device)
    wall = time.perf_counter() - t0
    return {
        "ddim_steps": steps,
        "n_gen": n_gen,
        "guidance": guidance,
        "clips": clips,
        "wall_s": wall,
        "tgen_per_clip_s": wall / clips,
        "peak_vram_gb": _peak_gb(),
    }


# --------------------------------------------------------------------------- preflight
def preflight(args, result):
    prov = git_provenance()
    result["git"] = prov
    if args.expect_commit and prov["commit"] != args.expect_commit:
        raise SystemExit(f"PREFLIGHT FAIL: commit {prov['commit']} != expected {args.expect_commit}")
    if prov["dirty"] and not args.allow_dirty and not args.dry_run_cpu:
        raise SystemExit("PREFLIGHT FAIL: working tree is dirty (use --allow-dirty only for CPU dev)")
    if not os.path.exists(BASE_CKPT):
        raise SystemExit(f"PREFLIGHT FAIL: base checkpoint missing: {BASE_CKPT}")
    if not os.path.exists(VAL_MANIFEST):
        raise SystemExit(f"PREFLIGHT FAIL: val manifest missing: {VAL_MANIFEST}")
    if not args.dry_run_cpu:
        if not torch.cuda.is_available():
            raise SystemExit("PREFLIGHT FAIL: CUDA not available and not --dry-run-cpu (refusing to invent Tgen)")
        name = torch.cuda.get_device_name(0)
        result["gpu_name"] = name
        if args.expect_gpu and args.expect_gpu.lower() not in name.lower():
            raise SystemExit(f"PREFLIGHT FAIL: GPU {name} != expected {args.expect_gpu}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-list", type=lambda s: [int(x) for x in s.split(",")], default=[50, 200],
                    help="DDIM step counts to measure (default 50,200)")
    ap.add_argument("--clips", type=int, default=8, help="clips per measured point")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--warmup-steps", type=int, default=8, help="DDIM steps for the discarded warm-up call")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true",
                    help="validate the whole flow on CPU; produces NO measurement and refuses --out")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu refuses --out (a dry run is never a measurement)")

    result = {"script": "measure_tgen.py", "dry_run_cpu": args.dry_run_cpu}
    preflight(args, result)

    os.makedirs(SCRATCH_LOG, exist_ok=True)
    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    config = load_config()

    if args.dry_run_cpu:
        args.steps_list = [4]
        args.clips = 2
        args.batch = 2
        args.warmup_steps = 2

    _load_ctx = _cpu_torch_load() if args.dry_run_cpu else _nullctx()
    with _load_ctx:
        t_build = time.perf_counter()
        model, load_info = build_model(config, device)
        result["load_info"] = load_info
        result["model_build_s"] = time.perf_counter() - t_build

        # warm-up (discarded): absorbs CUDA/cuDNN init and any first-call graph build
        warm_loader = build_loader(config, args.batch, args.batch)
        time_generation(model, warm_loader, device, args.warmup_steps, args.n_gen, args.guidance, args.batch)

        points = []
        for steps in args.steps_list:
            loader = build_loader(config, args.clips, args.batch)
            pt = time_generation(model, loader, device, steps, args.n_gen, args.guidance, args.clips)
            points.append(pt)
            print(f"  S={steps:>3}  Tgen={pt['tgen_per_clip_s']:.4f} s/clip  peak={pt['peak_vram_gb']:.3f} GB  (wall {pt['wall_s']:.1f}s / {args.clips} clips)")

    result["points"] = points
    result["measured"] = not args.dry_run_cpu

    out = json.dumps(result, indent=2)
    print(out)
    if args.dry_run_cpu:
        print("\nDRY RUN — NO MEASUREMENT WRITTEN. Flow validated on CPU.")
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"\nMEASUREMENT written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
