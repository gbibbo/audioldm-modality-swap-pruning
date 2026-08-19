#!/usr/bin/env python3
"""GPU benchmark for the compute gate (master plan §7.2), run as a Lightning JOB.

Measures every §7.2 variable on the real pruned `(1,2,3,1)` U-Net **with the real
published weights loaded**, so `docs/compute_budget.md` can be populated with MEASURED
values and Compute Gate CG resolved. It refuses to run without CUDA and never
fabricates a number.

**Why real weights are mandatory here (finding R7c, ledger M1-009).** A freshly
initialised U-Net has `zero_module`-ed final conv `out.2` (`sum|W| = 0.0`), so no
gradient propagates backward past it: only 1 of 284 LoRA adapters is exercised and the
TRAIN and SALIENCY timings measure an almost-empty backward graph. With the published
weights that tensor is `sum|W| = 174.98` and gradient reaches all 284. An earlier
version of this script benchmarked a fresh-init model and its numbers would have been
invalid. `assert_real_weights()` now guards this before any measurement.

Measured variables:
    GPU_MODEL, VRAM_GB
    TRAIN_SEC_PER_STEP, PEAK_TRAIN_VRAM_GB          (PEFT recovery step: fwd+bwd+opt)
    SALIENCY_SEC_PER_GRAD_EVAL_OR_BATCH, PEAK_SALIENCY_VRAM_GB
                                                    (Taylor: fwd+bwd, weight grads)
    FORWARD_SEC_PER_DIAGNOSTIC_BATCH, PEAK_FORWARD_VRAM_GB   (D_gen/D_mod fwd only)

Staged by design — never start at a large batch on a 16 GB T4 and discover the limit
via OOM:

    preflight  fail fast on any missing prerequisite: checkpoint, expected commit,
               CUDA device, and the R7a PEFT-backward invariant (on CPU, real weights).
    smoke      batch=1, warmup=2, 5 steps — proves CUDA, a real backward through the
               checkpointed blocks, and gradient reaching all 284 adapters.
    escalate   batch 1,2,4,8,... each probed briefly; OOM is caught, not fatal, and the
               largest stable batch plus the whole VRAM curve are recorded. This is the
               useful information about headroom.
    measure    the long run at the largest stable batch, enough steps to stabilise
               sec/step.

Every stage records the exact git commit. The JSON is written to `--out` AND printed to
stdout, so the numbers survive in the job log even if the artifact is not collected.

    # as a Lightning job (the Studio stays on free CPU):
    lightning job run --name gpu-benchmark --machine T4 \\
        --studio gabriel-allgd-deploy-model-devbox \\
        --command ".venv/bin/python scripts/research/gpu_benchmark.py --stage all \\
                   --out artifacts/m3_pilot/compute_budget_measured.json"

This benchmark does NOT constitute M1 GPU acceptance: that additionally requires a
several-hundred-step run with a resume test, also as a job.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import torch
import yaml
from torch import nn

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
CKPT = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
PREFIX = "model.diffusion_model."

N_ADAPTERS = 284          # R6a / R7a
DEFAULT_BATCHES = (1, 2, 4, 8)


# --------------------------------------------------------------------------- model
def build_pruned_unet(real_weights: bool = True):
    from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
    with open(CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    params = copy.deepcopy(cfg["model"]["params"]["unet_config"]["params"])
    params["channel_mult"] = [1, 2, 3, 1]
    unet = UNetModel(**params)
    if real_weights:
        load_real_weights(unet)
    return unet


def load_real_weights(unet) -> None:
    """Strict-load the published pruned weights (same path as test_peft_backward_real_unet)."""
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"{CKPT} is missing. The benchmark must not run on a fresh-init model "
            "(see R7c). Fetch it with scripts/research/fetch_public_artifacts.sh."
        )
    obj = torch.load(CKPT, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    weights = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    if not weights:
        raise RuntimeError(f"no '{PREFIX}*' tensors found in {CKPT}")
    unet.load_state_dict(weights, strict=True)


def _final_conv(unet):
    """The final output conv, unwrapping a LoRA adapter if PEFT already wrapped it."""
    mod = unet.out[2]
    return mod.base if hasattr(mod, "base") else mod


def assert_real_weights(unet) -> float:
    """Guard R7c: refuse to benchmark a model whose backward graph is mostly dead."""
    w_sum = _final_conv(unet).weight.abs().sum().item()
    if not w_sum > 0:
        raise RuntimeError(
            "REFUSING TO BENCHMARK: the final output conv `out.2` has sum|W| == 0, i.e. "
            "this is a fresh-init (zero_module) model. Gradient cannot propagate past it, "
            "so TRAIN and SALIENCY would time an almost-empty backward graph and the "
            "numbers would be invalid (finding R7c, ledger M1-009). Load the real "
            f"published weights from {CKPT}."
        )
    return w_sum


class Holder(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = unet


# --------------------------------------------------------------------------- utils
# `--dry-run-cpu` exercises the ENTIRE staged flow on the CPU Studio for free, so a typo
# in the escalation/measurement wiring can never waste a GPU job. It stubs the CUDA-only
# calls and forces DRY_RUN into the JSON; a dry run must never be mistaken for a
# measurement, so it also refuses to write --out.
DRY_RUN = False


def _sync():
    if not DRY_RUN:
        torch.cuda.synchronize()


def _peak_gb():
    if DRY_RUN:
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reset_peak():
    if not DRY_RUN:
        torch.cuda.reset_peak_memory_stats()


def _empty_cache():
    if not DRY_RUN:
        torch.cuda.empty_cache()


def _fake_batch(batch, device):
    z_t = torch.randn(batch, 8, 256, 16, device=device)
    t = torch.randint(0, 1000, (batch,), device=device)
    y = torch.randn(batch, 512, device=device)
    eps = torch.randn_like(z_t)
    return z_t, t, y, eps


def _call(unet, z_t, t, y):
    return unet(z_t, t, y=y, context_list=[], context_attn_mask_list=[])


def git_provenance() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "upstream_patch_diffstat": run("git", "diff", "--shortstat",
                                       "upstream-frozen", "--", "audioldm_train/"),
    }


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# --------------------------------------------------------------------------- peft
def _peft_model(device):
    from audioldm_peft import setup_peft, build_peft_optimizer, PeftConfig
    unet = build_pruned_unet(real_weights=True)
    assert_real_weights(unet)
    holder = Holder(unet).to(device)                     # move FIRST: exercises the F9 fix
    setup_peft(holder, PeftConfig(root_path="model.diffusion_model", rank=8, alpha=16))
    opt, _ = build_peft_optimizer(holder, lora_lr=1e-4, auxiliary_lr=1e-4)
    return holder, holder.model.diffusion_model, opt


def _adapter_grad_census(unet) -> dict:
    b = [p for n, p in unet.named_parameters() if n.endswith("lora_B")]
    a = [p for n, p in unet.named_parameters() if n.endswith("lora_A")]
    return {
        "lora_B_total": len(b),
        "lora_B_nonzero_grad": sum(1 for p in b
                                   if p.grad is not None and p.grad.abs().sum().item() > 0),
        "lora_A_total": len(a),
        "frozen_with_grad": sum(1 for n, p in unet.named_parameters()
                                if "lora_" not in n and not p.requires_grad
                                and p.grad is not None),
    }


# --------------------------------------------------------------------------- timings
def time_train_step(device, batch, steps, warmup, census=False):
    """PEFT recovery step: forward + backward + optimizer step."""
    holder, unet, opt = _peft_model(device)
    _reset_peak()
    times, info = [], {}
    for i in range(warmup + steps):
        z_t, t, y, eps = _fake_batch(batch, device)
        opt.zero_grad(set_to_none=True)
        _sync(); t0 = time.perf_counter()
        pred = _call(unet, z_t, t, y)
        loss = torch.nn.functional.mse_loss(pred, eps)
        loss.backward()
        if census and not info:
            info = _adapter_grad_census(unet)
        opt.step()
        _sync(); dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return sum(times) / len(times), _peak_gb(), info


def time_saliency(device, batch, iters, warmup):
    """Taylor saliency: forward + backward populating weight grads (no opt step)."""
    m = build_pruned_unet(real_weights=True)
    assert_real_weights(m)
    m = m.to(device)
    for p in m.parameters():
        p.requires_grad = True
    _reset_peak()
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
    m = build_pruned_unet(real_weights=True)
    assert_real_weights(m)
    m = m.to(device).eval()
    _reset_peak()
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


# --------------------------------------------------------------------------- stages
def preflight(args, result) -> None:
    """Fail fast, before a single measurement, on any missing prerequisite."""
    print("=== PREFLIGHT ===", flush=True)

    prov = git_provenance()
    result["git"] = prov
    print(f"  commit {prov['commit']} on {prov['branch']} (dirty={prov['dirty']})")
    print(f"  upstream patch: {prov['upstream_patch_diffstat'] or '(none)'}")
    if args.expect_commit and prov["commit"] != args.expect_commit:
        raise SystemExit(
            f"PREFLIGHT FAIL: expected commit {args.expect_commit}, found {prov['commit']}"
        )
    if prov["dirty"] and not args.allow_dirty:
        raise SystemExit("PREFLIGHT FAIL: working tree is dirty; measurements would not be "
                         "traceable to a commit. Commit first or pass --allow-dirty.")

    if not os.path.exists(CKPT):
        raise SystemExit(f"PREFLIGHT FAIL: checkpoint missing: {CKPT}")
    print(f"  checkpoint present: {CKPT}")

    if DRY_RUN:
        result["DRY_RUN"] = True
        result["GPU_MODEL"] = "DRY-RUN-CPU (no measurement)"
        result["VRAM_GB"] = float("nan")
        print("  DRY RUN on CPU: flow validation only, every timing below is MEANINGLESS "
              "and no value may reach docs/compute_budget.md")
    elif not torch.cuda.is_available():
        raise SystemExit(
            "PREFLIGHT FAIL: no CUDA device. This script never fabricates GPU numbers; "
            "docs/compute_budget.md must stay TBD_MEASURED. Submit it as a Lightning job "
            "with --machine T4 (do NOT switch the interactive Studio)."
        )
    if not DRY_RUN:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        result["GPU_MODEL"] = name
        result["VRAM_GB"] = round(vram, 3)
        print(f"  CUDA device: {name}, {vram:.2f} GB, torch {torch.__version__}")
        if args.expect_gpu and args.expect_gpu.lower() not in name.lower():
            raise SystemExit(f"PREFLIGHT FAIL: expected a {args.expect_gpu}, found {name}")

    if args.r7 != "skip":
        print("  R7a gate (CPU, real weights): one PEFT fwd+bwd+opt step ...", flush=True)
        from tests.research.test_peft_backward_real_unet import check_r7a_step
        if not check_r7a_step():
            raise SystemExit("PREFLIGHT FAIL: R7a did not pass; the PEFT backward path is "
                             "broken. Fix it on the CPU Studio, never on the GPU.")
        result["r7a_gate"] = "PASS"
        print("  R7a PASS")
    print("=== PREFLIGHT OK ===\n", flush=True)


def stage_smoke(args, device, result) -> None:
    print("=== SMOKE (batch=1) ===", flush=True)
    t, peak, census = time_train_step(device, batch=1, steps=args.smoke_steps,
                                      warmup=args.smoke_warmup, census=True)
    print(f"  train sec/step {t:.4f}  peak {peak:.3f} GB")
    print(f"  adapter census: {census}")
    ok = (census.get("lora_B_nonzero_grad") == N_ADAPTERS
          and census.get("frozen_with_grad") == 0)
    result["smoke"] = {"batch": 1, "sec_per_step": round(t, 6),
                       "peak_vram_gb": round(peak, 3), "census": census, "ok": ok}
    if not ok:
        raise SystemExit(
            f"SMOKE FAIL: expected {N_ADAPTERS} adapters with non-zero grad and 0 frozen "
            f"params with grad on CUDA, got {census}. Reproduce on the CPU Studio."
        )
    print("=== SMOKE OK ===\n", flush=True)


def stage_escalate(args, device, result) -> int:
    """Probe increasing batches; record the whole VRAM curve and the largest stable one."""
    print("=== ESCALATE ===", flush=True)
    curve, best = [], None
    for batch in args.batches:
        _empty_cache()
        _reset_peak()
        try:
            t, peak, _ = time_train_step(device, batch=batch, steps=args.probe_steps,
                                         warmup=args.probe_warmup)
            headroom = result["VRAM_GB"] - peak   # NaN in a dry run, by design
            curve.append({"batch": batch, "sec_per_step": round(t, 6),
                          "peak_vram_gb": round(peak, 3),
                          "headroom_gb": round(headroom, 3), "ok": True})
            best = batch
            print(f"  batch {batch:>3}: {t:.4f} s/step, peak {peak:.3f} GB, "
                  f"headroom {headroom:.3f} GB")
        except Exception as exc:                      # noqa: BLE001 - OOM must not be fatal
            if not _is_oom(exc):
                raise
            _empty_cache()
            curve.append({"batch": batch, "ok": False, "error": "CUDA out of memory"})
            print(f"  batch {batch:>3}: OOM — stopping escalation")
            break
    result["escalation"] = curve
    result["MAX_STABLE_BATCH"] = best
    if best is None:
        raise SystemExit("ESCALATE FAIL: even batch=1 did not complete.")
    print(f"=== ESCALATE OK — largest stable batch: {best} ===\n", flush=True)
    return best


def stage_measure(args, device, result, batch) -> None:
    # The escalation probes only a couple of steps, so the largest batch it accepted can
    # still OOM during the long run (allocator fragmentation, a larger transient peak).
    # Step down the ladder instead of losing the whole paid job.
    ladder = [b for b in sorted(args.batches, reverse=True) if b <= batch] or [batch]
    for candidate in ladder:
        try:
            _empty_cache()
            _measure_at(args, device, result, candidate)
            return
        except Exception as exc:                      # noqa: BLE001
            if not _is_oom(exc) or candidate == ladder[-1]:
                raise
            _empty_cache()
            result.setdefault("measure_oom_stepdowns", []).append(candidate)
            print(f"  OOM at batch {candidate} during the long run — stepping down", flush=True)


def _measure_at(args, device, result, batch) -> None:
    print(f"=== MEASURE (batch={batch}) ===", flush=True)
    result["measure_batch"] = batch
    result["steps"] = args.steps
    result["iters"] = args.iters
    result["warmup"] = args.warmup

    t_train, peak_train, _ = time_train_step(device, batch, args.steps, args.warmup)
    result["TRAIN_SEC_PER_STEP"] = round(t_train, 6)
    result["PEAK_TRAIN_VRAM_GB"] = round(peak_train, 3)
    print(f"  TRAIN    {t_train:.4f} s/step   peak {peak_train:.3f} GB", flush=True)

    t_sal, peak_sal = time_saliency(device, batch, args.iters, args.warmup)
    result["SALIENCY_SEC_PER_GRAD_EVAL_OR_BATCH"] = round(t_sal, 6)
    result["PEAK_SALIENCY_VRAM_GB"] = round(peak_sal, 3)
    print(f"  SALIENCY {t_sal:.4f} s/eval   peak {peak_sal:.3f} GB", flush=True)

    t_fwd, peak_fwd = time_forward(device, batch, args.iters, args.warmup)
    result["FORWARD_SEC_PER_DIAGNOSTIC_BATCH"] = round(t_fwd, 6)
    result["PEAK_FORWARD_VRAM_GB"] = round(peak_fwd, 3)
    print(f"  FORWARD  {t_fwd:.4f} s/batch  peak {peak_fwd:.3f} GB", flush=True)
    print("=== MEASURE OK ===\n", flush=True)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["preflight", "smoke", "escalate", "measure", "all"])
    ap.add_argument("--batch", type=int, default=None,
                    help="force the measure batch instead of the largest stable one")
    ap.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(DEFAULT_BATCHES), help="escalation ladder, e.g. 1,2,4,8")
    ap.add_argument("--smoke-steps", type=int, default=5)
    ap.add_argument("--smoke-warmup", type=int, default=2)
    ap.add_argument("--probe-steps", type=int, default=3)
    ap.add_argument("--probe-warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--r7", default="a", choices=["a", "skip"],
                    help="run the R7a PEFT-backward gate before measuring")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None, help="e.g. T4; substring match")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true",
                    help="validate the whole staged flow on CPU; produces NO measurement")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global DRY_RUN
    DRY_RUN = args.dry_run_cpu
    if DRY_RUN and args.out:
        raise SystemExit("--dry-run-cpu produces no measurement and refuses --out, so a "
                         "flow check can never be mistaken for a benchmark.")

    result = {"schema": "gpu_benchmark/2", "argv": sys.argv[1:], "DRY_RUN": DRY_RUN,
              "torch": torch.__version__, "cuda_build": torch.version.cuda}

    preflight(args, result)
    device = torch.device("cpu" if DRY_RUN else "cuda")

    if args.stage == "preflight":
        pass
    else:
        if args.stage in ("smoke", "all"):
            stage_smoke(args, device, result)
        batch = args.batch
        if args.stage in ("escalate", "all"):
            best = stage_escalate(args, device, result)
            batch = batch or best
        if args.stage in ("measure", "all"):
            if batch is None:
                raise SystemExit("MEASURE needs --batch or a preceding escalate stage.")
            stage_measure(args, device, result, batch)

    blob = json.dumps(result, indent=2, sort_keys=True)
    print("=== RESULT JSON ===")
    print(blob, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(blob + "\n")
        print(f"\nwrote {args.out}")
    print("\nNOTE: this benchmark is NOT M1 GPU acceptance. That additionally requires a "
          "several-hundred-step run with a resume test, also submitted as a job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
