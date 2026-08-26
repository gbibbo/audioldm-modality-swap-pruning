#!/usr/bin/env python3
"""Gate-0 GPU smoke: BOUNDED executions of the PRODUCTION trainer and PRODUCTION generator.

Measures execution/cost ONLY (no scientific inference). It calls the exact production per-step code
(`gate0_trainer.train_one_step`) and the exact production generator (`gate0_generator.generate`) so
the numbers correspond to the experiment we will pay for. FP32 (V4-11), EMA convention (V4-12),
recipe from prereg v4. Refuses to write a measurement without CUDA (no invented numbers);
`--dry-run-cpu` exercises the SAME logic on CPU and produces NO measurement.

Reports: GPU model; FP32 peak VRAM; warmup-discarded sec/train-step; #timed train steps;
warmup-discarded sec/generated clip @ 50 DDIM / guidance 2.5 / eta 0.0 / latent_t=96; projected cost of 19,400
training updates; projected cost of 64×3×2=384 Gate-0 generations; projected Gate-0 total;
projected remaining budget under the effective 3.0-cr cap.
"""
import argparse, json, os, subprocess, sys, time
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")

PREREG = "configs/research/icassp_gate0_prereg.yaml"
CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
BATTERY = "configs/research/icassp_gate0_battery.json"
KIM = "artifacts/icassp_gate0/kim193_train_manifest.json"


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-warmup", type=int, default=3)
    ap.add_argument("--train-timed", type=int, default=10)
    ap.add_argument("--gen-warmup", type=int, default=1)
    ap.add_argument("--gen-clips", type=int, default=3)
    ap.add_argument("--cr-per-gpu-hour", type=float, default=0.89)  # T4 anchor; re-anchored per compute_budget
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect-gpu", default="T4")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu writes no measurement; drop --out")
    if not args.dry_run_cpu and not torch.cuda.is_available():
        raise SystemExit("PREFLIGHT FAIL: no CUDA and not --dry-run-cpu (refusing to invent smoke numbers)")

    # provenance + preflight: a paid measurement must be traceable to a clean commit on the right GPU.
    commit = _git("rev-parse", "HEAD"); branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if not args.dry_run_cpu:
        if args.expect_commit and commit != args.expect_commit:
            raise SystemExit(f"PREFLIGHT FAIL: expected commit {args.expect_commit}, found {commit}")
        if dirty and not args.allow_dirty:
            raise SystemExit("PREFLIGHT FAIL: dirty tree; measurement would not be traceable to a commit "
                             "(commit first or pass --allow-dirty)")
        gname = torch.cuda.get_device_name(0)
        if args.expect_gpu and args.expect_gpu.lower() not in gname.lower():
            raise SystemExit(f"PREFLIGHT FAIL: expected a {args.expect_gpu}, found {gname}")

    pre = yaml.safe_load(open(PREREG))
    g0, bat = pre["gate0"], pre["battery"]
    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    R = {"dry_run_cpu": args.dry_run_cpu, "measured": not args.dry_run_cpu,
         "provenance": {"commit": commit, "branch": branch, "dirty": dirty},
         "recipe": {"ddim": bat["ddim_steps"], "guidance": bat["guidance_scale"],
                    "eta": bat.get("ddim_eta", 0.0), "latent_t": g0["data"]["latent_t_size"],
                    "mixed_precision": g0["optim"]["mixed_precision"],
                    "weight_convention": pre["weight_convention"]["convention"]}}
    if not args.dry_run_cpu:
        R["gpu"] = torch.cuda.get_device_name(0)

    # ---------- TRAIN: bounded production steps ----------
    import gate0_trainer as GT
    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = g0["data"]["train_clip_seconds"]
    from measure_tgen import build_model
    from audioldm_peft import setup_peft
    from research_pruning.eval.ema_weights import materialize_ema_into_unet
    torch.load = GT._cpu_load
    model, _ = build_model(config, device); model = model.float()
    dsd = GT._orig_load("data/checkpoints/audioldm-m-full.ckpt", map_location="cpu"); dsd = dsd.get("state_dict", dsd)
    materialize_ema_into_unet(model.model.diffusion_model, dsd, strict=True); model.use_ema = False
    setup_peft(model, GT.make_peft_cfg(g0)); model.train()
    opt, sched, lora_params = GT.build_optimizer_scheduler(model, g0)
    loader, _ = GT.build_loader(config, KIM, g0["optim"]["effective_batch_size"],
                                g0["optim"]["shuffle"], g0["optim"]["drop_last"], seed=20260826)
    it = iter(loader)
    def next_batch():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(loader); return next(it)
    for _ in range(args.train_warmup):
        GT.train_one_step(model, next_batch(), opt, sched, lora_params, g0)
    if not args.dry_run_cpu: torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.train_timed):
        GT.train_one_step(model, next_batch(), opt, sched, lora_params, g0)
    if not args.dry_run_cpu: torch.cuda.synchronize()
    sec_step = (time.perf_counter() - t0) / args.train_timed
    R["train"] = {"timed_steps": args.train_timed, "sec_per_step": sec_step, "fp32_peak_vram_gb": peak_gb()}

    # ---------- GEN: bounded production generations ----------
    import gate0_generator as GG
    import audioldm_train.modules.latent_diffusion.ddim as _ddim
    _oi = _ddim.DDIMSampler.__init__
    _ddim.DDIMSampler.__init__ = lambda s, m, schedule="linear", device=None, **k: _oi(s, m, schedule=schedule, device=(torch.device("cpu") if args.dry_run_cpu else device), **k)
    model.eval(); model._gate0_config = config; model.latent_t_size = g0["data"]["latent_t_size"]
    all_prompts = json.load(open(BATTERY))["prompts"]
    warm_prompts = all_prompts[: args.gen_warmup]
    timed_prompts = all_prompts[args.gen_warmup : args.gen_warmup + args.gen_clips]
    C, T, F = 8, g0["data"]["latent_t_size"], 16
    ddim = 6 if args.dry_run_cpu else bat["ddim_steps"]
    # WARMUP (discarded): same production generate() path/recipe as timed clips (DDIM=50, guidance 2.5,
    # eta 0.0, latent_t=96). CUDA context + DDIM schedule init must not inflate the timed sec/clip.
    for p in warm_prompts:
        x_T = GG.make_x_T(p["ytid"], 0, C, T, F).to(device)
        GG.generate(model, p["caption"], x_T, ddim, bat["guidance_scale"], bat.get("ddim_eta", 0.0))
    if not args.dry_run_cpu: torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    tg = time.perf_counter()
    for p in timed_prompts:
        x_T = GG.make_x_T(p["ytid"], 0, C, T, F).to(device)
        GG.generate(model, p["caption"], x_T, ddim, bat["guidance_scale"], bat.get("ddim_eta", 0.0))
    if not args.dry_run_cpu: torch.cuda.synchronize()
    sec_clip = (time.perf_counter() - tg) / len(timed_prompts)
    R["gen"] = {"clips": len(timed_prompts), "gen_warmup_discarded": len(warm_prompts),
                "ddim_steps": ddim, "sec_per_clip": sec_clip, "gen_peak_vram_gb": peak_gb()}

    # ---------- projections ----------
    rate = args.cr_per_gpu_hour
    n_updates = g0["optim"]["optimizer_updates_200ep"]           # 19400
    n_gens = bat["n_prompts"] * bat["n_seeds"] * 2                # 64*3*2 = 384
    train_cr = n_updates * sec_step / 3600.0 * rate
    gen_cr = n_gens * sec_clip / 3600.0 * rate
    total = train_cr + gen_cr
    bal = pre["budget"]["account_balance_cr"]; cap = pre["budget"]["effective_cap_cr"]
    R["projection"] = {
        "cr_per_gpu_hour": rate, "gate0_train_updates": n_updates, "gate0_generations": n_gens,
        "proj_train_cr": round(train_cr, 4), "proj_gen_cr": round(gen_cr, 4),
        "proj_gate0_total_cr": round(total, 4),
        "stop0_gate0_ceiling_cr": pre["budget"]["gate0_ceiling_cr"],
        "projected_over_stop0": total > pre["budget"]["gate0_ceiling_cr"],
        "effective_cap_cr": cap, "balance_cr": bal,
        "proj_remaining_after_gate0_cr": round(bal - total, 4),
        "note": "CPU dry-run numbers are NOT a measurement" if args.dry_run_cpu else "measured on GPU",
    }
    out = json.dumps(R, indent=2)
    print(out)
    if args.dry_run_cpu:
        print("\nSMOKE DRY-RUN (CPU) — wiring only, NOT a measurement")
        return 0
    if args.out:
        open(args.out, "w").write(out); print("\nMEASUREMENT written to", args.out)
    # hard-exit after persisting: no lingering thread can idle-bill the GPU job (SA3-SMOKE-T4-001).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
