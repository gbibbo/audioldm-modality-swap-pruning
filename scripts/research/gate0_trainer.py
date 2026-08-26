#!/usr/bin/env python3
"""Gate-0 PRODUCTION LoRA trainer: DENSE M-Full + Kim recipe at 3.84 s / latent_t=96.

ONE production training path. `--dry-run-cpu` and `--max-updates` are BOUNDED execution modes of
the SAME per-step code (`train_one_step`); the scientific run and the smoke share every line that
touches the model. Recipe frozen in configs/research/icassp_gate0_prereg.yaml v3 (DECISION-V4-09/10/11),
code-derived from Kim's train_audioldm_lora.py (peft==0.13.2, diffusers==0.32.2, accelerate==1.0.1):
  LoRA on to_q/to_v (all attention), r8/alpha16, GAUSSIAN init; AdamW lr 1e-5, betas (0.9,0.999),
  weight_decay 1e-5, eps 1e-8; grad clip max_norm 1.0; polynomial LR over the 97000-step horizon
  (0 warmup); FP32 (mixed_precision=None); NO CFG dropout in the LoRA loop (train uncond_prob 0.0);
  shuffle True, drop_last False => 97 steps/epoch; 200 epochs => EXACTLY 19,400 optimizer updates.

Full run (no flags, requires CUDA) trains 200 epochs and asserts global_step == 19400, then saves
adapter-only weights + config + sha256 + training manifest + optimizer/scheduler metadata.
"""
import argparse, hashlib, json, math, os, sys, time
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
PREREG = "configs/research/icassp_gate0_prereg.yaml"
KIM_MANIFEST = "artifacts/icassp_gate0/kim193_train_manifest.json"
CKPT_DIR = "artifacts/icassp_gate0/gate0_adapter"   # gitignored

_orig_load = torch.load
def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu"); return _orig_load(*a, **k)


# --------------------------------------------------------------------------- scheduler
def poly_lambda(sch, base_lr):
    """Polynomial decay == diffusers get_polynomial_decay_schedule_with_warmup (num_warmup=0)."""
    n = sch["num_training_steps"]; power = sch["power"]; end = sch["lr_end"]
    def fn(step):
        if step >= n:
            return end / base_lr
        return ((base_lr - end) * (1 - step / n) ** power + end) / base_lr
    return fn


def prove_scheduler_matches_diffusers(sch, base_lr):
    """Diffusers reference (v0.32.2 optimization.py, warmup=0). Assert equality at key steps."""
    n = sch["num_training_steps"]; power = sch["power"]; end = sch["lr_end"]
    def diffusers_ref(step):
        if step > n:
            return end / base_lr
        lr_range = base_lr - end
        pct = 1 - step / n
        return (lr_range * pct ** power + end) / base_lr
    ours = poly_lambda(sch, base_lr)
    checks = {}
    for s in (0, 1, 19400, 97000):
        a, b = ours(s), diffusers_ref(s)
        checks[str(s)] = {"ours": a, "diffusers": b, "equal": abs(a - b) < 1e-12}
    return checks, all(c["equal"] for c in checks.values())


# --------------------------------------------------------------------------- data / model
def build_loader(config, manifest, batch, shuffle, drop_last, seed):
    from torch.utils.data import DataLoader
    from audioldm_train.utilities.data.dataset import AudioDataset
    data = json.load(open(manifest))["data"]
    ds = AudioDataset(config=config, split="train", waveform_only=False,
                      dataset_json={"data": data})
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last,
                      num_workers=0, generator=g), len(data)


def make_peft_cfg(pre):
    from audioldm_peft import PeftConfig
    return PeftConfig(rank=pre["lora"]["rank"], alpha=pre["lora"]["alpha"], dropout=pre["lora"]["dropout"],
                      target_linear=True, target_conv2d=False, train_bias=False,
                      train_groupnorm_affine=False, train_layernorm_affine=False,
                      root_path="model.diffusion_model",
                      include_name_substrings=tuple(pre["lora"]["target_substrings"]),
                      init_lora_weights=pre["lora"]["init"])


def build_optimizer_scheduler(model, pre):
    o = pre["optim"]
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=o["lr"], betas=tuple(o["betas"]),
                            weight_decay=o["weight_decay"], eps=o["eps"])
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(opt, poly_lambda(o["lr_scheduler"], o["lr"]))
    return opt, scheduler, lora_params


# --------------------------------------------------------------------------- ONE per-step path
def train_one_step(model, batch, opt, scheduler, lora_params, pre):
    """The single per-step code shared by the full run, the dry-run, and the smoke."""
    z, cond = model.get_input(batch, model.first_stage_key,
                              unconditional_prob_cfg=pre["optim"]["train_unconditional_prob_cfg"])
    t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=z.device).long()
    loss, _ = model.p_losses(z, cond, t)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, pre["optim"]["grad_clip_max_norm"])
    opt.step()
    scheduler.step()
    return float(loss)


def save_adapter(model, pre, config, global_step, opt, scheduler, losses, out_dir):
    from audioldm_peft.layers import LoRALinear, LoRAConv2d
    os.makedirs(out_dir, exist_ok=True)
    adapter_sd = {}
    for name, m in model.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            adapter_sd[name + ".lora_A"] = m.lora_A.detach().cpu()
            adapter_sd[name + ".lora_B"] = m.lora_B.detach().cpu()
    wpath = os.path.join(out_dir, "gate0_adapter.pt")
    torch.save(adapter_sd, wpath)
    sha = hashlib.sha256(open(wpath, "rb").read()).hexdigest()
    meta = {
        "recipe": {"rank": pre["lora"]["rank"], "alpha": pre["lora"]["alpha"],
                   "init": pre["lora"]["init"], "targets": pre["lora"]["target_substrings"],
                   "lr": pre["optim"]["lr"], "betas": pre["optim"]["betas"],
                   "weight_decay": pre["optim"]["weight_decay"], "eps": pre["optim"]["eps"],
                   "grad_clip": pre["optim"]["grad_clip_max_norm"],
                   "mixed_precision": pre["optim"]["mixed_precision"],
                   "train_uncond_prob": pre["optim"]["train_unconditional_prob_cfg"],
                   "epochs": pre["epochs"], "clip_seconds": pre["data"]["train_clip_seconds"]},
        "global_step": global_step, "adapter_sha256": sha, "n_adapter_tensors": len(adapter_sd),
        "final_lr": scheduler.get_last_lr()[0], "last_losses": losses[-5:],
        "train_manifest": KIM_MANIFEST,
        "train_manifest_sha256": json.load(open(KIM_MANIFEST)).get("manifest_sha256"),
    }
    json.dump(meta, open(os.path.join(out_dir, "gate0_adapter_meta.json"), "w"), indent=1)
    return wpath, sha, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-cpu", action="store_true",
                    help="bounded CPU execution of the SAME per-step code + assertion checklist")
    ap.add_argument("--max-updates", type=int, default=None,
                    help="cap optimizer updates (bounded modes / smoke); full run leaves it None")
    ap.add_argument("--out", default=CKPT_DIR)
    args = ap.parse_args()
    if not args.dry_run_cpu and not torch.cuda.is_available():
        raise SystemExit("refusing to run the full trainer without CUDA (use --dry-run-cpu)")

    pre = yaml.safe_load(open(PREREG))["gate0"]
    R = {"assertions": {}, "report": {}}
    torch.load = _cpu_load
    if args.dry_run_cpu:  # patch DDIM device just in case downstream reuse; harmless
        pass

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = pre["data"]["train_clip_seconds"]
    exp_wave = int(round(pre["data"]["train_clip_seconds"] * config["preprocessing"]["audio"]["sampling_rate"]))
    torch.manual_seed(20260826)

    from measure_tgen import build_model
    from audioldm_peft import setup_peft
    from audioldm_peft.report import parameter_report
    from audioldm_peft.inject import assert_peft_ready
    from audioldm_peft.layers import LoRALinear, LoRAConv2d

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    model, load_info = build_model(config, device)
    if pre["optim"]["mixed_precision"] == "fp32":
        model = model.float()

    # V4-12 EMA convention: materialize dense EMA (inference weights) into the U-Net BEFORE LoRA,
    # and disable runtime EMA (its name-map is frozen pre-injection and would KeyError after PEFT).
    wc = yaml.safe_load(open(PREREG)).get("weight_convention", {})
    if wc.get("convention") == "ema":
        from research_pruning.eval.ema_weights import materialize_ema_into_unet
        dsd = _orig_load("data/checkpoints/audioldm-m-full.ckpt", map_location="cpu")
        dsd = dsd.get("state_dict", dsd)
        n_ema, ema_unusable = materialize_ema_into_unet(model.model.diffusion_model, dsd, strict=True)
        if wc.get("disable_runtime_ema"):
            model.use_ema = False
        R["report"]["ema_materialized"] = {"n": n_ema, "unusable": len(ema_unusable), "use_ema_disabled": True}
        R["assertions"]["ema_convention_applied"] = (n_ema == 690 and len(ema_unusable) == 0 and model.use_ema is False)

    cfg = make_peft_cfg(pre)
    setup_peft(model, cfg)
    model.train()

    # scheduler-equality proof vs diffusers (documented before any timing)
    sch_checks, sch_ok = prove_scheduler_matches_diffusers(pre["optim"]["lr_scheduler"], pre["optim"]["lr"])
    R["assertions"]["scheduler_equals_diffusers_polynomial"] = sch_ok
    R["report"]["scheduler_checks"] = sch_checks

    opt, scheduler, lora_params = build_optimizer_scheduler(model, pre)

    # --- static assertions (recipe wiring) ---
    lora_names = [n for n, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv2d))]
    R["assertions"]["lora_only_to_q_to_v"] = all(("to_q" in n or "to_v" in n) for n in lora_names)
    R["assertions"]["lora_no_conv2d"] = all(not isinstance(m, LoRAConv2d) for _, m in model.named_modules())
    rep = parameter_report(model)
    R["report"]["n_lora_modules"] = len(lora_names)
    R["report"]["parameter_report"] = rep
    R["assertions"]["only_lora_trainable"] = (rep.get("trainable_total", 0) - rep.get("lora", 0) == 0)
    assert_peft_ready(model, cfg)
    R["assertions"]["assert_peft_ready"] = True
    R["report"]["optimizer"] = {"betas": pre["optim"]["betas"], "weight_decay": pre["optim"]["weight_decay"],
                                "eps": pre["optim"]["eps"], "grad_clip": pre["optim"]["grad_clip_max_norm"],
                                "lora_init": pre["lora"]["init"], "mixed_precision": pre["optim"]["mixed_precision"],
                                "train_uncond_prob": pre["optim"]["train_unconditional_prob_cfg"]}

    batch_sz = pre["optim"]["effective_batch_size"]
    loader, n_data = build_loader(config, KIM_MANIFEST, batch_sz,
                                  pre["optim"]["shuffle"], pre["optim"]["drop_last"], seed=20260826)
    steps_per_epoch = n_data // batch_sz if pre["optim"]["drop_last"] else math.ceil(n_data / batch_sz)
    planned_updates = steps_per_epoch * pre["epochs"]
    R["report"]["step_accounting"] = {"n_data": n_data, "batch": batch_sz, "drop_last": pre["optim"]["drop_last"],
                                      "steps_per_epoch": steps_per_epoch, "epochs": pre["epochs"],
                                      "planned_updates": planned_updates,
                                      "matches_prereg_19400": planned_updates == pre["optim"]["optimizer_updates_200ep"]}

    # one-batch shape assertions (dry-run only; the full run trusts them + trains)
    if args.dry_run_cpu:
        b0 = next(iter(loader))
        R["assertions"]["waveform_61440"] = (b0["waveform"].shape[-1] == exp_wave)
        R["assertions"]["mel_time_384"] = (b0["log_mel_spec"].shape[1] == pre["data"]["mel_target_length"])
        z0, _ = model.get_input(b0, model.first_stage_key, unconditional_prob_cfg=0.0)
        R["assertions"]["latent_time_96"] = (z0.shape[2] == pre["data"]["latent_t_size"])

    # --- THE production loop (bounded by --max-updates / dry-run) ---
    cap = args.max_updates if args.max_updates is not None else (4 if args.dry_run_cpu else planned_updates)
    b_ref = next(m.lora_B.detach().clone() for _, m in model.named_modules() if isinstance(m, LoRALinear))
    base_ref = next(m.base.weight.detach().clone() for _, m in model.named_modules() if isinstance(m, LoRALinear))
    losses, global_step, t0 = [], 0, time.perf_counter()
    stop = False
    for epoch in range(pre["epochs"]):
        for batch in loader:
            losses.append(train_one_step(model, batch, opt, scheduler, lora_params, pre))
            global_step += 1
            if global_step >= cap:
                stop = True; break
        if stop:
            break
    wall = time.perf_counter() - t0
    R["report"]["ran_updates"] = global_step
    R["report"]["sec_per_step"] = wall / max(1, global_step)
    R["assertions"]["loss_finite"] = all(math.isfinite(l) for l in losses)
    b_new = next(m.lora_B.detach() for _, m in model.named_modules() if isinstance(m, LoRALinear))
    base_new = next(m.base.weight.detach() for _, m in model.named_modules() if isinstance(m, LoRALinear))
    R["assertions"]["lora_updated"] = bool((b_new - b_ref).abs().max() > 0)
    R["assertions"]["base_unchanged"] = bool((base_new - base_ref).abs().max() == 0)

    is_full = (not args.dry_run_cpu) and (args.max_updates is None)
    if is_full:
        assert global_step == pre["optim"]["optimizer_updates_200ep"], \
            f"global_step {global_step} != {pre['optim']['optimizer_updates_200ep']}"
        R["assertions"]["global_step_19400"] = True
        wpath, sha, meta = save_adapter(model, pre, config, global_step, opt, scheduler, losses, args.out)
        R["report"]["saved_adapter"] = {"path": wpath, "sha256": sha, "meta": meta}

    all_ok = all(R["assertions"].values())
    R["ALL_ASSERTIONS_PASS"] = all_ok
    R["mode"] = "full" if is_full else ("dry-run-cpu" if args.dry_run_cpu else f"bounded({cap})")
    print(json.dumps(R, indent=2))
    print("\nTRAINER", "PASS" if all_ok else "FAIL", "| mode:", R["mode"], "| ran", global_step, "updates")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
