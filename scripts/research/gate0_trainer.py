#!/usr/bin/env python3
"""Gate-0 real-data LoRA trainer: DENSE M-Full + Kim recipe at 3.84 s / latent_t=96.

Recipe frozen in configs/research/icassp_gate0_prereg.yaml (DECISION-V4-09/10), code-derived from
Kim's train_audioldm_lora.py: LoRA on to_q/to_v (all attention), r8/alpha16, AdamW lr 1e-5,
batch 2, shuffle True, drop_last False (=> 97 steps/epoch), polynomial LR schedule over the
97000-step 1000-epoch horizon (0 warmup). 200 epochs => 19,400 optimizer updates.

`--dry-run-cpu` validates the ENTIRE path + every preregistered assertion on CPU (no GPU), runs a
couple of real optimizer steps to prove the loop, and writes NO checkpoint. It is NOT a scientific
result. The GPU job runs the same code without --dry-run-cpu (guarded on CUDA).
"""
import argparse, json, math, os, sys
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
PREREG = "configs/research/icassp_gate0_prereg.yaml"
KIM_MANIFEST = "artifacts/icassp_gate0/kim193_train_manifest.json"

# CPU-only torch.load shim (vocoder/DDIM hardcode CUDA); mirrors measure_tgen._cpu_torch_load.
_orig_load = torch.load
def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu"); return _orig_load(*a, **k)


def build_loader(config, manifest, batch, shuffle, drop_last, seed):
    from torch.utils.data import DataLoader
    from audioldm_train.utilities.data.dataset import AudioDataset
    data = json.load(open(manifest))["data"]
    ds = AudioDataset(config=config, split="train", waveform_only=False,
                      dataset_json={"data": data})
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last,
                      num_workers=0, generator=g), len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=2, help="dry-run optimizer steps")
    args = ap.parse_args()
    if not args.dry_run_cpu and not torch.cuda.is_available():
        raise SystemExit("refusing to run the real trainer without CUDA (use --dry-run-cpu)")

    pre = yaml.safe_load(open(PREREG))["gate0"]
    R = {"assertions": {}, "report": {}}

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = pre["data"]["train_clip_seconds"]  # 3.84
    exp_wave = int(round(pre["data"]["train_clip_seconds"] * config["preprocessing"]["audio"]["sampling_rate"]))
    exp_mel = pre["data"]["mel_target_length"]           # 384
    exp_lat = pre["data"]["latent_t_size"]               # 96

    torch.manual_seed(pre["optim"].get("seed", 20260826) if isinstance(pre["optim"], dict) else 20260826)
    torch.load = _cpu_load
    from measure_tgen import build_model
    from audioldm_peft import PeftConfig, setup_peft
    from audioldm_peft.report import parameter_report
    from audioldm_peft.inject import assert_peft_ready
    from audioldm_peft.layers import LoRALinear, LoRAConv2d

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    model, load_info = build_model(config, device)
    R["report"]["base_load"] = load_info

    # --- attach LoRA to EXACTLY to_q/to_v (Kim recipe) ---
    cfg = PeftConfig(rank=pre["lora"]["rank"], alpha=pre["lora"]["alpha"], dropout=pre["lora"]["dropout"],
                     target_linear=True, target_conv2d=False, train_bias=False,
                     train_groupnorm_affine=False, train_layernorm_affine=False,
                     root_path="model.diffusion_model",
                     include_name_substrings=tuple(pre["lora"]["target_substrings"]))
    setup_peft(model, cfg)

    # ASSERT: LoRA only on to_q/to_v
    lora_names = [n for n, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv2d))]
    only_qv = all(("to_q" in n or "to_v" in n) for n in lora_names)
    n_conv = sum(1 for n, m in model.named_modules() if isinstance(m, LoRAConv2d))
    R["assertions"]["lora_only_to_q_to_v"] = only_qv
    R["assertions"]["lora_no_conv2d"] = (n_conv == 0)
    R["report"]["n_lora_modules"] = len(lora_names)
    R["report"]["n_to_q"] = sum("to_q" in n for n in lora_names)
    R["report"]["n_to_v"] = sum("to_v" in n for n in lora_names)

    # ASSERT: base frozen, only LoRA trainable
    rep = parameter_report(model)
    R["report"]["parameter_report"] = rep
    trainable_non_lora = rep.get("trainable_total", 0) - rep.get("lora", 0)
    R["assertions"]["only_lora_trainable"] = (trainable_non_lora == 0)
    assert_peft_ready(model, cfg)   # raises if any base param has grad or LoRA lacks grad
    R["assertions"]["assert_peft_ready"] = True

    # --- one real batch through the preregistered 3.84-s path ---
    batch_sz = pre["optim"]["effective_batch_size"]
    loader, n_data = build_loader(config, KIM_MANIFEST, batch_sz,
                                  pre["optim"]["shuffle"], pre["optim"]["drop_last"], seed=20260826)
    batch = next(iter(loader))

    wave_len = batch["waveform"].shape[-1]
    mel_t = batch["log_mel_spec"].shape[1]   # [B, T_mel, 64]
    R["assertions"]["waveform_61440"] = (wave_len == exp_wave)
    R["assertions"]["mel_time_384"] = (mel_t == exp_mel)
    R["report"]["waveform_samples"] = int(wave_len)
    R["report"]["mel_time"] = int(mel_t)

    # get_input -> latent z + cond; assert latent time = 96
    model.train()
    z, cond = model.get_input(batch, model.first_stage_key,
                              unconditional_prob_cfg=config["model"]["params"].get("unconditional_prob_cfg", 0.1))
    R["assertions"]["latent_time_96"] = (z.shape[2] == exp_lat)
    R["report"]["latent_shape"] = list(z.shape)

    # ASSERT: U-Net output shape == noise target exactly
    t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=device).long()
    noise = torch.randn_like(z)
    x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
    model_out = model.apply_model(x_noisy, t, cond)
    R["assertions"]["unet_out_matches_noise_target"] = (list(model_out.shape) == list(noise.shape) == list(z.shape))
    R["report"]["unet_out_shape"] = list(model_out.shape)

    # --- real optimizer step(s): AdamW + polynomial LR schedule (Kim horizon) ---
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=pre["optim"]["lr"])
    sch = pre["optim"]["lr_scheduler"]
    from torch.optim.lr_scheduler import LambdaLR
    def poly_lambda(step):
        n = sch["num_training_steps"]; p = sch["power"]; end = sch["lr_end"]; base = pre["optim"]["lr"]
        if step >= n:
            return end / base
        return ((base - end) * (1 - step / n) ** p + end) / base
    scheduler = LambdaLR(opt, poly_lambda)

    # snapshot to prove LoRA changes and base does not
    b_ref = next(m.lora_B for _, m in model.named_modules() if isinstance(m, LoRALinear)).detach().clone()
    base_ref = None
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            base_ref = m.base.weight.detach().clone(); break

    losses = []
    for i in range(args.steps):
        batch = next(iter(loader))
        z, cond = model.get_input(batch, model.first_stage_key,
                                  unconditional_prob_cfg=config["model"]["params"].get("unconditional_prob_cfg", 0.1))
        t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=device).long()
        loss, _ = model.p_losses(z, cond, t)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        losses.append(float(loss))
    R["report"]["dry_run_losses"] = losses
    R["assertions"]["loss_finite"] = all(math.isfinite(l) for l in losses)

    b_new = next(m.lora_B for _, m in model.named_modules() if isinstance(m, LoRALinear)).detach()
    base_new = None
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            base_new = m.base.weight.detach(); break
    R["assertions"]["lora_updated"] = bool((b_new - b_ref).abs().max() > 0)
    R["assertions"]["base_unchanged"] = bool((base_new - base_ref).abs().max() == 0)

    # --- optimizer-step accounting (exact, code-derived) ---
    steps_per_epoch = n_data // batch_sz if pre["optim"]["drop_last"] else math.ceil(n_data / batch_sz)
    updates_200 = steps_per_epoch * pre["epochs"]
    R["report"]["optimizer_step_accounting"] = {
        "n_data": n_data, "batch": batch_sz, "drop_last": pre["optim"]["drop_last"],
        "steps_per_epoch": steps_per_epoch, "epochs": pre["epochs"],
        "optimizer_updates": updates_200,
        "matches_prereg_19400": updates_200 == pre["optim"]["optimizer_updates_200ep"],
    }

    all_ok = all(R["assertions"].values())
    R["ALL_ASSERTIONS_PASS"] = all_ok
    print(json.dumps(R, indent=2))
    print("\nDRY-RUN", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
