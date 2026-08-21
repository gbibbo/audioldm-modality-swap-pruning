#!/usr/bin/env python3
"""Train single-block / backbone-only control + ecological LoRAs on the LOCAL small-sfx-base
(protocol §5.1, §5.2). Wraps the upstream DiffusionCondTrainingWrapper but points at our strictly
loaded local base (never the `base_models` registry) and FORCES standard `lora` r16 backbone-only.

Controls (§5.1):  L_6  = --include "transformer.layers.6."
                  L_13 = --include "transformer.layers.13."
Ecological (§5.2): --include "transformer.layers" (whole backbone), one domain per adapter.

The saved `.safetensors` embeds the lora_config (rank/alpha/adapter_type/include) so
`research_sa3.adapters.apply_trained_lora` re-attaches to exactly the trained layers.

Real (GPU):  _external/stable-audio-3/.venv/bin/python scripts/sa3/train_control_loras.py \
                 --block 6 --data_dir data/sa3/adapters/impact_percussion --steps 1000 \
                 --save data/sa3/adapters/L_6.safetensors --expect-commit <sha>
CPU dry-run: OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/train_control_loras.py \
                 --dry-run-cpu --block 6 --save artifacts/sa3/control_dry/L_6.safetensors
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "_external", "stable-audio-3"))
import torch
from research_sa3 import loading

BASE_DIR = "data/sa3/small-sfx-base"
CR_PER_GPU_HOUR = 0.89   # empirical T4 rate (docs/compute_budget.md)


class StepTimer:
    """pl.Callback measuring per-step wall (cuda-synced) + capturing the training loss."""
    def __init__(self):
        self.times, self.losses, self._t = [], [], None

    def _cb(self):
        import pytorch_lightning as pl
        import time as _time
        outer = self

        class _C(pl.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                outer._t = _time.perf_counter()

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if outer._t is not None:
                    outer.times.append(_time.perf_counter() - outer._t)
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                if loss is not None:
                    try:
                        outer.losses.append(float(loss.detach().float().item()))
                    except Exception:
                        pass
        return _C()


def reload_effect_check(save_path, state_file, device, half):
    """Export→reload→effect: apply the trained LoRA to a FRESH base and confirm ||δF(L)||^2 > 0."""
    import gc
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from research_sa3 import fields as Fmod, probes as Pmod, adapters as ADmod
    dtype = torch.float16 if half else torch.float32
    model, _ = build_local_base(device)
    if half:
        model = model.half()
    rep = ADmod.apply_trained_lora(model, save_path)
    d = torch.load(state_file); sts = d["states"][:2]; cap = d["caption"]
    xs = torch.cat([x for _, x in sts], dim=0).to(device, dtype)
    ts = torch.tensor([tau for tau, _ in sts], device=device, dtype=dtype)
    cc0 = Fmod.prepare_conditioning(model, cap, 10, device, latent_len=sts[0][1].shape[-1], dtype=dtype)
    cc = {k: (v.repeat(xs.shape[0], *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v) for k, v in cc0.items()}
    Pmod.set_strength(model, 0.0); FP = Fmod.raw_field(model, xs, ts, cc).detach().float()
    Pmod.set_strength(model, 1.0); FPL = Fmod.raw_field(model, xs, ts, cc).detach().float()
    effect = float(Fmod.state_sq_norm(FPL - FP).sum().item())
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"reload_blocks": rep["blocks"], "reload_n_layers": rep["n_layers"],
            "delta_F_sq": effect, "effect_nonzero": bool(effect > 0)}


def include_for(block, backbone):
    if backbone:
        return ["transformer.layers"]
    if block is None:
        raise SystemExit("either --block N (single-block control) or --backbone (ecological) is required")
    return [f"transformer.layers.{int(block)}."]


def make_synth_data(dir_, n, dur_s, sr):
    """Tiny wav+txt pairs for the CPU dry-run (no external data)."""
    import numpy as np, soundfile as sf
    os.makedirs(dir_, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
        x = (0.3 * np.sin(2 * np.pi * (220 + 20 * i) * t)).astype("float32")
        x = np.stack([x, x], 1)  # stereo
        sf.write(os.path.join(dir_, f"clip{i:03d}.wav"), x, sr)
        open(os.path.join(dir_, f"clip{i:03d}.txt"), "w").write(f"a synthetic control impact {i}")
    return dir_


def build_local_base(device):
    cfg = loading.load_json(f"{BASE_DIR}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{BASE_DIR}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{BASE_DIR}/model.safetensors", device=device)
    return model, cfg


def train(a):
    import pytorch_lightning as pl
    from stable_audio_3.data.dataset import LocalDatasetConfig, SampleDataset, collation_fn
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper
    from pathlib import Path

    dry = a.dry_run_cpu
    device = torch.device("cpu" if dry or not torch.cuda.is_available() else "cuda")
    if a.expect_commit and not dry:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    import time as _time
    pl.seed_everything(a.seed, workers=True)
    _t_load = _time.perf_counter()
    model, model_config = build_local_base(device)
    load_wall_s = _time.perf_counter() - _t_load
    sample_rate = model.sample_rate
    ds_ratio = model.pretransform.downsampling_ratio

    data_dir = a.data_dir
    if dry or (a.smoke and not a.data_dir):
        tmp = tempfile.mkdtemp(prefix="sa3_ctrl_")
        data_dir = make_synth_data(tmp, n=4, dur_s=a.duration, sr=sample_rate)

    sample_size = (int(a.duration * sample_rate) // ds_ratio) * ds_ratio

    def caption_metadata_fn(info, audio):
        txt = Path(info["path"]).with_suffix(".txt")
        return {"prompt": txt.read_text().strip()} if txt.exists() else {"__reject__": True}

    dataset = SampleDataset(
        [LocalDatasetConfig(id="train", path=data_dir, custom_metadata_fn=caption_metadata_fn)],
        sample_size=sample_size, sample_rate=sample_rate, force_channels="stereo",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True,
        collate_fn=collation_fn,
        worker_init_fn=lambda wid: torch.manual_seed(a.seed + wid),
    )

    lora_config = {
        "rank": a.rank, "alpha": a.rank, "adapter_type": "lora",
        "include": include_for(a.block, a.backbone), "exclude": None,
    }
    optimizer_config = {"diffusion": {"optimizer": {"type": "AdamW",
                        "config": {"lr": a.lr, "weight_decay": 0.01, "betas": [0.9, 0.95]}}}}

    wrapper = DiffusionCondTrainingWrapper(
        model, mask_loss_weight=1.0, mask_padding_attention=True,
        silence_extension_scale_seconds=4.0, use_ema=False, log_loss_info=False,
        optimizer_configs=optimizer_config, pre_encoded=False,
        timestep_sampler="trunc_logit_normal", timestep_sampler_options={},
        inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}},
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", 44100),
        sample_size=model_config.get("sample_size"),
        lora_config=lora_config, lora_state_dict=None, svd_bases_path=None,
        log_every_n_steps=a.log_every, ot_coupling=True,
        base_precision=(None if dry else a.base_precision),
    )
    from stable_audio_3.models.lora import get_lora_layers
    n_lora = len(get_lora_layers(wrapper.diffusion))
    print(f"[train] lora layers attached: {n_lora} (include={lora_config['include']})")
    assert n_lora > 0, "no LoRA layers attached -- include matched nothing"

    steps = a.smoke_steps if a.smoke else (1 if dry else a.steps)
    timer = StepTimer() if a.smoke else None
    if a.smoke and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(
        devices=1, accelerator=("cpu" if device.type == "cpu" else "gpu"), strategy="auto",
        # T4 has no bf16 tensor cores -> FP16 mixed on GPU (LoRA params stay fp32); fp32 on CPU dry-run
        precision=("32-true" if device.type == "cpu" else "16-mixed"),
        accumulate_grad_batches=1, callbacks=([timer._cb()] if timer else []), logger=False,
        max_steps=steps, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, log_every_n_steps=a.log_every,
    )
    _t_fit = _time.perf_counter()
    trainer.fit(wrapper, dataloader)
    fit_wall_s = _time.perf_counter() - _t_fit

    # export the trained LoRA (fp16 + embedded config)
    os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
    wrapper.export_lora_safetensors(a.save)
    from stable_audio_3.models.lora.utils import load_lora_checkpoint
    sd, cfg = load_lora_checkpoint(a.save)
    max_b = max((float(v.float().norm()) for k, v in sd.items() if k.endswith(".lora_B")), default=0.0)
    print(f"[train] saved {a.save}: {len(sd)} tensors, config include={cfg.get('include')}, "
          f"max|lora_B|={max_b:.4e} steps={steps} device={device.type}")

    if not a.smoke:
        return 0 if (n_lora > 0 and len(sd) > 0) else 1

    # ---- SMOKE: metrics + cost projection + export->reload->effect (INFRA ONLY, not scientific) ----
    import gc, json, statistics
    vram_peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None
    times = timer.times[1:] if len(timer.times) > 1 else timer.times   # drop warmup step
    sec_step = statistics.median(times) if times else float("nan")
    losses = timer.losses
    losses_finite = bool(losses) and all(l == l and abs(l) != float("inf") for l in losses)
    # free the training graph before the reload model
    del wrapper, model, trainer; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    state_file = a.state_file
    if not state_file:
        import glob as _g
        cands = sorted(_g.glob(os.path.join(a.state_store, "state_*.pt")))
        state_file = cands[0] if cands else None
    reload_info = {}
    if state_file:
        reload_info = reload_effect_check(a.save, state_file, device, half=(device.type == "cuda"))

    def proj(n_steps):
        gpu_h = (n_steps * sec_step) / 3600.0
        return {"gpu_hours": round(gpu_h, 4), "credits": round(gpu_h * CR_PER_GPU_HOUR, 4)}
    load_h = load_wall_s / 3600.0
    one_ctrl = {"steps": 1000, "sec_step": sec_step,
                "gpu_hours": round(load_h + 1000 * sec_step / 3600.0, 4),
                "credits": round((load_h + 1000 * sec_step / 3600.0) * CR_PER_GPU_HOUR, 4)}
    two_ctrl_credits = round(2 * one_ctrl["credits"], 4)
    smoke = {
        "phase": "train_smoke", "SYNTHETIC_INFRA_ONLY": True,
        "note": "synthetic data; NOT for any scientific conclusion / control / RQ2",
        "device": device.type, "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "precision": ("16-mixed" if device.type == "cuda" else "32-true"),
        "base_precision": a.base_precision, "block": a.block, "rank": a.rank,
        "steps_measured": steps, "n_lora_layers": n_lora,
        "sec_per_step_median": sec_step, "step_times_s": timer.times,
        "vram_peak_gb": vram_peak_gb, "load_wall_s": round(load_wall_s, 2), "fit_wall_s": round(fit_wall_s, 2),
        "loss_first": (losses[0] if losses else None), "loss_last": (losses[-1] if losses else None),
        "losses_finite": losses_finite, "max_lora_B": max_b, "lora_B_updated": bool(max_b > 0),
        "reload_effect": reload_info,
        "projection": {"cr_per_gpu_hour": CR_PER_GPU_HOUR, "per_1000_steps_compute_only": proj(1000),
                       "one_control_incl_load": one_ctrl, "two_controls_L6_L13_credits": two_ctrl_credits},
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.smoke_out)), exist_ok=True)
    json.dump(smoke, open(a.smoke_out, "w"), indent=2)
    print("SMOKE_JSON_BEGIN"); print(json.dumps(smoke)); print("SMOKE_JSON_END")
    ok = (n_lora > 0 and max_b > 0 and losses_finite
          and (not reload_info or reload_info.get("effect_nonzero")))
    print(f"[smoke] sec/step={sec_step:.4f} vram_peak_gb={vram_peak_gb} loss_finite={losses_finite} "
          f"lora_B={max_b:.3e} effect_nonzero={reload_info.get('effect_nonzero')} "
          f"proj_1000={one_ctrl['credits']}cr proj_L6+L13={two_ctrl_credits}cr OK={ok}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--block", type=int, default=None, help="single-block control host block (e.g. 6, 13)")
    ap.add_argument("--backbone", action="store_true", help="ecological: whole-backbone LoRA")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--duration", type=float, default=10.0,
                    help="FROZEN training crop = 10.0 s for controls AND ecological adapters "
                         "(matches the analysis panel SECONDS=10 / generation length); registered pre-data")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_precision", default="fp16", choices=["fp16", "float16", "fp32", "bf16", "bfloat16"],
                    help="T4 target -> fp16 (default); bf16 only on Ampere+; base fp32 ok if VRAM allows")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save", required=True)
    ap.add_argument("--expect-commit", default=None)
    # infra micro-smoke (synthetic data; NOT scientific)
    ap.add_argument("--smoke", action="store_true", help="infra micro-smoke: measure VRAM/sec-step/loss "
                    "+ export→reload→effect + 1000-step cost projection (synthetic data)")
    ap.add_argument("--smoke-steps", type=int, default=25)
    ap.add_argument("--state-store", default="artifacts/sa3/pilot_states_dry")
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--smoke-out", default="artifacts/sa3/train_smoke.json")
    a = ap.parse_args()
    if not a.dry_run_cpu and not a.smoke and not a.data_dir:
        ap.error("--data_dir is required for a real run (or use --dry-run-cpu / --smoke)")
    return train(a)


if __name__ == "__main__":
    sys.exit(main())
