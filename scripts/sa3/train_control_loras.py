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

    pl.seed_everything(a.seed, workers=True)
    model, model_config = build_local_base(device)
    sample_rate = model.sample_rate
    ds_ratio = model.pretransform.downsampling_ratio

    data_dir = a.data_dir
    if dry:
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

    steps = 1 if dry else a.steps
    trainer = pl.Trainer(
        devices=1, accelerator=("cpu" if device.type == "cpu" else "gpu"), strategy="auto",
        # T4 has no bf16 tensor cores -> FP16 mixed on GPU (LoRA params stay fp32); fp32 on CPU dry-run
        precision=("32-true" if device.type == "cpu" else "16-mixed"),
        accumulate_grad_batches=1, callbacks=[], logger=False,
        max_steps=steps, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, log_every_n_steps=a.log_every,
    )
    trainer.fit(wrapper, dataloader)

    # export the trained LoRA (fp16 + embedded config)
    os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
    wrapper.export_lora_safetensors(a.save)
    # verify a non-zero trained weight actually landed
    from stable_audio_3.models.lora.utils import load_lora_checkpoint
    sd, cfg = load_lora_checkpoint(a.save)
    max_b = max((float(v.float().norm()) for k, v in sd.items() if k.endswith(".lora_B")), default=0.0)
    print(f"[train] saved {a.save}: {len(sd)} tensors, config include={cfg.get('include')}, "
          f"max|lora_B|={max_b:.4e} steps={steps} device={device.type}")
    return 0 if (n_lora > 0 and len(sd) > 0) else 1


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
    a = ap.parse_args()
    if not a.dry_run_cpu and not a.data_dir:
        ap.error("--data_dir is required for a real run (or use --dry-run-cpu)")
    return train(a)


if __name__ == "__main__":
    sys.exit(main())
