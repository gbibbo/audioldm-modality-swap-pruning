#!/usr/bin/env python3
"""A1 evidence: real noised-latent construction for M3A (CPU-only).

Documents, with saved numbers, the correction of audit finding A1 (z_t was pure
Gaussian noise; it is now a noised REAL latent). Reports:
  * scale_factor read from the checkpoint;
  * whether the embedded first_stage VAE differs from the standalone
    vae_mel_16k_64bins.ckpt (and by how much);
  * schedule sanity (sqrt_alphas_cumprod endpoints, q_sample at t=0);
  * z_0 / z_t statistics and bit-identical determinism across two builds;
  * proof that z_t != pure noise and z_t == q_sample(z_0, t, noise).

Writes artifacts/m3_pilot/a1_latent_check.json. Uses real weights + real items.
NEVER loads the real pruned checkpoint.
"""
from __future__ import annotations

import json
import os
import sys

import torch

from research_pruning.diagnostics.conditioning import (
    FROZEN_CONFIG,
    NoiseSchedule,
    build_paired_slots,
    build_vae,
    load_config,
    read_scale_factor,
    tensor_hash,
    _torch_load,
)
from audioldm_train.utilities.data.dataset import AudioDataset

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
VAE_CKPT = "data/checkpoints/vae_mel_16k_64bins.ckpt"
OUT = "artifacts/m3_pilot"
INDICES = [0, 1, 2, 3]
SEED = 1234


def compare_vae_sources(config):
    """Compare first_stage_model.* embedded in the full ckpt vs the standalone VAE."""
    sd = _torch_load(BASE_CKPT)
    sd = sd.get("state_dict", sd)
    FS_full = {k[len("first_stage_model."):]: v
               for k, v in sd.items() if k.startswith("first_stage_model.")}
    # The standalone VAE ckpt carries non-tensor pickled objects, so weights_only
    # fails; it is a trusted local artifact. Comparison only, not used for encode.
    FS_vae = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    common = [k for k in FS_full if k in FS_vae and FS_full[k].shape == FS_vae[k].shape]
    identical = sum(1 for k in common
                    if torch.equal(FS_full[k].float(), FS_vae[k].float()))
    maxdiff = 0.0
    enc_maxdiff = 0.0
    for k in common:
        d = (FS_full[k].float() - FS_vae[k].float()).abs().max().item()
        maxdiff = max(maxdiff, d)
        if k.startswith("encoder.") or k.startswith("quant_conv."):
            enc_maxdiff = max(enc_maxdiff, d)
    return {
        "common_same_shape_tensors": len(common),
        "identical_tensors": identical,
        "differing_tensors": len(common) - identical,
        "max_abs_diff_all": maxdiff,
        "max_abs_diff_encode_path": enc_maxdiff,
        "verdict": ("DIFFER: embedded first_stage was jointly retrained; "
                    "LatentDiffusion uses the embedded weights, so M3 must too"
                    if identical < len(common) else "identical"),
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    config = load_config(FROZEN_CONFIG)

    scale_factor = read_scale_factor(BASE_CKPT)
    vae_cmp = compare_vae_sources(config)

    schedule = NoiseSchedule(config)
    sac = schedule.sqrt_alphas_cumprod
    somac = schedule.sqrt_one_minus_alphas_cumprod
    # sanity: t=0 -> almost all signal; t=T-1 -> almost all noise
    schedule_info = {
        "timesteps": schedule.timesteps,
        "linear_start": schedule.linear_start,
        "linear_end": schedule.linear_end,
        "sqrt_alphas_cumprod[0]": float(sac[0]),
        "sqrt_alphas_cumprod[-1]": float(sac[-1]),
        "sqrt_one_minus_alphas_cumprod[0]": float(somac[0]),
        "sqrt_one_minus_alphas_cumprod[-1]": float(somac[-1]),
    }

    vae = build_vae(config, BASE_CKPT)
    dataset = AudioDataset(config=config, split="test", waveform_only=False)

    kw = dict(vae=vae, schedule=schedule, scale_factor=scale_factor, seed=SEED)
    slots1 = build_paired_slots(dataset, INDICES, config, **kw)
    slots2 = build_paired_slots(dataset, INDICES, config, **kw)

    zt_recomputed = schedule.q_sample(slots1.z_0, slots1.t, slots1.noise)

    latent = {
        "scale_factor": scale_factor,
        "z_0_shape": list(slots1.z_0.shape),
        "z_0_mean": float(slots1.z_0.mean()),
        "z_0_std": float(slots1.z_0.std()),
        "z_t_mean": float(slots1.z_t.mean()),
        "z_t_std": float(slots1.z_t.std()),
        "t_values": slots1.t.tolist(),
        "z_0_bit_identical_across_builds": bool(torch.equal(slots1.z_0, slots2.z_0)),
        "z_t_bit_identical_across_builds": bool(torch.equal(slots1.z_t, slots2.z_t)),
        "z_t_equals_q_sample": bool(torch.equal(slots1.z_t, zt_recomputed)),
        "z_t_differs_from_pure_noise": not bool(torch.equal(slots1.z_t, slots1.noise)),
        "z_0_hash": tensor_hash(slots1.z_0),
        "z_t_hash": tensor_hash(slots1.z_t),
        "noise_hash": tensor_hash(slots1.noise),
    }

    result = {
        "provenance": {"config": FROZEN_CONFIG, "base_ckpt": BASE_CKPT,
                       "vae_ckpt": VAE_CKPT, "indices": INDICES, "seed": SEED},
        "vae_source_comparison": vae_cmp,
        "schedule": schedule_info,
        "latent": latent,
    }
    with open(f"{OUT}/a1_latent_check.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    ok = (latent["z_0_bit_identical_across_builds"]
          and latent["z_t_bit_identical_across_builds"]
          and latent["z_t_equals_q_sample"]
          and latent["z_t_differs_from_pure_noise"])
    print(f"\nA1 latent check: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
