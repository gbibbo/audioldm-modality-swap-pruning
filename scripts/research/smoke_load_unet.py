#!/usr/bin/env python3
"""M0 acceptance smoke test: rebuild both U-Net architectures and load real weights.

Instantiates `audioldm_train...UNetModel` from the frozen upstream config and
loads the matching `model.diffusion_model.*` weights with `strict=True`, for the
base `(1,2,3,5)` model and for the pruned `(1,2,3,1)` model. A strict load that
reports no missing/unexpected keys is the deterministic-reconstruction evidence
M0 requires.

CPU only. No sampling, no training, no GPU.
"""
from __future__ import annotations

import argparse
import copy
import sys

import torch
import yaml

from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel

def torch_load(path: str):
    """Load a checkpoint on CPU without executing pickled code.

    `mmap=True` keeps memory flat but only exists in torch >= 2.1; the pinned
    upstream environment is torch 1.13.1, so fall back to a normal load there.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
PREFIX = "model.diffusion_model."


def unet_params(config_path: str) -> dict:
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    return copy.deepcopy(cfg["model"]["params"]["unet_config"]["params"])


def diffusion_weights(ckpt_path: str) -> dict:
    obj = torch_load(ckpt_path)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}


def check(ckpt_path: str, channel_mult: list[int], config_path: str) -> bool:
    params = unet_params(config_path)
    params["channel_mult"] = channel_mult
    model = UNetModel(**params)
    n_params = sum(p.numel() for p in model.parameters())

    weights = diffusion_weights(ckpt_path)
    missing, unexpected = model.load_state_dict(weights, strict=True)

    print(f"{ckpt_path}")
    print(f"  channel_mult      {channel_mult}")
    print(f"  built params      {n_params/1e6:.3f} M")
    print(f"  ckpt tensors      {len(weights)}")
    print(f"  missing keys      {len(missing)}")
    print(f"  unexpected keys   {len(unexpected)}")
    ok = not missing and not unexpected
    print(f"  strict load       {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--base", default="data/checkpoints/audioldm-m-full.ckpt")
    ap.add_argument("--pruned", default="data/checkpoints/l1_audioldm-m-full_p1.ckpt")
    args = ap.parse_args()

    results = [
        check(args.base, [1, 2, 3, 5], args.config),
        check(args.pruned, [1, 2, 3, 1], args.config),
    ]
    print(f"\nSMOKE LOAD: {'PASS' if all(results) else 'FAIL'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
