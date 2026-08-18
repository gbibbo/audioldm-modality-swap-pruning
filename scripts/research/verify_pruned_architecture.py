#!/usr/bin/env python3
"""M0 acceptance check: derive the U-Net structural budget from checkpoint tensors.

Reads AudioLDM checkpoints on CPU and recovers `model_channels` and
`channel_mult` from the actual weight shapes, so the base and pruned
architectures are confirmed from the artifacts themselves rather than from a
config file or a README claim.

CPU only. Loads weights with `weights_only=True` and `mmap=True`; it never
constructs or runs a model.

Usage:
    python3 scripts/research/verify_pruned_architecture.py \
        data/checkpoints/audioldm-m-full.ckpt \
        data/checkpoints/l1_audioldm-m-full_p1.ckpt
"""
from __future__ import annotations

import re
import sys

import torch

RESBLOCK_OUT = re.compile(r"(?:^|\.)input_blocks\.(\d+)\.0\.out_layers\.3\.weight$")
STEM = re.compile(r"(?:^|\.)input_blocks\.0\.0\.weight$")


def load_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return obj


def derive(path: str) -> dict:
    sd = load_state_dict(path)

    model_channels = None
    for key, tensor in sd.items():
        if STEM.search(key):
            model_channels = tensor.shape[0]
            break
    if model_channels is None:
        raise SystemExit(f"{path}: no U-Net stem conv found; not an AudioLDM U-Net checkpoint")

    widths: list[int] = []
    for key, tensor in sd.items():
        match = RESBLOCK_OUT.search(key)
        if match:
            widths.append((int(match.group(1)), tensor.shape[0]))
    widths.sort()

    # Collapse consecutive duplicates: each resolution level repeats its width
    # across num_res_blocks, and downsample blocks carry no out_layers.3.
    levels: list[int] = []
    for _, width in widths:
        if not levels or levels[-1] != width:
            levels.append(width)

    if any(width % model_channels for width in levels):
        raise SystemExit(f"{path}: level widths {levels} are not multiples of {model_channels}")

    return {
        "path": path,
        "n_keys": len(sd),
        "model_channels": model_channels,
        "level_widths": levels,
        "channel_mult": [width // model_channels for width in levels],
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    results = [derive(p) for p in argv[1:]]
    for r in results:
        print(f"{r['path']}")
        print(f"  tensors        {r['n_keys']}")
        print(f"  model_channels {r['model_channels']}")
        print(f"  level widths   {r['level_widths']}")
        print(f"  channel_mult   {r['channel_mult']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
