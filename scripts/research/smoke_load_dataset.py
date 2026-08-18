#!/usr/bin/env python3
"""M0 smoke test: build the AudioCaps dataset from the frozen config and read samples.

Exercises metadata resolution, waveform loading and the STFT/mel front end on
CPU. No model, no training, no GPU.

Usage:
    .venv/bin/python scripts/research/smoke_load_dataset.py [--n 3]
"""
from __future__ import annotations

import argparse
import sys

import torch
import yaml

from audioldm_train.utilities.data.dataset import AudioDataset

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"


def describe(value) -> str:
    if torch.is_tensor(value):
        return f"tensor{tuple(value.shape)} {value.dtype}"
    if isinstance(value, str):
        return f"str {value[:48]!r}"
    return f"{type(value).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    dataset = AudioDataset(config=config, split=args.split, waveform_only=False)
    print(f"split            {args.split}")
    print(f"dataset size     {len(dataset)}")

    for i in range(min(args.n, len(dataset))):
        sample = dataset[i]
        if i == 0:
            print("first sample keys:")
            for key in sorted(sample):
                print(f"  {key:<24} {describe(sample[key])}")
        else:
            print(f"  sample {i} loaded ok")

    print("\nSMOKE DATASET: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
