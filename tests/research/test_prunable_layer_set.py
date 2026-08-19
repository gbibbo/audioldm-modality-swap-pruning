#!/usr/bin/env python3
"""P0-P3 prunable-layer-set verification on the REAL base U-Net (CPU-only) — M3B-001.

Structural / plumbing only. NO saliency and NO P0-P3 ranking is computed on the
real model; that is the M3B/M4 scientific run, blocked until the pilot protocol is
frozen. These checks confirm the criteria machinery targets exactly the L1-ranked
layer set (finding 9.4).

    V1 LAYER-SET     the 28 public L1 ranking keys resolve to Conv2d modules of the
                     base (1,2,3,5) U-Net with out_channels == the ranking full length.
    V2 GATE-INVARIANT attaching channel gates (init 1.0) to those 28 layers leaves the
                     U-Net output bit-identical; remove_gates restores the bare convs.

Run directly:

    .venv/bin/python tests/research/test_prunable_layer_set.py
"""
from __future__ import annotations

import copy
import sys

import torch
import yaml
from torch import nn

from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
from research_pruning.diagnostics.random_masks import load_l1_ranking
from research_pruning.taylor import (
    l1_prunable_layer_names, verify_prunable_layers, attach_gates, remove_gates,
    ChannelGate,
)

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
RANKING_PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"


def _base_unet():
    with open(CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    params = copy.deepcopy(cfg["model"]["params"]["unet_config"]["params"])
    params["channel_mult"] = [1, 2, 3, 5]  # saliency is computed on the UNPRUNED base
    return UNetModel(**params)


def check_v1_layer_set() -> bool:
    ranking = load_l1_ranking(RANKING_PKL)
    unet = _base_unet()
    mapping = verify_prunable_layers(unet, ranking)  # raises on any mismatch
    names = l1_prunable_layer_names(ranking)
    ok = len(mapping) == 28 and len(names) == 28
    widths = sorted(set(mapping.values()))
    print(f"  verified {len(mapping)} L1 conv layers; channel widths present: {widths}")
    return bool(ok)


def check_v2_gate_invariant() -> bool:
    ranking = load_l1_ranking(RANKING_PKL)
    unet = _base_unet()
    names = l1_prunable_layer_names(ranking)
    x = torch.randn(1, 8, 256, 16)
    t = torch.randint(0, 1000, (1,))
    y = torch.randn(1, 512)
    with torch.no_grad():
        y0 = unet(x, t, y=y, context_list=[], context_attn_mask_list=[])
    gates = attach_gates(unet, names)
    n_gates = len(gates)
    all_conv = all(isinstance(g, ChannelGate) for g in gates.values())
    with torch.no_grad():
        y1 = unet(x, t, y=y, context_list=[], context_attn_mask_list=[])
    diff = (y0 - y1).abs().max().item()
    remove_gates(unet, gates)
    restored = all(
        not isinstance(dict(unet.named_modules()).get(n), ChannelGate) for n in names
    )
    print(f"  gates={n_gates} all_ChannelGate={all_conv} max|gated-ungated|={diff:.2e} restored={restored}")
    return bool(n_gates == 28 and all_conv and diff == 0.0 and restored)


TESTS = [
    ("V1 LAYER-SET", check_v1_layer_set),
    ("V2 GATE-INVARIANT", check_v2_gate_invariant),
]


def main() -> int:
    results = {}
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False
    print("\n==== M3B-001 PRUNABLE LAYER SET (REAL U-NET) ====")
    for name, _ in TESTS:
        print(f"  {name:<20} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
