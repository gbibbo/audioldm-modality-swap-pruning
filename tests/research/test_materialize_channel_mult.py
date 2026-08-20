#!/usr/bin/env python3
"""Regression tests for the channel_mult-parameterized materializer (CPU queue Q4).

`build_pruned_unet`/`materialize` used to hardcode (1,2,3,1). They now take an optional
`channel_mult` (default (1,2,3,1)). These tests prove the parameterization is correct AND
did not disturb the published-artifact golden path.

    P1 DEFAULT-PARAMS   count_pruned_params() == 145,673,864 (exact) at the default budget.
    P2 MILD-PARAMS      count_pruned_params([1,2,3,4]) == 317,308,040 (exact) — the mild
                        saturation budget (DECISION-V4-04, −23.7 %; REVIEW-003).
    P3 MILD-MATERIALIZE materialize(..., channel_mult=[1,2,3,4]) strict-loads the base
                        weights and runs a forward (valid model at the mild budget).
    P4 R5-EQUALITY      materialize(base, L1 ranking, channel_mult=[1,2,3,1]) equals the
                        published l1_audioldm-m-full_p1.ckpt on ALL 690 tensors, bit-exact
                        — i.e. the EXPLICIT default budget still reproduces the artifact.

Run: .venv/bin/python tests/research/test_materialize_channel_mult.py
"""
from __future__ import annotations

import sys

import torch

from research_pruning.diagnostics.conditioning import load_config, FROZEN_CONFIG, _torch_load
from research_pruning.diagnostics import random_masks as rm

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
L1_CKPT = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
PREFIX = "model.diffusion_model."

EXACT = {(1, 2, 3, 1): 145_673_864, (1, 2, 3, 4): 317_308_040}


def check_p1_default_params():
    n = rm.count_pruned_params(load_config(FROZEN_CONFIG))
    exp = EXACT[(1, 2, 3, 1)]
    print(f"    P1 default params = {n:,} (expect {exp:,})")
    return n == exp


def check_p2_mild_params():
    n = rm.count_pruned_params(load_config(FROZEN_CONFIG), channel_mult=[1, 2, 3, 4])
    exp = EXACT[(1, 2, 3, 4)]
    print(f"    P2 (1,2,3,4) params = {n:,} = {n/1e6:.3f} M (expect {exp:,} = 317.308 M)")
    return n == exp


def check_p3_mild_materialize():
    config = load_config(FROZEN_CONFIG)
    l1 = rm.load_l1_ranking(PKL)
    rankings, _, _ = rm.build_random_null(config, l1, channel_mult=[1, 2, 3, 4])
    base_sd = rm.base_unet_state_dict(BASE_CKPT)
    model = rm.materialize(base_sd, rankings[0], config, channel_mult=[1, 2, 3, 4])  # strict load
    p = sum(x.numel() for x in model.parameters())
    z = torch.randn(2, 8, 256, 16)
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    y = torch.randn(2, 512)
    with torch.no_grad():
        out = model(z, timesteps=t, y=y, context_list=[], context_attn_mask_list=[])
    ok = tuple(out.shape) == (2, 8, 256, 16) and p == EXACT[(1, 2, 3, 4)]
    print(f"    P3 (1,2,3,4) strict-load OK, params={p:,}, forward out={tuple(out.shape)}")
    return ok


def check_p4_r5_equality_explicit():
    config = load_config(FROZEN_CONFIG)
    l1 = rm.load_l1_ranking(PKL)
    obj = _torch_load(L1_CKPT)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    pub = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    base_sd = rm.base_unet_state_dict(BASE_CKPT)
    # EXPLICIT channel_mult=[1,2,3,1] must equal the published artifact bit-for-bit.
    msd = rm.materialize(base_sd, l1, config, channel_mult=[1, 2, 3, 1]).state_dict()
    identical = sum(1 for k in msd
                    if k in pub and pub[k].shape == msd[k].shape
                    and torch.equal(pub[k].float(), msd[k].float()))
    n = len(pub)
    print(f"    P4 identical tensors: {identical}/{n} (materialized={len(msd)})")
    return identical == n == len(msd)


def main() -> int:
    checks = [
        ("P1 DEFAULT-PARAMS", check_p1_default_params),
        ("P2 MILD-PARAMS", check_p2_mild_params),
        ("P3 MILD-MATERIALIZE", check_p3_mild_materialize),
        ("P4 R5-EQUALITY", check_p4_r5_equality_explicit),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as e:
            print(f"    ERROR: {e!r}")
            results[name] = False
    print("\n==== MATERIALIZE channel_mult TESTS ====")
    for name, _ in checks:
        print(f"  {name:<22} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
