#!/usr/bin/env python3
"""Scenario-B executable seam proof: every M-Full ladder seam admits a deterministic
mask-induced LoRA transfer (ICASSP-PIVOT).

Builds the dense (1,2,3,5) and pruned U-Nets from the frozen config (random init —
only keys/shapes matter for mappability; weight-value fidelity is already proven by
R5/P4 bit-exactness), recovers per-tensor selections by mirroring
`prune_with_indices` exactly (LAYER_MAP ranked / positional fallback, with the
artifact's documented seam corrections), and checks:

    S1 CHANGED-SET      changed tensors identical at (1,2,3,4) and (1,2,3,1); 238 total;
                        all ndim in {1,2,4}.
    S2 SELECTIONS       every changed tensor gets a deterministic, shape-consistent
                        (out_sel, in_sel) at both budgets — no seam needs learned params.
    S3 LORA-IDENTITY    for EVERY changed 2D/4D tensor at (1,2,3,1): sliced-factor
                        composition == dense dW restricted to (out_sel, in_sel), exact.
    S4 AUX-1D           every changed 1D tensor (bias / GroupNorm-LayerNorm affine)
                        transfers by out_sel slicing, shape-consistent.
    S5 (1,2,3,3)-DROP   input_blocks.10.0.skip_connection.* vanishes (in==out -> Identity):
                        detected; deterministic rule = drop that adapter component.
    S6 CONV-FLAT        audioldm_peft LoRAConv2d stores lora_A flattened (r, in*kh*kw);
                        reshape->slice->flatten equals the 4D slicing path.

Run: .venv/bin/python tests/research/test_scenario_b_seam_transfer.py
"""
from __future__ import annotations

import sys

import torch

from research_pruning.diagnostics.conditioning import FROZEN_CONFIG, load_config
from research_pruning.diagnostics import random_masks as rm
from research_pruning.lora_mask_transfer import (
    delta_w_conv2d, delta_w_linear, slice_lora_conv2d, slice_lora_linear)

PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
DENSE_MULT = [1, 2, 3, 5]
EXPECT_CHANGED = 238

torch.manual_seed(0)


def state_shapes(config, mult):
    model = rm.build_pruned_unet(config, mult)
    sd = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    del model
    return sd


def record_selections(old_shapes, new_shapes, idx_dict):
    """Mirror prune_with_indices, returning index vectors instead of tensors.

    Returns {key: (out_sel, in_sel_or_None, mode)}; in_sel None = input untouched.
    Raises if any changed tensor lacks a shape-consistent deterministic selection.
    """
    sels = {}
    for k, shp_new in new_shapes.items():
        if k not in old_shapes:
            continue
        shp_old = old_shapes[k]
        if shp_old == shp_new:
            continue
        if k in rm.LAYER_MAP:  # CASE 1: ranked
            idx1, idx2 = rm.LAYER_MAP[k]
            out_sel = torch.tensor(idx_dict[idx1][: shp_new[0]], dtype=torch.long)
            in_sel = None
            if len(shp_old) in (2, 4):
                if idx2 is not None:
                    in_sel = torch.tensor(idx_dict[idx2][: shp_new[1]], dtype=torch.long)
                elif shp_old[1] != shp_new[1]:
                    raise AssertionError(f"{k}: input dim changes but map says identity")
            mode = "ranked"
        else:  # CASE 2: positional truncation
            out_sel = torch.arange(shp_new[0])
            in_sel = None
            if len(shp_old) in (2, 4) and shp_old[1] != shp_new[1]:
                in_sel = torch.arange(shp_new[1])
            mode = "positional"
        if len(out_sel) != shp_new[0]:
            raise AssertionError(f"{k}: out_sel len {len(out_sel)} != target {shp_new[0]}")
        if in_sel is not None and len(in_sel) != shp_new[1]:
            raise AssertionError(f"{k}: in_sel len {len(in_sel)} != target {shp_new[1]}")
        sels[k] = (out_sel, in_sel, mode)
    return sels


def check_s1_s2(config, idx_dict, dense_shapes):
    shapes_p4 = state_shapes(config, [1, 2, 3, 4])
    shapes_p1 = state_shapes(config, [1, 2, 3, 1])
    sels_p4 = record_selections(dense_shapes, shapes_p4, idx_dict)
    sels_p1 = record_selections(dense_shapes, shapes_p1, idx_dict)
    same_set = set(sels_p4) == set(sels_p1)
    n = len(sels_p1)
    ndims_ok = all(len(dense_shapes[k]) in (1, 2, 4) for k in sels_p1)
    ranked = sum(1 for v in sels_p1.values() if v[2] == "ranked")
    print(f"    S1 changed set: p4==p1 {same_set}; n={n} (expect {EXPECT_CHANGED}); "
          f"ndims ok {ndims_ok} ({ranked} ranked / {n - ranked} positional)")
    print(f"    S2 selections: every changed tensor mapped, shape-consistent (both budgets)")
    return same_set and n == EXPECT_CHANGED and ndims_ok, sels_p1, shapes_p1


def check_s3_s4(dense_shapes, shapes_p1, sels, rank=4):
    g = torch.Generator().manual_seed(7)
    n2d = n4d = n1d = 0
    for k, (out_sel, in_sel, _mode) in sels.items():
        shp = dense_shapes[k]
        if len(shp) == 1:
            db = torch.randn(shp[0], generator=g)
            if not torch.equal(db[out_sel], db.index_select(0, out_sel)):
                print(f"    S4 FAIL at {k}"); return False
            n1d += 1
            continue
        d_out, d_in = shp[0], shp[1]
        eff_out = out_sel
        eff_in = in_sel  # None = untouched input
        if len(shp) == 2:
            A = torch.randn(rank, d_in, generator=g)
            B = torch.randn(d_out, rank, generator=g)
            A2, B2 = slice_lora_linear(A, B, eff_in, eff_out)
            lhs = delta_w_linear(A2, B2)
            rhs = delta_w_linear(A, B).index_select(0, eff_out)
            if eff_in is not None:
                rhs = rhs.index_select(1, eff_in)
            n2d += 1
        else:
            kh, kw = shp[2], shp[3]
            A = torch.randn(rank, d_in, kh, kw, generator=g)
            B = torch.randn(d_out, rank, 1, 1, generator=g)
            A2, B2 = slice_lora_conv2d(A, B, eff_in, eff_out)
            lhs = delta_w_conv2d(A2, B2)
            rhs = delta_w_conv2d(A, B).index_select(0, eff_out)
            if eff_in is not None:
                rhs = rhs.index_select(1, eff_in)
            n4d += 1
        if not torch.equal(lhs, rhs):
            print(f"    S3 FAIL at {k}"); return False
        if tuple(lhs.shape[:2]) != tuple(shapes_p1[k][:2]):
            print(f"    S3 SHAPE FAIL at {k}: {tuple(lhs.shape[:2])} vs {shapes_p1[k][:2]}")
            return False
    print(f"    S3 LoRA identity exact on all changed 2D ({n2d}) + 4D ({n4d}) tensors")
    print(f"    S4 1D aux transfer shape-consistent on all {n1d} changed 1D tensors")
    return True


def check_s5(config, dense_shapes):
    shapes_p3 = state_shapes(config, [1, 2, 3, 3])
    gone = sorted(k for k in dense_shapes
                  if k.startswith("input_blocks.10.0.skip_connection") and k not in shapes_p3)
    expect = ["input_blocks.10.0.skip_connection.bias",
              "input_blocks.10.0.skip_connection.weight"]
    ok = gone == expect
    print(f"    S5 (1,2,3,3) drops exactly {gone} -> adapter component dropped "
          f"deterministically: {ok}")
    return ok


def check_s6():
    from audioldm_peft.layers import LoRAConv2d
    conv = torch.nn.Conv2d(12, 20, 3, padding=1)
    wrap = LoRAConv2d(conv, rank=4, alpha=8.0)
    A_flat = wrap.lora_A.detach()
    ok_layout = tuple(A_flat.shape) == (4, 12 * 3 * 3) and tuple(wrap.lora_B.shape)[0] == 20
    A4 = A_flat.reshape(4, 12, 3, 3)
    ki = torch.tensor([0, 2, 3, 5, 7, 11])
    ko = torch.arange(10)
    A4s, _ = slice_lora_conv2d(A4, torch.randn(20, 4, 1, 1), ki, None)
    flat_path = A_flat.reshape(4, 12, 3, 3).index_select(1, ki).reshape(4, -1)
    ok_slice = torch.equal(A4s.reshape(4, -1), flat_path) and len(ko) == 10
    print(f"    S6 LoRAConv2d flat layout {tuple(A_flat.shape)} ok {ok_layout}; "
          f"reshape->slice->flatten == 4D path: {ok_slice}")
    return ok_layout and ok_slice


def main():
    config = load_config(FROZEN_CONFIG)
    idx_dict = rm.load_l1_ranking(PKL)
    dense_shapes = state_shapes(config, DENSE_MULT)
    r12, sels_p1, shapes_p1 = check_s1_s2(config, idx_dict, dense_shapes)
    r34 = check_s3_s4(dense_shapes, shapes_p1, sels_p1)
    r5 = check_s5(config, dense_shapes)
    r6 = check_s6()
    ok = r12 and r34 and r5 and r6
    print(f"{'PASS' if ok else 'FAIL'}: S1-S6")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
