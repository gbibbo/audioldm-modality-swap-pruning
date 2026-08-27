#!/usr/bin/env python3
"""Tests for the deterministic mask-induced sliced-adapter mapping (research_pruning.sliced_adapter).

    S1 SLICE-IDENTITY   B'@A' == (B@A)[keep_out][:,keep_in] exactly (float64), Linear factors.
    S2 UNCHANGED-BITEXACT  keep=None on both dims -> sliced factors bit-identical to dense.
    S3 BUILD-AUDIT      build_sliced_adapter on a 2-module synthetic adapter (one changed
                        960->192, one unchanged) -> 64/64-style accounting, identity err 0.0,
                        positional keep vectors, unchanged bit-identical.
    S4 GUARDS           a dense factor whose in-dim disagrees with the Linear raises.

Run: .venv/bin/python tests/research/test_sliced_adapter.py
"""
from __future__ import annotations

import sys

import torch

from research_pruning.lora_mask_transfer import slice_lora_linear, delta_w_linear
from research_pruning.sliced_adapter import build_sliced_adapter, summarize_audit

PFX = "model.diffusion_model."


def check_s1_slice_identity():
    g = torch.Generator().manual_seed(0)
    r, do, di = 8, 960, 960
    A = torch.randn(r, di, generator=g, dtype=torch.float64)
    B = torch.randn(do, r, generator=g, dtype=torch.float64)
    keep = torch.arange(192)
    A2, B2 = slice_lora_linear(A, B, keep, keep)
    lhs = delta_w_linear(A2, B2)
    rhs = delta_w_linear(A, B).index_select(0, keep).index_select(1, keep)
    err = float((lhs - rhs).abs().max())
    ok = err == 0.0 and tuple(A2.shape) == (r, 192) and tuple(B2.shape) == (192, r)
    print(f"    S1 slice identity max|Δ|={err:.1e} shapes A{tuple(A2.shape)} B{tuple(B2.shape)}: {ok}")
    return ok


def check_s2_unchanged_bitexact():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(8, 384, generator=g)
    B = torch.randn(384, 8, generator=g)
    A2, B2 = slice_lora_linear(A, B, None, None)
    ok = torch.equal(A2, A) and torch.equal(B2, B)
    print(f"    S2 unchanged bit-identical: {ok}")
    return ok


def _mod(name, r, do, di, seed):
    g = torch.Generator().manual_seed(seed)
    return {PFX + name + ".lora_A": torch.randn(r, di, generator=g),
            PFX + name + ".lora_B": torch.randn(do, r, generator=g)}


def check_s3_build_audit():
    dense_sd = {}
    dense_sd.update(_mod("input_blocks.10.1.transformer_blocks.0.attn1.to_q", 8, 960, 960, 2))  # changed
    dense_sd.update(_mod("input_blocks.4.1.transformer_blocks.0.attn1.to_q", 8, 384, 384, 3))   # unchanged
    dense_shapes = {"input_blocks.10.1.transformer_blocks.0.attn1.to_q": (960, 960),
                    "input_blocks.4.1.transformer_blocks.0.attn1.to_q": (384, 384)}
    pruned_shapes = {"input_blocks.10.1.transformer_blocks.0.attn1.to_q": (192, 192),
                     "input_blocks.4.1.transformer_blocks.0.attn1.to_q": (384, 384)}
    sliced, audit = build_sliced_adapter(dense_sd, dense_shapes, pruned_shapes)
    s = summarize_audit(audit)
    changed = [a for a in audit if a["mapping_mode"] == "positional-slice"][0]
    unchanged = [a for a in audit if a["mapping_mode"] == "identity"][0]
    ok = (s["n_modules"] == 2 and s["n_changed_positional"] == 1 and s["n_unchanged_identity"] == 1
          and s["max_restricted_dW_abs_err_float64"] == 0.0 and s["all_unchanged_bit_identical"]
          and changed["keep_in"]["changed"] and changed["keep_out"]["changed"]
          and changed["sliced_A_shape"] == [8, 192] and changed["sliced_B_shape"] == [192, 8]
          and not unchanged["in_changed"] and not unchanged["out_changed"]
          and len(sliced) == 4)
    print(f"    S3 build audit {s}: {ok}")
    return ok


def check_s4_guards():
    bad = 0
    dense_sd = _mod("m.to_q", 8, 960, 960, 4)
    # dense Linear claims in=512 but adapter A has in=960 -> mismatch raises
    try:
        build_sliced_adapter(dense_sd, {"m.to_q": (960, 512)}, {"m.to_q": (192, 128)})
    except ValueError:
        bad += 1
    # target missing from the U-Net shapes -> raises
    try:
        build_sliced_adapter(dense_sd, {}, {})
    except ValueError:
        bad += 1
    ok = bad == 2
    print(f"    S4 guards raised: {bad}/2: {ok}")
    return ok


def main():
    checks = [check_s1_slice_identity, check_s2_unchanged_bitexact, check_s3_build_audit,
              check_s4_guards]
    res = []
    for c in checks:
        print(f"  {c.__name__}")
        res.append(c())
    ok = all(res)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(res)}/{len(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
