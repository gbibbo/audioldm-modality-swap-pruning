#!/usr/bin/env python3
"""Exactness tests for mask-induced LoRA transfer (Scenario B core identity).

    T1 LINEAR-IDENTITY   B'A' == (BA)[K_out][:, K_in] exactly (fp32, non-square, r=16).
    T2 CONV-IDENTITY     composed conv update sliced == slice of composed update.
    T3 NESTING           slicing to K1 then K2⊆K1 == slicing to K2 directly (nested ladder).
    T4 ONE-SIDED         only-in / only-out pruning (None on the unpruned side).
    T5 FP16-EXACT        index_select is exact in fp16 (no arithmetic on values).
    T6 REJECTS           duplicate / out-of-range / empty index vectors raise.

Run: .venv/bin/python tests/research/test_lora_mask_transfer.py
"""
from __future__ import annotations

import sys

import torch

from research_pruning.lora_mask_transfer import (
    delta_w_conv2d, delta_w_linear, slice_lora_conv2d, slice_lora_linear)

torch.manual_seed(0)


def _keep(n, k, g):
    return torch.randperm(n, generator=g)[:k].sort().values


def check_t1_linear_identity():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(16, 448, generator=g)
    B = torch.randn(704, 16, generator=g)
    ki, ko = _keep(448, 320, g), _keep(704, 512, g)
    A2, B2 = slice_lora_linear(A, B, ki, ko)
    lhs = delta_w_linear(A2, B2)
    rhs = delta_w_linear(A, B).index_select(0, ko).index_select(1, ki)
    ok = torch.equal(lhs, rhs)
    print(f"    T1 linear identity exact: {ok} (max|diff|={float((lhs-rhs).abs().max()):.3e})")
    return ok


def check_t2_conv_identity():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(16, 192, 3, 3, generator=g)
    B = torch.randn(256, 16, 1, 1, generator=g)
    ki, ko = _keep(192, 128, g), _keep(256, 160, g)
    A2, B2 = slice_lora_conv2d(A, B, ki, ko)
    lhs = delta_w_conv2d(A2, B2)
    rhs = delta_w_conv2d(A, B).index_select(0, ko).index_select(1, ki)
    ok = torch.equal(lhs, rhs)
    print(f"    T2 conv identity exact:   {ok} (max|diff|={float((lhs-rhs).abs().max()):.3e})")
    return ok


def check_t3_nesting():
    g = torch.Generator().manual_seed(3)
    A = torch.randn(16, 400, generator=g)
    B = torch.randn(600, 16, generator=g)
    ki1, ko1 = _keep(400, 300, g), _keep(600, 450, g)
    # K2 as subsets of K1, expressed both absolutely and relative to K1.
    sel_i = _keep(300, 200, g)
    sel_o = _keep(450, 300, g)
    ki2_abs, ko2_abs = ki1[sel_i], ko1[sel_o]
    A1, B1 = slice_lora_linear(A, B, ki1, ko1)
    A12, B12 = slice_lora_linear(A1, B1, sel_i, sel_o)          # two-step (ladder)
    A2, B2 = slice_lora_linear(A, B, ki2_abs, ko2_abs)          # direct
    ok = torch.equal(A12, A2) and torch.equal(B12, B2)
    print(f"    T3 nested-ladder composition exact: {ok}")
    return ok


def check_t4_one_sided():
    g = torch.Generator().manual_seed(4)
    A = torch.randn(16, 100, generator=g)
    B = torch.randn(200, 16, generator=g)
    ko = _keep(200, 150, g)
    A2, B2 = slice_lora_linear(A, B, None, ko)
    ok_out = torch.equal(A2, A) and torch.equal(delta_w_linear(A2, B2),
                                                delta_w_linear(A, B).index_select(0, ko))
    ki = _keep(100, 60, g)
    A3, B3 = slice_lora_linear(A, B, ki, None)
    ok_in = torch.equal(B3, B) and torch.equal(delta_w_linear(A3, B3),
                                               delta_w_linear(A, B).index_select(1, ki))
    print(f"    T4 one-sided (out-only {ok_out}, in-only {ok_in})")
    return ok_out and ok_in


def check_t5_fp16_exact():
    g = torch.Generator().manual_seed(5)
    A = torch.randn(8, 64, generator=g).half()
    B = torch.randn(96, 8, generator=g).half()
    ki, ko = _keep(64, 40, g), _keep(96, 48, g)
    A2, B2 = slice_lora_linear(A, B, ki, ko)
    ok = torch.equal(A2, A.index_select(1, ki)) and torch.equal(B2, B.index_select(0, ko))
    print(f"    T5 fp16 slicing bit-exact: {ok}")
    return ok


def check_t6_rejects():
    A, B = torch.zeros(4, 10), torch.zeros(12, 4)
    bad = 0
    for ki, ko in [(torch.tensor([1, 1]), None), (torch.tensor([10]), None),
                   (torch.tensor([], dtype=torch.long), None), (None, torch.tensor([12]))]:
        try:
            slice_lora_linear(A, B, ki, ko)
        except ValueError:
            bad += 1
    print(f"    T6 invalid index vectors rejected: {bad}/4")
    return bad == 4


def main():
    checks = [check_t1_linear_identity, check_t2_conv_identity, check_t3_nesting,
              check_t4_one_sided, check_t5_fp16_exact, check_t6_rejects]
    results = []
    for c in checks:
        print(f"  {c.__name__}")
        results.append(c())
    ok = all(results)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(results)}/{len(results)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
