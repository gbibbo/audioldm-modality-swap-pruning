#!/usr/bin/env python3
"""M1 LoRA-layer tests (CPU-only).

    L1 LINEAR-MERGE   unmerged == merged == unmerge-restored for LoRALinear.
    L2 CONV-MERGE     same for LoRAConv2d; also proves the factorised forward (F5)
                      equals the materialised-delta path, since `merge()` adds the
                      materialised delta_weight and the unmerged forward is the
                      factorised conv2d(x,A)->conv2d(.,B). Agreement == equivalence.
    L3 CONV-FACTOR    factorised forward matches an explicit reference built from
                      the full materialised kernel (independent of merge()).

Run directly (no pytest dependency in this environment):

    .venv/bin/python tests/research/test_lora_layers.py

Exit code 0 iff all checks pass.
"""
from __future__ import annotations

import sys

import torch
from torch import nn
from torch.nn import functional as F

from audioldm_peft.layers import LoRALinear, LoRAConv2d


def _nudge(module):
    with torch.no_grad():
        module.lora_B.normal_(0, 0.05)


def check_l1_linear_merge() -> bool:
    torch.manual_seed(0)
    base = nn.Linear(7, 5)
    layer = LoRALinear(base, rank=3, alpha=6, dropout=0.0)
    _nudge(layer)
    x = torch.randn(4, 7)
    y_unmerged = layer(x)
    layer.merge()
    y_merged = layer(x)
    ok_merge = torch.allclose(y_unmerged, y_merged, atol=1e-5, rtol=1e-5)
    layer.unmerge()
    y_restored = layer(x)
    ok_restore = torch.allclose(y_unmerged, y_restored, atol=1e-5, rtol=1e-5)
    print(f"  max|unmerged-merged|={ (y_unmerged-y_merged).abs().max():.2e}  "
          f"max|unmerged-restored|={ (y_unmerged-y_restored).abs().max():.2e}")
    return bool(ok_merge and ok_restore)


def check_l2_conv_merge() -> bool:
    torch.manual_seed(0)
    base = nn.Conv2d(4, 6, 3, padding=1)
    layer = LoRAConv2d(base, rank=3, alpha=6, dropout=0.0)
    _nudge(layer)
    x = torch.randn(2, 4, 12, 12)
    y_unmerged = layer(x)
    layer.merge()
    y_merged = layer(x)
    ok_merge = torch.allclose(y_unmerged, y_merged, atol=1e-5, rtol=1e-5)
    layer.unmerge()
    y_restored = layer(x)
    ok_restore = torch.allclose(y_unmerged, y_restored, atol=1e-5, rtol=1e-5)
    print(f"  max|unmerged-merged|={ (y_unmerged-y_merged).abs().max():.2e}  "
          f"max|unmerged-restored|={ (y_unmerged-y_restored).abs().max():.2e}")
    return bool(ok_merge and ok_restore)


def check_l3_conv_factorised() -> bool:
    """Factorised forward must equal base_out + conv2d(x, materialised delta)."""
    torch.manual_seed(1)
    base = nn.Conv2d(5, 7, 3, stride=1, padding=1)
    layer = LoRAConv2d(base, rank=4, alpha=8, dropout=0.0)
    _nudge(layer)
    x = torch.randn(2, 5, 10, 10)
    got = layer(x)  # factorised
    with torch.no_grad():
        ref = base(x) + F.conv2d(x, layer.delta_weight(), bias=None, padding=1)
    diff = (got - ref).abs().max().item()
    print(f"  max|factorised-materialised|={diff:.2e}")
    return diff < 1e-5


TESTS = [
    ("L1 LINEAR-MERGE", check_l1_linear_merge),
    ("L2 CONV-MERGE", check_l2_conv_merge),
    ("L3 CONV-FACTOR", check_l3_conv_factorised),
]


def main() -> int:
    results = {}
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPTION: {exc}")
            results[name] = False
    print("\n==== M1 LoRA-LAYER TESTS ====")
    for name, _ in TESTS:
        print(f"  {name:<16} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
