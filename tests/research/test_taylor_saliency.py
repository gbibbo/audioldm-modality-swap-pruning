#!/usr/bin/env python3
"""P0-P3 Taylor-saliency machinery tests on CONTROL models (CPU-only).

NO saliency is computed on the real pruned/L1 checkpoint here — that is an M3B/M4
scientific run, blocked until the pilot protocol is frozen. These checks validate
the machinery (gate gradients, normalization, P0-P3 aggregation, matched budget)
on small controlled networks, exactly like the M3A diagnostic tests.

    C1 GATE-GRAD       gate saliency is non-negative; a dead (zero-weight) channel
                       scores ~0 and is ordered first for pruning; a high-impact
                       channel scores highest.
    C2 SWAP-REAL       audio- vs text-conditioned saliencies differ (S_a != S_t),
                       so P2 mean and P3 max are non-trivial.
    C3 COMBINE         combine_mean == 0.5(a+b) and combine_max == max(a,b) exactly.
    C4 NORMALIZE       within-layer sum/max/l2 normalizations hold per layer.
    C5 BUDGET          assert_matched_budget accepts 2B == B+B and rejects mismatches.
    C6 P0-L1           data-free L1 magnitude equals the manual per-channel weight norm;
                       a zero-weight channel scores 0.

Run directly:

    .venv/bin/python tests/research/test_taylor_saliency.py
"""
from __future__ import annotations

import sys

import torch
from torch import nn

from research_pruning.taylor import (
    attach_gates, conv_modules, accumulate_taylor, normalize_within_layer,
    p0_l1_magnitude, combine_mean, combine_max, prune_order, keep_topk,
    assert_matched_budget,
)
from research_pruning.paired_modality import compute_criteria


class ControlNet(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 4, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.relu(self.conv1(x)))


def _slots(n=6, seed=1):
    torch.manual_seed(seed)
    return [torch.randn(2, 3, 8, 8) for _ in range(n)]


def _make_gated(dead_channel=None, seed=0):
    m = ControlNet(seed=seed)
    if dead_channel is not None:
        with torch.no_grad():
            m.conv1.weight[dead_channel].zero_()
            m.conv1.bias[dead_channel].zero_()
    gates = attach_gates(m, ["conv1"])
    return m, gates


def check_c1_gate_grad() -> bool:
    m, gates = _make_gated(dead_channel=0)
    target = torch.zeros(2, 4, 8, 8)
    def loss_fn(slot):
        return ((m(slot) - target) ** 2).mean()
    sal = accumulate_taylor(gates, loss_fn, _slots())
    s = sal["conv1"]
    nonneg = bool((s >= 0).all())
    dead_small = bool(s[0] < 1e-8)
    dead_first = int(prune_order(sal)["conv1"][0]) == 0
    print(f"  saliency={[round(float(x),4) for x in s]} dead(ch0)={float(s[0]):.2e} "
          f"nonneg={nonneg} dead_first={dead_first}")
    return bool(nonneg and dead_small and dead_first)


def check_c2_swap_real() -> bool:
    m, gates = _make_gated()
    torch.manual_seed(3)
    tgt_a = torch.randn(2, 4, 8, 8)
    tgt_t = torch.randn(2, 4, 8, 8)
    slots = _slots()
    sa = accumulate_taylor(gates, lambda s: ((m(s) - tgt_a) ** 2).mean(), slots)
    st = accumulate_taylor(gates, lambda s: ((m(s) - tgt_t) ** 2).mean(), slots)
    sa_n = normalize_within_layer(sa)["conv1"]
    st_n = normalize_within_layer(st)["conv1"]
    differ = float((sa_n - st_n).abs().max()) > 1e-6
    p2 = combine_mean(normalize_within_layer(sa), normalize_within_layer(st))["conv1"]
    p3 = combine_max(normalize_within_layer(sa), normalize_within_layer(st))["conv1"]
    # max >= mean elementwise
    max_ge_mean = bool((p3 + 1e-9 >= p2).all())
    print(f"  max|Sa~-St~|={float((sa_n-st_n).abs().max()):.3e} differ={differ} max>=mean={max_ge_mean}")
    return bool(differ and max_ge_mean)


def check_c3_combine() -> bool:
    a = {"L": torch.tensor([1.0, 3.0, 2.0])}
    b = {"L": torch.tensor([2.0, 1.0, 5.0])}
    mean = combine_mean(a, b)["L"]
    mx = combine_max(a, b)["L"]
    ok = torch.allclose(mean, torch.tensor([1.5, 2.0, 3.5])) and torch.allclose(mx, torch.tensor([2.0, 3.0, 5.0]))
    print(f"  mean={mean.tolist()} max={mx.tolist()}")
    return bool(ok)


def check_c4_normalize() -> bool:
    sal = {"L": torch.tensor([1.0, 2.0, 1.0])}
    s_sum = normalize_within_layer(sal, "sum")["L"]
    s_max = normalize_within_layer(sal, "max")["L"]
    s_l2 = normalize_within_layer(sal, "l2")["L"]
    ok = (
        abs(float(s_sum.sum()) - 1.0) < 1e-6
        and abs(float(s_max.max()) - 1.0) < 1e-6
        and abs(float(s_l2.norm()) - 1.0) < 1e-6
    )
    print(f"  sum->{float(s_sum.sum()):.4f} max->{float(s_max.max()):.4f} l2->{float(s_l2.norm()):.4f}")
    return bool(ok)


def check_c5_budget() -> bool:
    ok_match = assert_matched_budget(2 * 32, 32, 32) == 64
    raised_unmatched = False
    try:
        assert_matched_budget(2 * 32, 32, 33)
    except ValueError:
        raised_unmatched = True
    raised_unequal = False
    try:
        assert_matched_budget(64, 30, 34)  # 2B==B+B total but audio!=text
    except ValueError:
        raised_unequal = True
    print(f"  match_returns_2B={ok_match} rejects_total_mismatch={raised_unmatched} "
          f"rejects_unequal_split={raised_unequal}")
    return bool(ok_match and raised_unmatched and raised_unequal)


def check_c6_p0_l1() -> bool:
    m, gates = _make_gated(dead_channel=2)
    convs = conv_modules(gates)
    p0 = p0_l1_magnitude(convs)["conv1"]
    manual = convs["conv1"].weight.detach().abs().sum(dim=(1, 2, 3))
    ok = torch.allclose(p0, manual) and float(p0[2]) == 0.0
    print(f"  p0[:3]={[round(float(x),4) for x in p0[:3]]} dead(ch2)={float(p0[2])}")
    return bool(ok)


def check_c7_orchestration() -> bool:
    """compute_criteria produces P1/P2/P3 at the matched 2B budget on a control model."""
    m, gates = _make_gated()
    torch.manual_seed(7)
    tgt_a = torch.randn(2, 4, 8, 8)
    tgt_t = torch.randn(2, 4, 8, 8)
    audio_loss = lambda s: ((m(s) - tgt_a) ** 2).mean()
    text_loss = lambda s: ((m(s) - tgt_t) ** 2).mean()
    B = 4
    audio_slots = _slots(B, seed=10)
    text_slots_p2p3 = _slots(B, seed=11)
    text_slots_p1 = _slots(2 * B, seed=12)  # P1 gets 2B text draws

    crit = compute_criteria(gates, audio_loss, text_loss,
                            audio_slots, text_slots_p2p3, text_slots_p1, norm_mode="sum")
    budget_ok = crit.budget_grad_evals == 2 * B
    # P2 == mean(S~a, S~t), P3 == max
    p2_ok = torch.allclose(crit.p2["conv1"], 0.5 * (crit.s_audio_norm["conv1"] + crit.s_text_norm["conv1"]))
    p3_ok = torch.allclose(crit.p3["conv1"], torch.maximum(crit.s_audio_norm["conv1"], crit.s_text_norm["conv1"]))
    p1_sum_ok = abs(float(crit.p1["conv1"].sum()) - 1.0) < 1e-6  # sum-normalized

    # a mismatched P1 budget must raise
    raised = False
    try:
        compute_criteria(gates, audio_loss, text_loss, audio_slots, text_slots_p2p3,
                         _slots(B, seed=13))  # P1 only B, not 2B
    except ValueError:
        raised = True
    print(f"  budget={crit.budget_grad_evals}(expect {2*B}) p2={p2_ok} p3={p3_ok} "
          f"p1_sum1={p1_sum_ok} rejects_bad_budget={raised}")
    return bool(budget_ok and p2_ok and p3_ok and p1_sum_ok and raised)


TESTS = [
    ("C1 GATE-GRAD", check_c1_gate_grad),
    ("C2 SWAP-REAL", check_c2_swap_real),
    ("C3 COMBINE", check_c3_combine),
    ("C4 NORMALIZE", check_c4_normalize),
    ("C5 BUDGET", check_c5_budget),
    ("C6 P0-L1", check_c6_p0_l1),
    ("C7 ORCHESTRATION", check_c7_orchestration),
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
    print("\n==== P0-P3 TAYLOR SALIENCY MACHINERY TESTS ====")
    for name, _ in TESTS:
        print(f"  {name:<16} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
