"""Taylor saliency criteria: the faithful text-only P1 baseline and the shared
channel-gate machinery used by P2/P3.

STATUS: machinery implemented and CONTROL-MODEL tested (see
tests/research/test_taylor_saliency.py). It computes NO saliency on the real
pruned/L1 checkpoint — that is an M3B/M4 scientific run, blocked until the pilot
protocol is frozen. P1 is scientifically load-bearing (the master plan makes any
cross-modal claim depend on a correctly implemented P1); this code must pass
`/auditar` review before any real use.
"""
from .gates import ChannelGate, attach_gates, remove_gates, conv_modules, zero_gate_grads
from .saliency import (
    accumulate_taylor, normalize_within_layer, p0_l1_magnitude,
    p0_importance, P0_CONVENTION,
    combine_mean, combine_max, prune_order, keep_topk, assert_matched_budget,
)
from .layer_set import l1_prunable_layer_names, verify_prunable_layers, load_and_verify

__all__ = [
    "ChannelGate", "attach_gates", "remove_gates", "conv_modules", "zero_gate_grads",
    "accumulate_taylor", "normalize_within_layer", "p0_l1_magnitude",
    "p0_importance", "P0_CONVENTION",
    "combine_mean", "combine_max", "prune_order", "keep_topk", "assert_matched_budget",
    "l1_prunable_layer_names", "verify_prunable_layers", "load_and_verify",
]
