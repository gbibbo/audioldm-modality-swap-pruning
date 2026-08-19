"""First-order Taylor saliency accumulation and P0-P3 aggregation (master plan §4-5).

Generic and model-agnostic: `accumulate_taylor` takes a `loss_fn(slot)` closure so
the same machinery is exercised by control models in tests and by the real
conditioning path (audio vs text) in `research_pruning.paired_modality`. No
saliency is computed on the real pruned checkpoint here.

Criteria:
    P0  L1 magnitude          data-free per-output-channel weight norm.
    P1  text-only Taylor      S_t from 2B text gradient evaluations (mandatory baseline).
    P2  paired-mean Taylor    (S~_a + S~_t) / 2, from B audio + B text evaluations.
    P3  swap-robust Taylor    max(S~_a, S~_t), sharing P2's S_a and S_t.

`~` denotes within-layer normalization, applied before combining modalities so the
two conditioning modes are on a comparable scale. P1 and P2/P3 both spend 2B
gradient evaluations (the calibration-budget contract, §5).
"""
from typing import Callable, Dict, Iterable, List

import torch

from .gates import ChannelGate, zero_gate_grads


LayerSaliency = Dict[str, torch.Tensor]  # name -> [out_channels]


def accumulate_taylor(gates: Dict[str, ChannelGate],
                      loss_fn: Callable[[object], torch.Tensor],
                      slots: Iterable) -> LayerSaliency:
    """S_c = mean_slots |g_c · dL/dg_c| at g_c = 1.

    `loss_fn(slot)` must run the gated model and return a scalar loss whose graph
    reaches the gates. Gate grads are zeroed before each slot's backward.
    """
    accum: LayerSaliency = {name: torch.zeros(g.gate.numel()) for name, g in gates.items()}
    n = 0
    for slot in slots:
        zero_gate_grads(gates)
        loss = loss_fn(slot)
        loss.backward()
        for name, g in gates.items():
            if g.gate.grad is None:
                raise RuntimeError(f"gate {name} received no gradient; is it on the loss path?")
            accum[name] += (g.gate.detach() * g.gate.grad.detach()).abs()
        n += 1
    if n == 0:
        raise ValueError("no slots provided")
    return {name: v / n for name, v in accum.items()}


def normalize_within_layer(sal: LayerSaliency, mode: str = "sum", eps: float = 1e-12) -> LayerSaliency:
    """Normalize each layer's saliency so modalities are comparable before combining.

    mode: "sum" (channels sum to 1), "max" (max is 1), or "l2" (unit L2 norm). The
    choice is a protocol parameter to be frozen in docs/pilot_protocol.md; "sum" is
    the default (a within-layer importance distribution).
    """
    out: LayerSaliency = {}
    for name, v in sal.items():
        v = v.detach()
        if mode == "sum":
            denom = v.sum()
        elif mode == "max":
            denom = v.max()
        elif mode == "l2":
            denom = v.norm(p=2)
        else:
            raise ValueError(f"unknown normalization mode {mode!r}")
        out[name] = v / (denom + eps)
    return out


def p0_l1_magnitude(convs: Dict[str, torch.nn.Conv2d]) -> LayerSaliency:
    """Data-free P0: per-output-channel L1 norm of the convolution weight."""
    return {name: conv.weight.detach().abs().sum(dim=(1, 2, 3)) for name, conv in convs.items()}


def combine_mean(sa: LayerSaliency, st: LayerSaliency) -> LayerSaliency:
    """P2 paired-mean: (S~_a + S~_t) / 2."""
    _assert_same_layers(sa, st)
    return {name: 0.5 * (sa[name] + st[name]) for name in sa}


def combine_max(sa: LayerSaliency, st: LayerSaliency) -> LayerSaliency:
    """P3 swap-robust: max(S~_a, S~_t)."""
    _assert_same_layers(sa, st)
    return {name: torch.maximum(sa[name], st[name]) for name in sa}


def prune_order(sal: LayerSaliency) -> Dict[str, torch.Tensor]:
    """Per layer, channel indices sorted by ASCENDING saliency (prune lowest first)."""
    return {name: torch.argsort(v, descending=False) for name, v in sal.items()}


def keep_topk(sal: LayerSaliency, k_per_layer: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Per layer, indices of the top-`k` highest-saliency channels (the kept set)."""
    out = {}
    for name, v in sal.items():
        k = k_per_layer[name]
        out[name] = torch.argsort(v, descending=True)[:k].sort().values
    return out


def _assert_same_layers(a: LayerSaliency, b: LayerSaliency) -> None:
    if set(a) != set(b):
        raise ValueError("saliency dicts cover different layer sets")
    for name in a:
        if a[name].shape != b[name].shape:
            raise ValueError(f"channel-count mismatch at {name}: {a[name].shape} vs {b[name].shape}")


def assert_matched_budget(p1_text_evals: int, p2p3_audio_evals: int, p2p3_text_evals: int) -> int:
    """Enforce §5: P1 spends 2B text evals; P2/P3 spend B audio + B text = 2B; equal.

    Returns the common gradient-evaluation count. Raises if they differ.
    """
    p2p3 = p2p3_audio_evals + p2p3_text_evals
    if p1_text_evals != p2p3:
        raise ValueError(
            f"unmatched calibration budget: P1={p1_text_evals} vs "
            f"P2/P3={p2p3} ({p2p3_audio_evals} audio + {p2p3_text_evals} text)"
        )
    if p2p3_audio_evals != p2p3_text_evals:
        raise ValueError(
            f"P2/P3 must use equal audio/text evals, got "
            f"{p2p3_audio_evals} audio vs {p2p3_text_evals} text"
        )
    return p1_text_evals
