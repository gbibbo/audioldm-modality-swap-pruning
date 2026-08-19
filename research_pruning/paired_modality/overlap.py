"""Gate B statistic: audio/text saliency disagreement as KEPT-set overlap (master
plan §6, as amended by DECISION-M3B-003).

The master plan states Gate B against the **prune-set** overlap with thresholds
``<= 0.80`` weighted and ``>= 2`` layers ``<= 0.70``. At the ``(1,2,3,1)`` budget each
ranking-driven layer prunes ``p = 768`` of ``N = 960`` channels, so prune-set overlap
is confined by pigeonhole to ``[(2p - N)/p, 1] = [0.75, 1]`` with chance at
``p/N = 0.80``: condition 1 would demand "no more agreement than pure chance" and
condition 2 would be mathematically impossible. Gate B could never PASS as written
(audit finding G1; the draft protocol also carried both definitions at once, finding
G2). Gabriel's amendment (2026-08-19, option (a)) keeps the plan's two numerals and
transfers them onto the **kept-set** definition, where the range is the full ``[0, 1]``
and chance is ``k/N = 0.20``.

Definitions, per ranking-driven layer ``l`` with ``k_l`` kept of ``N_l`` channels::

    overlap_l        = |K_a ∩ K_t| / k_l                        in [0, 1]
    weighted overlap = sum_l k_l * overlap_l / sum_l k_l
    chance_l         = k_l / N_l
    adjusted_l       = (overlap_l - chance_l) / (1 - chance_l)  0 at chance, 1 at identity

``prune_overlap_l`` is reported for transparency only, via the exact identity
``(N_l - 2*k_l + |K_a ∩ K_t|) / p_l`` — it is never the gate.

This module is pure set algebra over saliency dicts. It loads no checkpoint, computes
no saliency, and decides nothing about the real model: the M3B scientific run stays
blocked until ``docs/pilot_protocol.md`` is frozen.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from research_pruning.taylor import keep_topk

LayerSaliency = Dict[str, torch.Tensor]

# Pre-registered Gate B thresholds (master plan numerals, kept-set definition;
# DECISION-M3B-003). Do not tune these after seeing a result.
GATE_B_WEIGHTED_MAX = 0.80
GATE_B_LAYER_MAX = 0.70
GATE_B_MIN_LAYERS = 2


@dataclass(frozen=True)
class LayerOverlap:
    """Kept-set agreement between the audio and text saliency at one layer."""
    name: str
    n_channels: int          # N_l
    k_kept: int              # k_l
    intersection: int        # |K_a ∩ K_t|
    overlap: float           # intersection / k_l
    chance: float            # k_l / N_l
    adjusted: float          # chance-corrected overlap
    prune_overlap: float     # reporting only, never the gate


@dataclass(frozen=True)
class GateBResult:
    per_layer: List[LayerOverlap]
    weighted_overlap: float
    weighted_adjusted: float
    layers_at_or_below_layer_max: List[str]
    weighted_max: float
    layer_max: float
    min_layers: int
    passed: bool

    def summary_lines(self) -> List[str]:
        head = [
            f"Gate B (kept-set overlap, DECISION-M3B-003): "
            f"{'PASS' if self.passed else 'FAIL'}",
            f"  weighted overlap   = {self.weighted_overlap:.4f} "
            f"(threshold <= {self.weighted_max})",
            f"  weighted adjusted  = {self.weighted_adjusted:.4f} (0 = chance)",
            f"  layers <= {self.layer_max}: {len(self.layers_at_or_below_layer_max)} "
            f"(need >= {self.min_layers}) {self.layers_at_or_below_layer_max}",
        ]
        rows = [
            f"  {r.name}: overlap={r.overlap:.4f} adj={r.adjusted:.4f} "
            f"chance={r.chance:.4f} |K_a∩K_t|={r.intersection}/{r.k_kept} "
            f"(prune_overlap={r.prune_overlap:.4f})"
            for r in self.per_layer
        ]
        return head + rows


def kept_set_overlap(keep_a: Dict[str, torch.Tensor],
                     keep_t: Dict[str, torch.Tensor],
                     n_per_layer: Dict[str, int]) -> List[LayerOverlap]:
    """Per-layer kept-set overlap records from two already-selected kept sets.

    ``keep_a``/``keep_t`` map layer -> kept channel indices (as returned by
    ``research_pruning.taylor.keep_topk``). Both must cover the same layers with the
    same per-layer ``k``; that equality is what makes the comparison budget-matched.
    """
    if set(keep_a) != set(keep_t):
        raise ValueError("audio and text kept-set dicts cover different layer sets")
    missing = sorted(set(keep_a) - set(n_per_layer))
    if missing:
        raise ValueError(f"no channel count for layer(s): {missing}")

    records: List[LayerOverlap] = []
    for name in sorted(keep_a):
        a = keep_a[name].reshape(-1).tolist()
        t = keep_t[name].reshape(-1).tolist()
        sa, st = set(a), set(t)
        if len(sa) != len(a) or len(st) != len(t):
            raise ValueError(f"repeated channel index in the kept set at {name}")
        if len(sa) != len(st):
            raise ValueError(
                f"unmatched kept count at {name}: audio {len(sa)} vs text {len(st)} "
                "(P1/P2/P3 must be compared at the same per-layer budget)"
            )
        k = len(sa)
        n = int(n_per_layer[name])
        if k == 0:
            raise ValueError(f"empty kept set at {name}")
        if k > n:
            raise ValueError(f"kept count {k} exceeds channel count {n} at {name}")
        if max(sa | st) >= n:
            raise ValueError(f"channel index out of range at {name} (N={n})")

        inter = len(sa & st)
        chance = k / n
        overlap = inter / k
        # (1 - chance) == 0 only when k == n, i.e. nothing is pruned: no disagreement
        # is definable there, so the adjusted value is reported as 0 (exactly chance).
        adjusted = 0.0 if k == n else (overlap - chance) / (1.0 - chance)
        p = n - k
        prune_overlap = 1.0 if p == 0 else (n - 2 * k + inter) / p

        records.append(LayerOverlap(
            name=name, n_channels=n, k_kept=k, intersection=inter,
            overlap=overlap, chance=chance, adjusted=adjusted,
            prune_overlap=prune_overlap,
        ))
    return records


def weighted_overlap(records: Sequence[LayerOverlap]) -> float:
    """Weighted aggregate ``sum_l k_l * overlap_l / sum_l k_l``.

    Evaluated as ``sum_l |K_a ∩ K_t|_l / sum_l k_l`` — algebraically identical, since
    ``k_l * overlap_l == intersection_l``, but exact: both sums are integers, so the
    result is a single correctly-rounded division instead of an accumulation of
    rounded products. This matters because the gate compares against the threshold
    with ``<=``, and the plan's numerals are reachable exactly (e.g. 32/40 == 0.80).
    """
    if not records:
        raise ValueError("no layers to aggregate")
    total_k = sum(r.k_kept for r in records)
    return sum(r.intersection for r in records) / total_k


def weighted_adjusted(records: Sequence[LayerOverlap]) -> float:
    """Weighted aggregate of the chance-adjusted per-layer values."""
    if not records:
        raise ValueError("no layers to aggregate")
    total_k = sum(r.k_kept for r in records)
    return sum(r.k_kept * r.adjusted for r in records) / total_k


def evaluate_gate_b(saliency_audio: LayerSaliency,
                    saliency_text: LayerSaliency,
                    k_per_layer: Dict[str, int],
                    n_per_layer: Optional[Dict[str, int]] = None,
                    layers: Optional[Sequence[str]] = None,
                    weighted_max: float = GATE_B_WEIGHTED_MAX,
                    layer_max: float = GATE_B_LAYER_MAX,
                    min_layers: int = GATE_B_MIN_LAYERS) -> GateBResult:
    """Evaluate the amended Gate B from audio and text saliencies.

    ``layers`` restricts the comparison to the ranking-driven layers (see
    ``research_pruning.diagnostics.random_masks.ranking_driven_layers``); the
    positional seams have no ranking to disagree on and must be excluded, or the gate
    is diluted by layers that agree by construction. ``n_per_layer`` defaults to each
    saliency vector's length.

    PASS requires weighted kept-set overlap ``<= weighted_max`` AND at least
    ``min_layers`` layers with overlap ``<= layer_max``. Both bounds are inclusive.
    """
    if set(saliency_audio) != set(saliency_text):
        raise ValueError("audio and text saliency cover different layer sets")
    if layers is not None:
        layers = list(layers)
        if not layers:
            raise ValueError("empty layer restriction: Gate B needs at least one layer")
        unknown = sorted(set(layers) - set(saliency_audio))
        if unknown:
            raise ValueError(f"restricted to layer(s) absent from the saliency: {unknown}")
        sal_a = {name: saliency_audio[name] for name in layers}
        sal_t = {name: saliency_text[name] for name in layers}
    else:
        sal_a, sal_t = dict(saliency_audio), dict(saliency_text)

    if n_per_layer is None:
        n_per_layer = {name: int(v.reshape(-1).numel()) for name, v in sal_a.items()}
    for name, v in sal_a.items():
        n = int(v.reshape(-1).numel())
        if n != int(sal_t[name].reshape(-1).numel()):
            raise ValueError(f"channel-count mismatch between modalities at {name}")
        if n != int(n_per_layer[name]):
            raise ValueError(
                f"declared channel count {n_per_layer[name]} != saliency length {n} at {name}"
            )

    keep_a = keep_topk(sal_a, k_per_layer)
    keep_t = keep_topk(sal_t, k_per_layer)
    records = kept_set_overlap(keep_a, keep_t, n_per_layer)

    below = [r.name for r in records if r.overlap <= layer_max]
    w = weighted_overlap(records)
    passed = (w <= weighted_max) and (len(below) >= min_layers)

    return GateBResult(
        per_layer=records,
        weighted_overlap=w,
        weighted_adjusted=weighted_adjusted(records),
        layers_at_or_below_layer_max=below,
        weighted_max=weighted_max,
        layer_max=layer_max,
        min_layers=min_layers,
        passed=passed,
    )
