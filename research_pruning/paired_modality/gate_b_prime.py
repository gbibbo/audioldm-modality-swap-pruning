"""Gate B' machinery (RQ3'): per-slot saliency storage, CPU recomposition, and the
null-split kept-set-overlap distribution.

Plan v4 §6, Gate B': an intervention changes the mask iff the kept-set overlap between the
natural P1 mask and the P1 intervention variant falls **below the 5th percentile** of the
null distribution of `overlap(P1-half_i, P1-half_j)` over >= 1000 random natural
half-splits of matched size and budget. The null is built from STORED per-slot saliency
contributions, so the >=1000 splits reuse a single calibration pass (no extra gradient
evaluations) — that is the reason the M3B saliency run must persist per-slot contributions.

All CPU. `per_slot_saliency` is the only piece that needs the model+gates (done on GPU in
the real run); everything downstream (recomposition, splits, overlaps, the gate) is pure
tensor arithmetic on the stored contributions and runs on the free CPU Studio.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from research_pruning.taylor import keep_topk
from research_pruning.taylor.gates import zero_gate_grads
from research_pruning.paired_modality.overlap import kept_set_overlap, weighted_overlap

PerSlot = Dict[str, torch.Tensor]      # layer -> (n_slots, n_channels)
LayerSaliency = Dict[str, torch.Tensor]


def per_slot_saliency(gates, loss_fn: Callable, slots: Iterable) -> PerSlot:
    """Store, per gated layer, the |g_c · dL/dg_c| contribution of EVERY slot.

    Same per-slot term as `accumulate_taylor`, but kept separately instead of summed, so
    Gate B' can recompute saliency on arbitrary slot subsets. Returns {layer: (S, C)}.
    """
    contribs: Dict[str, List[torch.Tensor]] = {name: [] for name in gates}
    for slot in slots:
        zero_gate_grads(gates)
        loss = loss_fn(slot)
        loss.backward()
        for name, g in gates.items():
            if g.gate.grad is None:
                raise RuntimeError(f"gate {name} received no gradient; is it on the loss path?")
            contribs[name].append((g.gate.detach() * g.gate.grad.detach()).abs().clone())
    if not next(iter(contribs.values()), None):
        raise ValueError("no slots provided")
    return {name: torch.stack(v, dim=0) for name, v in contribs.items()}


def recompose_mean(per_slot: PerSlot, idx: Optional[torch.Tensor] = None) -> LayerSaliency:
    """Mean saliency over all slots (idx=None) or a slot subset. idx=all reproduces
    `accumulate_taylor` exactly."""
    out: LayerSaliency = {}
    for name, mat in per_slot.items():
        m = mat if idx is None else mat.index_select(0, torch.as_tensor(idx, dtype=torch.long))
        out[name] = m.mean(dim=0)
    return out


def _weighted_overlap_between(sal_a: LayerSaliency, sal_b: LayerSaliency,
                              k_per_layer: Dict[str, int],
                              n_per_layer: Dict[str, int]) -> float:
    keep_a = keep_topk(sal_a, k_per_layer)
    keep_b = keep_topk(sal_b, k_per_layer)
    return weighted_overlap(kept_set_overlap(keep_a, keep_b, n_per_layer))


def overlap_between_saliencies(sal_a, sal_b, k_per_layer, n_per_layer) -> float:
    """Public: weighted kept-set overlap between two saliency maps at a fixed budget."""
    return _weighted_overlap_between(sal_a, sal_b, k_per_layer, n_per_layer)


def null_split_overlaps(per_slot: PerSlot, k_per_layer: Dict[str, int],
                        n_per_layer: Dict[str, int], n_splits: int = 1000,
                        seed: int = 20260818) -> np.ndarray:
    """Null distribution of overlap(P1-half_i, P1-half_j) over `n_splits` random halves."""
    n_slots = next(iter(per_slot.values())).shape[0]
    if n_slots < 4:
        raise ValueError(f"need >= 4 slots to split; got {n_slots}")
    half = n_slots // 2
    rng = np.random.default_rng(seed)
    base = np.arange(n_slots)
    overlaps = np.empty(n_splits, dtype=np.float64)
    for i in range(n_splits):
        perm = rng.permutation(base)
        a = torch.as_tensor(perm[:half], dtype=torch.long)
        b = torch.as_tensor(perm[half:2 * half], dtype=torch.long)
        sal_a = recompose_mean(per_slot, a)
        sal_b = recompose_mean(per_slot, b)
        overlaps[i] = _weighted_overlap_between(sal_a, sal_b, k_per_layer, n_per_layer)
    return overlaps


def gate_b_prime(overlap_observed: float, null_overlaps: np.ndarray,
                 alpha: float = 0.05) -> dict:
    """PASS (mask changed) iff observed overlap < the alpha-percentile of the half-half null.

    p_value = fraction of null overlaps at or below the observed value (a lower observed
    overlap = a bigger change = smaller p).
    """
    null_overlaps = np.asarray(null_overlaps, dtype=np.float64)
    thr = float(np.percentile(null_overlaps, alpha * 100.0))
    p = float((null_overlaps <= overlap_observed).mean())
    return {
        "pass": bool(overlap_observed < thr),
        "observed_overlap": float(overlap_observed),
        "threshold_pctile": thr,
        "alpha": alpha,
        "p_value": p,
        "n_splits": int(null_overlaps.size),
        "null_median": float(np.median(null_overlaps)),
        "null_min": float(null_overlaps.min()),
    }


def save_per_slot(per_slot: PerSlot, path: str, meta: Optional[dict] = None) -> None:
    torch.save({"per_slot": {k: v.cpu() for k, v in per_slot.items()},
                "meta": meta or {}}, path)


def load_per_slot(path: str) -> PerSlot:
    obj = torch.load(path, map_location="cpu")
    return obj["per_slot"] if isinstance(obj, dict) and "per_slot" in obj else obj
