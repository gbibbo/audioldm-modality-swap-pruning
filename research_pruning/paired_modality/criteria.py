"""P1/P2/P3 orchestration at the matched gradient-evaluation budget (master plan §4-5).

Generic over the conditioning: the caller supplies `audio_loss_fn` / `text_loss_fn`
closures (built from `research_pruning.diagnostics.conditioning.paired_eps` for the
real model, or simple control losses in tests) and the pre-registered slot sets.
This module spends and *checks* the calibration budget, so P1 and P2/P3 always cost
the same 2B gradient evaluations.

It runs on whatever gated model it is handed. It does NOT itself load the real
pruned/L1 checkpoint or freeze any slot construction — those belong to the M3B/M4
scientific run behind the (still-unfrozen) pilot protocol. Wiring notes for the
real model are at the bottom of this file.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from research_pruning.taylor import (
    ChannelGate, accumulate_taylor, normalize_within_layer,
    combine_mean, combine_max, assert_matched_budget,
)

LayerSaliency = Dict[str, torch.Tensor]


@dataclass
class Criteria:
    """P1/P2/P3 saliencies plus the shared normalized per-modality saliencies and the
    common gradient-evaluation budget actually spent."""
    p1: LayerSaliency          # text-only Taylor (2B text evals)
    p2: LayerSaliency          # paired mean
    p3: LayerSaliency          # swap-robust max
    s_audio_norm: LayerSaliency
    s_text_norm: LayerSaliency
    budget_grad_evals: int


def compute_criteria(gates: Dict[str, ChannelGate],
                     audio_loss_fn: Callable,
                     text_loss_fn: Callable,
                     audio_slots: List,
                     text_slots_p2p3: List,
                     text_slots_p1: List,
                     norm_mode: str = "sum") -> Criteria:
    """Compute P1/P2/P3 with the §5 matched budget.

    * P2/P3 spend ``len(audio_slots)`` audio + ``len(text_slots_p2p3)`` text evals,
      sharing S_a and S_t (no duplicate compute to switch mean vs max).
    * P1 spends ``len(text_slots_p1)`` text evals.
    ``assert_matched_budget`` enforces P1 == audio + text and audio == text (= B).
    """
    budget = assert_matched_budget(
        len(text_slots_p1), len(audio_slots), len(text_slots_p2p3)
    )

    s_a = accumulate_taylor(gates, audio_loss_fn, audio_slots)
    s_t = accumulate_taylor(gates, text_loss_fn, text_slots_p2p3)
    s_a_n = normalize_within_layer(s_a, norm_mode)
    s_t_n = normalize_within_layer(s_t, norm_mode)

    p2 = combine_mean(s_a_n, s_t_n)
    p3 = combine_max(s_a_n, s_t_n)

    s_t_p1 = accumulate_taylor(gates, text_loss_fn, text_slots_p1)
    p1 = normalize_within_layer(s_t_p1, norm_mode)

    return Criteria(p1=p1, p2=p2, p3=p3, s_audio_norm=s_a_n, s_text_norm=s_t_n,
                    budget_grad_evals=budget)


# --- Real-model wiring notes (NOT executed until the pilot protocol is frozen) ---
#
# For the real AudioLDM run, build the gated model and loss closures like this:
#
#   from research_pruning.diagnostics.conditioning import (
#       build_unet, build_clap, build_paired_slots, paired_eps)
#   from research_pruning.taylor import attach_gates
#
#   unet = build_unet(...)                       # the base (1,2,3,5) or pruned model
#   gates = attach_gates(unet, PRUNABLE_LAYER_NAMES)   # the 28 L1 conv layers,
#                                                       # mapped to Conv2d module paths
#   # slots carry (z_t, t, noise) and the paired CLAP audio/text embeddings; a loss
#   # closure runs eps_pred under one modality and returns the diffusion MSE so the
#   # gate gradients populate:
#   def audio_loss_fn(slot): return mse(eps_pred(unet, slot.z_t, slot.t, slot.e_audio), slot.noise)
#   def text_loss_fn(slot):  return mse(eps_pred(unet, slot.z_t, slot.t, slot.e_text),  slot.noise)
#
# PRUNABLE_LAYER_NAMES must be exactly the L1 layer set (research_pruning.diagnostics
# .random_masks.load_l1_ranking) so P0-P3 are structure-matched to the published L1
# pruning (finding 9.4). The slot sets and B are frozen in docs/pilot_protocol.md
# before any saliency is inspected. P1 is scientifically load-bearing — this whole
# path must pass /auditar before any real result is drawn.
