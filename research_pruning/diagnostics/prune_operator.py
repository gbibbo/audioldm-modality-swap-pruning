"""Source-agnostic Singh pruning operator with the principled 'identity-when-full' rule.

Generalizes `random_masks.materialize` (which hard-codes the (1,2,3,1) seam corrections and cannot run
at multi-level severities) so the EXACT Singh channel-selection transform can be applied to ANY compatible
source U-Net state (dense RAW or dense EMA) at ANY severity, deriving the input-propagation from a
principle instead of a per-layer hack:

  * CASE 0 (shape match) -> keep base tensor verbatim.
  * CASE 1 (k in layer_map) -> out_idx = ranking(idx1)[:out_k]; in_idx = ranking(idx2)[:in_k] or all.
  * CASE 2 (not in map, shape differs) -> positional [:out_k, :in_k].
  * IDENTITY-WHEN-FULL: a dimension uses the ranking selection ONLY when actually pruned (k < full);
    when k == full it keeps identity order (no reorder). This reproduces the hand-coded
    `IDENTITY_INPUT_LAYERS` correction from a principle and propagates multi-level (b3+b4) pruning
    through the `input_blocks.10` transition that the old materializer mishandled.

Two seam conventions differ ONLY at three decoder concat-seam tensors where the PUBLISHED severity-1 and
severity-2 checkpoints are mutually inconsistent (`cross-checkpoint pruning-convention inconsistency`):
  APRIME (severity-1 convention): out_blocks.0/1.in positional, out_blocks.2 bias ranked.
  BPRIME (severity-2 / published-dp1 convention): out_blocks.0/1.in ranked, out_blocks.2 bias positional.
Oracle-validated (CPU, bit-exact):
  APRIME: RAW->[1,2,3,1] == published p1 (690/690); EMA->[1,2,3,1] == frozen p1_pruned_ema_reconstructed.
  BPRIME: RAW->[1,2,1,1] == published dp1 (688/688).
No single operator can satisfy both published RAW oracles (proven impossible; source inconsistency).
"""
from __future__ import annotations
import torch

from research_pruning.diagnostics import random_masks as rm

SEAM_TENSORS = (
    "output_blocks.0.0.in_layers.2.weight",
    "output_blocks.1.0.in_layers.2.weight",
    "output_blocks.2.0.in_layers.2.bias",
)


def _aprime_map() -> dict:
    """Severity-1 seam convention: drop out_blocks.0/1 in (positional); add out_blocks.2 bias (ranked)."""
    m = dict(rm._REFERENCE_LAYER_MAP)
    for k in rm.POSITIONAL_OUT_LAYERS:      # output_blocks.0/1.0.in_layers.2.weight -> positional
        m.pop(k, None)
    m.update(rm.RANKED_BIAS_OVERRIDES)      # output_blocks.2.0.in_layers.2.bias -> ranked
    # NOTE: rm.IDENTITY_INPUT_LAYERS is intentionally NOT applied; identity-when-full generalizes it.
    return m


def _bprime_map() -> dict:
    """Severity-2/published-dp1 seam convention: verbatim map (out_blocks.0/1 in ranked; bias positional)."""
    return dict(rm._REFERENCE_LAYER_MAP)


LAYER_MAP_APRIME = _aprime_map()
LAYER_MAP_BPRIME = _bprime_map()


def prune_general(old_sd: dict, new_sd: dict, ranking: dict, layer_map: dict) -> dict:
    """Singh CASE0/1/2 + identity-when-full. old_sd/new_sd are relative-key U-Net state dicts."""
    out = {}
    for k, v_new in new_sd.items():
        if k not in old_sd:
            continue
        v_old = old_sd[k]
        if v_old.shape == v_new.shape:            # CASE 0
            out[k] = v_old
            continue
        if k in layer_map:                        # CASE 1
            idx1, idx2 = layer_map[k]
            out_k = v_new.shape[0]
            out_idx = (torch.arange(out_k) if out_k == v_old.shape[0]
                       else torch.as_tensor(ranking[idx1][:out_k]))
            if v_old.ndim in (2, 4):
                if idx2 is None:
                    in_idx = slice(None)
                else:
                    in_k = v_new.shape[1]
                    in_idx = (slice(None) if in_k == v_old.shape[1]
                              else torch.as_tensor(ranking[idx2][:in_k]))
                out[k] = v_old[out_idx][:, in_idx]
            elif v_old.ndim == 1:
                out[k] = v_old[out_idx]
            else:
                out[k] = v_old
        else:                                     # CASE 2 (positional fallback)
            if v_old.ndim == 4:
                out[k] = v_old[:v_new.shape[0], :v_new.shape[1]]
            elif v_old.ndim == 2:
                out[k] = v_old[:v_new.shape[0], :v_new.shape[1]]
            elif v_old.ndim == 1:
                out[k] = v_old[:v_new.shape[0]]
            else:
                out[k] = v_old
    return out


def build_pruned_ema(base_sd: dict, ranking: dict, config: dict, channel_mult, convention: str):
    """Return a UNetModel with `prune_general(base_sd, target[channel_mult])` strict-loaded.

    convention: 'A' (LAYER_MAP_APRIME, method-consistent primary) or 'B' (LAYER_MAP_BPRIME, sensitivity).
    """
    lm = LAYER_MAP_APRIME if convention.upper() == "A" else LAYER_MAP_BPRIME
    model = rm.build_pruned_unet(config, channel_mult).float()
    target_sd = model.state_dict()
    pruned = prune_general(base_sd, target_sd, ranking, lm)
    model.load_state_dict(pruned, strict=True)
    return model
