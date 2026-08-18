"""Matched random structured-pruning masks for M3A (CPU-only).

M3A needs `Krand = 20` random structured masks at the `(1,2,3,1)` budget with
EXACTLY the same per-layer channel counts as the public L1 baseline, over the same
28 ranked conv layers, so that random pruning is a fair matched null for L1.

The `prune_with_indices` function and `_REFERENCE_LAYER_MAP` are ported VERBATIM
from the frozen reference
`_external/PruningAudioLDM/scripts/pruned_unet_dict_creation.py`. Two facts that are
easy to get wrong and are handled exactly as the reference does:

* `sorted_indexes_dict.pkl` contains **no counts**. Each of its 28 entries is a full
  **permutation** of that layer's channels (lengths 384/576/960). The number of
  channels each layer keeps (`k`) comes from the **shapes** of the pruned U-Net's
  state_dict, built from the frozen config with `channel_mult=[1,2,3,1]` — NOT from
  the L1 checkpoint.
* The prune keeps the **first `k`** indices of the ranking (`out_idx_full[:out_k]`)
  and propagates the input selection between adjacent layers via the layer map
  (`in_idx_full[:in_k]`). Layers absent from the map fall back to positional
  truncation `v_old[:new0, :new1]`.

## Faithful to the published artifact, not to the reference script (M3-002)

The reference script does NOT reproduce the published `l1_audioldm-m-full_p1.ckpt`:
`materialize(base, L1 ranking)` with `_REFERENCE_LAYER_MAP` gives 686/690 tensors
bit-identical and **4 wrong**. The published artifact is internally INCONSISTENT at
its seams; `LAYER_MAP` below is corrected so that `materialize(L1 ranking)` equals
the published checkpoint **690/690, bit-exact** (test R5). Each deviation was proven
bit-exact against the published tensor (see `scripts/research/verify_l1_bitexact.py`):

| Tensor | Published convention (bit-exact) | Reference script did |
|---|---|---|
| `input_blocks.10.0.in_layers.2.weight` | out=`perm[:192]`, **in=identity** (all 576, no reorder) | in reordered by `input_blocks.9.0.op` ranking → **fixed: idx2=None** |
| `output_blocks.0.0.in_layers.2.weight` | **fully positional** `[:192,:384]` | out=`perm[:192]` (ranked) → **fixed: removed from map** |
| `output_blocks.1.0.in_layers.2.weight` | **fully positional** `[:192,:384]` | out=`perm[:192]` (ranked) → **fixed: removed from map** |
| `output_blocks.2.0.in_layers.2.bias`  | **ranked** `base[perm[:192]]` (perm of the *weight*) | positional `[:192]` (reference comments it out) → **fixed: added, ranked** |

Note the artifact's own inconsistency: `output_blocks.2.0.in_layers.2.weight` keeps
**positional** rows while its **bias** keeps **ranked** rows — bias values are
attached to channels other than the ones whose weight rows were kept. This is a
property of the published pre-recovery baseline (see `docs/m0_baseline_reproduction/`
and the question logged for Arshdeep in PROGRESS), reproduced here exactly because
M3A will diagnose that exact checkpoint.

## Consequence for the random null

Random masks are produced by the SAME corrected materializer with random rankings.
Where the artifact's construction is NOT ranking-driven (the positional/identity
seams above), a random mask coincides with L1 by construction; the null therefore
differs from L1 **only in the ranking-driven output layers** and is matched exactly
everywhere else. Distinctness (test R2) is measured only over
`ranking_driven_layers`. The base WEIGHTS always come from `audioldm-m-full.ckpt`
(base `[1,2,3,5]`); the L1 checkpoint is opened ONLY for the R5 bit-equality test.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import torch

from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
from research_pruning.diagnostics.conditioning import _torch_load, load_config

PRUNED_CHANNEL_MULT = [1, 2, 3, 1]
DIFFUSION_PREFIX = "model.diffusion_model."

# 20 pre-registered deterministic seeds for the matched random null.
MASTER_SEED = 20260818
PREREGISTERED_SEEDS = [MASTER_SEED + i for i in range(20)]

# Ported verbatim from _external/PruningAudioLDM/scripts/pruned_unet_dict_creation.py
_REFERENCE_LAYER_MAP = {
    'input_blocks.7.0.in_layers.2.weight': ('input_blocks.7.0.in_layers.2.weight', None),
    'input_blocks.7.0.out_layers.3.weight': ('input_blocks.7.0.out_layers.3.weight', 'input_blocks.7.0.in_layers.2.weight'),
    'input_blocks.8.0.in_layers.2.weight': ('input_blocks.8.0.in_layers.2.weight', 'input_blocks.7.0.in_layers.2.weight'),
    'input_blocks.8.0.out_layers.3.weight': ('input_blocks.8.0.out_layers.3.weight', 'input_blocks.8.0.in_layers.2.weight'),
    'input_blocks.9.0.op.weight': ('input_blocks.9.0.op.weight', 'input_blocks.8.0.in_layers.2.weight'),
    'input_blocks.10.0.in_layers.2.weight': ('input_blocks.10.0.in_layers.2.weight', 'input_blocks.9.0.op.weight'),
    'input_blocks.10.0.out_layers.3.weight': ('input_blocks.10.0.out_layers.3.weight', 'input_blocks.10.0.in_layers.2.weight'),
    'input_blocks.11.0.in_layers.2.weight': ('input_blocks.11.0.in_layers.2.weight', 'input_blocks.10.0.in_layers.2.weight'),
    'input_blocks.11.0.out_layers.3.weight': ('input_blocks.11.0.out_layers.3.weight', 'input_blocks.11.0.in_layers.2.weight'),
    'middle_block.0.in_layers.2.weight': ('middle_block.0.in_layers.2.weight', 'input_blocks.11.0.in_layers.2.weight'),
    'middle_block.0.out_layers.3.weight': ('middle_block.0.out_layers.3.weight', 'middle_block.0.in_layers.2.weight'),
    'middle_block.2.in_layers.2.weight': ('middle_block.2.in_layers.2.weight', 'middle_block.0.out_layers.3.weight'),
    'middle_block.2.out_layers.3.weight': ('middle_block.2.out_layers.3.weight', 'middle_block.2.in_layers.2.weight'),
    'output_blocks.0.0.in_layers.2.weight': ('output_blocks.0.0.in_layers.2.weight', 'middle_block.2.out_layers.3.weight'),
    'output_blocks.0.0.out_layers.3.weight': ('output_blocks.0.0.out_layers.3.weight', 'output_blocks.0.0.in_layers.2.weight'),
    'output_blocks.1.0.in_layers.2.weight': ('output_blocks.1.0.in_layers.2.weight', 'output_blocks.0.0.out_layers.3.weight'),
    'output_blocks.1.0.out_layers.3.weight': ('output_blocks.1.0.out_layers.3.weight', 'output_blocks.1.0.in_layers.2.weight'),
    'output_blocks.2.0.out_layers.3.weight': ('output_blocks.2.0.out_layers.3.weight', 'output_blocks.2.0.in_layers.2.weight'),
    'output_blocks.2.2.conv.weight': ('output_blocks.2.2.conv.weight', 'output_blocks.2.0.out_layers.3.weight'),
    'output_blocks.3.0.out_layers.3.weight': ('output_blocks.3.0.out_layers.3.weight', 'output_blocks.3.0.in_layers.2.weight'),
    'output_blocks.4.0.out_layers.3.weight': ('output_blocks.4.0.out_layers.3.weight', 'output_blocks.4.0.in_layers.2.weight'),
    'output_blocks.5.0.out_layers.3.weight': ('output_blocks.5.0.out_layers.3.weight', 'output_blocks.5.0.in_layers.2.weight'),
    'output_blocks.5.2.conv.weight': ('output_blocks.5.2.conv.weight', 'output_blocks.5.0.out_layers.3.weight'),
    'input_blocks.7.0.in_layers.2.bias': ('input_blocks.7.0.in_layers.2.weight', None),
    'input_blocks.7.0.out_layers.3.bias': ('input_blocks.7.0.out_layers.3.weight', None),
    'input_blocks.8.0.in_layers.2.bias': ('input_blocks.8.0.in_layers.2.weight', None),
    'input_blocks.8.0.out_layers.3.bias': ('input_blocks.8.0.out_layers.3.weight', None),
    'input_blocks.9.0.op.bias': ('input_blocks.9.0.op.weight', None),
    'input_blocks.10.0.in_layers.2.bias': ('input_blocks.10.0.in_layers.2.weight', None),
    'input_blocks.10.0.out_layers.3.bias': ('input_blocks.10.0.out_layers.3.weight', None),
    'input_blocks.11.0.in_layers.2.bias': ('input_blocks.11.0.in_layers.2.weight', None),
    'input_blocks.11.0.out_layers.3.bias': ('input_blocks.11.0.out_layers.3.weight', None),
    'middle_block.0.in_layers.2.bias': ('middle_block.0.in_layers.2.weight', None),
    'middle_block.0.out_layers.3.bias': ('middle_block.0.out_layers.3.weight', None),
    'middle_block.2.in_layers.2.bias': ('middle_block.2.in_layers.2.weight', None),
    'middle_block.2.out_layers.3.bias': ('middle_block.2.out_layers.3.weight', None),
    'output_blocks.0.0.in_layers.2.bias': ('output_blocks.0.0.in_layers.2.weight', None),
    'output_blocks.0.0.out_layers.3.bias': ('output_blocks.0.0.out_layers.3.weight', None),
    'output_blocks.1.0.in_layers.2.bias': ('output_blocks.1.0.in_layers.2.weight', None),
    'output_blocks.1.0.out_layers.3.bias': ('output_blocks.1.0.out_layers.3.weight', None),
    'output_blocks.2.0.out_layers.3.bias': ('output_blocks.2.0.out_layers.3.weight', None),
    'output_blocks.2.2.conv.bias': ('output_blocks.2.2.conv.weight', None),
    'output_blocks.3.0.out_layers.3.bias': ('output_blocks.3.0.out_layers.3.weight', None),
    'output_blocks.4.0.out_layers.3.bias': ('output_blocks.4.0.out_layers.3.weight', None),
    'output_blocks.5.0.out_layers.3.bias': ('output_blocks.5.0.out_layers.3.weight', None),
    'output_blocks.5.2.conv.bias': ('output_blocks.5.2.conv.weight', None),
}

# --- Corrections that make materialize(L1 ranking) == published ckpt (690/690) ---
# Each is proven bit-exact against l1_audioldm-m-full_p1.ckpt (test R5).
POSITIONAL_OUT_LAYERS = frozenset({
    # published keeps positional rows/cols here; reference wrongly ranked them.
    "output_blocks.0.0.in_layers.2.weight",
    "output_blocks.1.0.in_layers.2.weight",
})
IDENTITY_INPUT_LAYERS = frozenset({
    # published keeps input columns in identity order (no ranking reorder).
    "input_blocks.10.0.in_layers.2.weight",
})
RANKED_BIAS_OVERRIDES = {
    # published ranks this bias by the *weight* ranking, though the weight is positional.
    "output_blocks.2.0.in_layers.2.bias": ("output_blocks.2.0.in_layers.2.weight", None),
}


def _build_artifact_layer_map() -> dict:
    m = dict(_REFERENCE_LAYER_MAP)
    for k in POSITIONAL_OUT_LAYERS:      # drop -> positional fallback (case 2)
        m.pop(k, None)
    for k in IDENTITY_INPUT_LAYERS:      # keep ranked rows, identity columns
        idx1, _ = m[k]
        m[k] = (idx1, None)
    m.update(RANKED_BIAS_OVERRIDES)      # add ranked-bias override
    return m


# The artifact-faithful map used everywhere (materialize, L1 and random masks).
LAYER_MAP = _build_artifact_layer_map()


def prune_with_indices(old_sd, new_sd, idx_dict, layer_map):
    """Ported verbatim from the frozen reference (structured + fallback cases)."""
    pruned_sd = OrderedDict()
    for k, v_new in new_sd.items():
        if k not in old_sd:
            continue
        v_old = old_sd[k]
        # CASE 0: shape match -> keep
        if v_old.shape == v_new.shape:
            pruned_sd[k] = v_old
            continue
        # CASE 1: structured (layer_map)
        if k in layer_map:
            idx1_name, idx2_name = layer_map[k]
            out_k = v_new.shape[0]
            out_idx_full = idx_dict[idx1_name]
            out_idx = out_idx_full[:out_k]
            if v_old.ndim == 4 or v_old.ndim == 2:
                if idx2_name is not None:
                    in_k = v_new.shape[1]
                    in_idx_full = idx_dict[idx2_name]
                    in_idx = in_idx_full[:in_k]
                else:
                    in_idx = slice(None)
                pruned_sd[k] = v_old[out_idx][:, in_idx]
            elif v_old.ndim == 1:
                pruned_sd[k] = v_old[out_idx]
            else:
                pruned_sd[k] = v_old
        # CASE 2: fallback positional truncation
        else:
            if v_old.ndim == 4:
                pruned_sd[k] = v_old[:v_new.shape[0], :v_new.shape[1]]
            elif v_old.ndim == 2:
                pruned_sd[k] = v_old[:v_new.shape[0], :v_new.shape[1]]
            elif v_old.ndim == 1:
                pruned_sd[k] = v_old[:v_new.shape[0]]
            else:
                pruned_sd[k] = v_old
    return pruned_sd


def build_pruned_unet(config: dict) -> UNetModel:
    """UNetModel at the (1,2,3,1) budget, from the frozen config."""
    params = copy.deepcopy(config["model"]["params"]["unet_config"]["params"])
    params["channel_mult"] = list(PRUNED_CHANNEL_MULT)
    return UNetModel(**params)


def base_unet_state_dict(base_ckpt: str) -> dict:
    """The base [1,2,3,5] U-Net weights from audioldm-m-full.ckpt (NEVER the L1 ckpt)."""
    obj = _torch_load(base_ckpt)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {k[len(DIFFUSION_PREFIX):]: v
            for k, v in state.items() if k.startswith(DIFFUSION_PREFIX)}


def load_l1_ranking(pkl_path: str) -> dict:
    """The public L1 ranking: 28 layer keys -> full channel permutation (list)."""
    import pickle
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    return {k: list(v) for k, v in d.items()}


def ranking_full_lengths(l1_ranking: dict) -> dict:
    return {k: len(v) for k, v in l1_ranking.items()}


def kept_counts(config: dict, ranked_layers) -> dict:
    """Per-layer k (channels kept in the OUTPUT dim), from the pruned target shapes."""
    tsd = build_pruned_unet(config).state_dict()
    return {k: int(tsd[k].shape[0]) for k in ranked_layers if k in tsd}


def ranking_driven_layers(config: dict, l1_ranking: dict) -> list[str]:
    """The layers whose OUTPUT rows are actually selected by the ranking (out=perm[:k]
    with k<full) under the artifact-faithful LAYER_MAP.

    These are the ONLY layers where a random mask differs from L1 in its kept
    output channels; the positional/identity seams coincide with L1 by construction.
    This is also the set over which M3B's prune-tail overlap competes. Excludes the
    3 positional-out seams (output_blocks.0/1/2.0.in_layers.2.weight).
    """
    counts = kept_counts(config, list(l1_ranking.keys()))
    full = ranking_full_lengths(l1_ranking)
    out = []
    for k in l1_ranking:
        if k not in counts:
            continue
        entry = LAYER_MAP.get(k)
        ranked_out = entry is not None and entry[0] == k  # idx1 == self → ranked rows
        if ranked_out and counts[k] < full[k]:
            out.append(k)
    return sorted(out)


def random_ranking(seed: int, full_lengths: dict) -> dict:
    """A seeded random permutation per ranked layer (same lengths as the L1 pkl)."""
    g = torch.Generator().manual_seed(int(seed))
    return {k: torch.randperm(n, generator=g).tolist() for k, n in full_lengths.items()}


def materialize(base_sd: dict, ranking: dict, config: dict) -> UNetModel:
    """Build the (1,2,3,1) U-Net and load the structurally-pruned base weights."""
    model = build_pruned_unet(config)
    target_sd = model.state_dict()
    pruned_sd = prune_with_indices(base_sd, target_sd, ranking, LAYER_MAP)
    model.load_state_dict(pruned_sd, strict=True)  # raises if any key missing/misshaped
    model.eval()
    return model


def kept_sets(ranking: dict, counts: dict, layers=None) -> dict:
    """Per-layer SET of kept output channels (order-independent, for comparing masks).

    Pass `layers` (e.g. `ranking_driven_layers(...)`) to restrict the comparison to
    the layers where masks actually differ; the positional/identity seams coincide
    with L1 by construction and would spuriously read as 'identical'.
    """
    keys = layers if layers is not None else list(counts.keys())
    return {k: frozenset(ranking[k][:counts[k]]) for k in keys}


def mask_signature(ranking: dict) -> bytes:
    """Complete canonical fingerprint of a mask = its full random rankings for every
    layer (the raw randomness that fully determines the materialised model)."""
    parts = []
    for k in sorted(ranking):
        parts.append(k + ":" + ",".join(map(str, ranking[k])))
    return "|".join(parts).encode()


def masks_sha256(rankings: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rankings:
        h.update(mask_signature(r))
        h.update(b"\n")
    return h.hexdigest()


def build_random_null(config: dict, l1_ranking: dict, seeds=PREREGISTERED_SEEDS):
    """Return (list of 20 random rankings, kept_counts, sha256 of the mask set)."""
    full = ranking_full_lengths(l1_ranking)
    counts = kept_counts(config, list(l1_ranking.keys()))
    rankings = [random_ranking(s, full) for s in seeds]
    sha = masks_sha256(rankings)
    return rankings, counts, sha
