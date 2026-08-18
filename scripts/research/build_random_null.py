#!/usr/bin/env python3
"""Persist the M3A matched random-null record (seeds + per-layer k + sha256).

The 20 masks themselves are reproducible from the pre-registered seeds, so only
the seeds, the per-layer kept counts (derived from the pruned target shapes), the
pkl full lengths, and the sha256 of the mask set are stored. Nothing here loads or
touches `l1_audioldm-m-full_p1.ckpt`.

Writes artifacts/m3_pilot/random_null_masks.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from research_pruning.diagnostics.conditioning import load_config, FROZEN_CONFIG
from research_pruning.diagnostics import random_masks as rm

PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
OUT = "artifacts/m3_pilot/random_null_masks.json"


def main() -> int:
    os.makedirs("artifacts/m3_pilot", exist_ok=True)
    config = load_config(FROZEN_CONFIG)
    l1 = rm.load_l1_ranking(PKL)
    full = rm.ranking_full_lengths(l1)
    rankings, counts, sha = rm.build_random_null(config, l1)
    driven = rm.ranking_driven_layers(config, l1)

    # L1 fingerprint (full-ranking hash) — computed from the pkl, never from the ckpt.
    l1_sig = rm.masks_sha256([l1])

    record = {
        "krand": len(rankings),
        "preregistered_seeds": rm.PREREGISTERED_SEEDS,
        "master_seed": rm.MASTER_SEED,
        "ranked_layers": sorted(counts.keys()),
        "kept_counts_per_layer": counts,
        "full_lengths_per_layer": full,
        "total_kept_channels_per_mask": sum(counts.values()),
        "k_histogram": {str(k): sum(1 for v in counts.values() if v == k)
                        for k in sorted(set(counts.values()))},
        "ranking_driven_layers": driven,
        "n_ranking_driven_layers": len(driven),
        "positional_out_seams": sorted(rm.POSITIONAL_OUT_LAYERS),
        "identity_input_seams": sorted(rm.IDENTITY_INPUT_LAYERS),
        "ranked_bias_overrides": sorted(rm.RANKED_BIAS_OVERRIDES.keys()),
        "random_masks_sha256": sha,
        "l1_reference_mask_sha256": l1_sig,
        "pruned_channel_mult": rm.PRUNED_CHANNEL_MULT,
        "weight_source": "data/checkpoints/audioldm-m-full.ckpt (base [1,2,3,5]); L1 ckpt opened only for R5 equality",
        "mechanic": ("prune_with_indices ported from _external/PruningAudioLDM; "
                     "LAYER_MAP corrected to reproduce the published artifact 690/690 (M3-002)"),
    }
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps({k: record[k] for k in
                      ["krand", "k_histogram", "total_kept_channels_per_mask",
                       "random_masks_sha256", "l1_reference_mask_sha256"]}, indent=2))
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
