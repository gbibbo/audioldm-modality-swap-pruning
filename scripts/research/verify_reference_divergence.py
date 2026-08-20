#!/usr/bin/env python3
"""Which tensors does the PUBLIC reconstruction logic get different from the release?

Claim under test (AUDIT-M3-001, restated for external communication): applying the
PruningAudioLDM reference channel-mapping to the base checkpoint plus the published
ranking reproduces **686 of 690** U-Net tensors of `l1_audioldm-m-full_p1.ckpt`, and the
four that differ need per-tensor conventions that the public script does not apply. Our
corrected map reproduces all 690.

Until now that "four tensors" number existed only as prose in the ledger. This script
re-derives it from the artifacts so the claim is reproducible — by us, by a reviewer, and
by the original authors if we send it to them.

**Scope caveat, stated plainly.** `_REFERENCE_LAYER_MAP` in
`research_pruning/diagnostics/random_masks.py` is *our reading* of the reference
implementation's channel mapping, not the upstream script executed verbatim. So the honest
claim is "the public reconstruction logic, as we implement it, diverges on these four
tensors" — which is exactly why the finding is offered to the original authors as a
question with the tensor names attached, rather than asserted as their bug.

Run: .venv/bin/python scripts/research/verify_reference_divergence.py
Exit 0 if the divergence is exactly the four expected tensors AND our map is bit-exact.
"""
from __future__ import annotations

import json
import sys

import torch
import yaml

import research_pruning.diagnostics.random_masks as rm

CONFIG = ("audioldm_train/config/2023_08_23_reproduce_audioldm/"
          "audioldm_original_medium.yaml")
BASE = "data/checkpoints/audioldm-m-full.ckpt"
PUBLISHED = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
RANKING = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
PREFIX = "model.diffusion_model."

EXPECTED_DIVERGENT = {
    "input_blocks.10.0.in_layers.2.weight",    # input columns kept in identity order
    "output_blocks.0.0.in_layers.2.weight",    # output rows kept positionally
    "output_blocks.1.0.in_layers.2.weight",    # output rows kept positionally
    "output_blocks.2.0.in_layers.2.bias",      # bias ranked while its weight is positional
}


def _unet_tensors(path: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}


def _divergence(base_u, target, ranking, layer_map, published) -> list:
    sd = rm.prune_with_indices(base_u, target, ranking, layer_map)
    out = []
    for name, ref in published.items():
        got = sd.get(name)
        if got is None or got.shape != ref.shape or not torch.equal(got, ref):
            out.append(name)
    return sorted(out)


def main() -> int:
    config = yaml.safe_load(open(CONFIG))
    published = _unet_tensors(PUBLISHED)
    base_u = _unet_tensors(BASE)
    ranking = rm.load_l1_ranking(RANKING)
    target = rm.build_pruned_unet(config).state_dict()

    print(f"published U-Net tensors: {len(published)}")

    ref_diff = _divergence(base_u, target, ranking, rm._REFERENCE_LAYER_MAP, published)
    our_diff = _divergence(base_u, target, ranking, rm.LAYER_MAP, published)

    print(f"\nreference mapping (as we implement it): "
          f"{len(published) - len(ref_diff)}/{len(published)} identical, "
          f"{len(ref_diff)} divergent")
    for name in ref_diff:
        print(f"    - {name}  {tuple(published[name].shape)}")
    print(f"\nour corrected mapping: "
          f"{len(published) - len(our_diff)}/{len(published)} identical, "
          f"{len(our_diff)} divergent")

    ok = set(ref_diff) == EXPECTED_DIVERGENT and not our_diff
    if set(ref_diff) != EXPECTED_DIVERGENT:
        print(f"\nUNEXPECTED divergence set!"
              f"\n  missing: {sorted(EXPECTED_DIVERGENT - set(ref_diff))}"
              f"\n  extra:   {sorted(set(ref_diff) - EXPECTED_DIVERGENT)}")
    print(json.dumps({
        "published_tensors": len(published),
        "reference_identical": len(published) - len(ref_diff),
        "reference_divergent": ref_diff,
        "ours_identical": len(published) - len(our_diff),
        "ours_bit_exact": not our_diff,
    }, indent=2, sort_keys=True))
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
