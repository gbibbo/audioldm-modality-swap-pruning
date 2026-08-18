#!/usr/bin/env python3
"""Acceptance check (M3-002): materialize(base, L1 ranking) == published L1 ckpt.

Opens `l1_audioldm-m-full_p1.ckpt` for TENSOR-EQUALITY ONLY (the same class of
check as M0's prerecovery_check). No D_gen/D_mod/R_mod, saliency, or any diagnostic
is computed on it — the M3 pre-registration is not touched.

PASS iff all 690 U-Net tensors of the artifact-faithful materializer are
bit-identical to the published checkpoint. Writes
artifacts/m3_pilot/l1_bitexact_check.json.
"""
from __future__ import annotations

import json
import os
import sys

import torch

from research_pruning.diagnostics.conditioning import load_config, FROZEN_CONFIG, _torch_load
from research_pruning.diagnostics import random_masks as rm

BASE = "data/checkpoints/audioldm-m-full.ckpt"
L1CKPT = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
PREFIX = "model.diffusion_model."
OUT = "artifacts/m3_pilot/l1_bitexact_check.json"


def main() -> int:
    os.makedirs("artifacts/m3_pilot", exist_ok=True)
    config = load_config(FROZEN_CONFIG)

    obj = _torch_load(L1CKPT)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    pub = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}

    l1 = rm.load_l1_ranking(PKL)
    base_sd = rm.base_unet_state_dict(BASE)
    model = rm.materialize(base_sd, l1, config)
    msd = model.state_dict()

    identical, differing = 0, []
    for k in msd:
        if k not in pub:
            differing.append({"tensor": k, "reason": "missing_in_published"})
        elif pub[k].shape != msd[k].shape:
            differing.append({"tensor": k, "reason": "shape",
                              "pub": list(pub[k].shape), "ours": list(msd[k].shape)})
        elif torch.equal(pub[k].float(), msd[k].float()):
            identical += 1
        else:
            d = (pub[k].float() - msd[k].float()).abs().max().item()
            differing.append({"tensor": k, "reason": "value", "max_abs_diff": d})

    result = {
        "published_tensors": len(pub),
        "materialized_tensors": len(msd),
        "identical": identical,
        "differing": differing,
        "bit_exact": identical == len(pub) == len(msd) and not differing,
        "l1_reference_mask_sha256": rm.masks_sha256([l1]),
    }
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nR5 BIT-EXACT: {'PASS' if result['bit_exact'] else 'FAIL'} "
          f"({identical}/{len(pub)} identical)")
    return 0 if result["bit_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
