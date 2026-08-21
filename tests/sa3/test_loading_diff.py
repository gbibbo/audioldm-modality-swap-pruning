#!/usr/bin/env python3
"""Pure-Python tests for the Step 0 diff helpers (no model, no torch needed).

    V1 CONFIG-DIFF   flatten/diff finds only_a / only_b / changed keys exactly.
    V2 INDEX-DIFF    key-set and shape mismatches are reported; identical indices -> empty diff.
    V3 INVENTORY     block inventory counts tensors/params per `transformer.layers.N`.

Run: .venv-sa3/bin/python tests/sa3/test_loading_diff.py   (or any python3)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from research_sa3.loading import diff_configs, diff_indices, block_inventory, total_params  # noqa: E402


def main() -> int:
    ok = True
    a = {"model": {"diffusion": {"diffusion_objective": "rectified_flow", "config": {"depth": 20, "embed_dim": 1024}}}}
    b = {"model": {"diffusion": {"diffusion_objective": "rf_denoiser", "config": {"depth": 20, "embed_dim": 1024},
                                 "sampling_distribution_shift_options": {"type": "full"}}}}
    d = diff_configs(a, b)
    v1 = (d["only_a"] == [] and d["only_b"] == ["model.diffusion.sampling_distribution_shift_options.type"]
          and d["changed"] == [("model.diffusion.diffusion_objective", "rectified_flow", "rf_denoiser")])
    print(f"  V1 config diff: {d}"); ok &= v1

    ia = {"model.model.transformer.layers.0.w": ((4, 4), "F32"), "model.model.transformer.layers.1.w": ((4, 4), "F32"), "x": ((2,), "F32")}
    ib = {"model.model.transformer.layers.0.w": ((4, 4), "F32"), "model.model.transformer.layers.1.w": ((4, 8), "F32"), "y": ((2,), "F32")}
    d2 = diff_indices(ia, ib)
    v2 = (d2["only_a"] == ["x"] and d2["only_b"] == ["y"] and d2["shape_mismatch"] == [("model.model.transformer.layers.1.w", (4, 4), (4, 8))]
          and diff_indices(ia, ia) == {"only_a": [], "only_b": [], "shape_mismatch": [], "dtype_mismatch": []})
    print(f"  V2 index diff: {d2}"); ok &= v2

    inv = block_inventory(ia)
    v3 = inv == {0: {"tensors": 1, "params": 16}, 1: {"tensors": 1, "params": 16}} and total_params(ia) == 34
    print(f"  V3 inventory: {inv}, total={total_params(ia)}"); ok &= v3

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
