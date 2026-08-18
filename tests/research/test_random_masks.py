#!/usr/bin/env python3
"""Tests for the matched random structured-pruning masks (M3A, CPU-only).

    R1 PER-LAYER k   every mask keeps exactly the per-layer k derived from the
                     pruned target SHAPES (not from the L1 checkpoint), with valid
                     in-range indices.
    R2 DISTINCT      the 20 random masks are distinct from each other and from L1
                     (compared as kept-channel SETS at the genuinely pruned layers).
    R3 REPRODUCIBLE  a mask is bit-identical when regenerated from its seed.
    R4 MATERIALISE   a random mask materialises to a (1,2,3,1) U-Net that loads
                     strict=True and forwards at the correct latent shape, using
                     the BASE checkpoint as weight source (never the L1 ckpt).

Run: .venv/bin/python tests/research/test_random_masks.py
"""
from __future__ import annotations

import sys

import torch

from research_pruning.diagnostics.conditioning import load_config, FROZEN_CONFIG
from research_pruning.diagnostics import random_masks as rm

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"

_CTX = {}


def context():
    if _CTX:
        return _CTX
    config = load_config(FROZEN_CONFIG)
    l1 = rm.load_l1_ranking(PKL)
    full = rm.ranking_full_lengths(l1)
    counts = rm.kept_counts(config, list(l1.keys()))
    rankings, _, sha = rm.build_random_null(config, l1)
    _CTX.update(config=config, l1=l1, full=full, counts=counts,
                rankings=rankings, sha=sha)
    return _CTX


def check_r1_per_layer_k():
    ctx = context()
    counts, full, rankings = ctx["counts"], ctx["full"], ctx["rankings"]
    ok = True
    for mi, r in enumerate(rankings):
        for layer, k in counts.items():
            kept = r[layer][:k]
            if len(kept) != k or len(set(kept)) != k:
                ok = False
            if any(idx < 0 or idx >= full[layer] for idx in kept):
                ok = False
    print(f"    R1 20 masks x {len(counts)} layers; k histogram="
          f"{ {kk: sum(1 for v in counts.values() if v==kk) for kk in sorted(set(counts.values()))} }")
    print(f"    R1 total kept channels/mask = {sum(counts.values())}")
    print(f"    R1 {'ok ' if ok else 'FAIL'} every mask respects per-layer k (from shapes) with valid indices")
    return ok


def check_r2_distinct():
    ctx = context()
    counts, l1, rankings = ctx["counts"], ctx["l1"], ctx["rankings"]
    l1_sets = rm.kept_sets(l1, counts)
    rand_sets = [rm.kept_sets(r, counts) for r in rankings]

    def differ(a, b):
        return any(a[k] != b[k] for k in counts)

    vs_l1 = all(differ(rs, l1_sets) for rs in rand_sets)
    pairwise = all(
        differ(rand_sets[i], rand_sets[j])
        for i in range(len(rand_sets)) for j in range(i + 1, len(rand_sets))
    )
    n_pruned_layers = sum(1 for k in counts if counts[k] < ctx["full"][k])
    print(f"    R2 masks differ from L1: {vs_l1}; pairwise distinct: {pairwise}; "
          f"genuinely-pruned layers (k<full): {n_pruned_layers}")
    ok = vs_l1 and pairwise
    print(f"    R2 {'ok ' if ok else 'FAIL'} 20 masks distinct from each other and from L1")
    return ok


def check_r3_reproducible():
    ctx = context()
    full = ctx["full"]
    seed = rm.PREREGISTERED_SEEDS[3]
    a = rm.random_ranking(seed, full)
    b = rm.random_ranking(seed, full)
    ok = all(a[k] == b[k] for k in full)
    # and a different seed gives a different ranking
    c = rm.random_ranking(rm.PREREGISTERED_SEEDS[4], full)
    diff_seed = any(a[k] != c[k] for k in full)
    print(f"    R3 same seed identical: {ok}; different seed differs: {diff_seed}")
    print(f"    R3 mask-set sha256 = {ctx['sha'][:16]}…")
    ok = ok and diff_seed
    print(f"    R3 {'ok ' if ok else 'FAIL'} bit-identical reproducibility from seed")
    return ok


def check_r4_materialise():
    ctx = context()
    config, l1, rankings = ctx["config"], ctx["l1"], ctx["rankings"]
    base_sd = rm.base_unet_state_dict(BASE_CKPT)  # BASE weights, never L1
    # materialise L1 (sanity: known param count) and one random mask
    m_l1 = rm.materialize(base_sd, l1, config)
    m_r = rm.materialize(base_sd, rankings[0], config)
    p_l1 = sum(p.numel() for p in m_l1.parameters()) / 1e6
    p_r = sum(p.numel() for p in m_r.parameters()) / 1e6
    z = torch.randn(2, 8, 256, 16)
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    y = torch.randn(2, 512)
    with torch.no_grad():
        out = m_r(z, timesteps=t, y=y, context_list=[], context_attn_mask_list=[])
    shape_ok = tuple(out.shape) == (2, 8, 256, 16)
    param_ok = abs(p_l1 - 145.674) < 0.01 and abs(p_r - 145.674) < 0.01
    print(f"    R4 L1 params={p_l1:.3f}M random params={p_r:.3f}M (expect 145.674M)")
    print(f"    R4 random forward out shape={tuple(out.shape)}")
    ok = shape_ok and param_ok
    print(f"    R4 {'ok ' if ok else 'FAIL'} strict-load + forward at (1,2,3,1); base ckpt weights")
    return ok


def test_r1_per_layer_k():
    assert check_r1_per_layer_k()


def test_r2_distinct():
    assert check_r2_distinct()


def test_r3_reproducible():
    assert check_r3_reproducible()


def test_r4_materialise():
    assert check_r4_materialise()


def main() -> int:
    checks = [
        ("R1 PER-LAYER k", check_r1_per_layer_k),
        ("R2 DISTINCT", check_r2_distinct),
        ("R3 REPRODUCIBLE", check_r3_reproducible),
        ("R4 MATERIALISE", check_r4_materialise),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        results[name] = bool(fn())
    print("\n==== M3A RANDOM MASK TESTS ====")
    for name, _ in checks:
        print(f"  {name:<18} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
