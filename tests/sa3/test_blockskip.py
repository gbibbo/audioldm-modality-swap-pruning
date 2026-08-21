#!/usr/bin/env python3
"""BlockMask tests on a tiny random-weights DiffusionTransformer built from the upstream code.

    K1 EMPTY-MASK    block_mask(model, []) leaves the output bit-identical.
    K2 SKIP-CHANGES  skipping one block changes the output; the swapped block is restored after.
    K3 SKIP-ALL      skipping every block equals a forward through only project_in/project_out
                     (memory-token stripping included) -- i.e. the identity semantics are exact.
    K4 RANGE         out-of-range index raises; mask is restored even after an exception.

Run: .venv-sa3/bin/python tests/sa3/test_blockskip.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from research_sa3.blockskip import block_mask, find_layers, depth, IdentityBlock  # noqa: E402


def tiny_dit(depth_=4, dim=128, heads=2, io=8, cond=16, seed=0):  # dim_heads=64 like the real DiT
    from stable_audio_3.models.dit import DiffusionTransformer
    torch.manual_seed(seed)
    m = DiffusionTransformer(io_channels=io, embed_dim=dim, depth=depth_, num_heads=heads,
                             cond_token_dim=cond, global_cond_dim=cond, global_cond_type="adaLN",
                             diffusion_objective="rectified_flow", num_memory_tokens=2)
    # random non-zero branch outputs so that blocks actually do something
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and (p == 0).all():
                p.normal_(0, 0.02)
    return m.eval()


def inputs(io=8, T=12, cond=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, io, T, generator=g)
    t = torch.rand(2, generator=g)
    ctx = torch.randn(2, 5, cond, generator=g)
    glob = torch.randn(2, cond, generator=g)
    return x, t, ctx, glob


def fwd(m, x, t, ctx, glob):
    with torch.no_grad():
        return m._forward(x, t, cross_attn_cond=ctx, global_embed=glob)


def main() -> int:
    m = tiny_dit()
    x, t, ctx, glob = inputs()
    ref = fwd(m, x, t, ctx, glob)
    ok = True

    with block_mask(m, []):
        y = fwd(m, x, t, ctx, glob)
    k1 = torch.equal(y, ref)
    print(f"  K1 empty mask bit-identical: {k1}"); ok &= k1

    layers = find_layers(m)
    orig1 = layers[1]
    with block_mask(m, [1]):
        y1 = fwd(m, x, t, ctx, glob)
        swapped = isinstance(layers[1], IdentityBlock)
    restored = layers[1] is orig1
    k2 = (not torch.equal(y1, ref)) and swapped and restored
    print(f"  K2 skip one block changes output={not torch.equal(y1, ref)}, swapped={swapped}, restored={restored}"); ok &= k2

    # K3: manual project_in / (memory tokens) / project_out path
    with block_mask(m, range(depth(m))):
        y_all = fwd(m, x, t, ctx, glob)
    with torch.no_grad():
        xin = x.transpose(1, 2) if hasattr(m, "transformer") else x
        # replicate DiffusionTransformer._forward pre/post-processing by calling the transformer
        # with an empty layer list: swap layers for an empty ModuleList
        tr = m.transformer
        saved = tr.layers
        tr.layers = torch.nn.ModuleList([])
        try:
            y_manual = fwd(m, x, t, ctx, glob)
        finally:
            tr.layers = saved
    k3 = torch.equal(y_all, y_manual)
    print(f"  K3 skip-all == empty-layer-list forward: {k3}"); ok &= k3

    raised = False
    try:
        with block_mask(m, [depth(m)]):
            pass
    except IndexError:
        raised = True
    try:
        with block_mask(m, [2]):
            raise ValueError("boom")
    except ValueError:
        pass
    k4 = raised and all(not isinstance(l, IdentityBlock) for l in find_layers(m))
    print(f"  K4 out-of-range raises={raised}, restored after exception={k4}"); ok &= k4

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
