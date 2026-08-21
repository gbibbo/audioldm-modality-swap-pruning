#!/usr/bin/env python3
"""Field + norm tests (protocol section 1.4, 3.1). Norm math is exact; field behaviour on a
tiny random DiffusionTransformer (no 15 GB load).

    FF1 NORM      state_sq_norm / diff_sq_norm exact, padding-masked, divided by valid T.
    FF2 GETDIT    get_dit resolves the DiffusionTransformer under model.model.model.
    FF3 FIELD     raw_field finite; block removal changes it; deploy_field(cfg=7) != raw(cfg=1);
                  APG(1.0) != vanilla CFG(0.0).

Run: .venv-sa3/bin/python tests/sa3/test_fields.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from research_sa3 import fields as F
from research_sa3.blockskip import block_mask


def ff1_norm():
    v = torch.ones(1, 2, 3)
    ok = abs(F.state_sq_norm(v).item() - 2.0) < 1e-12   # sum sq 6 / T 3
    pm = torch.tensor([[1.0, 1.0, 0.0]])
    ok &= abs(F.state_sq_norm(v, pm).item() - 2.0) < 1e-12  # valid T=2, sum 4 /2
    a = torch.zeros(1, 2, 3); b = torch.ones(1, 2, 3)
    ok &= abs(F.diff_sq_norm(a, b).item() - 2.0) < 1e-12
    # non-uniform
    w = torch.tensor([[[1.0, 2.0, 2.0]]])  # sumsq = 1+4+4=9 /3 = 3
    ok &= abs(F.state_sq_norm(w).item() - 3.0) < 1e-12
    print(f"    FF1 norms ok={ok}")
    return ok


def ff2_getdit():
    class DiffusionTransformer:
        pass
    dit = DiffusionTransformer()
    class DiTWrapper: pass
    w = DiTWrapper(); w.model = dit
    class Cond: pass
    c = Cond(); c.model = w
    ok = F.get_dit(c) is dit and F.get_dit(dit) is dit
    print(f"    FF2 get_dit ok={ok}")
    return ok


def tiny_dit(seed=0):
    from stable_audio_3.models.dit import DiffusionTransformer
    torch.manual_seed(seed)
    m = DiffusionTransformer(io_channels=8, embed_dim=128, depth=4, num_heads=2,
                             cond_token_dim=16, global_cond_dim=16, global_cond_type="adaLN",
                             diffusion_objective="rectified_flow", num_memory_tokens=2)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and (p == 0).all():
                p.normal_(0, 0.02)
    return m.eval()


def ff3_field():
    m = tiny_dit()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, 12, generator=g)
    t = torch.rand(1, generator=g)
    ctx = torch.randn(1, 5, 16, generator=g)
    glob = torch.randn(1, 16, generator=g)
    cc = {"cross_attn_cond": ctx, "cross_attn_cond_mask": None, "global_embed": glob, "local_add_cond": None}
    with torch.no_grad():
        raw = F.raw_field(m, x, t, cc)
        with block_mask(m, [1]):
            raw_m = F.raw_field(m, x, t, cc)
        dep = F.deploy_field(m, x, t, cc, cfg_scale=7.0, apg_scale=1.0)
        dep_van = F.deploy_field(m, x, t, cc, cfg_scale=7.0, apg_scale=0.0)
    finite = bool(torch.isfinite(raw).all() and torch.isfinite(dep).all())
    changed = not torch.equal(raw, raw_m)
    cfg_effect = not torch.allclose(dep, raw, atol=1e-6)          # cfg=7 != raw velocity
    apg_effect = not torch.allclose(dep, dep_van, atol=1e-6)      # APG vs vanilla differ
    ok = finite and changed and cfg_effect and apg_effect
    print(f"    FF3 finite={finite} block_removal_changes={changed} cfg_effect={cfg_effect} apg!=vanilla={apg_effect}")
    return ok


def main():
    checks = [("FF1", ff1_norm), ("FF2", ff2_getdit), ("FF3", ff3_field)]
    ok_all = True
    for n, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {n}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
