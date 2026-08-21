#!/usr/bin/env python3
"""Probe tests on a tiny DiT (protocol section 3.4).
    P1 KAPPA    per-layer ||dW||/||W0|| == kappa after build.
    P2 F_P      strength=0 gives the original field bit-exactly (F_P recovery).
    P3 DELTA    strength=1 changes the field (dF != 0).
    P4 LINEAR   ||dF(2u)||/||dF(u)|| in [1.9,2.1] at small kappa (tangent regime).
    P5 RESTRICT restrict_to_surviving(g) zeroes block g's strength, others 1.
Run: .venv-sa3/bin/python tests/sa3/test_probes.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from research_sa3 import probes as P
from research_sa3 import fields as F


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


def inp():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, 12, generator=g); t = torch.rand(1, generator=g)
    ctx = torch.randn(1, 5, 16, generator=g); glob = torch.randn(1, 16, generator=g)
    cc = {"cross_attn_cond": ctx, "cross_attn_cond_mask": None, "global_embed": glob, "local_add_cond": None}
    return x, t, cc


def field(m, x, t, cc):
    with torch.no_grad():
        return F.raw_field(m, x, t, cc)


def main():
    torch.set_default_dtype(torch.float32)
    ok_all = True

    # reference model (no probe)
    m0 = tiny_dit(); x, t, cc = inp()
    ref = field(m0, x, t, cc)

    # P1 + P2 + P3
    m = tiny_dit()
    P.build_probe(m, family="U_gen", kappa=0.01, rank=8, seed=3)
    kap = P.per_layer_kappa(m)
    p1 = all(abs(v - 0.01) < 1e-4 for v in kap.values()) and len(kap) > 0
    print(f"    P1 n_layers={len(kap)} kappa range=({min(kap.values()):.5f},{max(kap.values()):.5f})")
    P.set_strength(m, 0.0)
    f_off = field(m, x, t, cc)
    p2 = torch.equal(f_off, ref)
    print(f"    P2 strength=0 == reference (F_P recovery): {p2}")
    P.set_strength(m, 1.0)
    f_on = field(m, x, t, cc)
    p3 = not torch.equal(f_on, ref)
    print(f"    P3 strength=1 changes field: {p3}")
    ok_all &= p1 and p2 and p3

    # P4 linearity at small kappa
    m2 = tiny_dit()
    P.build_probe(m2, family="U_gen", kappa=1e-3, rank=8, seed=5)
    P.set_strength(m2, 1.0)
    fP = field(tiny_dit(), x, t, cc)  # unperturbed (same seed as m2's base? use m2 strength0)
    P.set_strength(m2, 0.0); fP = field(m2, x, t, cc); P.set_strength(m2, 1.0)
    d1 = (field(m2, x, t, cc) - fP).norm().item()
    P.probe_scale(m2, 2.0)
    d2 = (field(m2, x, t, cc) - fP).norm().item()
    ratio = d2 / d1 if d1 > 0 else float("nan")
    p4 = 1.9 <= ratio <= 2.1
    print(f"    P4 linearity ||dF(2u)||/||dF(u)||={ratio:.3f} in [1.9,2.1]: {p4}")
    ok_all &= p4

    # P5 restriction
    m3 = tiny_dit()
    P.build_probe(m3, family="U_gen", kappa=0.01, rank=8, seed=7)
    P.restrict_to_surviving(m3, removed_block=1)
    strengths = {}
    for name, pr in P._probe_params(m3):
        import re
        mm = re.search(r"\.layers\.(\d+)\.", name)
        b = int(mm.group(1)) if mm else -1
        strengths.setdefault(b, set()).add(float(pr.lora_strength))
    p5 = strengths.get(1) == {0.0} and all(s == {1.0} for b, s in strengths.items() if b != 1)
    print(f"    P5 restrict block1 -> strengths per block: { {b: sorted(v) for b,v in sorted(strengths.items())} }")
    ok_all &= p5

    for nm, v in [("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4), ("P5", p5)]:
        print(f"  {nm}: {'PASS' if v else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
