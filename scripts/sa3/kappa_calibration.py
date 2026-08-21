#!/usr/bin/env python3
"""kappa micro-calibration for A_tan (review round 3), CPU only, NO block removal, NO A_tan gate.

Freezes the LARGEST kappa for which ALL n_u=8 frozen probes are in the tangent regime, BEFORE the
N=32 A_tan run and WITHOUT looking at the A_tan-vs-D_P divergence. Per probe u (averaged over a few
prompts x 8 states):
  * linearity  ||dF(2u)|| / ||dF(u)||  must be in [1.9, 2.1]  (regime is linear)
  * precision  ||dF(u)||^2 / ||F_P||^2  must be >= FACTOR*eta   (fp16-resolvable on the T4; eta=6.7e-5)
Both are computed in fp32 (the RATIOS are precision-independent). Chooses the largest passing kappa
from the grid; if none passes, reports so. dF(u)=F_{P+u}-F_P (probe on vs off), batched over states.

Run: OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/kappa_calibration.py --n 4 --n-u 8
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, fields as F, probes as P

SECONDS = 10
ETA = 6.7e-5      # GPU fp16 eta (smoke); precision floor = FACTOR*ETA
FACTOR = 10.0


def load_post():
    d = "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    m, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device="cpu")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--n-u", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--states", type=int, default=0)
    ap.add_argument("--kappa-grid", default="0.01,0.005,0.0025")
    ap.add_argument("--state-store", default="artifacts/sa3/pilot_states")
    ap.add_argument("--out", default="artifacts/sa3/kappa_calibration.json")
    a = ap.parse_args()
    kgrid = sorted([float(x) for x in a.kappa_grid.split(",")], reverse=True)  # largest first
    prec_floor = FACTOR * ETA
    sfs = sorted(glob.glob(os.path.join(a.state_store, "state_*.pt")),
                 key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))[:a.n]
    assert sfs, f"no states in {a.state_store}"
    post = load_post()
    dtype = torch.float32

    def batched(sts):
        return (torch.cat([x for _, x in sts], 0).to(dtype),
                torch.tensor([float(t) for t, _ in sts], dtype=dtype))

    def rep(cc, B):
        return {k: (v.repeat(B, *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v) for k, v in cc.items()}

    # precompute baselines F_P per prompt
    base = []
    for sf in sfs:
        d = torch.load(sf); sts = (d["states"][:a.states] if a.states else d["states"]); cap = d["caption"]
        cc = rep(F.prepare_conditioning(post, cap, SECONDS, "cpu", latent_len=sts[0][1].shape[-1], dtype=dtype), len(sts))
        bx, bt = batched(sts)
        with torch.no_grad():
            FP = F.raw_field(post, bx, bt, cc)
        base.append({"bx": bx, "bt": bt, "cc": cc, "FP": FP,
                     "fp_sq": float(F.state_sq_norm(FP).sum().item())})

    results = {}
    for kap in kgrid:
        per_probe = []
        for u in range(a.n_u):
            lin_ratios = []; num_df = 0.0; den_fp = 0.0
            P.build_probe(post, family="U_gen", kappa=kap, rank=a.rank, seed=1000 + u)  # BUILD ONCE
            for b in base:
                with torch.no_grad():
                    P.set_strength(post, 1.0)
                    dfu = F.raw_field(post, b["bx"], b["bt"], b["cc"]).float() - b["FP"].float()
                    du = float(F.state_sq_norm(dfu).sum().item())
                    P.set_strength(post, 2.0)
                    d2 = float(F.state_sq_norm(F.raw_field(post, b["bx"], b["bt"], b["cc"]).float() - b["FP"].float()).sum().item())
                lin_ratios.append((d2 ** 0.5) / (du ** 0.5) if du > 0 else float("nan"))
                num_df += du; den_fp += b["fp_sq"]
            P.remove_probe(post)
            import numpy as np
            lin = float(np.mean(lin_ratios)); prec = num_df / den_fp
            per_probe.append({"u": u, "linearity_mean": lin, "precision_ratio": prec,
                              "lin_ok": bool(1.9 <= lin <= 2.1), "prec_ok": bool(prec >= prec_floor)})
        all_lin = all(p["lin_ok"] for p in per_probe)
        all_prec = all(p["prec_ok"] for p in per_probe)
        results[str(kap)] = {"all_probes_linear": all_lin, "all_probes_precision": all_prec,
                             "passes": bool(all_lin and all_prec),
                             "linearity_range": [min(p["linearity_mean"] for p in per_probe),
                                                 max(p["linearity_mean"] for p in per_probe)],
                             "precision_min": min(p["precision_ratio"] for p in per_probe),
                             "per_probe": per_probe}
        print(f"kappa={kap}: all_linear={all_lin} all_precision={all_prec} PASSES={results[str(kap)]['passes']} "
              f"lin_range=[{results[str(kap)]['linearity_range'][0]:.3f},{results[str(kap)]['linearity_range'][1]:.3f}] "
              f"prec_min={results[str(kap)]['precision_min']:.2e} (floor {prec_floor:.1e})", flush=True)

    chosen = next((k for k in kgrid if results[str(k)]["passes"]), None)
    out = {"kappa_grid": kgrid, "eta": ETA, "precision_floor": prec_floor, "n_prompts": a.n, "n_u": a.n_u,
           "results": results, "frozen_kappa": chosen,
           "note": "largest kappa passing linearity[1.9,2.1]+precision(>=10 eta) for ALL probes; "
                   "chosen WITHOUT looking at A_tan-vs-D_P."}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nFROZEN kappa = {chosen}  (wrote {a.out})")
    return 0 if chosen is not None else 2


if __name__ == "__main__":
    sys.exit(main())
