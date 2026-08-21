#!/usr/bin/env python3
"""Field-norm recompute (review task 6), CPU, NO block removal, NO generation-scoring.

The pilot stored D_P=dp_num/f_den and D_B=db_num/fb_den but not the raw numerators/denominators,
and the field-store is gone. To decide whether D_P/D_B ~= 10x is a true functional amplification
or an artifact of the DIFFERENT normalizers ||F_P||^2 vs ||F_B||^2, we recompute the panel field
norms on the SAME (deterministic) S_traj states:

    f_den  = sum_states ||F_P||^2      fb_den = sum_states ||F_B||^2
    ratio_norm = f_den / fb_den
    non_normalized_damage_ratio(g) = dp_num(g)/db_num(g) = (D_P(g)/D_B(g)) * ratio_norm

If ratio_norm ~= 1, the 10x is real amplification; if ratio_norm << 1, it is inflated by
normalization. Two passes (post then base) on the pilot panel. NO GPU.

Run: OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/rq1_field_norms.py --n 32
"""
from __future__ import annotations
import argparse, gc, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, fields as F, e2e, states
from research_sa3 import seeds as S

SECONDS = 10
OUT = "artifacts/sa3/rq1_field_norms.json"


def load(which):
    d = "data/sa3/small-sfx-base" if which == "base" else "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    m, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device="cpu")
    return m, cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=32); a = ap.parse_args()
    panel = json.load(open("configs/sa3/panel_pilot.json"))
    prompts = sorted(panel["items"], key=lambda x: int(x["audiocap_id"]))[:a.n]
    store = os.path.join(os.environ.get("SCRATCH", "/tmp"), "sa3_fieldnorm_states")
    os.makedirs(store, exist_ok=True)

    # PASS 1: post -> S_traj + ||F_P||^2 per prompt
    post, cfg = load("post"); sa = e2e.wrap_model(post, cfg, "cpu", model_half=False)
    fP_per = {}; f_den = 0.0
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]; seed = S.derive_seed(0, aid, "init", 0)
        tr = states.capture_trajectory(sa, cap, SECONDS, seed, steps=8, cfg_scale=1.0, apg_scale=1.0)
        sts = tr["states"]
        cc = F.prepare_conditioning(post, cap, SECONDS, "cpu", latent_len=sts[0][1].shape[-1])
        s = 0.0
        for tau, x in sts:
            xt = x.to(torch.float32); tt = torch.full((xt.shape[0],), tau, dtype=torch.float32)
            s += F.state_sq_norm(F.raw_field(post, xt, tt, cc)).item()
        fP_per[aid] = s; f_den += s
        torch.save({"states": [(t, x) for t, x in sts]}, os.path.join(store, f"st_{aid}.pt"))
        print(f"[fn] post {aid} ||F_P||^2={s:.4f}", flush=True)
    del post, sa; gc.collect()

    # PASS 2: base -> ||F_B||^2 on the same states
    base, bcfg = load("base")
    fB_per = {}; fb_den = 0.0
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]
        sts = torch.load(os.path.join(store, f"st_{aid}.pt"))["states"]
        cc = F.prepare_conditioning(base, cap, SECONDS, "cpu", latent_len=sts[0][1].shape[-1])
        s = 0.0
        for tau, x in sts:
            xt = x.to(torch.float32); tt = torch.full((xt.shape[0],), tau, dtype=torch.float32)
            s += F.state_sq_norm(F.raw_field(base, xt, tt, cc)).item()
        fB_per[aid] = s; fb_den += s
        print(f"[fn] base {aid} ||F_B||^2={s:.4f}", flush=True)
    del base; gc.collect()

    ratio_norm = f_den / fb_den if fb_den > 0 else float("nan")
    # combine with pilot D_P/D_B to back out non-normalized damage ratios
    pil = json.load(open("artifacts/sa3/pilot_fields.json"))
    gs = sorted(pil["D_P"], key=lambda k: int(k))
    nonnorm = {}
    for g in gs:
        dp, db = pil["D_P"][g], pil["D_B_common"][g]
        r_norm_dp_db = (dp / db) * ratio_norm if db > 0 else float("nan")
        nonnorm[g] = {"D_P": dp, "D_B": db, "ratio_DP_DB": (dp/db if db>0 else None),
                      "nonnormalized_damage_ratio": r_norm_dp_db}
    out = {"N": len(prompts), "f_den_sumFP2": f_den, "fb_den_sumFB2": fb_den,
           "ratio_norm_FP2_over_FB2": ratio_norm,
           "per_prompt_FP2": fP_per, "per_prompt_FB2": fB_per,
           "nonnormalized_damage_ratio_per_block": nonnorm,
           "interpretation": ("ratio_norm~=1 => D_P/D_B is a REAL amplification; "
                              "ratio_norm<<1 => the 10x is inflated by normalization")}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[fn] ||F_P||^2/||F_B||^2 (panel) = {ratio_norm:.4f}")
    print(f"[fn] => non-normalized damage ratio dp_num/db_num = (D_P/D_B) * {ratio_norm:.4f}")
    import numpy as np
    rr = [nonnorm[g]["nonnormalized_damage_ratio"] for g in gs if nonnorm[g]["nonnormalized_damage_ratio"] is not None]
    print(f"[fn] non-normalized ratio: mean={np.mean(rr):.2f} median={np.median(rr):.2f} range=({min(rr):.2f},{max(rr):.2f})")
    print(f"[fn] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
