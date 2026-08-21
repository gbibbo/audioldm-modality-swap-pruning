#!/usr/bin/env python3
"""CPU dry-run of the field + I_PT path on the REAL base/post pair (protocol section 10 dry-run).

2 smoke prompts x 2 synthetic noised states, fp32, no decode, cached conditioning, block removal
g in {5,13}. Loads POST then BASE sequentially (15 GB RAM). Validates: raw/deploy fields finite;
D_P(g) > 0; base vs post fields differ; I_PT(g) pooled+per-level finite; block_mask empty bit-exact.
NOT a scientific result -- states are seeded noise, not S_traj.

Run: OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/dryrun_fields.py
"""
from __future__ import annotations
import gc, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, fields as F
from research_sa3.blockskip import block_mask

DEV = "cpu"
SECONDS = 10
BLOCKS = [5, 13]
T_LATENT = 108
SCRATCH = os.environ.get("SCRATCH", "/tmp")


def load(which):
    d = f"data/sa3/small-sfx-base" if which == "base" else "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfg = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfg, f"{d}/model.safetensors", device=DEV)
    return model


def smoke_prompts():
    d = json.load(open("configs/sa3/panel_smoke.json"))
    items = sorted(d["items"], key=lambda x: int(x["audiocap_id"]))[:2]
    return [(it["audiocap_id"], it["caption"]) for it in items]


def states_for(aid):
    # 2 seeded synthetic noised states at tau levels i=5 (0.74555) and i=7 (0.27389)
    sched = json.load(open("configs/sa3/schedule_post_10s.json"))
    taus = [sched["tau_levels"][5], sched["tau_levels"][7]]
    out = []
    for k, tau in enumerate(taus):
        g = torch.Generator().manual_seed(1000 + int(aid) % 997 + k)
        x = torch.randn(1, 256, T_LATENT, generator=g, dtype=torch.float32)
        t = torch.tensor([tau], dtype=torch.float32)
        out.append((tau, x, t))
    return out


def main():
    torch.manual_seed(0)
    prompts = smoke_prompts()
    print(f"[dryrun] prompts: {[p[0] for p in prompts]}")

    # ---- POST ----
    t0 = time.time()
    post = load("post")
    print(f"[dryrun] post loaded ({time.time()-t0:.0f}s)")
    cache = {}  # aid -> {'cc':.., 'states':[(tau,x,t)], 'FP':[...], 'FPmg':{g:[...]}}
    empty_ok = True
    dep_finite = True
    eta_vals = []
    for aid, cap in prompts:
        cc = F.prepare_conditioning(post, cap, SECONDS, DEV, latent_len=T_LATENT)
        sts = states_for(aid)
        rec = {"cc_shapes": [tuple(cc["cross_attn_cond"].shape), tuple(cc["global_embed"].shape)],
               "states": sts, "FP": [], "FPmg": {g: [] for g in BLOCKS}, "taus": [s[0] for s in sts]}
        for tau, x, t in sts:
            fp = F.raw_field(post, x, t, cc)
            rec["FP"].append(fp)
            # empty BlockMask bit-exact
            with block_mask(post, []):
                fp0 = F.raw_field(post, x, t, cc)
            empty_ok &= bool(torch.equal(fp, fp0))
            # block-removed
            for g in BLOCKS:
                with block_mask(post, [g]):
                    rec["FPmg"][g].append(F.raw_field(post, x, t, cc))
            # deploy field
            dep = F.deploy_field(post, x, t, cc, cfg_scale=7.0, apg_scale=1.0)
            dep_finite &= bool(torch.isfinite(dep).all())
            # eta (fp16 vs fp32) on CPU -- code-path only (real eta from GPU smoke)
            try:
                ph = post.half()
                fp16 = F.raw_field(ph, x.half(), t.half(), {k: (v.half() if torch.is_tensor(v) else v) for k, v in cc.items()})
                post.float()
                num = (fp16.float() - fp).pow(2).sum().item()
                den = fp.pow(2).sum().item()
                eta_vals.append(num / den if den > 0 else float("nan"))
            except Exception as e:
                eta_vals.append(None)
        cache[aid] = rec
    # D_P on post states
    dp = {}
    for g in BLOCKS:
        num = den = 0.0
        for aid, _ in prompts:
            rec = cache[aid]
            for i in range(len(rec["states"])):
                num += F.diff_sq_norm(rec["FP"][i], rec["FPmg"][g][i]).item()
                den += F.state_sq_norm(rec["FP"][i]).item()
        dp[g] = num / den
    del post; gc.collect()
    print(f"[dryrun] POST: empty_mask_bitexact={empty_ok} deploy_finite={dep_finite} D_P={ {g: round(v,4) for g,v in dp.items()} }")
    print("[dryrun] eta(fp16-vs-fp32, CPU, non-representative)=" + str([None if e is None else round(e,6) for e in eta_vals]))

    # ---- BASE ----
    t0 = time.time()
    base = load("base")
    print(f"[dryrun] base loaded ({time.time()-t0:.0f}s)")
    # I_PT(g) with Delta = F_P - F_B ; Delta^{-g} = F_P^{-g} - F_B^{-g}
    ipt = {}
    base_vs_post_reldiff = []
    per_level_num = {g: [0.0, 0.0] for g in BLOCKS}  # 2 levels
    per_level_den = [0.0, 0.0]
    # recompute F_B and F_B^{-g} on identical states
    for aid, cap in prompts:
        cc = F.prepare_conditioning(base, cap, SECONDS, DEV, latent_len=T_LATENT)
        rec = cache[aid]
        for i, (tau, x, t) in enumerate(rec["states"]):
            fb = F.raw_field(base, x, t, cc)
            fp = rec["FP"][i]
            rel = (fp - fb).pow(2).sum().item() / fp.pow(2).sum().item()
            base_vs_post_reldiff.append(rel)
            delta = fp - fb
            per_level_den[i] += delta.pow(2).sum().item()
            for g in BLOCKS:
                with block_mask(base, [g]):
                    fbg = F.raw_field(base, x, t, cc)
                delta_g = rec["FPmg"][g][i] - fbg
                per_level_num[g][i] += (delta - delta_g).pow(2).sum().item()
    for g in BLOCKS:
        pooled = sum(per_level_num[g]) / sum(per_level_den)
        ipt[g] = {"pooled": pooled, "per_level": [per_level_num[g][i]/per_level_den[i] for i in range(2)]}
    del base; gc.collect()
    print(f"[dryrun] base<->post field rel-diff (per state): {[round(r,4) for r in base_vs_post_reldiff]}")
    print(f"[dryrun] I_PT pooled: { {g: round(v['pooled'],4) for g,v in ipt.items()} }")
    print(f"[dryrun] I_PT per-level: { {g: [round(x,4) for x in v['per_level']] for g,v in ipt.items()} }")
    ok = empty_ok and dep_finite and all(v > 0 for v in dp.values()) and all(v['pooled'] >= 0 for v in ipt.values()) \
         and all(r > 0 for r in base_vs_post_reldiff)
    print("DRYRUN-FIELDS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
