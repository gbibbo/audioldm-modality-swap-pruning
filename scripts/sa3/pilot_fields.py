#!/usr/bin/env python3
"""RQ1 field pilot (protocol section 3.1-3.3, 6.0) -- forward-only, no probes, no generation.

Single-block leave-one-out tables over `blocks` on the disjoint PILOT panel:
  D_P(g)         post-field damage
  D_B^common(g)  base-field damage on the SAME S_traj states
  I_PT^raw(g)    normalized post-training-delta distortion (pooled + per level), eta guard
  I_PT^dep(g)    same with the base DEPLOY (CFG/APG) field
  W(g)           parameter-delta covariate (never an effect estimate)
optional: D_P sequential greedy path (k<=KMAX) + additivity gaps.

Two passes (memory-safe, identical on CPU/GPU): PASS 1 loads the post, captures S_traj per prompt,
computes F_P + F_P^{-g}, stores them to --field-store; PASS 2 loads the base, recomputes F_B/F_B^{-g}
+ F_B^dep on the SAME stored states, and combines. Conditioning is model-independent (byte-identical
T5Gemma) but recomputed per model for faithfulness. Pilot numbers SIZE the experiment; no section-8
decision is read from them.

Run (GPU):  _external/stable-audio-3/.venv/bin/python scripts/sa3/pilot_fields.py --device cuda \
                --n 16 --greedy --expect-commit <sha> --out artifacts/sa3/pilot_fields.json
CPU dry:   OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/pilot_fields.py --dry-run-cpu
"""
from __future__ import annotations
import argparse, gc, json, os, subprocess, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, fields as F, e2e, states, metrics as M, greedy as G
from research_sa3 import seeds as S
from research_sa3.blockskip import block_mask

SECONDS = 10


def sha256_file(p):
    import hashlib; return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(which, device, half):
    d = "data/sa3/small-sfx-base" if which == "base" else "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device=device)
    if half:
        model = model.half()
    return model, cfg, f"{d}/model.safetensors"


def block_W(base_st, post_st, g):
    """W(g) = sum ||theta_P-theta_B||_F^2 / sum ||theta_B||_F^2 over transformer.layers.g params."""
    from safetensors import safe_open
    pref = f"model.model.transformer.layers.{g}."
    num = den = 0.0
    with safe_open(base_st, framework="pt", device="cpu") as fb, safe_open(post_st, framework="pt", device="cpu") as fp:
        keys = [k for k in fb.keys() if k.startswith(pref)]
        for k in keys:
            b = fb.get_tensor(k).to(torch.float64); p = fp.get_tensor(k).to(torch.float64)
            num += (p - b).pow(2).sum().item(); den += b.pow(2).sum().item()
    return num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--panel", default="configs/sa3/panel_pilot.json")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--blocks", default=None, help="comma list; default all 20")
    ap.add_argument("--greedy", action="store_true"); ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--field-store", default=None)
    ap.add_argument("--out", default="artifacts/sa3/pilot_fields.json")
    a = ap.parse_args()
    if a.dry_run_cpu:
        a.device = "cpu"; a.n = a.n or 2; a.blocks = a.blocks or "5,13"; a.kmax = min(a.kmax, 3)
    dev = a.device; half = (dev == "cuda")
    dtype = torch.float16 if half else torch.float32
    blocks = [int(x) for x in a.blocks.split(",")] if a.blocks else list(range(20))
    store = a.field_store or os.path.join(os.environ.get("SCRATCH", "/tmp"), "sa3_pilot_fields")
    os.makedirs(store, exist_ok=True)

    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    panel = json.load(open(a.panel))
    prompts = sorted(panel["items"], key=lambda x: int(x["audiocap_id"]))
    if a.n:
        prompts = prompts[:a.n]
    N = len(prompts)
    R = {"phase": "pilot_fields", "device": dev, "dry_run_cpu": a.dry_run_cpu, "N": N,
         "blocks": blocks, "panel": a.panel, "panel_sha256": sha256_file(a.panel),
         "schedule_post_sha256": sha256_file("configs/sa3/schedule_post_10s.json"),
         "git_commit": subprocess.getoutput("git rev-parse HEAD"),
         "upstream_commit": loading.SA3_UPSTREAM_COMMIT}
    t_start = time.time()

    # ---------------- PASS 1: POST ----------------
    post, cfg, post_st = load("post", dev, half)
    sa = e2e.wrap_model(post, cfg, dev, model_half=half)
    dp_num = {g: 0.0 for g in blocks}; f_den = 0.0
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]; seed = S.derive_seed(0, aid, "init", 0)
        tr = states.capture_trajectory(sa, cap, SECONDS, seed, steps=8, cfg_scale=1.0, apg_scale=1.0)
        sts = tr["states"]  # [(tau, x_cpu)]
        cc = F.prepare_conditioning(post, cap, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=dtype)
        FP = []; FPmg = {g: [] for g in blocks}
        for tau, x in sts:
            xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
            fp = F.raw_field(post, xt, tt, cc); FP.append(fp.detach())
            f_den += F.state_sq_norm(fp).item()
            for g in blocks:
                with block_mask(post, [g]):
                    fpg = F.raw_field(post, xt, tt, cc)
                FPmg[g].append(fpg.detach())
                dp_num[g] += F.diff_sq_norm(fp, fpg).item()
        torch.save({"states": [(t, x.half()) for t, x in sts],
                    "FP": [f.to("cpu").half() for f in FP],
                    "FPmg": {g: [f.to("cpu").half() for f in FPmg[g]] for g in blocks}},
                   os.path.join(store, f"post_{aid}.pt"))
    D_P = {g: dp_num[g] / f_den for g in blocks}

    greedy_out = None
    if a.greedy:
        # aggregate D_P greedy: score_fn(M) = sum_{p,i} ||F_P - F_P^{-M}||^2 / f_den (post loaded)
        cache = {it["audiocap_id"]: torch.load(os.path.join(store, f"post_{it['audiocap_id']}.pt"))
                 for it in prompts}
        ccs = {it["audiocap_id"]: F.prepare_conditioning(post, it["caption"], SECONDS, dev,
                 latent_len=cache[it["audiocap_id"]]["states"][0][1].shape[-1], dtype=dtype) for it in prompts}
        def score(Mset):
            Mset = list(Mset); num = 0.0
            for it in prompts:
                aid = it["audiocap_id"]; rec = cache[aid]; cc = ccs[aid]
                for i, (tau, x) in enumerate(rec["states"]):
                    xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
                    with block_mask(post, Mset):
                        fpm = F.raw_field(post, xt, tt, cc)
                    num += F.diff_sq_norm(rec["FP"][i].to(dev, dtype), fpm).item()
            return num / f_den
        gp = G.greedy_path(20, a.kmax, score)
        gp["additivity_gap"] = {k: G.additivity_gap(frozenset(gp["sets"][k]), score) for k in gp["sets"]}
        gp["sets"] = {k: sorted(v) for k, v in gp["sets"].items()}
        greedy_out = {"order": gp["order"], "sets": gp["sets"], "n_evals": gp["n_evals"],
                      "additivity_gap": gp["additivity_gap"]}
    del post, sa; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---------------- PASS 2: BASE ----------------
    base, bcfg, base_st = load("base", dev, half)
    db_num = {g: 0.0 for g in blocks}; fb_den = 0.0
    ipt_num = {g: [0.0] * 8 for g in blocks}; ipt_den = [0.0] * 8
    fp_sq_lvl = [0.0] * 8
    iptdep_num = {g: [0.0] * 8 for g in blocks}; iptdep_den = [0.0] * 8
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]
        rec = torch.load(os.path.join(store, f"post_{aid}.pt"))
        sts = rec["states"]
        cc = F.prepare_conditioning(base, cap, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=dtype)
        for i, (tau, x) in enumerate(sts):
            xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
            fb = F.raw_field(base, xt, tt, cc); fb_den += F.state_sq_norm(fb).item()
            fp = rec["FP"][i].to(dev, dtype)
            delta = fp - fb
            lvl = min(i, 7)
            ipt_den[lvl] += F.state_sq_norm(delta).item()
            fp_sq_lvl[lvl] += F.state_sq_norm(fp).item()
            fbdep = F.deploy_field(base, xt, tt, cc, cfg_scale=7.0, apg_scale=1.0)
            delta_dep = fp - fbdep
            iptdep_den[lvl] += F.state_sq_norm(delta_dep).item()
            for g in blocks:
                with block_mask(base, [g]):
                    fbg = F.raw_field(base, xt, tt, cc)
                db_num[g] += F.diff_sq_norm(fb, fbg).item()
                delta_g = rec["FPmg"][g][i].to(dev, dtype) - fbg
                ipt_num[g][lvl] += F.diff_sq_norm(delta, delta_g).item()
                with block_mask(base, [g]):
                    fbgdep = F.deploy_field(base, xt, tt, cc, cfg_scale=7.0, apg_scale=1.0)
                delta_gdep = rec["FPmg"][g][i].to(dev, dtype) - fbgdep
                iptdep_num[g][lvl] += F.diff_sq_norm(delta_dep, delta_gdep).item()
    D_B = {g: db_num[g] / fb_den for g in blocks}
    # eta guard: use the smoke's eta_max if available else 0
    eta_max = 6.7e-5
    eta_by_level = [eta_max] * 8
    ipt = M.i_pt(ipt_num, ipt_den, fp_sq_lvl, eta_by_level)
    iptdep = M.i_pt(iptdep_num, iptdep_den, fp_sq_lvl, eta_by_level)
    del base; gc.collect()

    # W(g)
    W = {g: block_W(base_st, post_st, g) for g in blocks}

    R.update({
        "D_P": D_P, "D_B_common": D_B,
        "I_PT_raw": {str(g): ipt["per_block"][g] for g in blocks}, "I_PT_raw_excluded_levels": ipt["excluded_levels"],
        "I_PT_dep": {str(g): iptdep["per_block"][g] for g in blocks}, "I_PT_dep_excluded_levels": iptdep["excluded_levels"],
        "W": W, "greedy_D_P": greedy_out,
        "note": "PILOT sizing only; no section-8 decision. eta_max from smoke (6.7e-5).",
        "wall_s": round(time.time() - t_start, 1),
    })
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print("PILOT_FIELDS_JSON_BEGIN"); print(json.dumps(R)); print("PILOT_FIELDS_JSON_END")
    print(f"[pilot] wrote {a.out}  N={N} blocks={len(blocks)} wall={R['wall_s']}s")
    print(f"[pilot] D_P={ {g: round(v,4) for g,v in D_P.items()} }")
    print(f"[pilot] D_B={ {g: round(v,4) for g,v in D_B.items()} }")
    print(f"[pilot] I_PT_raw pooled={ {g: round(ipt['per_block'][g]['pooled'],4) for g in blocks} }")
    print(f"[pilot] W={ {g: round(v,5) for g,v in W.items()} }")
    if greedy_out: print(f"[pilot] greedy_D_P sets={greedy_out['sets']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
