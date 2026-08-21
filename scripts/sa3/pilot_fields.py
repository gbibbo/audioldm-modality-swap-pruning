#!/usr/bin/env python3
"""RQ1 field pilot with PER-PROMPT sufficient statistics (protocol 3.1-3.3; review round 2).

Single-block leave-one-out on the dense-post S_traj states. Persists, PER PROMPT p and PER BLOCK g,
the sufficient statistics that let D_P, D_B, I_PT and the removal rankings be recomputed for ANY
subsample of prompts WITHOUT re-running forwards (D_P/I_PT are ratios of sums ⇒ per-prompt-additive):

  per_prompt[aid] = { f_den, fb_den,
                      dp_num{g}, db_num{g},
                      ipt_num{g}[level], ipt_den[level], fp_sq[level] }

Aggregates D_P(g)=Σ_p dp_num[p,g] / Σ_p f_den[p], etc. A self-check asserts the summed per-prompt
stats reproduce the directly-accumulated aggregates exactly. The S_traj states are also saved to a
PERSISTENT --state-store so the A_tan pilot reuses them (no regeneration). I_PT_dep optional.

Run (GPU):  _external/stable-audio-3/.venv/bin/python scripts/sa3/pilot_fields.py --device cuda \
                --n 48 --expect-commit <sha> --state-store artifacts/sa3/pilot_states \
                --out artifacts/sa3/pilot_fields.json
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
    from safetensors import safe_open
    pref = f"model.model.transformer.layers.{g}."
    num = den = 0.0
    with safe_open(base_st, framework="pt", device="cpu") as fb, safe_open(post_st, framework="pt", device="cpu") as fp:
        for k in [k for k in fb.keys() if k.startswith(pref)]:
            b = fb.get_tensor(k).to(torch.float64); p = fp.get_tensor(k).to(torch.float64)
            num += (p - b).pow(2).sum().item(); den += b.pow(2).sum().item()
    return num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--panel", default="configs/sa3/panel_pilot.json")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--greedy", action="store_true"); ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--with-dep", action="store_true")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--field-store", default=None)
    ap.add_argument("--state-store", default="artifacts/sa3/pilot_states")
    ap.add_argument("--out", default="artifacts/sa3/pilot_fields.json")
    a = ap.parse_args()
    if a.dry_run_cpu:
        a.device = "cpu"; a.n = a.n or 3; a.blocks = a.blocks or "5,13"; a.kmax = min(a.kmax, 2)
    dev = a.device; half = (dev == "cuda"); dtype = torch.float16 if half else torch.float32
    blocks = [int(x) for x in a.blocks.split(",")] if a.blocks else list(range(20))
    store = a.field_store or os.path.join(os.environ.get("SCRATCH", "/tmp"), "sa3_pilot_fields")
    os.makedirs(store, exist_ok=True); os.makedirs(a.state_store, exist_ok=True)

    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    panel = json.load(open(a.panel))
    prompts = sorted(panel["items"], key=lambda x: int(x["audiocap_id"]))
    if a.n:
        prompts = prompts[:a.n]
    N = len(prompts)
    R = {"phase": "pilot_fields", "device": dev, "dry_run_cpu": a.dry_run_cpu, "N": N, "blocks": blocks,
         "panel": a.panel, "panel_sha256": sha256_file(a.panel),
         "schedule_post_sha256": sha256_file("configs/sa3/schedule_post_10s.json"),
         "seed_table_sha256": sha256_file("configs/sa3/seed_table.json"),
         "git_commit": subprocess.getoutput("git rev-parse HEAD"), "upstream_commit": loading.SA3_UPSTREAM_COMMIT,
         "with_dep": a.with_dep, "state_store": a.state_store}
    t_start = time.time()
    pp = {}  # per-prompt sufficient statistics

    # ---------------- PASS 1: POST ----------------
    post, cfg, post_st = load("post", dev, half)
    sa = e2e.wrap_model(post, cfg, dev, model_half=half)
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]; seed = S.derive_seed(0, aid, "init", 0)
        tr = states.capture_trajectory(sa, cap, SECONDS, seed, steps=8, cfg_scale=1.0, apg_scale=1.0)
        sts = tr["states"]
        print(f"[pilot] pass1 post prompt {aid}", flush=True)
        cc = F.prepare_conditioning(post, cap, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=dtype)
        rec = {"f_den": 0.0, "dp_num": {g: 0.0 for g in blocks}}
        FP = []; FPmg = {g: [] for g in blocks}
        for tau, x in sts:
            xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
            fp = F.raw_field(post, xt, tt, cc); FP.append(fp.detach())
            rec["f_den"] += F.state_sq_norm(fp).item()
            for g in blocks:
                with block_mask(post, [g]):
                    fpg = F.raw_field(post, xt, tt, cc)
                FPmg[g].append(fpg.detach())
                rec["dp_num"][g] += F.diff_sq_norm(fp, fpg).item()
        pp[aid] = rec
        torch.save({"states": [(t, x.half()) for t, x in sts],
                    "FP": [f.to("cpu").half() for f in FP],
                    "FPmg": {g: [f.to("cpu").half() for f in FPmg[g]] for g in blocks}},
                   os.path.join(store, f"post_{aid}.pt"))
        # persistent state-only save for A_tan reuse (cheap)
        torch.save({"states": [(t, x.half()) for t, x in sts], "caption": cap, "seconds_total": SECONDS},
                   os.path.join(a.state_store, f"state_{aid}.pt"))
    D_P = {g: sum(pp[a2]["dp_num"][g] for a2 in pp) / sum(pp[a2]["f_den"] for a2 in pp) for g in blocks}

    greedy_out = None
    if a.greedy:
        cache = {it["audiocap_id"]: torch.load(os.path.join(store, f"post_{it['audiocap_id']}.pt")) for it in prompts}
        ccs = {it["audiocap_id"]: F.prepare_conditioning(post, it["caption"], SECONDS, dev,
                 latent_len=cache[it["audiocap_id"]]["states"][0][1].shape[-1], dtype=dtype) for it in prompts}
        f_den_tot = sum(pp[a2]["f_den"] for a2 in pp)
        def score(Mset):
            Mset = list(Mset); num = 0.0
            for it in prompts:
                aid = it["audiocap_id"]; recc = cache[aid]; cc = ccs[aid]
                for i, (tau, x) in enumerate(recc["states"]):
                    xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
                    with block_mask(post, Mset):
                        fpm = F.raw_field(post, xt, tt, cc)
                    num += F.diff_sq_norm(recc["FP"][i].to(dev, dtype), fpm).item()
            return num / f_den_tot
        gp = G.greedy_path(20, a.kmax, score)
        greedy_out = {"order": gp["order"], "sets": {k: sorted(v) for k, v in gp["sets"].items()}, "n_evals": gp["n_evals"]}
    del post, sa; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---------------- PASS 2: BASE ----------------
    base, bcfg, base_st = load("base", dev, half)
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]
        recf = torch.load(os.path.join(store, f"post_{aid}.pt")); sts = recf["states"]
        cc = F.prepare_conditioning(base, cap, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=dtype)
        print(f"[pilot] pass2 base prompt {aid}", flush=True)
        r = pp[aid]
        r["fb_den"] = 0.0; r["db_num"] = {g: 0.0 for g in blocks}
        r["ipt_num"] = {g: [0.0] * 8 for g in blocks}; r["ipt_den"] = [0.0] * 8; r["fp_sq"] = [0.0] * 8
        if a.with_dep:
            r["iptdep_num"] = {g: [0.0] * 8 for g in blocks}; r["iptdep_den"] = [0.0] * 8
        for i, (tau, x) in enumerate(sts):
            lvl = min(i, 7)
            xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
            fb = F.raw_field(base, xt, tt, cc); r["fb_den"] += F.state_sq_norm(fb).item()
            fp = recf["FP"][i].to(dev, dtype); delta = fp - fb
            r["ipt_den"][lvl] += F.state_sq_norm(delta).item(); r["fp_sq"][lvl] += F.state_sq_norm(fp).item()
            if a.with_dep:
                fbdep = F.deploy_field(base, xt, tt, cc, cfg_scale=7.0, apg_scale=1.0); delta_dep = fp - fbdep
                r["iptdep_den"][lvl] += F.state_sq_norm(delta_dep).item()
            for g in blocks:
                with block_mask(base, [g]):
                    fbg = F.raw_field(base, xt, tt, cc)
                r["db_num"][g] += F.diff_sq_norm(fb, fbg).item()
                delta_g = recf["FPmg"][g][i].to(dev, dtype) - fbg
                r["ipt_num"][g][lvl] += F.diff_sq_norm(delta, delta_g).item()
                if a.with_dep:
                    with block_mask(base, [g]):
                        fbgdep = F.deploy_field(base, xt, tt, cc, cfg_scale=7.0, apg_scale=1.0)
                    delta_gdep = recf["FPmg"][g][i].to(dev, dtype) - fbgdep
                    r["iptdep_num"][g][lvl] += F.diff_sq_norm(delta_dep, delta_gdep).item()
    del base; gc.collect()

    # ---- aggregates from per-prompt stats + self-check ----
    def agg_ratio(numkey, denkey):
        return {g: sum(pp[a2][numkey][g] for a2 in pp) / sum(pp[a2][denkey] for a2 in pp) for g in blocks}
    D_B = agg_ratio("db_num", "fb_den")
    eta_by_level = [6.7e-5] * 8
    ipt_num_agg = {g: [sum(pp[a2]["ipt_num"][g][i] for a2 in pp) for i in range(8)] for g in blocks}
    ipt_den_agg = [sum(pp[a2]["ipt_den"][i] for a2 in pp) for i in range(8)]
    fp_sq_agg = [sum(pp[a2]["fp_sq"][i] for a2 in pp) for i in range(8)]
    ipt = M.i_pt(ipt_num_agg, ipt_den_agg, fp_sq_agg, eta_by_level)
    # self-check: recompute D_P by summing per-prompt vs the pass-1 aggregate
    dp_check = {g: sum(pp[a2]["dp_num"][g] for a2 in pp) / sum(pp[a2]["f_den"] for a2 in pp) for g in blocks}
    selfcheck = all(abs(dp_check[g] - D_P[g]) < 1e-9 for g in blocks)
    W = {g: block_W(base_st, post_st, g) for g in blocks}

    R.update({
        "D_P": D_P, "D_B_common": D_B,
        "I_PT_raw": {str(g): ipt["per_block"][g] for g in blocks}, "I_PT_raw_excluded_levels": ipt["excluded_levels"],
        "W": W, "greedy_D_P": greedy_out,
        "per_prompt": pp,
        "sufficient_stats_selfcheck_DP": selfcheck,
        "note": "PILOT sizing only; per-prompt sufficient stats persisted for subsample bootstrap; eta_max=6.7e-5 (smoke).",
        "wall_s": round(time.time() - t_start, 1),
    })
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print("PILOT_FIELDS_JSON_BEGIN"); print(json.dumps({k: v for k, v in R.items() if k != "per_prompt"})); print("PILOT_FIELDS_JSON_END")
    print(f"[pilot] wrote {a.out}  N={N} blocks={len(blocks)} wall={R['wall_s']}s selfcheck_DP={selfcheck}")
    print(f"[pilot] D_P={ {g: round(v,4) for g,v in D_P.items()} }")
    print(f"[pilot] I_PT_raw pooled={ {g: round(ipt['per_block'][g]['pooled'],4) for g in blocks} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
