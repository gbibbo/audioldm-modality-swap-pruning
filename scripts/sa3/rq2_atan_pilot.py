#!/usr/bin/env python3
"""RQ2 primary A_tan pilot with PER-(prompt x probe x block x kappa) sufficient statistics.

A_tan(g) = E_u ||dF(u) - dF^{-g}(u)||^2 / E_u ||dF(u)||^2   (protocol 3.4), U_gen = standard LoRA r16.
dF(u)      = F_{P+u} - F_P                       (probe on vs off)
dF^{-g}(u) = F^{-g}_{P+u_{-g}} - F^{-g}_P        (probe restricted to surviving blocks; block g removed)

Persisted per (prompt p, probe u, kappa kf):  denom[p,u,kf] = Σ_states ||dF(u)||^2
                                  per block g:  num_tan[p,u,g,kf] = Σ_states ||dF(u) - dF^{-g}(u)||^2
so A_tan(g | prompt-subsample, probe-subsample) = Σ num / Σ denom is recomputable for ANY N and n_u
without re-running forwards (ratio of sums). Linearity check ||dF(2u)||/||dF(u)|| at the smallest kappa.
REUSES the S_traj states persisted by pilot_fields.py (--state-store) — no regeneration.

Run (GPU):  _external/stable-audio-3/.venv/bin/python scripts/sa3/rq2_atan_pilot.py --device cuda \
                --n 8 --n-u 8 --state-store artifacts/sa3/pilot_states --expect-commit <sha> \
                --out artifacts/sa3/atan_pilot.json
CPU dry:   OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/rq2_atan_pilot.py --dry-run-cpu \
                --state-store artifacts/sa3/pilot_states_dry
"""
from __future__ import annotations
import argparse, gc, json, os, subprocess, sys, time, glob
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, fields as F, probes as P
from research_sa3.blockskip import block_mask

SECONDS = 10


def sha256_file(p):
    import hashlib; return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_post(device, half):
    d = "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device=device)
    if half:
        model = model.half()
    return model, cfg


def sq(v):
    return F.state_sq_norm(v).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--n-u", type=int, default=8)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--kappa", type=float, default=0.01)
    ap.add_argument("--kappa-grid", default="1.0,0.5")  # multipliers of --kappa
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--state-store", default="artifacts/sa3/pilot_states")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--out", default="artifacts/sa3/atan_pilot.json")
    a = ap.parse_args()
    if a.dry_run_cpu:
        a.device = "cpu"; a.n = min(a.n, 2); a.n_u = min(a.n_u, 2); a.blocks = a.blocks or "5,13"; a.kappa_grid = "1.0"
    dev = a.device; half = (dev == "cuda"); dtype = torch.float16 if half else torch.float32
    blocks = [int(x) for x in a.blocks.split(",")] if a.blocks else list(range(20))
    kgrid = [float(x) for x in a.kappa_grid.split(",")]

    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    state_files = sorted(glob.glob(os.path.join(a.state_store, "state_*.pt")),
                         key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))[:a.n]
    assert state_files, f"no persisted states in {a.state_store} (run pilot_fields.py first)"
    post, cfg = load_post(dev, half)

    R = {"phase": "atan_pilot", "device": dev, "dry_run_cpu": a.dry_run_cpu, "n": len(state_files),
         "n_u": a.n_u, "blocks": blocks, "kappa": a.kappa, "kappa_grid": kgrid, "rank": a.rank,
         "family": "U_gen (standard lora r%d)" % a.rank, "state_store": a.state_store,
         "git_commit": subprocess.getoutput("git rev-parse HEAD"), "upstream_commit": loading.SA3_UPSTREAM_COMMIT}
    t0 = time.time()
    pp = {}   # per_prompt_probe sufficient stats
    lin_checks = []

    for sf in state_files:
        aid = os.path.basename(sf).split("_")[1].split(".")[0]
        d = torch.load(sf); sts = d["states"]; cap = d["caption"]
        cc = F.prepare_conditioning(post, cap, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=dtype)
        print(f"[atan] prompt {aid}", flush=True)
        # baselines (probe OFF): F_P and F_P^{-g}
        FP = []; FPmg = {g: [] for g in blocks}
        for tau, x in sts:
            xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
            FP.append(F.raw_field(post, xt, tt, cc).detach())
            for g in blocks:
                with block_mask(post, [g]):
                    FPmg[g].append(F.raw_field(post, xt, tt, cc).detach())
        prec = {}
        for u in range(a.n_u):
            for kf in kgrid:
                P.build_probe(post, family="U_gen", kappa=a.kappa * kf, rank=a.rank, seed=1000 + u)
                P.set_strength(post, 1.0)
                # dF(u) and num_tan per block
                denom = 0.0; num_tan = {g: 0.0 for g in blocks}
                dFu = []
                for i, (tau, x) in enumerate(sts):
                    xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
                    fpu = F.raw_field(post, xt, tt, cc)
                    dfu = (fpu.float() - FP[i].float())
                    dFu.append(dfu.detach()); denom += sq(dfu)
                for g in blocks:
                    # dF^{-g}(u_{-g}) = F^{-g}_{P+u_{-g}} - F^{-g}_P ; restrict probe off block g, remove block g
                    P.restrict_to_surviving(post, removed_block=g)
                    for i, (tau, x) in enumerate(sts):
                        xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
                        with block_mask(post, [g]):
                            fpu_mg = F.raw_field(post, xt, tt, cc)
                        dfu_mg = (fpu_mg.float() - FPmg[g][i].float())
                        num_tan[g] += sq(dFu[i] - dfu_mg)
                    P.set_strength(post, 1.0)  # restore full probe for next g
                # linearity at the smallest kappa multiplier
                if kf == min(kgrid):
                    base_norm = sum(sq(v) for v in dFu) ** 0.5
                    P.probe_scale(post, 2.0)
                    d2 = 0.0
                    for i, (tau, x) in enumerate(sts):
                        xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
                        d2 += sq(F.raw_field(post, xt, tt, cc).float() - FP[i].float())
                    ratio = (d2 ** 0.5) / base_norm if base_norm > 0 else float("nan")
                    lin_checks.append({"aid": aid, "u": u, "kappa": a.kappa * kf, "ratio_2u_over_u": ratio})
                P.remove_probe(post)
                prec[f"{u}|{kf}"] = {"denom": denom, "num_tan": {str(g): num_tan[g] for g in blocks}}
        pp[aid] = prec
        gc.collect()
    del post; gc.collect()

    # aggregate A_tan over all prompts+probes at each kappa multiplier
    def atan_at(kf):
        num = {g: 0.0 for g in blocks}; den = 0.0
        for aid in pp:
            for key, rec in pp[aid].items():
                if abs(float(key.split("|")[1]) - kf) < 1e-9:
                    den += rec["denom"]
                    for g in blocks:
                        num[g] += rec["num_tan"][str(g)]
        return {g: (num[g] / den if den > 0 else float("nan")) for g in blocks}
    atan = {str(kf): atan_at(kf) for kf in kgrid}
    import numpy as np
    lr = [c["ratio_2u_over_u"] for c in lin_checks]
    R.update({"per_prompt_probe": pp, "A_tan_by_kappa": atan,
              "linearity_checks": lin_checks,
              "linearity_ok_range_1p9_2p1": bool(len(lr) and all(1.9 <= r <= 2.1 for r in lr)),
              "linearity_median": float(np.median(lr)) if lr else None,
              "note": "PILOT; per-(prompt,probe,kappa) sufficient stats persisted for N/n_u bootstrap.",
              "wall_s": round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print("ATAN_JSON_BEGIN"); print(json.dumps({k: v for k, v in R.items() if k != "per_prompt_probe"})); print("ATAN_JSON_END")
    print(f"[atan] wrote {a.out} n={R['n']} n_u={a.n_u} wall={R['wall_s']}s linearity_ok={R['linearity_ok_range_1p9_2p1']} (median {R['linearity_median']})")
    print(f"[atan] A_tan (kappa mult {kgrid[0]}): { {g: round(v,4) for g,v in atan[str(kgrid[0])].items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
