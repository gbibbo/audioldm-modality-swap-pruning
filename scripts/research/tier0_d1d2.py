#!/usr/bin/env python3
"""Tier-0 D1+D2 forward diagnostics (plan v4 §7, first Tier-0 GPU job).

One forward-only job over the v4 system set — {base, P0-std, P0-pub, P1-nat, RAND×5} —
on 500 eval slots (disjoint val split, W×K), under three conditionings {audio, text,
unconditional}. From the epsilon predictions it derives:

  D1 (modality asymmetry / RQ-swap wording, DECISION-V4-05): per pruned system, the SIGNED
     ||E_a|| − ||E_t|| and R_mod, overall and per timestep stratum, vs base. Plus a
     Gate-A matched-null SCREEN of the primary baseline P0-std against RAND×5 (Tier-0 is a
     screen; the confirmatory Gate E is Tier 1 with K_rand=20).

  D2 (H-guidance direction): per system and modality m∈{audio,text}, the guidance-norm
     G_S,m = ||eps_S(m) − eps_S(∅)|| and the pruning error in that direction
     ΔG_S,m = ||[eps_S(m)−eps_S(∅)] − [eps_F(m)−eps_F(∅)]|| (H-guidance ΔG_P), with the
     base guidance G_F,m as reference. ∅ = empty-string CLAP text embedding.

Systems are materialized from the M3B saliency artifact (`--saliency`) exactly as the M4
screening does; base weights always come from audioldm-m-full.ckpt. Forward-only, no FAD
(so the F-eval-3 fix does not gate this). Refuses without CUDA unless --dry-run-cpu.
Uses no credit on CPU; the GPU run is ~0.35 credits (compute_budget.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3a_diagnostics import (                     # noqa: E402  (shared library functions)
    BASE_CKPT, RANKING_PKL, VAL_MANIFEST, MASTER_SEED,
    git_provenance, build_eval_slots, eps_for_model, eps_under_text, guidance_norm,
    d1_signed_per_stratum, per_wav_diag,
)
from research_pruning.diagnostics import (        # noqa: E402
    load_config, build_unet, build_clap, build_vae, read_scale_factor, NoiseSchedule,
)
from research_pruning.diagnostics.conditioning import clap_embed  # noqa: E402
from research_pruning.diagnostics.random_masks import (           # noqa: E402
    load_l1_ranking, ranking_full_lengths, random_ranking, materialize, PREREGISTERED_SEEDS,
)
from research_pruning.diagnostics.matched_null import point_estimate, bootstrap_delta_swap  # noqa: E402

SALIENCY = "artifacts/m3_pilot/m3b_saliency.pt"


def ranking_from_saliency(sal_layer_dict, l1_ranking):
    """Descending-saliency per-layer order (materialize keeps first-k = top saliency)."""
    return {k: np.argsort(-sal_layer_dict[k].detach().cpu().numpy()).astype(np.int64).tolist()
            for k in l1_ranking}


def system_rankings(sal, l1_ranking):
    """Per-system per-layer channel order for materialize (keeps FIRST-k of each layer).

    IMPORTANT: P0-std and P0-pub are NOT taken from the saved saliency — the M3B artifact
    stored `P0_L1` and `P0_published` identically because `normalize_within_layer('sum')`
    cancels the sign of ±L1 (`-L1/sum(-L1) == L1/sum(L1)`), collapsing both conventions.
    So we build them from the L1 ranking directly:
      * P0-pub = the published pkl (keep-LOWEST-L1; materialize reproduces the published
        artifact bit-exact, R5).
      * P0-std = the pkl reversed per layer (keep-HIGHEST-L1) — the opposite selection
        (0/2304 kept-set overlap with P0-pub on the ranking-driven layers).
    P1-nat comes from the P1 saliency (a genuine non-degenerate |g·∂L/∂g| distribution).
    """
    return {
        "P0-std": {k: list(reversed(l1_ranking[k])) for k in l1_ranking},
        "P0-pub": {k: list(l1_ranking[k]) for k in l1_ranking},
        "P1-nat": ranking_from_saliency(sal["P1"], l1_ranking),
    }


def preflight(args, result):
    prov = git_provenance()
    result["git"] = prov
    if args.expect_commit and prov["commit"] != args.expect_commit:
        raise SystemExit(f"PREFLIGHT FAIL: commit {prov['commit']} != {args.expect_commit}")
    if prov["dirty"] and not args.allow_dirty and not args.dry_run_cpu:
        raise SystemExit("PREFLIGHT FAIL: dirty tree")
    for p in (BASE_CKPT, RANKING_PKL, VAL_MANIFEST, args.saliency):
        if not os.path.exists(p):
            raise SystemExit(f"PREFLIGHT FAIL: missing {p}")
    if not args.dry_run_cpu:
        if not torch.cuda.is_available():
            raise SystemExit("PREFLIGHT FAIL: no CUDA and not --dry-run-cpu")
        name = torch.cuda.get_device_name(0)
        result["gpu_name"] = name
        if args.expect_gpu and args.expect_gpu.lower() not in name.lower():
            raise SystemExit(f"PREFLIGHT FAIL: GPU {name} != {args.expect_gpu}")


def guidance_block(eps_cond, eps_uncond, eps_base_cond, eps_base_uncond):
    """G_S = mean||eps_S(cond)-eps_S(∅)||; ΔG_S = mean||[S guidance] - [base guidance]||."""
    g_s = guidance_norm(eps_cond, eps_uncond)
    g_base = guidance_norm(eps_base_cond, eps_base_uncond)
    dg = ((eps_cond - eps_uncond) - (eps_base_cond - eps_base_uncond))
    dg = dg.reshape(dg.shape[0], -1).norm(p=2, dim=1)
    return {"G_S": float(g_s.mean()), "G_base": float(g_base.mean()),
            "dG_P": float(dg.mean()), "rel_dG": float((dg.mean() / (g_base.mean() + 1e-12)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=100)     # W (×K=5 → 500 slots)
    ap.add_argument("--k-strata", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-masks", type=int, default=5)      # RAND×5 (DECISION-V4-04, Tier 0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--saliency", default=SALIENCY)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu refuses --out")

    result = {"script": "tier0_d1d2.py", "dry_run_cpu": args.dry_run_cpu}
    preflight(args, result)
    if args.dry_run_cpu:
        args.n_eval, args.k_strata, args.batch, args.n_masks, args.n_boot = 4, 2, 2, 2, 200

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    config = load_config()

    t0 = time.perf_counter()
    full = build_unet(config, BASE_CKPT, channel_mult=None, strict=True).to(device).eval()
    clap = build_clap(config, unconditional_prob=0.0).to(device)
    vae = build_vae(config, BASE_CKPT)
    scale_factor = read_scale_factor(BASE_CKPT)
    schedule = NoiseSchedule(config)
    vae = vae.to(device)
    result["build_s"] = time.perf_counter() - t0

    slots = build_eval_slots(config, args.n_eval, args.k_strata, device, vae, clap, scale_factor, schedule)
    W, K = slots["W"], slots["K"]
    result["W"], result["K"], result["n_slots"] = W, K, slots["z_t"].shape[0]

    # empty-string unconditional text embedding
    e_uncond = clap_embed(clap, [""], "text").reshape(1, 512).cpu()

    # base eps under the three conditionings (computed once)
    t1 = time.perf_counter()
    eps_Fa = eps_for_model(full, slots, device, args.batch, slots["e_a"])
    eps_Ft = eps_for_model(full, slots, device, args.batch, slots["e_t"])
    eps_Fu = eps_under_text(full, slots, device, args.batch, e_uncond)
    result["eps_base_s"] = time.perf_counter() - t1
    del full
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # saliency artifact -> per-system rankings
    sal = torch.load(args.saliency, map_location="cpu")["saliency"]
    l1_ranking = load_l1_ranking(RANKING_PKL)
    full_lengths = ranking_full_lengths(l1_ranking)
    base_sd = torch.load(BASE_CKPT, map_location="cpu")
    base_sd = base_sd.get("state_dict", base_sd)
    base_unet_sd = {k[len("model.diffusion_model."):]: v for k, v in base_sd.items()
                    if k.startswith("model.diffusion_model.")}

    def eps_triplet_for(ranking):
        m = materialize(base_unet_sd, ranking, config).to(device).eval()
        ea = eps_for_model(m, slots, device, args.batch, slots["e_a"])
        et = eps_for_model(m, slots, device, args.batch, slots["e_t"])
        eu = eps_under_text(m, slots, device, args.batch, e_uncond)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return ea, et, eu

    def diagnose(ea, et, eu, label):
        d1 = d1_signed_per_stratum(eps_Fa, eps_Ft, ea, et, slots["stratum_idx"], K)
        dgen_w, rmod_w = per_wav_diag(eps_Fa, eps_Ft, ea, et, slots["wav_idx"], W, K)
        d2 = {"audio": guidance_block(ea, eu, eps_Fa, eps_Fu),
              "text": guidance_block(et, eu, eps_Ft, eps_Fu)}
        return {"label": label, "D1": d1, "D2": d2,
                "per_wav_D_gen_mean": float(dgen_w.mean()), "per_wav_R_mod_mean": float(rmod_w.mean()),
                "_dgen_w": dgen_w, "_rmod_w": rmod_w}

    result["systems"] = {}
    t2 = time.perf_counter()
    # named pruned systems (P0-std/P0-pub from the L1 ranking, NOT the collapsed saliency)
    rankings = system_rankings(sal, l1_ranking)
    sys_diag = {}
    for label in ("P0-std", "P0-pub", "P1-nat"):
        ea, et, eu = eps_triplet_for(rankings[label])
        d = diagnose(ea, et, eu, label)
        sys_diag[label] = d
        result["systems"][label] = {k: v for k, v in d.items() if not k.startswith("_")}
        print(f"  {label:8s} R_mod={d['per_wav_R_mod_mean']:.4f} "
              f"signed={d['D1']['overall']['signed_a_minus_t']:.4f} "
              f"dG_audio={d['D2']['audio']['dG_P']:.4f}")

    # RAND×5 masks (matched null for the P0-std Gate-A screen)
    rand_dgen, rand_rmod = [], []
    for seed in PREREGISTERED_SEEDS[:args.n_masks]:
        ea, et, eu = eps_triplet_for(random_ranking(seed, full_lengths))
        d = diagnose(ea, et, eu, f"RAND-{seed}")
        rand_dgen.append(d["_dgen_w"]); rand_rmod.append(d["_rmod_w"])
        print(f"  RAND {seed}: R_mod={d['per_wav_R_mod_mean']:.4f}")
    result["diag_s"] = time.perf_counter() - t2
    rand_dgen = np.array(rand_dgen); rand_rmod = np.array(rand_rmod)

    # Gate-A matched-null SCREEN: primary baseline = P0-std (DECISION-V4-01)
    prim = sys_diag["P0-std"]
    pe = point_estimate(prim["_dgen_w"], prim["_rmod_w"], rand_dgen, rand_rmod, form="linear")
    boot = bootstrap_delta_swap(slots["wav_ids"], prim["_dgen_w"], prim["_rmod_w"],
                                rand_dgen, rand_rmod, n_boot=args.n_boot, seed=MASTER_SEED, form="linear")
    result["gate_a_screen_P0std"] = {
        "primary": "P0-std", "n_masks": int(args.n_masks),
        "P0std_D_gen": float(prim["_dgen_w"].mean()), "P0std_R_mod": float(prim["_rmod_w"].mean()),
        "rand_D_gen_mean": float(rand_dgen.mean()), "rand_R_mod_mean": float(rand_rmod.mean()),
        "delta_swap": pe["delta_swap"], "standardized_residual": pe["standardized_residual"],
        "bootstrap_ci_95": [float(boot["ci_low"]), float(boot["ci_high"])],
        "note": "SCREEN only (RAND×5); the confirmatory heterogeneity test is Tier-1 Gate E (K_rand=20).",
    }
    result["measured"] = not args.dry_run_cpu

    out = json.dumps(result, indent=2)
    print(out)
    if args.dry_run_cpu:
        print("\nDRY RUN — NO RESULT WRITTEN. Flow validated on a tiny slot subset.")
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"\nRESULT written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
