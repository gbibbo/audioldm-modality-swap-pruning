#!/usr/bin/env python3
"""M3A — RQ1: does structured pruning cause MODALITY-DEPENDENT damage (Gate A)?

Computes the master-plan §3 diagnostics on the real models and runs the matched
random-pruning null (Gate A). For each evaluation slot `(z_t, t, noise)` and each
pruned model it forms the four epsilon predictions on the SAME slot — full/audio,
full/text, pruned/audio, pruned/text — and

    E_a = eps_Pa - eps_Fa,  E_t = eps_Pt - eps_Ft          (per-modality pruning error)
    D_gen = 0.5(||E_a|| + ||E_t||),  D_mod = ||E_a - E_t||,  R_mod = D_mod/(||E_a||+||E_t||)

Random pruning may have larger generic damage `D_gen` than L1, so raw `R_mod` is not
comparable; the matched null fits `R_mod ~ D_gen` across 20 pre-registered random masks
and evaluates `Delta_swap = R_mod^L1 - E[R_mod^random | D_gen^L1]`. **Gate A PASS = 95 %
bootstrap CI of Delta_swap above 0 AND standardized residual ≥ 0.5** (pilot protocol).

Forward-only (no gradients). Evaluation data = the disjoint validation split (never test,
never the calibration/train pool). Cost discipline mirrors the other runners: refuses
without CUDA unless --dry-run-cpu (tiny W/M), fail-fast preflight, prints JSON to stdout.

Device handling follows the m3b_saliency lessons: all data tensors are moved to the model
device before any model touches them, and the schedule buffers live on-device — a CPU
dry-run cannot catch a device mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from research_pruning.diagnostics import (
    load_config, build_unet, build_clap, build_vae, vae_encode,
    read_scale_factor, NoiseSchedule,
)
from research_pruning.diagnostics.conditioning import clap_embed, eps_pred
from research_pruning.diagnostics.modality_diagnostics import modality_diagnostics
from research_pruning.diagnostics.random_masks import (
    load_l1_ranking, ranking_full_lengths, random_ranking, materialize, PREREGISTERED_SEEDS,
)
from research_pruning.diagnostics.matched_null import point_estimate, bootstrap_delta_swap

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
RANKING_PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
VAL_MANIFEST = "configs/research/val_split_disjoint.json"
METADATA_ROOT = "data/dataset/metadata/dataset_root.json"
MASTER_SEED = 20260818
STRATA = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]


def git_provenance():
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], text=True).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def preflight(args, result):
    prov = git_provenance()
    result["git"] = prov
    if args.expect_commit and prov["commit"] != args.expect_commit:
        raise SystemExit(f"PREFLIGHT FAIL: commit {prov['commit']} != {args.expect_commit}")
    if prov["dirty"] and not args.allow_dirty and not args.dry_run_cpu:
        raise SystemExit("PREFLIGHT FAIL: dirty tree")
    for p in (BASE_CKPT, RANKING_PKL, VAL_MANIFEST):
        if not os.path.exists(p):
            raise SystemExit(f"PREFLIGHT FAIL: missing {p}")
    if not args.dry_run_cpu:
        if not torch.cuda.is_available():
            raise SystemExit("PREFLIGHT FAIL: no CUDA and not --dry-run-cpu")
        name = torch.cuda.get_device_name(0)
        result["gpu_name"] = name
        if args.expect_gpu and args.expect_gpu.lower() not in name.lower():
            raise SystemExit(f"PREFLIGHT FAIL: GPU {name} != {args.expect_gpu}")


def build_eval_slots(config, W, K, device, vae, clap, scale_factor, schedule):
    """Deterministic eval slots from the disjoint val split.

    Returns dict with z_t [S,C,H,W], t [S], e_a [S,1,512], e_t [S,1,512], and the
    per-slot wav index [S] (S = W*K), all on CPU. Slots are built with a seeded RNG so
    the (wav, stratum, timestep, noise) draw is reproducible.
    """
    import os as _os
    from audioldm_train.utilities.data.dataset import AudioDataset
    root = json.load(open(METADATA_ROOT))["audiocaps"]
    items = json.load(open(VAL_MANIFEST))["items"]
    # deterministic wav selection: seeded permutation of the val split, first W
    perm = np.random.default_rng(MASTER_SEED).permutation(len(items))
    chosen = [items[i] for i in perm[:W]]
    data = []
    for it in chosen:
        abs_wav = _os.path.join(root, it["wav"])
        if not _os.path.exists(abs_wav):
            raise SystemExit(f"eval wav missing: {abs_wav}")
        data.append({"wav": abs_wav, "caption": it["caption"]})
    ds = AudioDataset(config=config, split="test", waveform_only=False, dataset_json={"data": data})

    z0s, e_as, e_ts = [], [], []
    with torch.no_grad():
        for i in range(len(chosen)):
            s = ds[i]
            wav = s["waveform"]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav = wav.unsqueeze(0).float().to(device)
            mel = s["log_mel_spec"].unsqueeze(0).unsqueeze(0).float().to(device)
            z0s.append(vae_encode(vae, mel, scale_factor).squeeze(0).cpu())
            e_as.append(clap_embed(clap, wav, "audio").squeeze(0).cpu())
            e_ts.append(clap_embed(clap, [chosen[i]["caption"]], "text").squeeze(0).cpu())

    tgen = np.random.default_rng(MASTER_SEED + 2)  # eval timesteps (distinct sub-seed)
    z_t, t_list, e_a_list, e_t_list, wav_idx = [], [], [], [], []
    for w in range(len(chosen)):
        z0 = z0s[w]
        for k, (lo, hi) in enumerate(STRATA[:K]):
            t = int(tgen.integers(lo, hi))
            g = torch.Generator().manual_seed((MASTER_SEED * 7919 + w * 101 + k) % (2**63 - 1))
            noise = torch.randn((1,) + tuple(z0.shape), generator=g)
            z_t.append(schedule.q_sample(z0.unsqueeze(0).cpu(), torch.tensor([t]), noise).squeeze(0))
            t_list.append(t)
            e_a_list.append(e_as[w])
            e_t_list.append(e_ts[w])
            wav_idx.append(w)
    return {
        "z_t": torch.stack(z_t), "t": torch.tensor(t_list, dtype=torch.long),
        "e_a": torch.stack(e_a_list), "e_t": torch.stack(e_t_list),
        "wav_idx": np.array(wav_idx), "W": len(chosen), "K": K,
        "wav_ids": [c["wav"] for c in chosen],
    }


@torch.no_grad()
def eps_for_model(unet, slots, device, batch, modality_emb):
    """eps_pred for every slot under one fixed conditioning tensor set. Returns [S,C,H,W] on CPU."""
    S = slots["z_t"].shape[0]
    out = []
    for i in range(0, S, batch):
        z = slots["z_t"][i:i+batch].to(device)
        t = slots["t"][i:i+batch].to(device)
        y = modality_emb[i:i+batch].to(device)  # [b,1,512]
        eps = eps_pred(unet, z, t, y)
        out.append(eps.cpu())
    return torch.cat(out, dim=0)


def per_wav_diag(eps_Fa, eps_Ft, eps_Pa, eps_Pt, wav_idx, W, K):
    """modality_diagnostics per slot, then mean over each wav's K strata -> [W]."""
    d = modality_diagnostics(eps_Fa, eps_Ft, eps_Pa, eps_Pt)
    dgen = d["D_gen"].numpy(); rmod = d["R_mod"].numpy()
    dgen_w = np.zeros(W); rmod_w = np.zeros(W); cnt = np.zeros(W)
    for s, w in enumerate(wav_idx):
        dgen_w[w] += dgen[s]; rmod_w[w] += rmod[s]; cnt[w] += 1
    return dgen_w / cnt, rmod_w / cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--k-strata", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-masks", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu refuses --out")

    result = {"script": "m3a_diagnostics.py", "dry_run_cpu": args.dry_run_cpu}
    preflight(args, result)
    if args.dry_run_cpu:
        args.n_eval, args.k_strata, args.batch, args.n_masks, args.n_boot = 4, 2, 2, 3, 200

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    config = load_config()

    t0 = time.perf_counter()
    full = build_unet(config, BASE_CKPT, channel_mult=None, strict=True).to(device).eval()
    clap = build_clap(config, unconditional_prob=0.0).to(device)
    vae = build_vae(config, BASE_CKPT).to(device)
    scale_factor = read_scale_factor(BASE_CKPT)
    # Schedule stays on CPU: it is used ONLY inside build_eval_slots, which builds z_t on
    # CPU (z0/t/noise all CPU) and stores it on CPU; eps_for_model moves each chunk to the
    # device. Moving the buffers to the device would make q_sample a cross-device op.
    schedule = NoiseSchedule(config)
    result["build_s"] = time.perf_counter() - t0

    slots = build_eval_slots(config, args.n_eval, args.k_strata, device, vae, clap, scale_factor, schedule)
    W, K = slots["W"], slots["K"]
    result["W"], result["K"], result["n_slots"] = W, K, slots["z_t"].shape[0]

    # eps for the FULL model under both modalities (computed once)
    t1 = time.perf_counter()
    eps_Fa = eps_for_model(full, slots, device, args.batch, slots["e_a"])
    eps_Ft = eps_for_model(full, slots, device, args.batch, slots["e_t"])
    result["eps_full_s"] = time.perf_counter() - t1
    del full
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ranking = load_l1_ranking(RANKING_PKL)
    full_lengths = ranking_full_lengths(ranking)
    base_sd = torch.load(BASE_CKPT, map_location="cpu")
    base_sd = base_sd.get("state_dict", base_sd)
    base_unet_sd = {k[len("model.diffusion_model."):]: v for k, v in base_sd.items()
                    if k.startswith("model.diffusion_model.")}

    def diag_for_ranking(rk):
        m = materialize(base_unet_sd, rk, config).to(device).eval()
        pa = eps_for_model(m, slots, device, args.batch, slots["e_a"])
        pt = eps_for_model(m, slots, device, args.batch, slots["e_t"])
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return per_wav_diag(eps_Fa, eps_Ft, pa, pt, slots["wav_idx"], W, K)

    # L1 (published pruning artifact)
    t2 = time.perf_counter()
    l1_dgen, l1_rmod = diag_for_ranking(ranking)
    # 20 pre-registered random masks
    rand_dgen, rand_rmod = [], []
    seeds = PREREGISTERED_SEEDS[:args.n_masks]
    for si, seed in enumerate(seeds):
        dg, rm = diag_for_ranking(random_ranking(seed, full_lengths))
        rand_dgen.append(dg); rand_rmod.append(rm)
        print(f"  mask {si+1}/{len(seeds)} (seed {seed}): D_gen={dg.mean():.4f} R_mod={rm.mean():.4f}")
    result["diag_s"] = time.perf_counter() - t2
    rand_dgen = np.array(rand_dgen); rand_rmod = np.array(rand_rmod)

    # matched null + Gate A
    pe = point_estimate(l1_dgen, l1_rmod, rand_dgen, rand_rmod, form="linear")
    boot = bootstrap_delta_swap(slots["wav_ids"], l1_dgen, l1_rmod, rand_dgen, rand_rmod,
                                n_boot=args.n_boot, seed=MASTER_SEED, form="linear")
    ci_lo, ci_hi = boot["ci_low"], boot["ci_high"]
    std_resid = pe["standardized_residual"]
    # Gate A PASS: 95% bootstrap CI of Delta_swap above zero AND standardized residual >= 0.5
    gate_a_pass = bool(ci_lo > 0 and std_resid is not None and std_resid >= 0.5)
    result.update({
        "l1_D_gen": float(l1_dgen.mean()), "l1_R_mod": float(l1_rmod.mean()),
        "rand_D_gen_mean": float(rand_dgen.mean()), "rand_R_mod_mean": float(rand_rmod.mean()),
        "delta_swap": pe["delta_swap"],
        "standardized_residual": std_resid,
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "gate_a_pass": gate_a_pass,
        "n_masks": len(seeds),
        "measured": not args.dry_run_cpu,
    })

    out = json.dumps(result, indent=2)
    print(out)
    print(f"\nGATE A: {'PASS' if gate_a_pass else 'FAIL'}  "
          f"Delta_swap={result['delta_swap']}  CI95=[{ci_lo:.4f},{ci_hi:.4f}]  std_resid={std_resid}")
    if args.dry_run_cpu:
        print("DRY RUN — NO RESULT WRITTEN.")
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"RESULT written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
