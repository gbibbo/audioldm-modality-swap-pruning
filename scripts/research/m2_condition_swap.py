#!/usr/bin/env python3
"""M2 evidence generator for docs/condition_swap_validation.md (CPU-only).

Produces, from real AudioLDM-M-Full weights and real AudioCaps items:

  (c) L2-norm distribution of CLAP audio/text embeddings over N>=32 items;
  (d) paired cosine (e_a vs e_t of the SAME item) vs cross-item cosine, with a
      same-item > different-item sanity check;
  (e) CPU seconds per U-Net forward at batch 1 and batch 4, and an estimated CPU
      cost of one full M3A diagnostic pass (plan B if no GPU arrives);
  (f) two figures: norm histograms and paired-cosine histogram.

All numbers are written to artifacts/m2_condition_swap/ so every reported value
is backed by a saved log/JSON. No synthetic audio, no random weights.

Usage:
    .venv/bin/python scripts/research/m2_condition_swap.py --n 48
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research_pruning.diagnostics.conditioning import (  # noqa: E402
    FROZEN_CONFIG,
    build_clap,
    build_unet,
    clap_embed,
    eps_pred,
    load_config,
)
from audioldm_train.utilities.data.dataset import AudioDataset  # noqa: E402

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
OUT_DIR = "artifacts/m2_condition_swap"


def quantiles(values: list[float]) -> dict:
    s = sorted(values)
    def q(p):
        if len(s) == 1:
            return s[0]
        idx = p * (len(s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    return {"p05": q(0.05), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p95": q(0.95)}


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def embed_all(clap, dataset, indices, batch=4):
    """Return (E_a, E_t) each [N,512] for the given item indices (batched)."""
    e_a_list, e_t_list = [], []
    for start in range(0, len(indices), batch):
        chunk = indices[start:start + batch]
        wavs, texts = [], []
        for idx in chunk:
            s = dataset[idx]
            wav = s["waveform"]
            wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
            wavs.append(wav)
            texts.append(s["text"])
        wav_batch = torch.stack(wavs, dim=0)  # [b,1,T]
        e_a_list.append(clap_embed(clap, wav_batch, "audio").squeeze(1))
        e_t_list.append(clap_embed(clap, texts, "text").squeeze(1))
    return torch.cat(e_a_list, 0), torch.cat(e_t_list, 0)


def time_unet(unet, config, batch_sizes=(1, 4), reps=5):
    params = config["model"]["params"]
    C, H, W = params["channels"], params["latent_t_size"], params["latent_f_size"]
    timings = {}
    for bs in batch_sizes:
        z = torch.randn(bs, C, H, W)
        t = torch.randint(0, params["timesteps"], (bs,), dtype=torch.long)
        y = torch.randn(bs, 1, 512)
        # warmup
        eps_pred(unet, z, t, y)
        samples = []
        for _ in range(reps):
            t0 = time.time()
            eps_pred(unet, z, t, y)
            samples.append(time.time() - t0)
        timings[bs] = {
            "sec_per_forward_mean": statistics.fmean(samples),
            "sec_per_forward_min": min(samples),
            "sec_per_item_mean": statistics.fmean(samples) / bs,
            "reps": reps,
        }
    return timings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="number of real items (>=32)")
    ap.add_argument("--config", default=FROZEN_CONFIG)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    assert args.n >= 32, "master plan requires N>=32 real items for norm stats"

    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    config = load_config(args.config)
    clap = build_clap(config, unconditional_prob=0.0)
    unet = build_unet(config, BASE_CKPT, channel_mult=None)  # base [1,2,3,5]
    dataset = AudioDataset(config=config, split=args.split, waveform_only=False)
    indices = list(range(args.n))

    # ---- (c) norms ----
    t0 = time.time()
    E_a, E_t = embed_all(clap, dataset, indices, batch=4)
    embed_sec = time.time() - t0
    norms_a = E_a.norm(dim=-1).tolist()
    norms_t = E_t.norm(dim=-1).tolist()

    # ---- (d) cosine ----
    E_a_n = torch.nn.functional.normalize(E_a, dim=-1)
    E_t_n = torch.nn.functional.normalize(E_t, dim=-1)
    paired_cos = (E_a_n * E_t_n).sum(-1).tolist()  # same item, audio vs text
    # cross-item: audio_i vs text_j (j = i+1 cyclic), different items
    perm = list(range(1, len(indices))) + [0]
    cross_cos = (E_a_n * E_t_n[perm]).sum(-1).tolist()
    # full off-diagonal mean for a broader cross baseline
    sim_matrix = E_a_n @ E_t_n.t()
    n = sim_matrix.size(0)
    off_diag = sim_matrix[~torch.eye(n, dtype=torch.bool)].tolist()

    # ---- (e) timing + M3A estimate ----
    timings = time_unet(unet, config)
    tfwd_item_b4 = timings[4]["sec_per_item_mean"]
    # M3A plan-B estimate: full + L1 + 20 random masks = 22 models; 2 modalities;
    # N_eval items; S timestep strata; one forward per (model, modality, item, t).
    est = {}
    for n_eval in (64, 200):
        for strata in (1, 5):
            models = 22
            modalities = 2
            forwards = models * modalities * n_eval * strata
            est[f"n_eval={n_eval},strata={strata}"] = {
                "forwards": forwards,
                "cpu_hours_at_batch4_rate": forwards * tfwd_item_b4 / 3600.0,
            }

    results = {
        "provenance": {
            "config": args.config,
            "base_ckpt": BASE_CKPT,
            "split": args.split,
            "n_items": args.n,
            "channel_mult": config["model"]["params"]["unet_config"]["params"]["channel_mult"],
            "embed_seconds_total": embed_sec,
        },
        "c_norms": {
            "audio": {**describe(norms_a)},
            "text": {**describe(norms_t)},
        },
        "d_cosine": {
            "paired_same_item": {**describe(paired_cos), **quantiles(paired_cos)},
            "cross_next_item": {**describe(cross_cos), **quantiles(cross_cos)},
            "cross_off_diagonal": {**describe(off_diag), **quantiles(off_diag)},
            "sanity_same_gt_cross": statistics.fmean(paired_cos) > statistics.fmean(off_diag),
        },
        "e_timing": {
            "cpu_sec_per_forward": timings,
            "m3a_plan_b_estimate": est,
            "estimate_assumptions": (
                "22 models (full + L1 + 20 random masks) x 2 modalities x N_eval "
                "items x S timestep strata, one U-Net forward each, at the measured "
                "batch-4 per-item CPU rate. Excludes CLAP embedding, VAE encode and "
                "backward passes (M3A diagnostics are forward-only)."
            ),
        },
    }

    with open(f"{OUT_DIR}/condition_swap_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- (f) figures ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(norms_a, bins=20, alpha=0.6, label=f"||e_a|| (audio), n={len(norms_a)}")
    ax.hist(norms_t, bins=20, alpha=0.6, label=f"||e_t|| (text), n={len(norms_t)}")
    ax.set_xlabel("L2 norm of CLAP embedding")
    ax.set_ylabel("count")
    ax.set_title("M2: CLAP embedding L2-norm distribution (real AudioCaps items)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/embedding_norm_hist.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(paired_cos, bins=20, alpha=0.7, label="paired (same item)")
    ax.hist(cross_cos, bins=20, alpha=0.7, label="cross (different items)")
    ax.set_xlabel("cosine(e_a, e_t)")
    ax.set_ylabel("count")
    ax.set_title("M2: paired vs cross-item audio/text CLAP cosine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/paired_cosine_hist.png", dpi=120)
    plt.close(fig)

    # ---- console summary ----
    print(json.dumps(results, indent=2))
    print(f"\nfigures: {OUT_DIR}/embedding_norm_hist.png, {OUT_DIR}/paired_cosine_hist.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
