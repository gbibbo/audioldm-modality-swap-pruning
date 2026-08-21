#!/usr/bin/env python3
"""Paired per-prompt OpenL3 drift diagnostic (review task 4) — replaces set-level FD at N=16.

FD_openl3 estimates a 512-dim covariance from N=16 clips (rank <= 15): a degenerate small-sample
Frechet. Instead, a PAIRED per-prompt drift avoids covariance estimation entirely.

PRE-REGISTERED config (fixed BEFORE looking at results):
  * OpenL3: content_type=env, input_repr=mel256, embedding_size=512, hop_size=1.0 s; clip embedding
    = mean over the 11 frames (L2-normalized).
  * Metric: cosine drift  d(g,p) = 1 - cos( emb(skip_g, p), emb(dense8_s0, p) )  per prompt p
    (paired: same prompt, same seed). Chosen: COSINE (not L2). One metric, fixed here.
  * Per block g: mean drift over prompts + bootstrap 95% CI (resample prompts, B=1000).
  * Null floor: the same drift between equivalent dense-8 seed streams (s_r vs s_0), pooled -> the
    seed-noise drift a genuinely-equivalent system incurs. A block is "drifts beyond seed noise"
    iff its drift CI lower bound exceeds the 95th pct of the null.
FD_openl3 (set-level) is retained descriptively for larger panels; the smoke FD is NOT deleted.

Run: OPENBLAS_CORETYPE=Haswell HF_HOME=... .venv-metrics/bin/python scripts/sa3/paired_openl3_drift.py \
        --manifest <abs manifest> --out artifacts/sa3/openl3_drift.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torchvision  # noqa: F401 -- torchvision-first guard

HOP = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260818); a = ap.parse_args()
    import torch, librosa, torchopenl3
    man = json.load(open(a.manifest)); S = man["systems"]; ref = "dense8_s0"
    aids = sorted(S[ref], key=lambda x: int(x))

    cache = {}
    def emb(fp):
        if fp not in cache:
            w, _ = librosa.load(fp, sr=48000, mono=True)
            e, _ = torchopenl3.get_audio_embedding(torch.tensor(w).unsqueeze(0), 48000,
                     content_type="env", input_repr="mel256", embedding_size=512, hop_size=HOP)
            v = e.squeeze(0).mean(0).detach().cpu().numpy()
            cache[fp] = v / (np.linalg.norm(v) + 1e-9)
        return cache[fp]

    def drift(sid):
        d = []
        for p in aids:
            if p in S[sid] and p in S[ref]:
                d.append(1.0 - float(np.dot(emb(S[sid][p]), emb(S[ref][p]))))
        return np.array(d)

    def boot_ci(d, B=1000):
        rng = np.random.default_rng(a.seed)
        idx = rng.integers(0, len(d), size=(B, len(d)))
        means = d[idx].mean(1)
        return float(np.mean(d)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    # null: dense-8 stream drift vs s0 (seed-noise), pooled
    null = []
    for r in range(1, man["R"]):
        sid = f"dense8_s{r}"
        if sid in S:
            null.extend(drift(sid).tolist())
    null = np.array(null)
    null_p95 = float(np.percentile(null, 95)) if len(null) else float("nan")
    null_mean = float(np.mean(null)) if len(null) else float("nan")

    blocks = {}
    for sid in sorted(S, key=lambda s: s):
        if not sid.startswith("skip"):
            continue
        g = int(sid.replace("skip", "")); d = drift(sid)
        m, lo, hi = boot_ci(d)
        blocks[g] = {"mean_drift": m, "ci95": [lo, hi], "beyond_seed_noise": lo > null_p95}
    # dense step ladder drift (descriptive)
    ladder = {}
    for st in [7, 6, 5, 4]:
        sid = f"dense{st}_s0"
        if sid in S:
            d = drift(sid); ladder[f"dense{st}"] = {"mean_drift": float(np.mean(d))}

    beyond = sorted([g for g in blocks if blocks[g]["beyond_seed_noise"]])
    out = {"metric": "1 - cosine(mean OpenL3 emb), paired per prompt; env/mel256/512 hop=1.0s",
           "pre_registered": True, "N": len(aids),
           "null_seed_drift": {"mean": null_mean, "p95": null_p95, "n": len(null)},
           "dense_ladder_drift": ladder, "per_block": blocks,
           "blocks_beyond_seed_noise": beyond, "n_beyond": len(beyond),
           "note": "paired per-prompt drift (avoids FD covariance rank-deficiency at N=16); "
                   "descriptive pilot, not a main-panel CASE-E decision."}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"null seed drift: mean={null_mean:.4f} p95={null_p95:.4f}")
    print("dense ladder drift:", {k: round(v['mean_drift'],4) for k,v in ladder.items()})
    print(f"{'blk':>3} {'mean':>7} {'ci_lo':>7} {'ci_hi':>7} {'>seed':>6}")
    for g in sorted(blocks):
        b=blocks[g]; print(f"{g:>3} {b['mean_drift']:7.4f} {b['ci95'][0]:7.4f} {b['ci95'][1]:7.4f} {str(b['beyond_seed_noise']):>6}")
    print(f"\nblocks with drift beyond seed noise: {beyond} ({len(beyond)}/20)")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
