#!/usr/bin/env python3
"""Power simulation for the SEVERITY-SWEEP overdispersion test (proposal, 2026-08-21).

Context. After Tier 0, the event-level heterogeneity of P0-std at −65 % is mostly sampling
noise (noise-corrected between-event SD of loss δ ≈ 0.11 on 35 events, ≈ 0 on the 8 events
with ≥ 5 prompts; `docs/tier1_proposal.md` §0). Gate E (RAND×20 exact rank test) needs a
30×30 sentinel (20 700 clips ≈ 49 cr) to reach 87 % power at δ = 0.15. Before funding that,
the cheaper prior question is whether there is ANY event structure beyond a homogeneous
("generic capacity loss") degradation, and whether it grows with pruning severity — a
severity sweep over the measured nested budgets (1,2,3,4) −23.7 % / (1,2,3,3) −42.5 % /
(1,2,3,2) −56.2 % / (1,2,3,1) −65.0 % with the SAME P0-std ranking.

Statistic (per severity level, panel of E events × n prompts, base + pruned recall):
  V = var_e( Lhat(e) ),   Lhat(e) = recall_base(e) − recall_pruned(e).
Null H0 ("generic"): a common logit shift, q_e = expit(logit(p_e) − β), with p_e the event's
base recall — i.e. every event degrades by the same log-odds, differences in Lhat come from
(i) finite-sample binomial noise and (ii) the MECHANICAL heterogeneity of a bounded outcome
(an event with p_e = 0.1 cannot lose 0.2). Both are absorbed by the null.
Test: parametric bootstrap — plug in smoothed p̂_e = (y_base+½)/(n+1) and the 1-parameter
MLE β̂ (bisection), draw B replicate panels under H0, p = (1 + #{V* ≥ V}) / (B + 1).
H1: q_e = expit(logit(p_e) − β + u_e), u_e ~ N(0, τ²). The effect is reported as
δ = SD over events of the TRUE loss p_e − q_e, so it is on the same scale as the Gate-E
power simulation (`gate_e_power_sim.py`, Q7).

p_e ~ Beta(a, b) fitted to the Tier-0 screen base recalls (35 events with ≥ 3 prompts,
sampling-deflated: mean 0.568, SD 0.269 → a = 1.354, b = 1.031). Base and pruned draws are
independent; the real design is seed-paired, which only reduces Var(Lhat) → conservative.

What this test does NOT establish: criterion-specificity vs random structured masks (that is
Gate E proper, which needs RAND×20 audio). It answers the prior question "is there event
structure at all, at which severity does it appear, and does it track mean damage".

    .venv/bin/python scripts/research/severity_sweep_power_sim.py \
        --out artifacts/gate_e_power/severity_sweep_power.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BETA_A, BETA_B = 1.354, 1.031       # Tier-0 screen base-recall Beta fit (thr>=3, deflated)
COST_PER_CLIP_CR = 0.00235          # measured, screen-1 (600 clips -> 1.412 cr), DDIM S=50


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def fit_common_shift(yb, yp, n, iters=40):
    """Smoothed base recall and the 1-parameter MLE of the common logit shift β (per row)."""
    p_hat = (yb + 0.5) / (n + 1.0)
    lo = np.full(yb.shape[:-1], -8.0)
    hi = np.full(yb.shape[:-1], 8.0)
    target = yp.sum(-1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pred = (n * expit(logit(p_hat) - mid[..., None])).sum(-1)   # decreasing in β
        hi = np.where(pred > target, hi, mid)
        lo = np.where(pred > target, mid, lo)
    return p_hat, 0.5 * (lo + hi)


def v_stat(yb, yp, n):
    return ((yb - yp) / n).var(-1)


def overdispersion_pvalue(rng, yb, yp, n, B):
    """Parametric-bootstrap p-value of V under the common-logit-shift null. yb, yp: [..., E]."""
    v_obs = v_stat(yb, yp, n)
    p_hat, beta_hat = fit_common_shift(yb, yp, n)
    q_hat = expit(logit(p_hat) - beta_hat[..., None])
    ge = np.zeros(v_obs.shape)
    for _ in range(B):
        ge += (v_stat(rng.binomial(n, p_hat), rng.binomial(n, q_hat), n) >= v_obs)
    samp = ((p_hat * (1 - p_hat) + q_hat * (1 - q_hat)) / n).mean(-1)
    latent_sd = np.sqrt(np.clip(v_obs - samp, 0.0, None))      # method-of-moments read-out
    return (1 + ge) / (B + 1), latent_sd


def simulate(rng, *, n_events, n_prompts, tau, beta, a, b, n_sim, B, alpha):
    p = np.clip(rng.beta(a, b, size=(n_sim, n_events)), 0.02, 0.98)
    u = rng.normal(0.0, tau, size=(n_sim, n_events)) if tau > 0 else 0.0
    q = expit(logit(p) - beta + u)
    yb = rng.binomial(n_prompts, p)
    yp = rng.binomial(n_prompts, q)
    pval, latent_sd = overdispersion_pvalue(rng, yb, yp, n_prompts, B)
    return {
        "tau": float(tau),
        "delta_true": float((p - q).std(-1).mean()),      # SD over events of TRUE loss
        "mu_true": float((p - q).mean()),
        "power": float((pval <= alpha).mean()),
        "latent_sd_mean": float(latent_sd.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="35x6,50x20,25x20,20x30,30x30,50x40",
                    help="events x prompts per event; 50x20 = the frozen Q2 mechanism set (999 prompts)")
    ap.add_argument("--betas", default="0.22,0.46,1.0",
                    help="common logit shifts (mean loss ≈ 0.05 / 0.10 / 0.20 at the fitted Beta)")
    ap.add_argument("--taus", default="0.0,0.3,0.5,0.7,1.0,1.4")
    ap.add_argument("--beta-a", type=float, default=BETA_A)
    ap.add_argument("--beta-b", type=float, default=BETA_B)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-sim", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=199)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    designs = [tuple(int(x) for x in d.split("x")) for d in args.designs.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    taus = [float(x) for x in args.taus.split(",")]

    rows = []
    for n_events, n_prompts in designs:
        for beta in betas:
            for tau in taus:
                r = simulate(rng, n_events=n_events, n_prompts=n_prompts, tau=tau, beta=beta,
                             a=args.beta_a, b=args.beta_b, n_sim=args.n_sim, B=args.n_boot,
                             alpha=args.alpha)
                r.update(n_events=n_events, n_prompts=n_prompts, prompts=n_events * n_prompts,
                         beta=beta, clips_per_system=n_events * n_prompts,
                         cr_per_system=round(n_events * n_prompts * COST_PER_CLIP_CR, 2))
                rows.append(r)
                print(f"E={n_events:>3} n={n_prompts:>3} mu≈{r['mu_true']:.2f} tau={tau:.1f} "
                      f"delta={r['delta_true']:.3f} power={r['power']:.3f} "
                      f"latentSD={r['latent_sd_mean']:.3f}", flush=True)

    result = {
        "inputs": {"beta_a": args.beta_a, "beta_b": args.beta_b, "alpha": args.alpha,
                   "n_sim": args.n_sim, "n_boot": args.n_boot, "seed": args.seed,
                   "cost_per_clip_cr": COST_PER_CLIP_CR,
                   "note": "p_e ~ Beta fitted to Tier-0 screen base recalls (thr>=3, deflated); "
                           "independent base/pruned draws (conservative vs seed pairing)"},
        "rows": rows,
    }
    fp = [r for r in rows if r["tau"] == 0.0]
    print("Type-I (tau=0) range: %.3f–%.3f" % (min(r["power"] for r in fp), max(r["power"] for r in fp)))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
