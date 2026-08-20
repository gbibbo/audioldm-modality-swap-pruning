#!/usr/bin/env python3
"""Gate E power simulation (CPU queue Q7; milestone M4-1b, DECISION-V4-07).

Gate E (plan v4 §6) declares event-level forgetting heterogeneous iff, on the sentinel
panel, P0-std's between-event variance of per-event loss `V_P0` beats a `K_rand`-mask
random null by the EXACT rank test `p = (1 + #{V_rand >= V_P0}) / (K_rand + 1) <= 0.05`.
With K_rand = 20 the attainable minimum p is 1/21 ≈ 0.048, so the design can in principle
reject; whether it reaches adequate POWER at a plausible effect is what this script decides
BEFORE any Tier-1 spend. If power < 80 % at the pre-set minimum detectable effect (MDE),
the sentinel panel must be resized.

Parametric bootstrap (all CPU, no model):
  * Per event e, per prompt: base capture ~ Bernoulli(p_base); pruned capture ~
    Bernoulli(p_base − loss_e). Per-event loss L(e) = recall_base(e) − recall_pruned(e),
    estimated from `n_prompts` prompts (finite-sample noise included).
  * H1 (heterogeneous): loss_e ~ clip(Normal(mu_loss, delta), 0, p_base) — `delta` is the
    between-event SD of the true loss = the effect size / MDE.
  * H0 / random masks: loss_e = mu_loss for every event (matched on generic damage, no
    event-specific structure); their between-event variance is finite-sample noise only.
  * Statistic V = variance over events of the estimated L(e); rank test as above.

`p_base` and `mu_loss` come from the Tier-0 screening (200 stratified prompts × {base,
P0-std, P1-nat}); until that lands they are CLI args with documented placeholders. Base and
pruned captures are drawn independently (the real design is seed-paired, which only
*reduces* loss-estimate variance → this power estimate is conservative).

    .venv/bin/python scripts/research/gate_e_power_sim.py \
        --p-base 0.6 --mu-loss 0.15 --out artifacts/gate_e_power/power.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _estimated_loss_var(rng, delta, n_events, n_prompts, p_base, mu_loss, n_sim):
    """Return an array [n_sim] of between-event variances of the estimated per-event loss.

    delta = 0 gives the homogeneous (null / random-mask) case.
    """
    # true per-event loss: Normal(mu_loss, delta) clipped to [0, p_base]
    if delta > 0:
        loss = rng.normal(mu_loss, delta, size=(n_sim, n_events))
    else:
        loss = np.full((n_sim, n_events), mu_loss, dtype=float)
    loss = np.clip(loss, 0.0, p_base)
    p_pruned = p_base - loss                                    # [n_sim, n_events]
    # finite-sample recall estimates from n_prompts Bernoulli trials
    recall_base = rng.binomial(n_prompts, p_base, size=(n_sim, n_events)) / n_prompts
    recall_pruned = rng.binomial(n_prompts, p_pruned) / n_prompts
    est_loss = recall_base - recall_pruned                     # [n_sim, n_events]
    return est_loss.var(axis=1)                                # [n_sim]


def simulate_power(delta, *, n_events, n_prompts, k_rand, p_base, mu_loss,
                   n_sim, alpha, seed):
    """Fraction of simulations where Gate E's exact rank test rejects, at effect `delta`."""
    rng = np.random.default_rng(seed)
    # V for P0-std at this effect
    v_p0 = _estimated_loss_var(rng, delta, n_events, n_prompts, p_base, mu_loss, n_sim)
    # V for each of k_rand random masks (homogeneous), per simulation
    v_rand = np.stack([
        _estimated_loss_var(rng, 0.0, n_events, n_prompts, p_base, mu_loss, n_sim)
        for _ in range(k_rand)
    ], axis=1)                                                 # [n_sim, k_rand]
    ge = (v_rand >= v_p0[:, None]).sum(axis=1)                 # #{V_rand >= V_P0}
    p = (1 + ge) / (k_rand + 1)
    return float((p <= alpha).mean())


def find_mde(*, deltas, power_target, **kw):
    """Smallest delta on the grid whose power >= power_target (None if none reach it)."""
    curve = [(float(d), simulate_power(d, **kw)) for d in deltas]
    mde = next((d for d, pw in curve if pw >= power_target), None)
    return mde, curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=20)      # sentinel events (V4-07)
    ap.add_argument("--n-prompts", type=int, default=15)     # prompts per event (V4-07)
    ap.add_argument("--k-rand", type=int, default=20)        # RAND masks (V4-04)
    ap.add_argument("--p-base", type=float, default=0.6, help="base recall of requested events (Tier-0)")
    ap.add_argument("--mu-loss", type=float, default=0.15, help="mean recall drop under pruning (Tier-0)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power-target", type=float, default=0.80)
    ap.add_argument("--n-sim", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--deltas", default="0.0,0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.35,0.40")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    kw = dict(n_events=args.n_events, n_prompts=args.n_prompts, k_rand=args.k_rand,
              p_base=args.p_base, mu_loss=args.mu_loss, n_sim=args.n_sim,
              alpha=args.alpha, seed=args.seed)

    mde, curve = find_mde(deltas=deltas, power_target=args.power_target, **kw)
    fp_rate = simulate_power(0.0, **kw)   # calibration: power at delta=0 ≈ alpha

    result = {
        "design": {"n_events": args.n_events, "n_prompts": args.n_prompts,
                   "k_rand": args.k_rand, "alpha": args.alpha,
                   "power_target": args.power_target},
        "screening_inputs": {"p_base": args.p_base, "mu_loss": args.mu_loss,
                             "note": "placeholders until the Tier-0 screening lands (M4-1b)"},
        "n_sim": args.n_sim, "seed": args.seed,
        "false_positive_rate_at_delta0": fp_rate,
        "power_curve": [{"delta": d, "power": pw} for d, pw in curve],
        "mde_delta": mde,
        "mde_reaches_target": mde is not None,
    }
    print(f"design {args.n_events}×{args.n_prompts}, K_rand={args.k_rand}, "
          f"p_base={args.p_base}, mu_loss={args.mu_loss}")
    print(f"FP rate at delta=0: {fp_rate:.3f} (target ≈ alpha={args.alpha})")
    for d, pw in curve:
        mark = "  <- MDE" if d == mde else ""
        print(f"  delta={d:.3f}  power={pw:.3f}{mark}")
    print(f"MDE (>= {args.power_target:.0%} power): "
          f"{'delta=%.3f' % mde if mde is not None else 'NOT REACHED on this grid — resize panel'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
