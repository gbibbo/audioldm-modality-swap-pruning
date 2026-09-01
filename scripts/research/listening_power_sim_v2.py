#!/usr/bin/env python3
"""Persistent-rater power + Type-I simulation for the listening study (v1.1 audit).

Adds the real repeated-rater structure absent from listening_power_sim.py:
  y[prompt i, listener L, duration d] =
      mu_d                      # duration mean (the effect of interest)
    + a[i,d]                    # prompt x duration latent (rho_dur-correlated across d)
    + b[L]                      # listener persistent intercept  (persistent heterogeneity)
    + c[L,d]                    # listener x duration interaction
    + p[i,L]                    # prompt x listener residual
    + s[L]*side_sign            # listener side-A tendency (A/B counterbalanced -> ~cancels)
    + eps                       # trial noise
  -> ordinal round-and-clip to {-2,-1,0,+1,+2}; positive = recovered.

Compares three designs under the same 160-judgment sev-1 budget, and two inference
procedures (pooled prompt bootstrap vs listener-stratified prompt bootstrap = fixed-panel,
equal listener weight). Reports H1 power, H1&H2 hierarchical power, and null Type-I.

Only the automatic effect geometry bounds plausible effects; NO human data exists.
Frozen seed. Run: .venv-loudness/bin/python scripts/research/listening_power_sim_v2.py
"""
import json, os, hashlib
import numpy as np

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
OUT = os.path.join(ROOT, "configs/research/listening_study_power_v2.json")
SEED = int.from_bytes(hashlib.sha256(b"LISTENING-STUDY|POWER-SIM-V2|2026-08-31").digest()[:4], "big")
NL = 6
N_SIMS = 2000
B = 1500
ALPHA_LOW = 2.5  # lower bound of two-sided 95% CI


def assign_partition(n_prompts, rng):
    base = n_prompts // NL; rem = n_prompts - base * NL
    sizes = [base + 1] * rem + [base] * (NL - rem)
    owners = np.concatenate([np.full(s, L) for L, s in enumerate(sizes)])
    return owners[rng.permutation(n_prompts)]


def design_assignment(design, rng):
    """Return list of (prompt_id, listener) primary + bridge second-rater ratings.
    prompt ids are integers; each rating covers both durations."""
    if design == "D1":
        owners = assign_partition(80, rng)
        ratings = [(i, int(owners[i])) for i in range(80)]
        n_prompts = 80
    elif design == "D2":
        # 40 prompts, each rated by 2 distinct listeners
        n_prompts = 40
        ratings = []
        for i in range(40):
            Ls = rng.choice(NL, size=2, replace=False)
            ratings += [(i, int(Ls[0])), (i, int(Ls[1]))]
    elif design == "D3":
        owners = assign_partition(80, rng)
        ratings = [(i, int(owners[i])) for i in range(80)]
        n_prompts = 80
        bridge = rng.permutation(80)[:18]
        # assign each bridge prompt a 2nd listener != owner, balanced ~3/listener
        second_slots = np.concatenate([np.full(3, L) for L in range(NL)])
        second_slots = second_slots[rng.permutation(len(second_slots))]
        bi = 0
        for L2 in second_slots:
            # find a bridge prompt whose owner != L2 and not already 2nd-rated by L2
            for _ in range(200):
                p = int(bridge[bi % len(bridge)]); bi += 1
                if owners[p] != L2 and (p, int(L2)) not in ratings:
                    ratings.append((p, int(L2))); break
    return ratings, n_prompts


def simulate(design, mu_n, mu_s, params, rng, center_panel=True):
    """center_panel=True => the six-listener panel-average recovered tendency is exactly
    mu_d (fixed-panel DGP: listener tendencies centered to mean 0). center_panel=False =>
    listener tendencies left random (models NAIVE generalization to a listener population)."""
    sp, sL, sLd, spl, se, rho, sside = (params[k] for k in
        ("sigma_p", "sigma_L", "sigma_Ld", "sigma_pl", "sigma_e", "rho_dur", "sigma_side"))
    ratings, n_prompts = design_assignment(design, rng)
    # latent components
    z1 = rng.standard_normal(n_prompts); z2 = rng.standard_normal(n_prompts)
    a_n = sp * z1
    a_s = sp * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    b = sL * rng.standard_normal(NL)
    c_n = sLd * rng.standard_normal(NL); c_s = sLd * rng.standard_normal(NL)
    if center_panel:
        b = b - b.mean(); c_n = c_n - c_n.mean(); c_s = c_s - c_s.mean()
    sside_L = sside * rng.standard_normal(NL)
    rows = []  # (prompt, listener, Hn, Hs)
    for (i, L) in ratings:
        # side counterbalanced per rating/duration
        side_n = 1 if rng.random() < 0.5 else -1
        side_s = 1 if rng.random() < 0.5 else -1
        yn = mu_n + a_n[i] + b[L] + c_n[L] + spl*rng.standard_normal() + sside_L[L]*side_n + se*rng.standard_normal()
        ys = mu_s + a_s[i] + b[L] + c_s[L] + spl*rng.standard_normal() + sside_L[L]*side_s + se*rng.standard_normal()
        Hn = float(np.clip(round(yn), -2, 2)); Hs = float(np.clip(round(ys), -2, 2))
        rows.append((i, L, Hn, Hs))
    return rows


def pooled_lower(rows, which, rng):
    # unit = prompt; average raters within prompt; resample prompts
    from collections import defaultdict
    agg = defaultdict(list)
    for i, L, Hn, Hs in rows:
        agg[i].append(Hn if which == "n" else (Hn - Hs))
    vals = np.array([np.mean(v) for v in agg.values()])
    n = len(vals)
    idx = rng.integers(0, n, size=(B, n))
    means = vals[idx].mean(axis=1)
    return np.percentile(means, ALPHA_LOW)


def stratified_lower(rows, which, rng):
    # fixed-panel: per listener mean, equal weight; resample prompts within listener.
    # Vectorized across B: per listener draw (B, n_L) indices -> row means -> average listeners.
    from collections import defaultdict
    byL = defaultdict(list)
    for i, L, Hn, Hs in rows:
        byL[L].append(Hn if which == "n" else (Hn - Hs))
    acc = np.zeros(B)
    nL = 0
    for L in sorted(byL.keys()):
        a = np.asarray(byL[L], float); m = len(a)
        acc += a[rng.integers(0, m, size=(B, m))].mean(axis=1)
        nL += 1
    boot = acc / nL
    return np.percentile(boot, ALPHA_LOW)


def run(design, mu_n, mu_s, params, boot, rng, n_sims=N_SIMS, center_panel=True):
    lo = stratified_lower if boot == "strat" else pooled_lower
    h1 = h12 = 0
    for _ in range(n_sims):
        rows = simulate(design, mu_n, mu_s, params, rng, center_panel=center_panel)
        if lo(rows, "n", rng) > 0:
            h1 += 1
            if lo(rows, "j", rng) > 0:
                h12 += 1
    return h1 / n_sims, h12 / n_sims


def main():
    os.chdir(ROOT)
    # persistent-rater params: central + stressed
    P = dict(sigma_p=0.6, sigma_L=0.5, sigma_Ld=0.3, sigma_pl=0.3, sigma_e=0.6, rho_dur=0.5, sigma_side=0.4)
    P_stress = dict(P, sigma_L=0.8, sigma_Ld=0.5)
    designs = ["D1", "D2", "D3"]
    res = {"seed": SEED, "n_sims": N_SIMS, "B": B, "params_central": P, "params_stress": P_stress,
           "model": "prompt x duration + listener intercept + listener x duration + prompt x listener "
                    "+ side bias + trial noise; ordinal round-clip; positive=recovered.",
           "cells": {}}

    def cell(tag, design, mu_n, mu_s, params, boot, center=True):
        rng = np.random.default_rng(SEED + hash(tag) % 9973)
        p1, p12 = run(design, mu_n, mu_s, params, boot, rng, center_panel=center)
        res["cells"][tag] = {"H1": round(p1, 3), "H1_H2": round(p12, 3)}
        print(f"{tag:44s} H1={p1:.3f}  H1&H2={p12:.3f}")

    print("== FIXED-PANEL NULL Type-I (panel-avg=0, listener tendencies centered; nominal 0.025) ==")
    for d in designs:
        for boot in ("pooled", "strat"):
            cell(f"nullA-fixed|{d}|{boot}|central", d, 0.0, 0.0, P, boot, center=True)
    cell("nullA-fixed|D3|strat|stress", "D3", 0.0, 0.0, P_stress, "strat", center=True)
    cell("nullA-fixed|D3|pooled|stress", "D3", 0.0, 0.0, P_stress, "pooled", center=True)

    print("\n== NAIVE-GENERALIZATION Type-I (listener tendencies NOT centered; shows danger) ==")
    for d in designs:
        for boot in ("pooled", "strat"):
            cell(f"nullA-naive|{d}|{boot}|central", d, 0.0, 0.0, P, boot, center=False)

    print("\n== FIXED-PANEL NULL Type-I for J_H (interaction 0; mu_n=mu_s=0.3) ==")
    for d in designs:
        cell(f"nullJ-fixed|{d}|strat|central", d, 0.30, 0.30, P, "strat", center=True)
        cell(f"nullJ-fixed|{d}|pooled|central", d, 0.30, 0.30, P, "pooled", center=True)

    print("\n== POWER (anchor mu_n=0.5, mu_s=0.15), fixed-panel ==")
    for d in designs:
        for boot in ("pooled", "strat"):
            cell(f"pow|{d}|{boot}|anchor", d, 0.50, 0.15, P, boot, center=True)
    print("\n== POWER stressed heterogeneity, strat ==")
    for d in designs:
        cell(f"pow|{d}|strat|stress", d, 0.50, 0.15, P_stress, "strat", center=True)
    print("\n== POWER conservative (mu_n=0.35, mu_s=0.15), strat ==")
    for d in designs:
        cell(f"pow|{d}|strat|conserv", d, 0.35, 0.15, P, "strat", center=True)

    payload = json.dumps(res, indent=2, sort_keys=True)
    res["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json.dump(res, open(OUT, "w"), indent=2, sort_keys=True)
    print("\nWROTE", OUT, res["self_sha256"][:16])


if __name__ == "__main__":
    main()
