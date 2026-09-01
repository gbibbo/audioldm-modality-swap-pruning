#!/usr/bin/env python3
"""v1.2 estimator calibration/power sim. CPU only, no human data.

Implements the EXACT deployed D3 estimator (unique-prompt weighting): average a
prompt's ratings WITHIN prompt (bridge prompts average their two listeners' ratings),
then take the equal-weight mean over the 80 unique prompts and bootstrap the 80 unique
prompt records. Compares against the v1.1 listener-stratified estimator (which
double-weights the 18 bridge prompts) and the pooled-all-ratings estimator.

Reuses the persistent-rater DGP from listening_power_sim_v2.py (listener intercept +
listener x duration + prompt x listener + correlated durations + side bias + ordinal).

Run: .venv-loudness/bin/python scripts/research/listening_power_sim_v3.py
"""
import json, os, hashlib, importlib.util
import numpy as np

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
OUT = os.path.join(ROOT, "configs/research/listening_study_power_v3.json")
SEED = int.from_bytes(hashlib.sha256(b"LISTENING-STUDY|POWER-SIM-V3|2026-08-31").digest()[:4], "big")
B = 3000
N_SIMS = 3000
ALPHA_LOW = 2.5

spec = importlib.util.spec_from_file_location("v2", os.path.join(ROOT, "scripts/research/listening_power_sim_v2.py"))
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)


def uniqueprompt_lower(rows, which, rng, B=B):
    """v1.2 PRIMARY: average within prompt (unique-prompt weighting), bootstrap 80 unique records."""
    from collections import defaultdict
    byP = defaultdict(list)
    for i, L, Hn, Hs in rows:
        byP[i].append(Hn if which == "n" else (Hn - Hs))
    vals = np.array([np.mean(v) for v in byP.values()])  # one value per unique prompt
    n = len(vals)
    means = vals[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(np.percentile(means, ALPHA_LOW))


def strat_lower(rows, which, rng, B=B):
    from collections import defaultdict
    byL = defaultdict(list)
    for i, L, Hn, Hs in rows:
        byL[L].append(Hn if which == "n" else (Hn - Hs))
    acc = np.zeros(B); nL = 0
    for L in sorted(byL):
        a = np.asarray(byL[L], float); m = len(a)
        acc += a[rng.integers(0, m, size=(B, m))].mean(axis=1); nL += 1
    return float(np.percentile(acc / nL, ALPHA_LOW))


def pooled_all_lower(rows, which, rng, B=B):
    vals = np.array([(Hn if which == "n" else Hn - Hs) for i, L, Hn, Hs in rows])
    n = len(vals)
    means = vals[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(np.percentile(means, ALPHA_LOW))


EST = {"uniqueprompt": uniqueprompt_lower, "strat_v11": strat_lower, "pooled_all": pooled_all_lower}


def run(mu_n, mu_s, params, est, rng, center=True, n_sims=N_SIMS):
    lo = EST[est]; h1 = h12 = 0
    for _ in range(n_sims):
        rows = v2.simulate("D3", mu_n, mu_s, params, rng, center_panel=center)
        if lo(rows, "n", rng) > 0:
            h1 += 1
            if lo(rows, "j", rng) > 0:
                h12 += 1
    return h1 / n_sims, h12 / n_sims


def main():
    os.chdir(ROOT)
    P = dict(sigma_p=0.6, sigma_L=0.5, sigma_Ld=0.3, sigma_pl=0.3, sigma_e=0.6, rho_dur=0.5, sigma_side=0.4)
    Pst = dict(P, sigma_L=0.8, sigma_Ld=0.5)
    res = {"seed": SEED, "n_sims": N_SIMS, "B": B, "design": "D3",
           "estimators": list(EST), "params": P, "params_stress": Pst, "cells": {}}

    def cell(tag, mu_n, mu_s, params, est, center=True):
        rng = np.random.default_rng(SEED + hash(tag) % 9973)
        p1, p12 = run(mu_n, mu_s, params, est, rng, center=center)
        res["cells"][tag] = {"H1": round(p1, 3), "H1_H2": round(p12, 3)}
        print(f"{tag:38s} H1={p1:.3f}  H1&H2={p12:.3f}")

    print("== FIXED-PANEL NULL Type-I (centered; nominal 0.025) ==")
    for e in EST:
        cell(f"nullA|{e}", 0.0, 0.0, P, e)
    print("== NAIVE-GENERALIZATION Type-I (uncentered) ==")
    for e in EST:
        cell(f"nullA-naive|{e}", 0.0, 0.0, P, e, center=False)
    print("== FIXED-PANEL J_H NULL (interaction 0; mu=0.3) ==")
    for e in EST:
        cell(f"nullJ|{e}", 0.30, 0.30, P, e)
    print("== POWER anchor (0.5,0.15) ==")
    for e in EST:
        cell(f"pow-anchor|{e}", 0.50, 0.15, P, e)
    print("== POWER conservative (0.35,0.15) ==")
    for e in EST:
        cell(f"pow-conserv|{e}", 0.35, 0.15, P, e)
    print("== POWER stressed heterogeneity (0.5,0.15) ==")
    for e in EST:
        cell(f"pow-stress|{e}", 0.50, 0.15, Pst, e)

    payload = json.dumps(res, indent=2, sort_keys=True)
    res["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json.dump(res, open(OUT, "w"), indent=2, sort_keys=True)
    print("\nWROTE", OUT, res["self_sha256"][:16])


if __name__ == "__main__":
    main()
