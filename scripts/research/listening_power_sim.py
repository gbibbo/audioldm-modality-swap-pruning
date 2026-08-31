#!/usr/bin/env python3
"""CPU Monte-Carlo power / MDE simulation for the blinded listening study.

Compares two balanced-incomplete designs under a matched sev-1 listening budget
(160 severity-1 judgments total):

  D1  (maximize prompt diversity): 80 sev-1 prompts, 1 rater each, both durations.
  D2  (rater replication):         40 sev-1 prompts, 2 raters each, both durations.

Primary human endpoints simulated (prompt is the unit):
  A_native = mean_i H_native_i               ; H1 PASS iff lower95(A_native) > 0
  J_H      = mean_i (H_native_i - H_short_i)  ; H2 PASS iff lower95(J_H) > 0 (only if H1)

Inference = paired prompt percentile bootstrap (matches frozen protocol), B_boot.
Latent per-prompt signed preference on the -2..+2 comparative scale is mapped to the
ordinal by round-and-clip (natural tie band |y|<0.5 -> 0). Rater noise averaged
within prompt. Scenario ranges are grounded in the EXISTING AUTOMATIC effect geometry
(FineLAP native sev-1 recovered mass 0.304 vs pruned 0.127; CLAP native R_alt +0.052;
short rec~pru) only to bound plausible human effects -- NOT to select prompts and NOT
using any human outcome (none exist).

No GPU. Frozen seed namespace. Run:
  .venv-loudness/bin/python scripts/research/listening_power_sim.py
"""
import json, os, hashlib
import numpy as np

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
OUT = os.path.join(ROOT, "configs/research/listening_study_power.json")
SEED_NS = "LISTENING-STUDY|POWER-SIM|2026-08-31"
SEED = int.from_bytes(hashlib.sha256(SEED_NS.encode()).digest()[:4], "big")

N_SIMS = 3000
B_BOOT = 1500
ALPHA = 0.05  # two-sided 95% percentile CI -> lower 2.5%


def to_ordinal(y):
    return np.clip(np.rint(y), -2, 2)


def sim_dataset(rng, n_prompts, n_raters, mu_n, mu_s, sigma_p, sigma_r, rho):
    """Return per-prompt rater-averaged H_native, H_short (length n_prompts)."""
    theta_n = mu_n + sigma_p * rng.standard_normal(n_prompts)
    z = rng.standard_normal(n_prompts)
    theta_s = mu_s + rho * (theta_n - mu_n) + np.sqrt(max(0.0, 1 - rho ** 2)) * sigma_p * z
    # rater observations: (n_prompts, n_raters)
    yn = theta_n[:, None] + sigma_r * rng.standard_normal((n_prompts, n_raters))
    ys = theta_s[:, None] + sigma_r * rng.standard_normal((n_prompts, n_raters))
    Hn = to_ordinal(yn).mean(axis=1)
    Hs = to_ordinal(ys).mean(axis=1)
    return Hn, Hs


def boot_lower(rng, vals, B):
    n = len(vals)
    idx = rng.integers(0, n, size=(B, n))
    means = vals[idx].mean(axis=1)
    return np.percentile(means, 100 * (ALPHA / 2))


def power_cell(rng, design, mu_n, mu_s, sigma_p, sigma_r, rho,
               n_sims=N_SIMS, b_boot=B_BOOT):
    n_prompts, n_raters = design
    pass_h1 = 0
    pass_h1h2 = 0
    for _ in range(n_sims):
        Hn, Hs = sim_dataset(rng, n_prompts, n_raters, mu_n, mu_s, sigma_p, sigma_r, rho)
        lo_a = boot_lower(rng, Hn, b_boot)
        h1 = lo_a > 0
        if h1:
            pass_h1 += 1
            lo_j = boot_lower(rng, Hn - Hs, b_boot)
            if lo_j > 0:
                pass_h1h2 += 1
    return pass_h1 / n_sims, pass_h1h2 / n_sims


def mde(rng, design, kind, sigma_p, sigma_r, rho, mu_s_base=0.1, target=0.80):
    """Smallest effect reaching `target` power. kind='A' sweeps mu_n; kind='J' sweeps delta."""
    grid = np.round(np.arange(0.10, 0.86, 0.05), 3)
    found = None
    curve = []
    for g in grid:
        if kind == "A":
            p1, _ = power_cell(rng, design, g, 0.0, sigma_p, sigma_r, rho, n_sims=1500, b_boot=1200)
            curve.append((float(g), round(p1, 3)))
            if p1 >= target and found is None:
                found = float(g)
        else:  # J: delta = mu_n - mu_s
            mu_n = mu_s_base + g
            _, p12 = power_cell(rng, design, mu_n, mu_s_base, sigma_p, sigma_r, rho, n_sims=1500, b_boot=1200)
            curve.append((float(g), round(p12, 3)))
            if p12 >= target and found is None:
                found = float(g)
    return found, curve


def main():
    os.chdir(ROOT)
    designs = {"D1": (80, 1), "D2": (40, 2)}
    result = {
        "artifact": "listening_study_power",
        "seed_namespace": SEED_NS, "seed": SEED,
        "n_sims": N_SIMS, "b_boot": B_BOOT,
        "model": "per-prompt latent signed preference (-2..+2), rho-correlated across duration, "
                 "rater noise averaged within prompt, ordinal=round-and-clip(|y|<0.5->0 tie), "
                 "paired prompt percentile bootstrap lower95>0",
        "designs": {k: {"sev1_prompts": v[0], "raters_per_prompt": v[1],
                         "sev1_judgments": v[0] * v[1] * 2} for k, v in designs.items()},
        "scenario_grounding": "FineLAP native sev1 recovered mass 0.304 vs pruned 0.127 (large); "
                              "CLAP native R_alt +0.052 [0.009,0.093]; short rec~pru; used only to bound "
                              "plausible human effect sizes, not to select prompts or using any human data.",
    }

    # central plausible dispersion
    SP, SR, RHO = 0.6, 0.8, 0.5

    # 1) Anchor planning scenario + robustness corners
    scenarios = {
        "anchor":        dict(mu_n=0.50, mu_s=0.15, sigma_p=0.6, sigma_r=0.8, rho=0.5),
        "conservative":  dict(mu_n=0.35, mu_s=0.15, sigma_p=0.6, sigma_r=0.8, rho=0.5),
        "optimistic":    dict(mu_n=0.70, mu_s=0.15, sigma_p=0.6, sigma_r=0.8, rho=0.5),
        "noisy_raters":  dict(mu_n=0.50, mu_s=0.15, sigma_p=0.6, sigma_r=1.1, rho=0.5),
        "heterog":       dict(mu_n=0.50, mu_s=0.15, sigma_p=0.8, sigma_r=0.8, rho=0.5),
        "low_rho":       dict(mu_n=0.50, mu_s=0.15, sigma_p=0.6, sigma_r=0.8, rho=0.3),
    }
    result["scenarios"] = {}
    for name, sc in scenarios.items():
        rng = np.random.default_rng(SEED)
        row = {"params": sc, "power": {}}
        for dk, dv in designs.items():
            p1, p12 = power_cell(rng, dv, sc["mu_n"], sc["mu_s"], sc["sigma_p"], sc["sigma_r"], sc["rho"])
            row["power"][dk] = {"H1_A_native": round(p1, 3), "H1_and_H2_J": round(p12, 3)}
        result["scenarios"][name] = row
        print(name, sc)
        for dk in designs:
            print("   ", dk, row["power"][dk])

    # 2) MDE for A_native and J_H, both designs (central dispersion)
    result["mde"] = {}
    for dk, dv in designs.items():
        rng = np.random.default_rng(SEED + 7)
        mA, curveA = mde(rng, dv, "A", SP, SR, RHO)
        mJ, curveJ = mde(rng, dv, "J", SP, SR, RHO)
        result["mde"][dk] = {
            "central_dispersion": {"sigma_p": SP, "sigma_r": SR, "rho": RHO},
            "MDE80_A_native_mu_n": mA, "curveA": curveA,
            "MDE80_J_H_delta": mJ, "curveJ": curveJ,
        }
        print(f"MDE {dk}: A_native mu_n80={mA}  J_H delta80={mJ}")

    # 3) sev-2 secondary A_native_2 power (single-rater, N=36), anchor effect
    rng = np.random.default_rng(SEED + 13)
    p1_36, _ = power_cell(rng, (36, 1), 0.50, 0.15, 0.6, 0.8, 0.5)
    p1_30, _ = power_cell(rng, (30, 1), 0.50, 0.15, 0.6, 0.8, 0.5)
    # sev-2 automatic native advantage is LARGE (R_native +0.244); model a bigger effect too
    rng = np.random.default_rng(SEED + 17)
    p1_36_big, _ = power_cell(rng, (36, 1), 0.80, 0.15, 0.6, 0.8, 0.5)
    result["sev2_secondary_A_native_2"] = {
        "N36_muN0.50": round(p1_36, 3), "N30_muN0.50": round(p1_30, 3),
        "N36_muN0.80_large": round(p1_36_big, 3),
        "note": "secondary, single rater; sev-2 automatic native advantage is large (R_native +0.244).",
    }
    print("sev2 A_native_2:", result["sev2_secondary_A_native_2"])

    payload = json.dumps(result, indent=2, sort_keys=True)
    result["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("WROTE", OUT, "self_sha256", result["self_sha256"][:16])


if __name__ == "__main__":
    main()
