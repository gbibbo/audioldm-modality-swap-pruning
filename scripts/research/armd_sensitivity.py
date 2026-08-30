#!/usr/bin/env python3
"""ARM D (OP-DURATION-DISCRIMINATOR-1) — corrected sensitivity for the EXACT matched design (CPU, 0 cr).

Design (per supervisor D1/D3): both arms single-replicate r0, paired at ytid, n=80.
  control = V1.1 r0 CLAP (existing WAVs, to be RE-SCORED as one 80-item group; NOT the 192-item hist).
  alt     = new 10.24 s r0 CLAP (not yet generated).
  J_i = (rec_alt_i - pru_alt_i) - (rec_ctrl_r0_i - pru_ctrl_r0_i);  J_CLAP = mean_i J_i (bootstrap ytids).

Estimates Var(J_hat) from the observed V1.1 r0 contrast distribution, assuming the alt arm shares the
same per-ytid contrast variance structure and a difficulty correlation rho across operating points.
GATE: proceed to freeze only if MDE(J, 80% power) <= 0.075 CLAP at rho=0 (conservative).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/armd_sensitivity.py
"""
from __future__ import annotations
import hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np

V1 = "configs/research/reversal_v1_1_result.json"
MANIFEST = "configs/research/reversal_v1_1_audiocaps_manifest.json"
SUBSET_SALT = "OP-DURATION-DISCRIMINATOR-1|SUBSET|2026-08-30"
N_SUBSET = 80
GATE_MDE = 0.075
Z80 = 2.80  # ~ (z_{.975}+z_{.80}) for 80% power, two-sided 5%


def subset_ytids():
    ys = [p["ytid"] for p in json.load(open(MANIFEST))["prompts"]]
    keyed = sorted(ys, key=lambda y: hashlib.sha256(f"{SUBSET_SALT}|YTID|{y}".encode("utf-8")).hexdigest())
    return set(keyed[:N_SUBSET]), keyed[:N_SUBSET], ys


def r0_contrasts(ytids_all):
    d = json.load(open(V1))
    rc = d["raw_clap_scores"]
    rec = np.array(rc["p1_recovered"]); pru = np.array(rc["p1_pruned_ema_reconstructed"])
    # prompt-major: index 2i = r0, 2i+1 = r1 (verified in op_discriminator design audit)
    rec_r0 = rec.reshape(96, 2)[:, 0]; pru_r0 = pru.reshape(96, 2)[:, 0]
    d_r0 = rec_r0 - pru_r0
    return {y: float(d_r0[i]) for i, y in enumerate(ytids_all)}, d_r0


def mde_matched(var_d, n, rho):
    # both arms single-rep; Var(J_i)=2*var_d*(1-rho) under equal-variance, difficulty-corr rho.
    se = (2 * var_d * (1 - rho) / n) ** 0.5
    return se, Z80 * se


def main():
    sel_set, sel_list, ys = subset_ytids()
    dmap, d_r0_all = r0_contrasts(ys)
    d_sel = np.array([dmap[y] for y in sel_list])

    var_all = float(np.var(d_r0_all, ddof=1))   # from all 96 (robust population estimate)
    var_sel = float(np.var(d_sel, ddof=1))       # from the 80 selected
    out = {
        "artifact": "armd_sensitivity", "design": "matched r0-only, paired, n=80",
        "subset_salt": SUBSET_SALT, "n_subset": N_SUBSET,
        "subset_sha256_of_sorted": hashlib.sha256(
            json.dumps(sel_list, sort_keys=False).encode()).hexdigest(),
        "var_d_r0_all96": var_all, "sd_d_r0_all96": var_all ** 0.5,
        "var_d_r0_selected80": var_sel, "sd_d_r0_selected80": var_sel ** 0.5,
        "mean_d_r0_all96": float(d_r0_all.mean()), "mean_d_r0_selected80": float(d_sel.mean()),
        "projection": {}, "gate_mde": GATE_MDE,
    }
    for label, var in [("from_all96", var_all), ("from_selected80", var_sel)]:
        out["projection"][label] = {}
        for rho in (0.0, 0.5, 0.7):
            se, mde = mde_matched(var, N_SUBSET, rho)
            out["projection"][label][f"rho={rho}"] = {"SE_J": round(se, 4),
                                                       "CI95_halfwidth": round(1.96 * se, 4),
                                                       "MDE_J_80pct": round(mde, 4)}
    mde_conservative = out["projection"]["from_all96"]["rho=0.0"]["MDE_J_80pct"]
    out["gate_pass"] = bool(mde_conservative <= GATE_MDE)
    out["gate_statistic"] = {"conservative_MDE_rho0_all96": mde_conservative, "threshold": GATE_MDE}
    os.makedirs("configs/research", exist_ok=True)
    json.dump(out, open("configs/research/armd_sensitivity.json", "w"), indent=2)

    print(f"subset: {N_SUBSET} of 96 ytids by SUBSET_SALT; sha(sorted80)="
          f"{out['subset_sha256_of_sorted'][:16]}")
    print(f"V1.1 r0 contrast SD: all96={var_all**0.5:.4f}  selected80={var_sel**0.5:.4f}  "
          f"(mean sel {d_sel.mean():+.4f})")
    print("MDE(J, 80% power) matched r0-only design, n=80:")
    for label in ("from_all96", "from_selected80"):
        row = out["projection"][label]
        print(f"  {label}: rho0={row['rho=0.0']['MDE_J_80pct']}  "
              f"rho.5={row['rho=0.5']['MDE_J_80pct']}  rho.7={row['rho=0.7']['MDE_J_80pct']}")
    print(f"\nGATE (conservative MDE rho0 from all96 <= {GATE_MDE}): "
          f"{mde_conservative} -> {'PASS' if out['gate_pass'] else 'FAIL — STOP before freeze'}")
    return 0 if out["gate_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
