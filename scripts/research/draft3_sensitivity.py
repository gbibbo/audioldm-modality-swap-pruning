#!/usr/bin/env python3
"""DRAFT3-SENSITIVITY — post-hoc sensitivity analyses of the frozen per-prompt CLAP scores (CPU, 0 cr).

Authorised by Gabriel 2026-09-02 (review docs/review/2026-09-02_manuscript_draft2_scientific_review.md,
items B1 + B2). Reads ONLY committed/frozen raw scores; changes no frozen verdict. Everything here is
labelled POST-HOC SENSITIVITY in the manuscript.

B1 — scale-free duration interaction, both severities:
    win-rate  w(op) = frac_prompts[ CLAP(P+FT) > CLAP(P) ]  at short / native
    dW = w(native) - w(short)          (paired per-prompt indicator difference; prompt bootstrap CI)
    paired standardised effect  d(op) = mean(r_op) / sd(r_op)  (descriptive)
    duration slopes  s(sys) = mean[ CLAP_native - CLAP_short ]  per system (descriptive, with CI)
B2 — matched-duration domain contrast at severity 2 (3.84 s):
    D_short = R_short(AudioCaps) - R_music     (independent two-sample prompt bootstrap, as K)
  (severity 1 is already frozen as I = R_AC - R_music = +0.092 [+0.054,+0.131], reversal_v1_1_result.json)

Bootstrap: unit = prompt, B = 10000, percentile 95%, seed namespace
"DRAFT3-SENSITIVITY|BOOTSTRAP|2026-09-02" -> PCG64(int(sha256(ns)[:8],16) % 2**31).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft3_sensitivity.py
"""
from __future__ import annotations
import hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np

NS = "DRAFT3-SENSITIVITY|BOOTSTRAP|2026-09-02"
SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000
OPD = "configs/research/op_duration_discriminator_1_result.json"
SEV2 = "artifacts/icassp_gate0/_score_tmp/xsev_sev2_groups_out.json"   # persisted frozen-scorer output
SEV2_FROZEN = "configs/research/xsev_result.json"
OUT = "configs/research/draft3_sensitivity_result.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def pct_ci(vals, stat, rng, b=B):
    """Percentile CI of a statistic over prompt-resampled arrays (vals: dict name->array, same n)."""
    n = len(next(iter(vals.values())))
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        boots[i] = stat({k: v[idx] for k, v in vals.items()})
    point = stat(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n": int(n)}


def two_sample_ci(a, b_, rng, b=B):
    """Independent prompt bootstrap of mean(a) - mean(b_)."""
    boots = np.empty(b)
    for i in range(b):
        boots[i] = a[rng.integers(0, len(a), len(a))].mean() - b_[rng.integers(0, len(b_), len(b_))].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(a.mean() - b_.mean()), "lo": float(lo), "hi": float(hi),
            "n_a": int(len(a)), "n_b": int(len(b_))}


def severity_block(rec_s, pru_s, rec_n, pru_n, rng):
    r_s, r_n = rec_s - pru_s, rec_n - pru_n
    v = {"r_s": r_s, "r_n": r_n, "rec_s": rec_s, "pru_s": pru_s, "rec_n": rec_n, "pru_n": pru_n}
    out = {
        "win_rate_short": float((r_s > 0).mean()),
        "win_rate_native": float((r_n > 0).mean()),
        "dW_native_minus_short": pct_ci(v, lambda x: (x["r_n"] > 0).mean() - (x["r_s"] > 0).mean(), rng),
        "paired_d_short": float(r_s.mean() / r_s.std(ddof=1)),
        "paired_d_native": float(r_n.mean() / r_n.std(ddof=1)),
        "slope_pruned": pct_ci(v, lambda x: (x["pru_n"] - x["pru_s"]).mean(), rng),
        "slope_postft": pct_ci(v, lambda x: (x["rec_n"] - x["rec_s"]).mean(), rng),
        "J_raw_check": pct_ci(v, lambda x: (x["r_n"] - x["r_s"]).mean(), rng),
    }
    return out


def main():
    rng = np.random.default_rng(np.random.PCG64(SEED))
    # severity 1: Arm-D matched 80-ytid raw cosines (frozen artifact)
    opd = json.load(open(OPD)); rc = {k: np.asarray(v, float) for k, v in opd["raw_cosines"].items()}
    sev1 = severity_block(rc["recovered_ctrl"], rc["pruned_ctrl"], rc["recovered_alt"], rc["pruned_alt"], rng)
    # severity 2: persisted frozen-scorer groups (the exact input of the frozen verdict; reproduces it)
    g = json.load(open(SEV2)); c = {r["name"]: np.asarray(r["cosines"], float) for r in g["results"]}
    sev2 = severity_block(c["recovered2__ac_short"], c["pruned2_A__ac_short"],
                          c["recovered2__ac_native"], c["pruned2_A__ac_native"], rng)
    # cross-check the persisted groups reproduce the frozen R_native / R_short points bit-for-bit
    frozen = json.load(open(SEV2_FROZEN))["PRIMARY_A"]
    chk = {"R_native": float((c["recovered2__ac_native"] - c["pruned2_A__ac_native"]).mean()),
           "R_short": float((c["recovered2__ac_short"] - c["pruned2_A__ac_short"]).mean())}
    for k in chk:
        if abs(chk[k] - frozen[k]["point"]) > 1e-12:
            raise SystemExit(f"persisted sev-2 groups do not reproduce frozen {k}: {chk[k]} vs {frozen[k]['point']}")
    # B2: matched-duration domain contrast at severity 2 (music: 3 replicates averaged per prompt first)
    r_short = c["recovered2__ac_short"] - c["pruned2_A__ac_short"]
    r_music = (c["recovered2__music"] - c["pruned2_A__music"]).reshape(64, 3).mean(1)
    dom2 = two_sample_ci(r_short, r_music, rng)
    music_win = float((r_music > 0).mean())
    rev = json.load(open("configs/research/reversal_v1_1_result.json"))["PRIMARY"]["I"]
    res = {
        "artifact": "draft3_sensitivity_result", "class": "POST-HOC SENSITIVITY (authorised 2026-09-02); changes no frozen verdict",
        "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": SEED, "unit": "prompt", "ci": "percentile 95%"},
        "inputs": {OPD: sha(OPD), SEV2: sha(SEV2), SEV2_FROZEN: sha(SEV2_FROZEN)},
        "B1_scale_free_interaction": {"sev1_armd80": sev1, "sev2_xsev192": sev2},
        "B2_matched_duration_domain_contrast": {
            "sev2_R_short_minus_R_music_at_3p84s": dom2, "sev2_music_win_rate": music_win,
            "sev1_frozen_I_R_AC_minus_R_music_at_3p84s": {"point": rev["point"], "lo": rev["lo"], "hi": rev["hi"],
                                                          "source": "reversal_v1_1_result.json (frozen, n=96/64)"}},
        "reproduction_check_vs_frozen_sev2": chk,
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("DRAFT3-SENSITIVITY PASS ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
