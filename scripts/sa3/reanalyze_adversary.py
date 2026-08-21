#!/usr/bin/env python3
"""Corrected single-block adversary re-analysis (review tasks 1, 3) — from disk, NO re-scoring.

Fixes vs analyze_adversary.py (whose OLD margins are kept, labelled OLD_INVALID):
  * SEED FLOOR at the right statistical scale: r_CLAP = spread of the PANEL-MEAN CLAP across the
    R=5 dense-8 streams (10 pairwise |mean_i - mean_j|), compared against panel-mean system diffs
    (not prompt-to-prompt diffs pooled). r_KL from the panel-mean KL of equivalent streams
    (the 4 KL(s_r||s_0) available on disk -- small-sample; a full pairwise floor needs re-scored
    posteriors, done separately in kl_floor_pairwise.py).
  * LATENCY COMPARATOR from MEASURED latencies (smoke): skip-g@8 = 0.560 s sits in the bracket
    {dense7 0.514, dense8 0.600}; the NEAREST is dense8 (delta 0.040 < 0.046). Report the bracket
    and apply the nearest-latency rule; verdicts given vs BOTH bracket members (point estimates,
    no CI -> directional).

Run: .venv-sa3/bin/python scripts/sa3/reanalyze_adversary.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

SRC = "artifacts/sa3/adversary_analysis.json"
OUT = "artifacts/sa3/adversary_reanalysis.json"
LAT = {"dense4": 0.3379, "dense5": 0.3934, "dense6": 0.4432, "dense7": 0.5139, "dense8": 0.5995,
       "skip@8": 0.5604}  # measured (smoke sa3-smoke-1); skip5@8 as the representative skip latency


def pct(v, q=95):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    r = json.load(open(SRC))
    S = r["systems"]
    R = r["manifest"] and 5  # R=5
    streams = [f"dense8_s{i}" for i in range(5) if f"dense8_s{i}" in S]
    aids = sorted(S["dense8_s0"]["CLAP_per"], key=lambda x: int(x))

    # panel-mean CLAP per stream, and the 10 pairwise |diff| null (CORRECT scale)
    panel_clap = {s: float(np.mean([S[s]["CLAP_per"][a] for a in aids])) for s in streams}
    clap_pairs = [abs(panel_clap[streams[i]] - panel_clap[streams[j]])
                  for i in range(len(streams)) for j in range(i + 1, len(streams))]
    r_CLAP = pct(clap_pairs, 95)  # 95th pctile of the 10 panel-mean pair diffs

    # KL floor at the panel-aggregate scale: panel-mean KL(s_r || s0) for r=1..4 (small-sample)
    kl_stream = [S[s]["KL_mean"] for s in streams if s != "dense8_s0"]
    r_KL_smallsample = float(np.max(kl_stream)) if kl_stream else float("nan")

    # 8->7 deterioration at the panel scale
    c8 = panel_clap["dense8_s0"]; c7 = float(np.mean([S["dense7_s0"]["CLAP_per"][a] for a in aids]))
    delta_CLAP = max(0.0, c8 - c7)
    delta_KL = S["dense7_s0"]["KL_mean"]  # panel-mean KL(dense7||dense8)

    m_CLAP = max(delta_CLAP, r_CLAP)
    m_KL = max(delta_KL, r_KL_smallsample)

    # comparator: nearest measured latency to skip@8
    skip_lat = LAT["skip@8"]
    bracket = ("dense7", "dense8")
    nearest = min(bracket, key=lambda d: abs(LAT[d] - skip_lat))
    comp_clap = {d: float(np.mean([S[f"{d}_s0"]["CLAP_per"][a] for a in aids])) for d in bracket}

    # paired bootstrap CI of the CLAP deficit vs the NEAREST comparator (protocol section 9.2 3-way rule)
    def boot_deficit(comp_id, sid, B=2000, seed=20260818):
        rng = np.random.default_rng(seed)
        paired = np.array([S[comp_id]["CLAP_per"][a] - S[sid]["CLAP_per"][a] for a in aids])  # per-prompt deficit
        idx = rng.integers(0, len(paired), size=(B, len(paired)))
        means = paired[idx].mean(1)
        return float(paired.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    comp_id = f"{nearest}_s0"
    verdicts = {}
    inferior = []; noninferior = []; indeterminate = []
    for sid in S:
        if not sid.startswith("skip"):
            continue
        g = int(sid.replace("skip", ""))
        sc = float(np.mean([S[sid]["CLAP_per"][a] for a in aids]))
        m, lo, hi = boot_deficit(comp_id, sid)
        # 3-way non-inferiority (X=skip_g vs Y=nearest dense): inferior iff lower CI of deficit > margin;
        # non-inferior iff upper CI <= margin; else indeterminate.
        if lo > m_CLAP:
            verdict = "inferior"; inferior.append(g)
        elif hi <= m_CLAP:
            verdict = "non_inferior"; noninferior.append(g)
        else:
            verdict = "indeterminate"; indeterminate.append(g)
        verdicts[sid] = {"block": g, "CLAP": sc, "comparator": comp_id,
                         "deficit_mean": m, "deficit_ci95": [lo, hi], "m_CLAP": m_CLAP,
                         "verdict": verdict}
    inferior_vs_nearest = sorted(inferior)  # kept name for downstream

    mids = list(range(2, 19))
    mids_inf = [g for g in mids if g in inferior]
    mids_within = [g for g in mids if g in noninferior]
    mids_indet = [g for g in mids if g in indeterminate]
    out = {
        "source": SRC,
        "OLD_INVALID_margins": r["margins"],
        "OLD_INVALID_note": "old r_CLAP/r_KL used prompt-to-prompt diffs vs system-mean diffs (wrong scale); "
                            "old comparator hardcoded dense7. Kept for the record; superseded below.",
        "corrected": {
            "panel_mean_CLAP_per_stream": panel_clap,
            "clap_pair_diffs_10": [round(x, 4) for x in sorted(clap_pairs)],
            "r_CLAP_panelmean_p95": r_CLAP,
            "delta_CLAP_8to7": delta_CLAP,
            "m_CLAP": m_CLAP,
            "r_KL_panelmean_smallsample_max": r_KL_smallsample,
            "delta_KL_8to7": delta_KL, "m_KL": m_KL,
            "kl_floor_caveat": "4-sample panel-mean KL null (s_r vs s0); full 10-pair pairwise floor "
                               "needs re-scored posteriors (kl_floor_pairwise.py).",
        },
        "latency_comparator": {"skip@8_measured": skip_lat, "bracket": {d: LAT[d] for d in bracket},
                               "nearest": nearest, "rule": "nearest measured latency; both bracket members reported",
                               "comparator_panel_CLAP": comp_clap},
        "verdicts": verdicts,
        "inferior_vs_nearest": sorted(inferior_vs_nearest),
        "non_inferior_blocks": sorted(noninferior), "indeterminate_blocks": sorted(indeterminate),
        "middle_blocks_2_18_INFERIOR": mids_inf, "middle_blocks_2_18_NON_INFERIOR": mids_within, "middle_blocks_2_18_INDETERMINATE": mids_indet,
        "caveat": "point estimates, no bootstrap CIs (pilot, N=16). CLAP primary; KL floor small-sample; "
                  "FD deferred (rank-deficient at N=16). NOT a main-panel CASE-E decision.",
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print("=== CORRECTED floors (panel-mean scale) ===")
    print(f"  r_CLAP (10 pair p95) = {r_CLAP:.4f}   delta_CLAP(8->7) = {delta_CLAP:.4f}   m_CLAP = {m_CLAP:.4f}")
    print(f"  OLD (invalid) m_CLAP = {r['margins']['m_CLAP']:.4f}")
    print(f"  r_KL (4-sample max) = {r_KL_smallsample:.3f}   m_KL = {m_KL:.3f}")
    print(f"\n=== comparator: skip@8={skip_lat:.3f}s nearest={nearest} (bracket dense7={LAT['dense7']}, dense8={LAT['dense8']}) ===")
    print(f"{'blk':>3} {'CLAP':>6} {'defmean':>8} {'ci_lo':>7} {'ci_hi':>7}  verdict (vs %s, m_CLAP=%.3f)" % (comp_id, m_CLAP))
    for sid in sorted(verdicts, key=lambda s: verdicts[s]['block']):
        x = verdicts[sid]
        print(f"{x['block']:>3} {x['CLAP']:6.3f} {x['deficit_mean']:8.4f} {x['deficit_ci95'][0]:7.4f} {x['deficit_ci95'][1]:7.4f}  {x['verdict']}")
    print(f"\nINFERIOR (lower CI>margin): {sorted(inferior)}")
    print(f"NON-INFERIOR (upper CI<=margin): {sorted(noninferior)}")
    print(f"INDETERMINATE: {sorted(indeterminate)}")
    print(f"middle 2-18: inferior={mids_inf} non_inf={mids_within} indeterminate={mids_indet}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
