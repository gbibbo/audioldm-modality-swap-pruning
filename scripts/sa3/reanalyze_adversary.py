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

    verdicts = {}
    inferior_vs_nearest = []; inferior_vs_both = []
    for sid in S:
        if not sid.startswith("skip"):
            continue
        g = int(sid.replace("skip", ""))
        sc = float(np.mean([S[sid]["CLAP_per"][a] for a in aids]))
        def_vs = {d: comp_clap[d] - sc for d in bracket}  # CLAP deficit vs each bracket member
        inf_near = def_vs[nearest] > m_CLAP
        inf_both = all(def_vs[d] > m_CLAP for d in bracket)
        verdicts[sid] = {"block": g, "CLAP": sc,
                         "clap_deficit_vs_dense7": def_vs["dense7"],
                         "clap_deficit_vs_dense8": def_vs["dense8"],
                         "inferior_vs_nearest(%s)" % nearest: inf_near,
                         "inferior_vs_both_bracket": inf_both}
        if inf_near:
            inferior_vs_nearest.append(g)
        if inf_both:
            inferior_vs_both.append(g)

    mids = list(range(2, 19))
    mids_within = [g for g in mids if g not in inferior_vs_nearest]
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
        "inferior_vs_both_bracket": sorted(inferior_vs_both),
        "middle_blocks_2_18_within_margin_vs_nearest": mids_within,
        "middle_blocks_2_18_INFERIOR_vs_nearest": [g for g in mids if g in inferior_vs_nearest],
        "caveat": "point estimates, no bootstrap CIs (pilot, N=16). CLAP primary; KL floor small-sample; "
                  "FD deferred (rank-deficient at N=16). NOT a main-panel CASE-E decision.",
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print("=== CORRECTED floors (panel-mean scale) ===")
    print(f"  r_CLAP (10 pair p95) = {r_CLAP:.4f}   delta_CLAP(8->7) = {delta_CLAP:.4f}   m_CLAP = {m_CLAP:.4f}")
    print(f"  OLD (invalid) m_CLAP = {r['margins']['m_CLAP']:.4f}")
    print(f"  r_KL (4-sample max) = {r_KL_smallsample:.3f}   m_KL = {m_KL:.3f}")
    print(f"\n=== comparator: skip@8={skip_lat:.3f}s nearest={nearest} (bracket dense7={LAT['dense7']}, dense8={LAT['dense8']}) ===")
    print(f"{'blk':>3} {'CLAP':>6} {'def_v7':>7} {'def_v8':>7} {'inf_near':>9} {'inf_both':>9}")
    for sid in sorted(verdicts, key=lambda s: verdicts[s]['block']):
        x = verdicts[sid]
        print(f"{x['block']:>3} {x['CLAP']:6.3f} {x['clap_deficit_vs_dense7']:7.3f} {x['clap_deficit_vs_dense8']:7.3f} "
              f"{str(x['inferior_vs_nearest(%s)'%nearest]):>9} {str(x['inferior_vs_both_bracket']):>9}")
    print(f"\ninferior vs nearest({nearest}): {sorted(inferior_vs_nearest)}")
    print(f"inferior vs BOTH bracket members: {sorted(inferior_vs_both)}")
    print(f"middle blocks 2-18 INFERIOR vs nearest: {[g for g in mids if g in inferior_vs_nearest]}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
