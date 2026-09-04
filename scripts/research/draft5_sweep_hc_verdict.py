#!/usr/bin/env python3
"""DRAFT5-OPSWEEP-1 — Human-CLAP corroboration of the sweep shape (CPU, 0 cr, post-hoc, no gate).

R_HC(d) = HC(P+FT) - HC(P) per prompt at 3.84 / 5.12 / 7.68 / 10.24 s (3.84 / 10.24 s from the committed
xsev_sev2_hc_groups_out.json; 5.12 / 7.68 s from draft5_sweep_hc.py), steps D1..D3, the CLAP shape rule
applied DESCRIPTIVELY (it was pre-specified for CLAP only). Guard: R_HC at 10.24 s must reproduce the
committed xsev_result.json human_clap R_native point.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_sweep_hc_verdict.py
"""
from __future__ import annotations
import hashlib, json, os
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np

TMP = "artifacts/icassp_gate0/_score_tmp"
NS = "DRAFT5-OPSWEEP-1|HC|BOOTSTRAP|2026-09-04"
B = 10000; SESOI = 0.025
OUT = "configs/research/draft5_sweep_hc.json"


def main():
    fro = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(f"{TMP}/xsev_sev2_hc_groups_out.json"))["results"]}
    new = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(f"{TMP}/draft5_sweep_hc_groups_out.json"))["results"]}
    cell = {3.84: (fro["pruned2_A__ac_short"], fro["recovered2__ac_short"], None),
            5.12: (new["pruned2_A__ac_d128"], new["recovered2__ac_d128"], new["dense__ac_d128"]),
            7.68: (new["pruned2_A__ac_d192"], new["recovered2__ac_d192"], new["dense__ac_d192"]),
            10.24: (fro["pruned2_A__ac_native"], fro["recovered2__ac_native"], None)}
    n = 192
    seed = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
    idx = np.random.default_rng(seed).integers(0, n, (B, n))

    def ci(v):
        bm = v[idx].mean(1); lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": n}
    R = {d: q - p for d, (p, q, _) in cell.items()}
    # guard vs committed xsev_result.json (human_clap R_native / R_short points)
    gnat = []
    hc = None
    def find_hc(o):
        nonlocal hc
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "human_clap": hc = v
                find_hc(v)
    find_hc(json.load(open("configs/research/xsev_result.json")))
    for k, v in (hc or {}).items():
        if isinstance(v, dict) and "point" in v and "native" in k.lower():
            gnat.append(v)
    out = {"artifact": "draft5_sweep_hc", "class": "POST-HOC corroborative Human-CLAP on the sweep; no gate; scale != primary CLAP",
           "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": seed, "unit": "prompt", "n": n},
           "levels": {f"{s}@{d}": float(v.mean()) for d, (p, q, den) in cell.items()
                      for s, v in (("P", p), ("PFT", q)) + ((("dense", den),) if den is not None else ())},
           "R_HC_by_duration": {str(d): ci(v) for d, v in R.items()},
           "steps": {"D1": ci(R[5.12] - R[3.84]), "D2": ci(R[7.68] - R[5.12]), "D3": ci(R[10.24] - R[7.68])},
           "J_HC_native_minus_short": ci(R[10.24] - R[3.84])}
    S = out["steps"]; pts = [S[k]["point"] for k in ("D1", "D2", "D3")]
    out["shape_descriptive"] = ("MONOTONE-INCREASING (all steps > 0, no hi95 < 0)" if all(p > 0 for p in pts) and all(S[k]["hi"] >= 0 for k in S)
                                else "NOT MONOTONE by the CLAP rule")
    out["guard_committed_xsev_human_clap"] = {"R_native_here": float(R[10.24].mean()), "committed_candidates": [g["point"] for g in gnat],
                                              "PASS": any(abs(g["point"] - R[10.24].mean()) < 1e-9 for g in gnat)}
    txt = json.dumps(out, indent=1, sort_keys=True); out["artifact_sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    json.dump(out, open(OUT, "w"), indent=1)
    for d in (3.84, 5.12, 7.68, 10.24):
        c = out["R_HC_by_duration"][str(d)]; print(f"R_HC@{d:5.2f} {c['point']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]")
    for k in ("D1", "D2", "D3"):
        print(k, f"{S[k]['point']:+.4f} [{S[k]['lo']:+.4f},{S[k]['hi']:+.4f}]")
    print("shape", out["shape_descriptive"], "| J_HC", out["J_HC_native_minus_short"]["point"], "| guard", out["guard_committed_xsev_human_clap"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
