#!/usr/bin/env python3
"""DRAFT5-OPSWEEP-1 — Human-CLAP SECONDARY scoring of the sweep cells (CPU, 0 cr, post-hoc, no gate).

Rescores the 5.12 s / 7.68 s groups emitted by draft5_opsweep_verdict.py --emit (P = pruned2_A,
P+FT = recovered2, dense) with the frozen HumanClapScorer convention used for the severity-2 cells
(sarulab-speech/human-clap-wsce-mae; np.random.seed(20260826) reset once per group; paired by position).
The 3.84 s / 10.24 s HC cosines are read from the committed xsev_sev2_hc_groups_out.json — NOT rescored.

Run: OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/draft5_sweep_hc.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
from reversal_humanclap import HumanClapScorer

TMP = "artifacts/icassp_gate0/_score_tmp"
IN = os.path.join(TMP, "draft5_sweep_groups_in.json")
OUT = os.path.join(TMP, "draft5_sweep_hc_groups_out.json")
WANT = ("pruned2_A__ac_d128", "recovered2__ac_d128", "dense__ac_d128",
        "pruned2_A__ac_d192", "recovered2__ac_d192", "dense__ac_d192")


def main():
    groups = [g for g in json.load(open(IN))["groups"] if g["name"] in WANT]
    assert len(groups) == len(WANT), [g["name"] for g in groups]
    sc = HumanClapScorer()
    results = []
    for g in groups:
        items = g["items"]
        np.random.seed(20260826)
        cos = sc.cosine([it["caption"] for it in items], [it["wav"] for it in items])
        results.append({"name": g["name"], "n": len(items), "cosines": [float(x) for x in cos]})
        print(f"HC scored {g['name']}: n={len(items)} mean={float(np.mean(cos)):.4f}", flush=True)
    json.dump({"results": results, "model": "sarulab-speech/human-clap-wsce-mae",
               "note": "SECONDARY corroborative Human-CLAP on the sweep cells; same convention as xsev_score_hc.py"},
              open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
