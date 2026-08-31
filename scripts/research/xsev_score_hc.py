#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — Human-CLAP SECONDARY scoring of the frozen groups (CPU, free).

Reuses the frozen HumanClapScorer (sarulab-speech/human-clap-wsce-mae; laion/clap-htsat-fused
processor; SR 48000; np.random.seed(20260826) reset once per group) to rescore the SAME 192-item
severity-2 groups emitted by xsev_score_emit.py. CORROBORATIVE, CLAP-family, NO PASS role, different
scale (the 0.025 CLAP SESOI does NOT apply). Writes HC cosines per group.

Run: OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/xsev_score_hc.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
from reversal_humanclap import HumanClapScorer

TMP = "artifacts/icassp_gate0/_score_tmp"
IN = os.path.join(TMP, "xsev_sev2_groups_in.json")
OUT = os.path.join(TMP, "xsev_sev2_hc_groups_out.json")


def main():
    groups = json.load(open(IN))["groups"]
    sc = HumanClapScorer()
    results = []
    for g in groups:
        items = g["items"]
        np.random.seed(20260826)                     # frozen per-group reset (paired by position)
        cos = sc.cosine([it["caption"] for it in items], [it["wav"] for it in items])
        results.append({"name": g["name"], "n": len(items), "cosines": [float(x) for x in cos]})
        print(f"HC scored {g['name']}: n={len(items)} mean={float(np.mean(cos)):.4f}")
    out = {"results": results, "model": "sarulab-speech/human-clap-wsce-mae",
           "note": "SECONDARY corroborative Human-CLAP; CLAP-family; no PASS role; scale != primary CLAP"}
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
