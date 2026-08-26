#!/usr/bin/env python3
"""F1 functional verdict (RQ2b). Consume the frozen paired CLAP audio-audio T_AA scores (score_taa.py
output, pairs F1_base and F1_post) and apply the SYMMETRIC SESOI gate (aeco_predict.f1_functional_verdict):
both base AND post require lower-CI > 0 AND point ΔT_AA >= SESOI (default 0.075).

  base fails              -> STOP_RQ2B_BASE_FAIL (task/training/measurement chain unqualified)
  base ok, post fails     -> STOP_RQ2B_POST_FAIL (meaningful base->post transfer not qualified)
  both pass               -> F1_PASS (F2 eligible)  [STILL requires a human sign-off before F2]

Run: .venv-sa3/bin/python scripts/sa3/f1_verdict.py --taa artifacts/sa3/f1_taa_scores.json \
        --out configs/sa3/adapters/f1_verdict.json [--sesoi 0.075]
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3.aeco_predict import f1_functional_verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taa", required=True)
    ap.add_argument("--out", default="configs/sa3/adapters/f1_verdict.json")
    ap.add_argument("--sesoi", type=float, default=0.075)
    a = ap.parse_args()
    deltas = json.load(open(a.taa))["deltas"]
    for k in ("F1_base", "F1_post"):
        if k not in deltas:
            raise SystemExit(f"scores missing pair {k}; found {list(deltas)}")
    v = f1_functional_verdict(deltas["F1_base"], deltas["F1_post"], sesoi=a.sesoi)
    out = {"phase": "F1_functional_verdict", "sesoi": a.sesoi, "verdict": v["verdict"],
           "pass": v["pass"], "base": v["base"], "post": v["post"],
           "post_over_base_retention": v["post_over_base_retention"],
           "consequence": ("F1 PASS -> F2 eligible (STOP and report before F2; no structural inspection, "
                           "no F2 in the same unattended job)." if v["pass"] else
                           "F1 FAIL -> STOP RQ2b (no F2, no structural inspection).")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
