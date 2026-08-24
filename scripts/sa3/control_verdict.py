#!/usr/bin/env python3
"""Combine FIELD + TASK evidence into the full §5.1/rc1.4 positive-control verdict for L_6 and L_13.

FIELD (mechanistic): precision_ok AND A_eco(host)≈1  (from control_field_eval.json).
TASK (functional, decisive): rc1.4 gate on ΔT_AA (from the score_taa output) —
  base uplift lower-CI>0, post uplift lower-CI>0, host^{-b} CI∋0, ≥1 external g∈G_ext(b) lower-CI>0.
Overall control PASS iff FIELD ok AND TASK pass. Only ΔT_AA sustains the functional claim; A_eco is
mechanistic evidence and cannot substitute for it.

Run: .venv-sa3/bin/python scripts/sa3/control_verdict.py --field configs/sa3/adapters/control_field_eval.json \
        --taa artifacts/sa3/control_taa_scores.json --out configs/sa3/adapters/control_verdict.json
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3.aeco_predict import task_control_verdict

GEXT = {6: [11, 12, 13], 13: [11, 12, 14]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True); ap.add_argument("--taa", required=True)
    ap.add_argument("--out", default="configs/sa3/adapters/control_verdict.json")
    a = ap.parse_args()
    field = json.load(open(a.field))["controls"]
    taa = json.load(open(a.taa))["deltas"]

    def ci(name):
        d = taa[name]; return (d["lo"], d["hi"])

    out = {"phase": "positive_control_verdict", "controls": {}}
    all_pass = True
    for b in (6, 13):
        fk = field[f"L_{b}"]
        field_ok = bool(fk["precision_ok"]) and abs(fk["A_eco_host"] - 1.0) <= 0.10
        tv = task_control_verdict(
            b, dT_base_ci=ci(f"L{b}_base"), dT_post_ci=ci(f"L{b}_post"), dT_host_ci=ci(f"L{b}_host"),
            dT_external_ci={g: ci(f"L{b}_ext{g}") for g in GEXT[b]})
        base_mean = taa[f"L{b}_base"]["dT_AA"]; post_mean = taa[f"L{b}_post"]["dT_AA"]
        retention = (post_mean / base_mean) if base_mean not in (0, 0.0) else None
        overall = field_ok and tv["pass"]
        all_pass = all_pass and overall
        out["controls"][f"L_{b}"] = {
            "field_ok": field_ok, "precision_ok": fk["precision_ok"], "A_eco_host": fk["A_eco_host"],
            "task": tv, "dT_AA_base": base_mean, "dT_AA_post": post_mean,
            "post_over_base_retention": retention,
            "OVERALL": "PASS" if overall else "FAIL"}
    out["all_controls_pass"] = all_pass
    out["consequence"] = ("Both controls PASS — the instrument can localise a known adaptation from "
                          "task outputs; proceed to ecological adapters." if all_pass else
                          "A control FAILED task observability → STOP RQ2 (do NOT train ecological adapters).")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
