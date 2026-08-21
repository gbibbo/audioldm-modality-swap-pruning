#!/usr/bin/env python3
"""Bootstrap sizing + collapse test from per-prompt sufficient statistics (review round 2), CPU only.

Consumes `pilot_fields.json['per_prompt']` and recomputes D_P/D_B/I_PT + removal rankings for ANY
subsample of prompts WITHOUT re-running forwards (D_P/I_PT are ratios of sums). Produces:
  * the pre-registered N_main ladder (16,32,64,...): for each rung and criterion, the 95th pct over B
    bootstrap subsample-PAIRS of the removal-set disagreement d_X(k); N_main = smallest rung with p95=0
    at every criterion/k (samplesize.choose_rung);
  * a FLOORED I_PT≈D_P collapse test at the full pilot N: delta_XY(k) vs max(floor_X, floor_Y) where
    floor_X(k) = 95th pct of bootstrap within-criterion disagreement (the floor the LOO test lacked).

Run: .venv-sa3/bin/python scripts/sa3/size_from_stats.py --src artifacts/sa3/pilot_fields.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import metrics as M, samplesize as SS, greedy as GG

ETA = [6.7e-5] * 8


def crit_values(pp, aids, criterion, blocks):
    """Aggregate a criterion over a prompt subsample from sufficient stats. Returns {g: value}."""
    if criterion == "D_P":
        den = sum(pp[a]["f_den"] for a in aids)
        return {g: sum(pp[a]["dp_num"][str(g)] if str(g) in pp[a]["dp_num"] else pp[a]["dp_num"][g] for a in aids) / den for g in blocks}
    if criterion == "D_B":
        den = sum(pp[a]["fb_den"] for a in aids)
        return {g: sum(pp[a]["db_num"].get(str(g), pp[a]["db_num"].get(g)) for a in aids) / den for g in blocks}
    if criterion == "I_PT":
        num = {g: [sum(pp[a]["ipt_num"].get(str(g), pp[a]["ipt_num"].get(g))[i] for a in aids) for i in range(8)] for g in blocks}
        den = [sum(pp[a]["ipt_den"][i] for a in aids) for i in range(8)]
        fp = [sum(pp[a]["fp_sq"][i] for a in aids) for i in range(8)]
        r = M.i_pt(num, den, fp, ETA)
        return {g: r["per_block"][g]["pooled"] for g in blocks}
    raise ValueError(criterion)


def removal_set(vals, k):
    return set(sorted(vals, key=lambda g: vals[g])[:k])


def bootstrap_ladder(pp, blocks, crits=("D_P", "I_PT"), ks=(2, 4, 6), B=1000, seed=20260818):
    aids = sorted(pp, key=lambda x: int(x)); N = len(aids)
    rng = np.random.default_rng(seed)
    ladder = [r for r in SS.ladder(N)] or [N]
    # rung -> criterion -> k -> [disagreements over B pairs]
    dis = {r: {c: {k: [] for k in ks} for c in crits} for r in ladder}
    for r in ladder:
        for _ in range(B):
            a = [aids[i] for i in rng.choice(N, size=r, replace=False)]
            b = [aids[i] for i in rng.choice(N, size=r, replace=False)]
            for c in crits:
                va, vb = crit_values(pp, a, c, blocks), crit_values(pp, b, c, blocks)
                for k in ks:
                    dis[r][c][k].append(GG.set_divergence(removal_set(va, k), removal_set(vb, k)))
    choose = SS.choose_rung(dis, ks=ks)
    return {"ladder": ladder, "n_main": choose["n_main"], "qualifies": choose["qualifies"], "trace": choose["trace"]}


def collapse_floor_test(pp, blocks, ks=(2, 4, 6), B=1000, seed=7):
    """delta(D_P,I_PT)(k) at full N vs the bootstrap within-criterion floor."""
    aids = sorted(pp, key=lambda x: int(x)); N = len(aids)
    rng = np.random.default_rng(seed)
    full = {c: crit_values(pp, aids, c, blocks) for c in ("D_P", "I_PT")}
    out = {}
    for k in ks:
        delta = GG.set_divergence(removal_set(full["D_P"], k), removal_set(full["I_PT"], k))
        floors = {}
        for c in ("D_P", "I_PT"):
            ds = []
            for _ in range(B):
                a = [aids[i] for i in rng.choice(N, size=N, replace=True)]
                b = [aids[i] for i in rng.choice(N, size=N, replace=True)]
                ds.append(GG.set_divergence(removal_set(crit_values(pp, a, c, blocks), k),
                                            removal_set(crit_values(pp, b, c, blocks), k)))
            floors[c] = float(np.percentile(ds, 95))
        real = delta > max(floors["D_P"], floors["I_PT"])
        out[k] = {"delta_DP_IPT": delta, "floor_DP": floors["D_P"], "floor_IPT": floors["I_PT"], "divergence_real": bool(real)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/sa3/pilot_fields.json")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default="artifacts/sa3/sizing.json")
    a = ap.parse_args()
    r = json.load(open(a.src))
    assert "per_prompt" in r, "pilot_fields.json lacks per_prompt (re-run the refactored pilot_fields.py)"
    pp = r["per_prompt"]; blocks = r["blocks"]
    # self-check: aggregate from per_prompt == stored aggregate
    dp = crit_values(pp, sorted(pp, key=lambda x: int(x)), "D_P", blocks)
    stored = {int(g): v for g, v in r["D_P"].items()}
    ok = all(abs(dp[g] - stored[g]) < 1e-9 for g in blocks)
    lad = bootstrap_ladder(pp, blocks, B=a.B)
    col = collapse_floor_test(pp, blocks, B=a.B)
    out = {"src": a.src, "N": r["N"], "aggregate_matches_per_prompt": ok,
           "N_main_ladder": lad, "collapse_floor_test": col}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"aggregate==per_prompt sum: {ok}")
    print(f"N_main ladder: {lad['ladder']}  -> N_main={lad['n_main']}  qualifies={lad['qualifies']}")
    print("collapse test (delta D_P vs I_PT with bootstrap floor):")
    for k, v in col.items():
        print(f"  k={k}: delta={v['delta_DP_IPT']} floor_DP={v['floor_DP']:.2f} floor_IPT={v['floor_IPT']:.2f} real_divergence={v['divergence_real']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
