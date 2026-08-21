#!/usr/bin/env python3
"""A_tan n_u sizing + the A_tan-vs-D_P decision gate (review round 2), CPU only.

From `atan_pilot.json['per_prompt_probe']` (per-(prompt,probe,kappa) denom + num_tan{g}) recompute
A_tan(g | prompt-subsample, probe-subsample) = Σ num_tan / Σ denom for ANY N and n_u (ratio of sums).
Produces:
  * n_u ladder (8,16,...): probe-bootstrap 95th pct of the A_tan removal-set disagreement d(k);
    n_u = smallest rung with p95=0 at every k (samplesize.choose_rung);
  * THE DECISION GATE: delta(A_tan, D_P)(k) at full (N,n_u) vs a bootstrap floor. If A_tan's removal
    ranking essentially coincides with D_P (divergence not real), the adapter-compatible-pruning
    hypothesis is not supported — close before training LoRAs. If a stable, decision-relevant
    divergence exists, RQ2 warrants real held-out adapters.

Run: .venv-sa3/bin/python scripts/sa3/size_atan_from_stats.py --atan artifacts/sa3/atan_pilot.json \
        --pilot artifacts/sa3/pilot_fields_n64.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import samplesize as SS, greedy as GG
from size_from_stats import crit_values as dp_crit_values  # reuse D_P aggregation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def atan_values(ppp, prompts, probes, kf, blocks):
    """A_tan(g) = Σ_{p,u} num_tan[p,u,g] / Σ_{p,u} denom[p,u] over the given prompt/probe subsamples."""
    num = {g: 0.0 for g in blocks}; den = 0.0
    for p in prompts:
        for u in probes:
            key = f"{u}|{kf}"
            if key in ppp[p]:
                rec = ppp[p][key]; den += rec["denom"]
                for g in blocks:
                    num[g] += rec["num_tan"][str(g)]
    return {g: (num[g] / den if den > 0 else float("nan")) for g in blocks}


def removal_set(vals, k):
    return set(sorted(vals, key=lambda g: vals[g])[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atan", default="artifacts/sa3/atan_pilot.json")
    ap.add_argument("--pilot", default="artifacts/sa3/pilot_fields_n64.json")
    ap.add_argument("--kappa-mult", type=float, default=1.0)
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--ks", default="2,4,6")
    ap.add_argument("--out", default="artifacts/sa3/atan_sizing.json")
    a = ap.parse_args()
    ks = tuple(int(x) for x in a.ks.split(","))
    at = json.load(open(a.atan)); ppp = at["per_prompt_probe"]; blocks = at["blocks"]
    kf = a.kappa_mult
    aids = sorted(ppp, key=lambda x: int(x)); N = len(aids)
    n_u = at["n_u"]; rng = np.random.default_rng(20260818)

    # n_u ladder: probe-bootstrap (disjoint probe pairs) at full N
    rungs = [r for r in SS.n_u_ladder(n_u) if 2 * r <= n_u] or [max(1, n_u // 2)]
    dis = {r: {"A_tan": {k: [] for k in ks}} for r in rungs}
    for r in rungs:
        for _ in range(a.B):
            perm = rng.permutation(n_u)
            ua = list(perm[:r]); ub = list(perm[r:2 * r])
            va = atan_values(ppp, aids, ua, kf, blocks); vb = atan_values(ppp, aids, ub, kf, blocks)
            for k in ks:
                dis[r]["A_tan"][k].append(GG.set_divergence(removal_set(va, k), removal_set(vb, k)))
    n_u_choice = SS.choose_rung(dis, ks=ks)

    out = {"atan_src": a.atan, "N": N, "n_u": n_u, "kappa_mult": kf,
           "n_u_ladder": rungs, "n_u_main": n_u_choice["n_main"], "n_u_qualifies": n_u_choice["qualifies"]}

    # DECISION GATE: A_tan vs D_P divergence with a bootstrap floor
    if os.path.exists(a.pilot):
        pil = json.load(open(a.pilot))
        if "per_prompt" in pil:
            pp = pil["per_prompt"]; pblocks = pil["blocks"]
            common = [x for x in aids if x in pp]  # prompts present in both
            full_atan = atan_values(ppp, common, list(range(n_u)), kf, blocks)
            full_dp = dp_crit_values(pp, common, "D_P", pblocks)
            Ncom = len(common); half = Ncom // 2; uhalf = max(1, n_u // 2)
            N_MAIN_DP = 32  # D_P removal sets are stable only at N>=32 (N=64 sizing)
            underpowered = Ncom < N_MAIN_DP
            gate = {}
            for k in ks:
                delta = GG.set_divergence(removal_set(full_atan, k), removal_set(full_dp, k))
                # DISJOINT-pair floor at size `half` (the largest disjoint size available); this is a
                # size-(N//2) floor -> a CONSERVATIVE (loose/upper) reference for the size-N delta.
                fa, fd = [], []
                for _ in range(a.B):
                    permP = rng.permutation(Ncom); Pa = [common[i] for i in permP[:half]]; Pb = [common[i] for i in permP[half:2*half]]
                    permU = rng.permutation(n_u); ua = list(permU[:uhalf]); ub = list(permU[uhalf:2*uhalf])
                    fa.append(GG.set_divergence(removal_set(atan_values(ppp, Pa, ua, kf, blocks), k),
                                                removal_set(atan_values(ppp, Pb, ub, kf, blocks), k)))
                    fd.append(GG.set_divergence(removal_set(dp_crit_values(pp, Pa, "D_P", pblocks), k),
                                                removal_set(dp_crit_values(pp, Pb, "D_P", pblocks), k)))
                floor_half = max(float(np.percentile(fa, 95)), float(np.percentile(fd, 95)))
                # a divergence is only credibly real if delta exceeds even the (looser) size-N/2 floor
                # AND N is large enough for D_P to be stable
                real = bool(delta > floor_half) and not underpowered
                gate[k] = {"delta_Atan_DP": delta, "floor_sizeNhalf": floor_half,
                           "delta_gt_looseFloor": bool(delta > floor_half), "credibly_real": real}
            out["gate_underpowered_N_lt_32"] = underpowered
            out["decision_gate_Atan_vs_DP"] = gate
            if underpowered:
                out["gate_verdict"] = ("UNDERPOWERED at N=%d: D_P removal sets are not stable below N_main=32, "
                                       "so the observed 1-2 block A_tan-vs-D_P divergence is comparable to D_P's "
                                       "own sampling noise and CANNOT be declared real. Resolve with A_tan at N>=32."
                                       % Ncom)
            elif any(v["credibly_real"] for v in gate.values()):
                out["gate_verdict"] = "A_tan DIVERGES from D_P (credible) -> RQ2 worth the real held-out-adapter test"
            else:
                out["gate_verdict"] = "A_tan ~= D_P -> close adapter-compatible-pruning before training LoRAs"
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"n_u ladder {rungs} -> n_u_main={out['n_u_main']}  qualifies={out['n_u_qualifies']}")
    if "decision_gate_Atan_vs_DP" in out:
        for k, v in out["decision_gate_Atan_vs_DP"].items():
            print(f"  GATE k={k}: delta(A_tan,D_P)={v['delta_Atan_DP']} loose_floor(N/2)={v['floor_sizeNhalf']:.2f} delta>floor={v['delta_gt_looseFloor']} credibly_real={v['credibly_real']}")
        print("underpowered (N<32):", out.get("gate_underpowered_N_lt_32"))
        print("VERDICT:", out["gate_verdict"])
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
