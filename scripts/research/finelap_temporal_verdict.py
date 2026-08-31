#!/usr/bin/env python3
"""Part A — FineLAP temporal-semantic FROZEN statistics + verdict (CPU, 0 GPU).

Consumes the persisted frame scores (`artifacts/finelap_temporal/scores_sev{1,2}.json`) and computes
the endpoints defined in `docs/finelap_temporal_protocol.md`:
  PRIMARY  T_2 = D_late_2 - D_early_2 (recovered - pruned2_A), gate lo95(T_2)>0
  SUPPORT  T_1 point>0 (directional; not "confirmed" if CI crosses 0)
  GUARD    D_late_2 point>0 to promote "late semantic gain" wording
  SEAM     sev-2 pruned2_B repeat (never rescues)
  SECOND   semantic mass / occupancy(tau=0.5) / quarter coverage / peak (descriptive)
Paired prompt bootstrap, B=10000, PCG64(1698610719) (frozen namespace
FINELAP-TEMPORAL-RECOVERY|BOOTSTRAP|2026-08-31). Independent unit = prompt; events averaged within
prompt before inference.

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/finelap_temporal_verdict.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

B = 10000
BOOT_SEED = 1698610719              # PCG64; namespace FINELAP-TEMPORAL-RECOVERY|BOOTSTRAP|2026-08-31
TAU = 0.5
BF = 24                            # boundary frame (3.84 s); EARLY 0..23, LATE 24..63
NF = 64
QUARTERS = [(0, 16), (16, 32), (32, 48), (48, 64)]   # four 2.56-s quarters


def within_prompt(scores_by_event, reducer):
    """Average a per-frame reducer over eligible events within a prompt."""
    vals = [reducer(np.asarray(v, dtype=np.float64)) for v in scores_by_event.values()]
    return float(np.mean(vals))


def r_early(v):   return v[:BF].mean()
def r_late(v):    return v[BF:].mean()
def r_mass(v):    return v.mean()
def r_occ(v):     return (v >= TAU).mean()
def r_peak(v):    return v.max()
def r_qcov(v):    return np.mean([1.0 if (v[a:b] >= TAU).any() else 0.0 for a, b in QUARTERS])


def boot_ci(per_prompt, rng, statfn):
    """statfn(resampled_index_array) -> scalar. Paired: one index draw per iteration."""
    n = len(next(iter(per_prompt.values())))
    idx = rng.integers(0, n, size=(B, n))
    pt = statfn(np.arange(n))
    dist = np.array([statfn(idx[b]) for b in range(B)])
    return {"point": float(pt), "lo": float(np.percentile(dist, 2.5)),
            "hi": float(np.percentile(dist, 97.5)), "n": int(n)}


def contrast_stats(sev_scores, systems, rng):
    """Per-prompt within-prompt event-averaged EARLY/LATE means per system; paired contrasts."""
    P = sev_scores  # list of prompt dicts
    def col(system, reducer):
        return np.array([within_prompt(p["scores"][system], reducer) for p in P])
    out = {}
    rec = "recovered"
    for pruned in [s for s in systems if s.startswith("pruned")]:
        e_rec, e_pru = col(rec, r_early), col(pruned, r_early)
        l_rec, l_pru = col(rec, r_late), col(pruned, r_late)
        D_early = e_rec - e_pru
        D_late = l_rec - l_pru
        T = D_late - D_early
        tag = pruned  # pruned_A / pruned_B
        out[tag] = {
            "D_early": boot_ci({"x": D_early}, np.random.default_rng(BOOT_SEED),
                               lambda ix, a=D_early: a[ix].mean()),
            "D_late": boot_ci({"x": D_late}, np.random.default_rng(BOOT_SEED),
                              lambda ix, a=D_late: a[ix].mean()),
            "T": boot_ci({"x": T}, np.random.default_rng(BOOT_SEED),
                         lambda ix, a=T: a[ix].mean()),
            "means": {"early_rec": float(e_rec.mean()), "early_pru": float(e_pru.mean()),
                      "late_rec": float(l_rec.mean()), "late_pru": float(l_pru.mean())},
            "frac_T_pos": float((T > 0).mean())}
    return out


def secondary_stats(sev_scores, systems, rng):
    P = sev_scores
    def col(system, reducer):
        return np.array([within_prompt(p["scores"][system], reducer) for p in P])
    reducers = {"semantic_mass": r_mass, "occupancy": r_occ, "quarter_coverage": r_qcov, "peak": r_peak}
    out = {"per_system_mean": {}, "recovered_minus_prunedA": {}}
    for s in systems:
        out["per_system_mean"][s] = {name: float(col(s, red).mean()) for name, red in reducers.items()}
    for name, red in reducers.items():
        d = col("recovered", red) - col("pruned_A", red)
        out["recovered_minus_prunedA"][name] = boot_ci(
            {"x": d}, np.random.default_rng(BOOT_SEED), lambda ix, a=d: a[ix].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "configs/research/finelap_temporal_result.json"))
    a = ap.parse_args()
    res = {"artifact": "finelap_temporal_result",
           "protocol_doc": "docs/finelap_temporal_protocol.md",
           "protocol_doc_sha256": open(os.path.join(ROOT, "docs/finelap_temporal_protocol.md.sha256")).read().strip(),
           "bootstrap": {"B": B, "seed_pcg64": BOOT_SEED,
                         "namespace": "FINELAP-TEMPORAL-RECOVERY|BOOTSTRAP|2026-08-31"},
           "windows": {"EARLY_frames": [0, BF - 1], "LATE_frames": [BF, NF - 1],
                       "seconds_per_frame": 0.16, "tau": TAU},
           "severities": {}}
    for sev in ("sev1", "sev2"):
        sc = json.load(open(os.path.join(ROOT, f"artifacts/finelap_temporal/scores_{sev}.json")))
        systems = sc["systems"]
        rng = np.random.default_rng(BOOT_SEED)
        cs = contrast_stats(sc["prompts"], systems, rng)
        sec = secondary_stats(sc["prompts"], systems, rng)
        res["severities"][sev] = {
            "n_eligible_prompts": len(sc["prompts"]),
            "n_event_occurrences": sum(len(p["events"]) for p in sc["prompts"]),
            "eligibility_manifest_sha256": sc["eligibility_manifest_sha256"],
            "scores_sha256": sc["scores_sha256"], "systems": systems,
            "contrasts": cs, "secondaries": sec}

    # ---- FROZEN verdict logic
    T2 = res["severities"]["sev2"]["contrasts"]["pruned_A"]["T"]
    Dl2 = res["severities"]["sev2"]["contrasts"]["pruned_A"]["D_late"]
    De2 = res["severities"]["sev2"]["contrasts"]["pruned_A"]["D_early"]
    T1 = res["severities"]["sev1"]["contrasts"]["pruned_A"]["T"]
    mass2 = res["severities"]["sev2"]["secondaries"]["recovered_minus_prunedA"]["semantic_mass"]
    occ2 = res["severities"]["sev2"]["secondaries"]["recovered_minus_prunedA"]["occupancy"]

    primary_pass = T2["lo"] > 0
    t1_dir = T1["point"] > 0
    dlate_guard = Dl2["point"] > 0
    if primary_pass and t1_dir and dlate_guard:
        branch = "A1_late_allocation_supported"
        strong = Dl2["lo"] > 0
    elif (not primary_pass) and (mass2["lo"] > 0 or occ2["lo"] > 0):
        branch = "A2_frame_gain_no_late_redistribution"
        strong = False
    elif primary_pass and not dlate_guard:
        branch = "A1_interaction_but_late_not_positive"  # T2>0 driven by less-negative late; guard fails
        strong = False
    else:
        branch = "A3_finelap_null_or_disagreement"
        strong = False
    res["verdict"] = {
        "primary_T2_gate_lo95_gt0": primary_pass,
        "T2": T2, "D_late2": Dl2, "D_early2": De2, "T1_directional": T1, "T1_point_gt0": t1_dir,
        "D_late2_guard_gt0": dlate_guard, "branch": branch,
        "late_gain_wording_allowed": bool(primary_pass and dlate_guard),
        "late_gain_strong_lo95_gt0": bool(strong and branch.startswith("A1_late")),
        "seam_sev2_prunedB": res["severities"]["sev2"]["contrasts"].get("pruned_B", {}).get("T"),
        "note": ("POST-RESULT diagnostic (frozen). Positive T = recovered-minus-pruned contrast larger "
                 "late than early WITHIN the long generation; not a causal mechanism; late semantic gain "
                 "requires D_late2>0 too.")}
    res["artifact_sha256"] = hashlib.sha256(
        json.dumps(res, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)

    def fmt(d): return f"{d['point']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] n={d['n']}"
    print("== PRIMARY (sev-2, recovered - pruned2_A) ==")
    print("  T_2      =", fmt(T2), "GATE lo95>0:", primary_pass)
    print("  D_early2 =", fmt(De2))
    print("  D_late2  =", fmt(Dl2), "guard>0:", dlate_guard)
    print("== SUPPORT (sev-1) ==  T_1 =", fmt(T1), "point>0:", t1_dir)
    pb = res["severities"]["sev2"]["contrasts"].get("pruned_B", {}).get("T")
    if pb: print("== SEAM (sev-2 pruned2_B) ==  T_2^B =", fmt(pb))
    print("== SECONDARY sev-2 recovered-prunedA ==")
    for k, v in res["severities"]["sev2"]["secondaries"]["recovered_minus_prunedA"].items():
        print(f"  {k:16s} {fmt(v)}")
    print("BRANCH:", branch, "| late_gain_wording_allowed:", res["verdict"]["late_gain_wording_allowed"])
    print("wrote", a.out, "sha", res["artifact_sha256"][:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
