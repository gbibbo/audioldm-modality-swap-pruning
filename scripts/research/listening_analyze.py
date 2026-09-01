#!/usr/bin/env python3
"""FROZEN analysis estimator for the listening study (v1.1). CPU only.

Inference target = FIXED PANEL: the average blinded preference across THESE six expert
listeners over the frozen prompt population. Listener sampling uncertainty is NOT
estimated (six listeners cannot support it). Wording later: "in our six-listener expert
panel", never "human listeners generally".

Estimator (equal listener weight):
  recovered_score(trial) = v            if recovered is side B
                         = -v           if recovered is side A       (v in {-2..+2}, +2='B much better')
  H1: A_native = mean_L [ mean_{i in L's native trials} recovered_score ]
      PASS iff lower95(A_native) > 0
  H2 (only if H1): per listener, over prompts that listener rated at BOTH durations,
      J^L = mean_i (native - short); J_H = mean_L J^L ; PASS iff lower95(J_H) > 0
  Bootstrap = LISTENER-STRATIFIED prompt bootstrap: within each listener stratum resample
      that listener's prompts with replacement, recompute the equal-listener-weight mean.
      B = 10000, seed namespace 'LISTENING-STUDY|HUMAN-BOOTSTRAP|2026-08-31', percentile 95%.
  Pre-specified (no gate): leave-one-listener-out for A_native and J_H; inter-rater
      agreement on bridge prompts (D3). Signed -2..+2 scale is fixed; no re-encoding.

NO human data is analyzed at freeze time. --self-test fabricates synthetic responses ONLY
to verify the estimator runs and is calibrated. Real analysis runs post-collection.

Usage:
  --self-test
  --private <private_key.json> --responses <dir-or-files...>   (post-collection only)
"""
import json, os, argparse, hashlib, glob
import numpy as np

SEED_NS = "LISTENING-STUDY|HUMAN-BOOTSTRAP|2026-08-31"
B = 10000
ALPHA_LOW = 2.5


def seed():
    return int.from_bytes(hashlib.sha256(SEED_NS.encode()).digest()[:4], "big")


def recovered_score(v, recovered_side):
    return v if recovered_side == "B" else -v


def strat_lower(strata, rng, B=B):
    """strata: list of 1-D arrays (per listener). Equal-listener-weight mean; resample
    prompts within each listener (vectorized across B). Returns (point, lower95, upper95)."""
    arrs = [np.asarray(a, float) for a in strata if len(a) > 0]
    point = float(np.mean([a.mean() for a in arrs]))
    acc = np.zeros(B)
    for a in arrs:
        m = len(a)
        acc += a[rng.integers(0, m, size=(B, m))].mean(axis=1)
    boot = acc / len(arrs)
    return point, float(np.percentile(boot, ALPHA_LOW)), float(np.percentile(boot, 100 - ALPHA_LOW))


def pooled_lower(strata, rng, B=B):
    """SENSITIVITY (no gate): prompt-weighted pooled bootstrap over all ratings."""
    flat = np.concatenate([np.asarray(a, float) for a in strata if len(a) > 0])
    n = len(flat); point = float(flat.mean())
    means = flat[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return point, float(np.percentile(means, ALPHA_LOW)), float(np.percentile(means, 100 - ALPHA_LOW))


def build_records(private, responses_by_participant):
    """Return per-listener dicts: native[ytid]=score, short[ytid]=score (experimental sev1 only)."""
    listeners = {}
    for code, pv in private["participants"].items():
        resp = responses_by_participant.get(code)
        if resp is None:
            continue
        rmap = {r["trial_id"]: r for r in resp["responses"]}
        L = listeners.setdefault(code, {"native": {}, "short": {}, "sev2_native": {},
                                        "catch": {}})
        for t in pv["trials"]:
            pub_id = t["public_trial_id"]
            if pub_id not in rmap:
                continue
            v = rmap[pub_id]["relevance"]
            if t["type"] == "experimental" and t["severity"] == "sev1":
                sc = recovered_score(v, t["recovered_side"])
                L[t["duration"]][t["ytid"]] = sc
            elif t["type"] == "experimental" and t["severity"] == "sev2":
                L["sev2_native"][t["ytid"]] = recovered_score(v, t["recovered_side"])
            elif t["type"] == "catch":
                L["catch"][pub_id] = {"kind": t.get("catch_kind"), "v": v,
                                       "quality": rmap[pub_id].get("quality"),
                                       "expected": t.get("expected"),
                                       "matched_side": t.get("matched_side")}
    return listeners


def analyze(listeners):
    rng = np.random.default_rng(seed())
    # H1 A_native: per listener mean of native scores
    a_strata = [list(L["native"].values()) for L in listeners.values() if L["native"]]
    A_pt, A_lo, A_hi = strat_lower(a_strata, rng)
    Ap_pt, Ap_lo, Ap_hi = pooled_lower(a_strata, np.random.default_rng(seed() + 2))
    out = {"H1_A_native": {"point": round(A_pt, 4), "lo95": round(A_lo, 4), "hi95": round(A_hi, 4),
                            "PASS": A_lo > 0, "estimator": "listener-stratified (fixed-panel)",
                            "sensitivity_pooled": {"point": round(Ap_pt, 4), "lo95": round(Ap_lo, 4),
                                                    "lo95_gt0": Ap_lo > 0}}}
    # H2 J_H: per listener mean of (native-short) over both-duration prompts
    j_strata = []
    for L in listeners.values():
        common = set(L["native"]) & set(L["short"])
        if common:
            j_strata.append([L["native"][y] - L["short"][y] for y in common])
    rng2 = np.random.default_rng(seed() + 1)
    if out["H1_A_native"]["PASS"] and j_strata:
        J_pt, J_lo, J_hi = strat_lower(j_strata, rng2)
        Jp_pt, Jp_lo, _ = pooled_lower(j_strata, np.random.default_rng(seed() + 3))
        out["H2_J_H"] = {"point": round(J_pt, 4), "lo95": round(J_lo, 4), "hi95": round(J_hi, 4),
                          "PASS": J_lo > 0, "gated_on_H1": True,
                          "estimator": "listener-stratified (fixed-panel)",
                          "sensitivity_pooled": {"point": round(Jp_pt, 4), "lo95": round(Jp_lo, 4),
                                                  "lo95_gt0": Jp_lo > 0}}
    else:
        out["H2_J_H"] = {"skipped_reason": "H1 did not pass" if not out["H1_A_native"]["PASS"] else "no paired data"}
    # leave-one-listener-out for A_native
    loo = []
    codes = [c for c, L in listeners.items() if L["native"]]
    for drop in codes:
        strata = [list(L["native"].values()) for c, L in listeners.items() if L["native"] and c != drop]
        rngd = np.random.default_rng(seed() + 100 + hash(drop) % 1000)
        pt, lo, hi = strat_lower(strata, rngd, B=3000)
        loo.append({"dropped": drop, "point": round(pt, 4), "lo95": round(lo, 4)})
    out["A_native_leave_one_listener_out"] = loo
    return out


# ---------------- self-test (synthetic; NOT human data) ----------------
def self_test():
    print("SELF-TEST (synthetic data; no human responses)")
    NL, rng = 6, np.random.default_rng(20260901)

    def sim_listeners(mu_n, mu_s, sL=0.5, sp=0.6, se=0.6, n_per=13, center=True):
        b = sL * rng.standard_normal(NL)
        if center:
            b = b - b.mean()
        L = {}
        for li in range(NL):
            nat, sh = {}, {}
            for k in range(n_per):
                yt = f"L{li}_p{k}"
                base = sp * rng.standard_normal()
                nat[yt] = float(np.clip(round(mu_n + b[li] + base + se * rng.standard_normal()), -2, 2))
                sh[yt] = float(np.clip(round(mu_s + b[li] + base + se * rng.standard_normal()), -2, 2))
            L[f"P{li}"] = {"native": nat, "short": sh, "sev2_native": {}, "catch": {}}
        return L

    # Type-I (fixed-panel null): panel-avg 0, centered
    reps, hits = 300, 0
    for _ in range(reps):
        L = sim_listeners(0.0, 0.0, center=True)
        strata = [list(v["native"].values()) for v in L.values()]
        _, lo, _ = strat_lower(strata, np.random.default_rng(rng.integers(1e9)), B=1500)
        hits += lo > 0
    print(f"  fixed-panel Type-I (A_native, nominal 0.025): {hits/reps:.3f}")
    # power at mu_n=0.5
    reps, hits = 200, 0
    for _ in range(reps):
        L = sim_listeners(0.5, 0.15, center=True)
        strata = [list(v["native"].values()) for v in L.values()]
        _, lo, _ = strat_lower(strata, np.random.default_rng(rng.integers(1e9)), B=1500)
        hits += lo > 0
    print(f"  power (A_native, mu_n=0.5): {hits/reps:.3f}")
    # full analyze() runs and returns structured output
    L = sim_listeners(0.5, 0.15, center=True)
    out = analyze(L)
    assert "H1_A_native" in out and "H2_J_H" in out and len(out["A_native_leave_one_listener_out"]) == 6
    assert {"point", "lo95", "hi95", "PASS", "sensitivity_pooled"}.issubset(out["H1_A_native"])
    print("  analyze() structure OK; example:", json.dumps(out["H1_A_native"]))
    print("SELF-TEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--private")
    ap.add_argument("--responses", nargs="*", default=[])
    args = ap.parse_args()
    if args.self_test or not args.private:
        self_test(); return
    private = json.load(open(args.private))
    resp_by = {}
    files = []
    for r in args.responses:
        files += glob.glob(os.path.join(r, "*.json")) if os.path.isdir(r) else [r]
    for f in files:
        d = json.load(open(f))
        resp_by[d["participant_code"]] = d
    listeners = build_records(private, resp_by)
    out = analyze(listeners)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
