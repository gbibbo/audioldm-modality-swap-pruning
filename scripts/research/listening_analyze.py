#!/usr/bin/env python3
"""FROZEN analysis estimator for the listening study (v1.2). CPU only.

Statistical unit = PROMPT. Ratings are averaged WITHIN prompt BEFORE any inference,
so every unique severity-1 prompt contributes EXACTLY ONCE (the 18 bridge prompts,
rated by two listeners, are averaged within prompt and do NOT get double weight).

Inference target = FIXED PANEL: mean prompt-level preference estimated from the assigned
members of THIS fixed six-listener panel over the frozen 80-prompt battery. Listener
sampling uncertainty is NOT estimated. Wording: "in our six-listener expert panel".

PRIMARY estimator (unique-prompt weighting):
  recovered_score(trial) = v if recovered is side B else -v   (v in {-2..+2}; +2='B much better')
  H_{i,d}  = mean over that prompt's assigned raters of recovered_score  (bridge -> mean of 2)
  H1: A_native = (1/80) sum_i H_{i,native} ; PASS iff lower95(A_native) > 0
  H2 (only if H1): J_H = (1/80) sum_i (H_{i,native} - H_{i,short}) ; PASS iff lower95(J_H) > 0
  Bootstrap = UNIQUE-PROMPT bootstrap: resample the 80 unique prompt records with
    replacement, B=10000, ns 'LISTENING-STUDY|HUMAN-BOOTSTRAP|V1.2|2026-08-31', 95% pctile.

SENSITIVITIES (descriptive/non-gating; cannot rescue a failed gate): v1.1 listener-stratified
  estimator, pooled-all-ratings estimator, leave-one-listener-out (unique-prompt; retain a
  prompt if >=1 assigned rating remains; no imputation), bridge inter-rater agreement.

NO human data is analyzed at freeze time. --self-test uses synthetic data only.
Usage: --self-test | --private <key.json> --responses <dir-or-files...>
"""
import json, os, argparse, hashlib, glob
import numpy as np

NS = "LISTENING-STUDY|HUMAN-BOOTSTRAP|V1.2|2026-08-31"
B = 10000
ALPHA_LOW = 2.5


def seed(salt=0):
    return int.from_bytes(hashlib.sha256((NS + "|" + str(salt)).encode()).digest()[:4], "big")


def recovered_score(v, side):
    return v if side == "B" else -v


def uprompt_ci(prompt_vals, rng, B=B):
    a = np.asarray(prompt_vals, float); n = len(a); pt = float(a.mean())
    boot = a[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return pt, float(np.percentile(boot, ALPHA_LOW)), float(np.percentile(boot, 100 - ALPHA_LOW))


def strat_ci(strata, rng, B=B):
    arrs = [np.asarray(a, float) for a in strata if len(a)]
    acc = np.zeros(B)
    for a in arrs:
        acc += a[rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1)
    boot = acc / len(arrs)
    return float(np.mean([a.mean() for a in arrs])), float(np.percentile(boot, ALPHA_LOW))


def pooled_ci(flat, rng, B=B):
    a = np.asarray(flat, float); n = len(a)
    boot = a[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(a.mean()), float(np.percentile(boot, ALPHA_LOW))


def build_records(private, resp_by):
    """prompt[ytid][dur] = list of (listener, score); also per-listener lists; bridge markers."""
    prompt = {}
    per_listener = {}
    bridge_ytids = set()
    for code, pv in private["participants"].items():
        resp = resp_by.get(code)
        if resp is None:
            continue
        rmap = {r["trial_id"]: r for r in resp["responses"]}
        pl = per_listener.setdefault(code, {"native": {}, "short": {}, "catch": {}})
        for t in pv["trials"]:
            pid = t["public_trial_id"]
            if pid not in rmap:
                continue
            v = rmap[pid]["relevance"]
            if t["type"] == "experimental" and t["severity"] == "sev1":
                sc = recovered_score(v, t["recovered_side"])
                prompt.setdefault(t["ytid"], {"native": [], "short": []})[t["duration"]].append((code, sc))
                pl[t["duration"]][t["ytid"]] = sc
                if t.get("bridge_role") == "bridge2":
                    bridge_ytids.add(t["ytid"])
            elif t["type"] == "catch":
                pl["catch"][pid] = {"kind": t.get("catch_kind"), "v": v,
                                     "quality": rmap[pid].get("quality"),
                                     "matched_side": t.get("matched_side")}
    return prompt, per_listener, bridge_ytids


def analyze(prompt, per_listener, bridge_ytids, strict=True):
    ytids = sorted(prompt)
    # structural asserts (fail closed) — real analysis expects the frozen 80-prompt battery
    n2 = sum(1 for y in ytids if len(prompt[y]["native"]) == 2 and len(prompt[y]["short"]) == 2)
    n1 = sum(1 for y in ytids if len(prompt[y]["native"]) == 1 and len(prompt[y]["short"]) == 1)
    if strict:
        assert len(ytids) == 80, f"unique prompts {len(ytids)} != 80"
        assert all(prompt[y]["native"] and prompt[y]["short"] for y in ytids), "prompt missing a duration"
        assert n2 == 18 and n1 == 62, f"bridge/non-bridge counts {n2}/{n1} != 18/62"

    Hn = {y: np.mean([s for _, s in prompt[y]["native"]]) for y in ytids}
    Hs = {y: np.mean([s for _, s in prompt[y]["short"]]) for y in ytids}
    nat = [Hn[y] for y in ytids]
    dif = [Hn[y] - Hs[y] for y in ytids]

    rng = np.random.default_rng(seed("A"))
    A_pt, A_lo, A_hi = uprompt_ci(nat, rng)
    out = {"n_unique_prompts": len(ytids), "n_bridge_2rater": n2, "n_single_rater": n1,
           "estimator": "unique-prompt (average within prompt; prompt is the unit)",
           "H1_A_native": {"point": round(A_pt, 4), "lo95": round(A_lo, 4), "hi95": round(A_hi, 4),
                            "PASS": A_lo > 0}}
    if A_lo > 0:
        rngj = np.random.default_rng(seed("J"))
        J_pt, J_lo, J_hi = uprompt_ci(dif, rngj)
        out["H2_J_H"] = {"point": round(J_pt, 4), "lo95": round(J_lo, 4), "hi95": round(J_hi, 4),
                          "PASS": J_lo > 0, "gated_on_H1": True}
    else:
        out["H2_J_H"] = {"skipped_reason": "H1 did not pass"}

    # --- sensitivities (non-gating) ---
    sens = {}
    # v1.1 listener-stratified (double-weights bridge; reported for comparison)
    st_nat = [list(pl["native"].values()) for pl in per_listener.values() if pl["native"]]
    st_dif = []
    for pl in per_listener.values():
        common = set(pl["native"]) & set(pl["short"])
        if common:
            st_dif.append([pl["native"][y] - pl["short"][y] for y in common])
    s_pt, s_lo = strat_ci(st_nat, np.random.default_rng(seed("s1")))
    sj_pt, sj_lo = strat_ci(st_dif, np.random.default_rng(seed("s2")))
    sens["listener_stratified_v11"] = {"A_native": {"point": round(s_pt, 4), "lo95": round(s_lo, 4)},
                                        "J_H": {"point": round(sj_pt, 4), "lo95": round(sj_lo, 4)}}
    # pooled all ratings
    flat_nat = [s for y in ytids for _, s in prompt[y]["native"]]
    flat_dif = []
    for y in ytids:
        dn = {c: s for c, s in prompt[y]["native"]}; ds = {c: s for c, s in prompt[y]["short"]}
        for c in set(dn) & set(ds):
            flat_dif.append(dn[c] - ds[c])
    p_pt, p_lo = pooled_ci(flat_nat, np.random.default_rng(seed("p1")))
    pj_pt, pj_lo = pooled_ci(flat_dif, np.random.default_rng(seed("p2")))
    sens["pooled_all_ratings"] = {"A_native": {"point": round(p_pt, 4), "lo95": round(p_lo, 4)},
                                  "J_H": {"point": round(pj_pt, 4), "lo95": round(pj_lo, 4)}}
    # leave-one-listener-out (unique-prompt; retain prompt if >=1 rating remains; no imputation)
    loo = []
    for drop in sorted(per_listener):
        kept_nat = []
        for y in ytids:
            vals = [s for c, s in prompt[y]["native"] if c != drop]
            if vals:
                kept_nat.append(np.mean(vals))
        rngd = np.random.default_rng(seed("loo" + drop))
        pt, lo, _ = uprompt_ci(kept_nat, rngd, B=3000)
        loo.append({"dropped": drop, "n_prompts": len(kept_nat), "A_native_point": round(pt, 4),
                     "A_native_lo95": round(lo, 4)})
    sens["leave_one_listener_out"] = loo
    # bridge inter-rater agreement (18 prompts x 2 durations)
    diffs, exact, adj, pairs = [], 0, 0, 0
    for y in bridge_ytids:
        for d in ("native", "short"):
            rs = [s for _, s in prompt[y][d]]
            if len(rs) == 2:
                pairs += 1; delta = abs(rs[0] - rs[1]); diffs.append(delta)
                exact += (delta == 0); adj += (delta <= 1)
    sens["bridge_agreement"] = {"n_pairs": pairs,
        "mean_abs_category_diff": round(float(np.mean(diffs)), 3) if diffs else None,
        "exact_agree_frac": round(exact / pairs, 3) if pairs else None,
        "within1_frac": round(adj / pairs, 3) if pairs else None}
    out["sensitivities_nongating"] = sens
    return out


# ---------------- self-test (synthetic; NOT human data) ----------------
def self_test():
    print("SELF-TEST (synthetic; no human responses)")
    NL, rng = 6, np.random.default_rng(20260901)

    def make(mu_n, mu_s, sL=0.5, sp=0.6, se=0.6, center=True):
        b = sL * rng.standard_normal(NL)
        if center:
            b = b - b.mean()
        # 80 prompts: assign owner listener round-robin-ish; 18 bridge get a 2nd distinct listener
        owners = [i % NL for i in range(80)]
        rng.shuffle(owners)
        bridge = set(range(18))
        prompt = {}; per_listener = {c: {"native": {}, "short": {}, "catch": {}} for c in
                                     [f"P{ll}" for ll in range(NL)]}
        bset = set()
        for i in range(80):
            yt = f"y{i}"; base = sp * rng.standard_normal()
            raters = [owners[i]]
            if i in bridge:
                second = (owners[i] + 1 + rng.integers(0, NL - 1)) % NL
                raters.append(int(second))
            prompt[yt] = {"native": [], "short": []}
            for L in raters:
                n = float(np.clip(round(mu_n + b[L] + base + se * rng.standard_normal()), -2, 2))
                s = float(np.clip(round(mu_s + b[L] + base + se * rng.standard_normal()), -2, 2))
                prompt[yt]["native"].append((f"P{L}", n)); prompt[yt]["short"].append((f"P{L}", s))
                per_listener[f"P{L}"]["native"][yt] = n; per_listener[f"P{L}"]["short"][yt] = s
                if i in bridge and L == raters[-1] and len(raters) == 2:
                    bset.add(yt)
        return prompt, per_listener, bset

    # Type-I (fixed-panel null)
    reps, hits = 400, 0
    for _ in range(reps):
        pr, pl, bs = make(0.0, 0.0, center=True)
        vals = [np.mean([s for _, s in pr[y]["native"]]) for y in pr]
        _, lo, _ = uprompt_ci(vals, np.random.default_rng(rng.integers(1e9)), B=1500)
        hits += lo > 0
    print(f"  unique-prompt fixed-panel Type-I (nominal 0.025): {hits/reps:.3f}")
    reps, hits = 250, 0
    for _ in range(reps):
        pr, pl, bs = make(0.5, 0.15, center=True)
        vals = [np.mean([s for _, s in pr[y]["native"]]) for y in pr]
        _, lo, _ = uprompt_ci(vals, np.random.default_rng(rng.integers(1e9)), B=1500)
        hits += lo > 0
    print(f"  power (mu_n=0.5): {hits/reps:.3f}")
    pr, pl, bs = make(0.5, 0.15, center=True)
    out = analyze(pr, pl, bs, strict=True)
    assert out["n_unique_prompts"] == 80 and out["n_bridge_2rater"] == 18 and out["n_single_rater"] == 62
    assert "sensitivities_nongating" in out and len(out["sensitivities_nongating"]["leave_one_listener_out"]) == 6
    print("  analyze() structure OK:", json.dumps(out["H1_A_native"]))
    print("  bridge agreement:", json.dumps(out["sensitivities_nongating"]["bridge_agreement"]))
    print("SELF-TEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--private"); ap.add_argument("--responses", nargs="*", default=[])
    args = ap.parse_args()
    if args.self_test or not args.private:
        self_test(); return
    private = json.load(open(args.private))
    resp_by, files = {}, []
    for r in args.responses:
        files += glob.glob(os.path.join(r, "*.json")) if os.path.isdir(r) else [r]
    for f in files:
        d = json.load(open(f)); resp_by[d["participant_code"]] = d
    prompt, per_listener, bridge = build_records(private, resp_by)
    print(json.dumps(analyze(prompt, per_listener, bridge), indent=2))


if __name__ == "__main__":
    main()
