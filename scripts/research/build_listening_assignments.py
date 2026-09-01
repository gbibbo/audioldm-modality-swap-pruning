#!/usr/bin/env python3
"""Generate FROZEN listening-study assignments (v1.1 amendment). CPU, deterministic.

v1.1 design = D3 (partially crossed primary), chosen by the persistent-rater power
simulation and the scientific priority order (sev1 A_native > sev1 J_H > inter-rater
interpretability > sev2). Under the same ~196-judgment budget:
  * severity 1: all 80 Arm-D prompts, 1 primary listener each, BOTH durations (160 judgments);
  * bridge: 18 outcome-blind hash-selected sev-1 prompts get a SECOND listener at BOTH
    durations (36 judgments) -> inter-rater reliability + robustness to persistent raters;
  * NO severity-2 human arm (expendable per the priority order; sev2 is corroborated by the
    automatic multi-metric + FineLAP frame-level evidence already);
  * 3 catch/listener: 2 identical-audio + 1 matched-vs-unrelated built from REAL AudioCaps
    references (robust attention control; may enter the gross-failure criterion).

Blinding: recovered/pruned/severity/duration/catch-identity live ONLY in the private key.
Public manifests carry opaque trial ids + hashed audio names + caption. NO 'type' field.

Outputs: design (public), private key (gitignored), 6 public manifests.
Run / --check as before.
"""
import json, os, sys, hashlib, argparse
import numpy as np

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
INV = os.path.join(ROOT, "configs/research/listening_study_inventory.json")
POOL = os.path.join(ROOT, "configs/research/listening_study_realref_pool.json")
DESIGN = os.path.join(ROOT, "configs/research/listening_study_design.json")
PRIV = os.path.join(ROOT, "configs/research/listening_study_assignments_private.json")
PUB_DIR = os.path.join(ROOT, "listening_study/public_manifests")
PROTO_BASE = os.path.join(ROOT, "docs/listening_study_protocol.md")
PROTO_AMEND = os.path.join(ROOT, "docs/listening_study_protocol_v1_1_amendment.md")
PROTO_AMEND_V12 = os.path.join(ROOT, "docs/listening_study_protocol_v1_2_amendment.md")

STUDY_VERSION = "LSTUDY-2026-08-31-v1.2"
N_LISTENERS = 6
BRIDGE_N = 18
N_CATCH = 3
MIN_SEP = 6
MIN_SEP_FALLBACK = 3
ORDER_TRIES = 60000

NS_MASTER = "LISTENING-STUDY|ASSIGN|2026-08-31|v1.1"
NS_BRIDGE = "LISTENING-STUDY|BRIDGE-SELECT|2026-08-31"
NS_CATCH = "LISTENING-STUDY|CATCH-REALREF|2026-08-31"


def h32(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


def parts(total, k):
    base = total // k
    rem = total - base * k
    return [base + 1] * rem + [base] * (k - rem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    protocol_hash = hashlib.sha256(
        open(PROTO_BASE, "rb").read() + open(PROTO_AMEND, "rb").read()
        + open(PROTO_AMEND_V12, "rb").read()).hexdigest()
    base_sha = hashlib.sha256(open(PROTO_BASE, "rb").read()).hexdigest()
    amend_sha = hashlib.sha256(open(PROTO_AMEND, "rb").read()).hexdigest()
    amend_v12_sha = hashlib.sha256(open(PROTO_AMEND_V12, "rb").read()).hexdigest()

    inv = json.load(open(INV))
    stim = inv["stimuli"]
    pool = json.load(open(POOL))["pool"]
    sev1_prompts = inv["prompts"]["sev1"]
    assert len(sev1_prompts) == 80

    rng = np.random.default_rng(h32(NS_MASTER))
    salt = hashlib.sha256((NS_MASTER + "|SALT|" + str(rng.integers(0, 2**63))).encode()).hexdigest()

    codes = ["P0%d" % (i + 1) for i in range(N_LISTENERS)]
    # primary partition of 80 prompts
    order = list(rng.permutation(80))
    sizes = parts(80, N_LISTENERS)   # [14,14,13,13,13,13]
    owner = {}                        # prompt_idx -> primary listener code
    prim_assign = {c: [] for c in codes}
    a = 0
    for c, n in zip(codes, sizes):
        for j in range(n):
            pidx = order[a + j]; owner[pidx] = c; prim_assign[c].append(pidx)
        a += n

    # bridge: 18 prompts outcome-blind hash-selected, each a 2nd listener != owner, 3/listener
    bridge_idx = sorted(range(80), key=lambda i: hashlib.sha256(
        (NS_BRIDGE + "|" + sev1_prompts[i]["ytid"]).encode()).hexdigest())[:BRIDGE_N]
    second_slots = []
    for c in codes:
        second_slots += [c] * 3
    # deterministic shuffle of slots
    brng = np.random.default_rng(h32(NS_BRIDGE + "|ASSIGN"))
    second_slots = [second_slots[k] for k in brng.permutation(len(second_slots))]
    bridge_second = {}   # prompt_idx -> 2nd listener code
    slot_i = 0
    remaining = list(bridge_idx)
    # greedy: assign each slot a bridge prompt with owner != slot-listener, not already assigned
    for L2 in second_slots:
        placed = False
        for k, pidx in enumerate(remaining):
            if owner[pidx] != L2:
                bridge_second[pidx] = L2; remaining.pop(k); placed = True; break
        if not placed:
            raise RuntimeError("bridge second-rater assignment failed")
    assert len(bridge_second) == BRIDGE_N

    # each listener's sev1 prompts to rate = primary + bridge-second
    rate_prompts = {c: list(prim_assign[c]) for c in codes}
    for pidx, L2 in bridge_second.items():
        rate_prompts[L2].append(pidx)

    hashes_map = {}  # hash_name -> {stim_id, src_path, src_sha256}

    def register(code, trial_id, side, stim_id, src_path, src_sha):
        hn = "a" + hashlib.sha256((salt + "|" + code + "|" + trial_id + "|" + side).encode()).hexdigest()[:16] + ".wav"
        hashes_map[hn] = {"stim_id": stim_id, "src_path": src_path, "src_sha256": src_sha}
        return hn

    def gen_src(stim_id):
        s = stim[stim_id]; return s["src_path"], s["sha256"]

    def rr_src(ytid):
        p = pool[ytid]; return p["staged_path"], p["sha256"]

    pub_all, priv_all = {}, {}
    ab_global = {"recovered_as_A": 0, "total": 0}
    pool_ytids = sorted(pool.keys(),
                        key=lambda y: hashlib.sha256((NS_CATCH + "|" + y).encode()).hexdigest())

    for code in codes:
        prng = np.random.default_rng(h32(NS_MASTER + "|" + code))

        def bal(n):
            f = [True] * (n // 2) + [False] * (n - n // 2)
            prng.shuffle(f); return f

        my = rate_prompts[code]
        fn = bal(len(my)); fs = bal(len(my))
        trials = []
        my_ytids = set()
        for j, pidx in enumerate(my):
            p = sev1_prompts[pidx]; my_ytids.add(p["ytid"])
            is_bridge = pidx in bridge_second and bridge_second[pidx] == code
            role = "bridge2" if is_bridge else "primary"
            pair_id = "s1_" + p["ytid"]
            for dur, tag, flag in (("native", "n", fn[j]), ("short", "s", fs[j])):
                rec = p["stim"][dur]["recovered"]; pru = p["stim"][dur]["pruned"]
                A, B, rside = (rec, pru, "A") if flag else (pru, rec, "B")
                trials.append(dict(trial_id="int_%s_%s_%s" % (code, pair_id, tag),
                    type="experimental", severity="sev1", duration=dur, ytid=p["ytid"],
                    caption=p["caption"], pair_id=pair_id, bridge_role=role,
                    A_stim=A, B_stim=B, recovered_side=rside, A_kind="gen", B_kind="gen"))

        # catch: pick recovered clips from prompts this listener does NOT rate
        others = [sev1_prompts[i] for i in range(80) if sev1_prompts[i]["ytid"] not in my_ytids]
        oshuf = [others[k] for k in prng.permutation(len(others))]
        c1, c2 = oshuf[0], oshuf[1]
        rn = c1["stim"]["native"]["recovered"]; rs = c2["stim"]["short"]["recovered"]
        trials.append(dict(trial_id="int_%s_catchIN" % code, type="catch",
            catch_kind="identical_native", severity="catch", duration="native",
            ytid=c1["ytid"], caption=c1["caption"], pair_id=None, bridge_role=None,
            A_stim=rn, B_stim=rn, recovered_side="both", A_kind="gen", B_kind="gen",
            expected="about_same"))
        trials.append(dict(trial_id="int_%s_catchIS" % code, type="catch",
            catch_kind="identical_short", severity="catch", duration="short",
            ytid=c2["ytid"], caption=c2["caption"], pair_id=None, bridge_role=None,
            A_stim=rs, B_stim=rs, recovered_side="both", A_kind="gen", B_kind="gen",
            expected="about_same"))
        # matched vs unrelated REAL refs: pick matched, then unrelated with disjoint labels
        idx0 = h32(NS_CATCH + "|" + code) % len(pool_ytids)
        matched_yt = pool_ytids[idx0]
        m_labels = set(pool[matched_yt]["labels"])
        unrelated_yt = None
        for k in range(1, len(pool_ytids)):
            cand = pool_ytids[(idx0 + k) % len(pool_ytids)]
            if not (set(pool[cand]["labels"]) & m_labels):
                unrelated_yt = cand; break
        assert unrelated_yt, "no disjoint-label unrelated real ref"
        m_flag = bool(prng.integers(0, 2))
        A_yt, B_yt, matched_side = (matched_yt, unrelated_yt, "A") if m_flag else (unrelated_yt, matched_yt, "B")
        trials.append(dict(trial_id="int_%s_catchMR" % code, type="catch",
            catch_kind="matched_vs_unrelated_real", severity="catch", duration="native",
            ytid=matched_yt, caption=pool[matched_yt]["caption"], pair_id=None, bridge_role=None,
            A_stim="realref|" + A_yt, B_stim="realref|" + B_yt, recovered_side="na",
            A_kind="real", B_kind="real", matched_side=matched_side,
            expected="prefer_%s" % matched_side, matched_ytid=matched_yt, unrelated_ytid=unrelated_yt))

        # order with separation + catch placement
        order_t = order_trials(trials, prng)

        pub_trials, priv_trials = [], []
        for k, t in enumerate(order_t):
            pub_id = "%s_%02d" % (code, k + 1)
            sA, shA = (gen_src(t["A_stim"]) if t["A_kind"] == "gen" else rr_src(t["A_stim"].split("|")[1]))
            sB, shB = (gen_src(t["B_stim"]) if t["B_kind"] == "gen" else rr_src(t["B_stim"].split("|")[1]))
            A_hn = register(code, t["trial_id"], "A", t["A_stim"], sA, shA)
            B_hn = register(code, t["trial_id"], "B", t["B_stim"], sB, shB)
            pub_trials.append(dict(trial_id=pub_id, prompt_text=t["caption"],
                                   audio_A="audio/" + A_hn, audio_B="audio/" + B_hn))
            pt = dict(t); pt["public_trial_id"] = pub_id
            pt["audio_A_hash"] = A_hn; pt["audio_B_hash"] = B_hn
            pt["A_sha256"] = shA; pt["B_sha256"] = shB
            priv_trials.append(pt)
            if t["type"] == "experimental":
                ab_global["total"] += 1
                ab_global["recovered_as_A"] += (t["recovered_side"] == "A")

        lvl = oshuf[3]["stim"]["native"]["recovered"]
        ls, lsh = gen_src(lvl)
        lvl_hn = register(code, "levelcheck", "L", lvl, ls, lsh)

        assignment_hash = hashlib.sha256(json.dumps(
            [(t["trial_id"], t["A_stim"], t["B_stim"]) for t in order_t], sort_keys=True).encode()).hexdigest()
        pub_all[code] = dict(study_version=STUDY_VERSION, protocol_hash=protocol_hash,
                             assignment_hash=assignment_hash, participant_code=code,
                             level_check_audio="audio/" + lvl_hn, n_trials=len(pub_trials),
                             trials=pub_trials)
        priv_all[code] = dict(participant_code=code, assignment_hash=assignment_hash,
                              level_check_stim=lvl, level_check_hash=lvl_hn, trials=priv_trials)

    n_by = {c: {"sev1_primary_prompts": len(prim_assign[c]),
                "bridge_second_prompts": sum(1 for pi, L in bridge_second.items() if L == c),
                "sev1_judgments": 2 * len(rate_prompts[c]), "catch": N_CATCH,
                "total": 2 * len(rate_prompts[c]) + N_CATCH} for c in codes}
    design = {
        "artifact": "listening_study_design", "study_version": STUDY_VERSION,
        "amends": "LSTUDY-2026-08-31-v1.1", "design": "D3",
        "protocol_base_doc": "docs/listening_study_protocol.md", "protocol_base_sha256": base_sha,
        "protocol_amendment_doc": "docs/listening_study_protocol_v1_1_amendment.md",
        "protocol_amendment_sha256": amend_sha,
        "protocol_amendment_v12_doc": "docs/listening_study_protocol_v1_2_amendment.md",
        "protocol_amendment_v12_sha256": amend_v12_sha, "protocol_sha256": protocol_hash,
        "primary_estimator": "unique-prompt (prompt is the unit; ratings averaged within prompt; "
                             "unique-prompt bootstrap B=10000)",
        "n_listeners": N_LISTENERS, "sev1_prompts_total": 80, "bridge_prompts": BRIDGE_N,
        "bridge_select_namespace": NS_BRIDGE,
        "bridge_selected_ytids": sorted(sev1_prompts[i]["ytid"] for i in bridge_idx),
        "sev2_human_arm": False, "catch_per_listener": N_CATCH,
        "catch_realref_namespace": NS_CATCH,
        "inference_target": "FIXED_PANEL (six-listener; listener-stratified prompt bootstrap, equal weight)",
        "min_pair_separation_target": MIN_SEP, "trials_by_listener": n_by,
        "ab_counterbalance_global": {**ab_global,
            "recovered_as_A_frac": round(ab_global["recovered_as_A"] / ab_global["total"], 4)},
        "loudness": inv["normalization_design"], "inventory_sha256": inv["self_sha256"],
        "realref_pool_sha256": json.load(open(POOL)).get("self_sha256"),
        "power_ref": "configs/research/listening_study_power_v2.json",
    }
    pub_payload = json.dumps({"design": design, "public": pub_all}, sort_keys=True, default=str)
    design["public_bundle_sha256"] = hashlib.sha256(pub_payload.encode()).hexdigest()

    if args.check:
        old = json.load(open(DESIGN)) if os.path.exists(DESIGN) else {}
        same = old.get("public_bundle_sha256") == design["public_bundle_sha256"]
        print("CHECK public_bundle_sha256", "PASS" if same else "FAIL",
              design["public_bundle_sha256"][:16], "vs", str(old.get("public_bundle_sha256"))[:16])
        sys.exit(0 if same else 2)

    json.dump(design, open(DESIGN, "w"), indent=2, sort_keys=True, default=str)
    json.dump({"artifact": "listening_study_assignments_private",
               "WARNING": "UNBLINDING KEY — do not deploy publicly; gitignored.",
               "study_version": STUDY_VERSION, "salt": salt, "audio_render_map": hashes_map,
               "bridge_second_rater": {sev1_prompts[p]["ytid"]: L for p, L in bridge_second.items()},
               "participants": priv_all, "public_bundle_sha256": design["public_bundle_sha256"]},
              open(PRIV, "w"), indent=2, sort_keys=True, default=str)
    os.makedirs(PUB_DIR, exist_ok=True)
    for c in codes:
        json.dump(pub_all[c], open(os.path.join(PUB_DIR, c + ".json"), "w"),
                  indent=2, sort_keys=True, default=str)
    print("WROTE v1.1 design + private key + 6 public manifests")
    print("trials_by_listener:", json.dumps(n_by))
    print("AB global recovered-as-A frac:", design["ab_counterbalance_global"]["recovered_as_A_frac"])
    print("audio render files:", len(hashes_map), "| bridge prompts:", BRIDGE_N)
    print("protocol_sha256:", protocol_hash[:16], "| public_bundle_sha256:", design["public_bundle_sha256"][:16])


def order_trials(trials, prng):
    n = len(trials)
    pair_ids = [t.get("pair_id") for t in trials]
    catch_idx = set(i for i, t in enumerate(trials) if t["type"] == "catch")

    def ok(perm, min_sep):
        pos = {}
        for pi, ti in enumerate(perm):
            pid = pair_ids[ti]
            if pid is not None:
                if pid in pos and (pi - pos[pid]) < min_sep:
                    return False
                pos[pid] = pi
        cpos = sorted(pi for pi, ti in enumerate(perm) if ti in catch_idx)
        if any(p < 2 for p in cpos):
            return False
        if any(cpos[k + 1] - cpos[k] == 1 for k in range(len(cpos) - 1)):
            return False
        return True

    for target in (MIN_SEP, MIN_SEP_FALLBACK, 2):
        for _ in range(ORDER_TRIES):
            perm = list(prng.permutation(n))
            if ok(perm, target):
                return [trials[i] for i in perm]
    raise RuntimeError("could not order trials")


if __name__ == "__main__":
    main()
