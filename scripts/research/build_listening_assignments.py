#!/usr/bin/env python3
"""Generate FROZEN listening-study assignments: blinded public manifests + private
unblinding key + design record. CPU, deterministic, no audio touched.

Design = D1 (chosen by scripts/research/listening_power_sim.py):
  * severity 1: all 80 Arm-D prompts, each assigned to exactly ONE listener, that
    listener judges BOTH durations (short 3.84 s + native 10.24 s) -> 160 sev-1 judgments.
  * severity 2: 36 outcome-blind hash-selected eligible prompts, native only,
    6 per listener, one judgment each (SECONDARY).
  * 3 catch trials / listener (2 identical-audio + 1 matched-vs-unrelated),
    built from prompts the listener never sees experimentally; outcome-independent.

Blinding: recovered/pruned identity, severity, and duration appear ONLY in the
private key. Public manifests carry hashed audio names + caption + generic type.
The hash salt is secret and lives ONLY in the private key (gitignored).

Outputs:
  configs/research/listening_study_design.json                 (public, committed)
  configs/research/listening_study_assignments_private.json    (PRIVATE, gitignored)
  listening_study/public_manifests/P0{1..6}.json               (public, committed)

Run:   .venv-loudness/bin/python scripts/research/build_listening_assignments.py
Check: add --check  (recomputes public manifests + design; compares hashes)
"""
import json, os, sys, hashlib, argparse
import numpy as np

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
INV = os.path.join(ROOT, "configs/research/listening_study_inventory.json")
DESIGN = os.path.join(ROOT, "configs/research/listening_study_design.json")
PRIV = os.path.join(ROOT, "configs/research/listening_study_assignments_private.json")
PUB_DIR = os.path.join(ROOT, "listening_study/public_manifests")
PROTOCOL_DOC = os.path.join(ROOT, "docs/listening_study_protocol.md")

STUDY_VERSION = "LSTUDY-2026-08-31-v1"
N_LISTENERS = 6
SEV2_N = 36
N_CATCH = 3
MIN_SEP = 6           # >=5 intervening trials between a prompt's two durations
MIN_SEP_FALLBACK = 3
ORDER_TRIES = 40000

NS_MASTER = "LISTENING-STUDY|ASSIGN|2026-08-31"
NS_SEV2 = "LISTENING-STUDY|SEV2-SELECT|2026-08-31"


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

    protocol_hash = hashlib.sha256(open(PROTOCOL_DOC, "rb").read()).hexdigest()
    inv = json.load(open(INV))
    stim = inv["stimuli"]
    sev1_prompts = inv["prompts"]["sev1"]          # 80
    sev2_prompts = inv["prompts"]["sev2"]          # 110 eligible
    assert len(sev1_prompts) == 80, len(sev1_prompts)

    rng = np.random.default_rng(h32(NS_MASTER))
    # secret blinding salt (persisted only in the private key)
    salt = hashlib.sha256((NS_MASTER + "|SALT|" +
                           str(rng.integers(0, 2**63))).encode()).hexdigest()

    # ---- sev-2 outcome-blind selection: hash eligible ytids by new salt ----
    sev2_sorted = sorted(sev2_prompts, key=lambda q: hashlib.sha256(
        (NS_SEV2 + "|" + q["ytid"]).encode()).hexdigest())
    sev2_sel = sev2_sorted[:SEV2_N]

    # ---- assign prompts to listeners ----
    codes = ["P0%d" % (i + 1) for i in range(N_LISTENERS)]
    s1_order = list(rng.permutation(len(sev1_prompts)))
    s2_order = list(rng.permutation(len(sev2_sel)))
    s1_sizes = parts(80, N_LISTENERS)              # [14,14,13,13,13,13]
    s2_sizes = parts(SEV2_N, N_LISTENERS)          # [6,6,6,6,6,6]

    s1_assign, s2_assign = {}, {}
    a = 0
    for c, n in zip(codes, s1_sizes):
        s1_assign[c] = [sev1_prompts[s1_order[a + j]] for j in range(n)]; a += n
    a = 0
    for c, n in zip(codes, s2_sizes):
        s2_assign[c] = [sev2_sel[s2_order[a + j]] for j in range(n)]; a += n

    # pool of sev-1 native recovered stimuli usable for catch (per participant: not theirs)
    def rec_native(p):
        return p["stim"]["native"]["recovered"]

    def rec_short(p):
        return p["stim"]["short"]["recovered"]

    hashes_map = {}   # hash_name -> {stim_id (source audio to render), ...}

    def audio_hash(code, trial_id, side):
        hn = "a" + hashlib.sha256((salt + "|" + code + "|" + trial_id + "|" + side)
                                  .encode()).hexdigest()[:16] + ".wav"
        return hn

    def register(code, trial_id, side, stim_id):
        hn = audio_hash(code, trial_id, side)
        hashes_map[hn] = stim_id
        return hn

    priv_all, pub_all = {}, {}
    ab_global = {"recovered_as_A": 0, "total": 0}

    for ci, code in enumerate(codes):
        prng = np.random.default_rng(h32(NS_MASTER + "|" + code))
        trials = []   # internal rich trials before ordering

        # balanced recovered-as-A within participant x stratum
        def balanced_flags(n):
            f = [True] * (n // 2) + [False] * (n - n // 2)
            prng.shuffle(f)
            return f

        s1 = s1_assign[code]
        fn = balanced_flags(len(s1))   # native stratum
        fs = balanced_flags(len(s1))   # short stratum
        for j, p in enumerate(s1):
            pair_id = "s1_" + p["ytid"]
            for dur, tag, flag in (("native", "n", fn[j]), ("short", "s", fs[j])):
                rec = p["stim"][dur]["recovered"]; pru = p["stim"][dur]["pruned"]
                A_stim, B_stim, rec_side = (rec, pru, "A") if flag else (pru, rec, "B")
                tid = "t_%s_%s_%s" % (code, pair_id, tag)
                trials.append(dict(trial_id=tid, type="experimental", pair_id=pair_id,
                                   severity="sev1", duration=dur, ytid=p["ytid"],
                                   caption=p["caption"], A_stim=A_stim, B_stim=B_stim,
                                   recovered_side=rec_side))

        # sev-2 native, one per prompt
        f2 = balanced_flags(len(s2_assign[code]))
        for j, q in enumerate(s2_assign[code]):
            rec = q["stim"]["native"]["recovered"]; pru = q["stim"]["native"]["pruned"]
            flag = f2[j]
            A_stim, B_stim, rec_side = (rec, pru, "A") if flag else (pru, rec, "B")
            tid = "t_%s_s2_%s" % (code, q["ytid"])
            trials.append(dict(trial_id=tid, type="experimental", pair_id=None,
                               severity="sev2", duration="native", ytid=q["ytid"],
                               caption=q["caption"], A_stim=A_stim, B_stim=B_stim,
                               recovered_side=rec_side))

        # ---- catch trials from OTHER participants' prompts ----
        others = [pp for cc in codes if cc != code for pp in s1_assign[cc]]
        oshuf = list(prng.permutation(len(others)))
        pick = [others[k] for k in oshuf]
        c_id1, c_id2, c_mA, c_mB = pick[0], pick[1], pick[2], pick[3]
        # ensure matched vs unrelated captions are clearly different
        for cand in pick[4:]:
            if c_mB["ytid"] != c_mA["ytid"]:
                break
        # catch1: identical native (recovered native of c_id1), A==B content
        trials.append(dict(trial_id="t_%s_catchN" % code, type="catch",
                           catch_kind="identical_native", pair_id=None,
                           severity="catch", duration="native", ytid=c_id1["ytid"],
                           caption=c_id1["caption"], A_stim=rec_native(c_id1),
                           B_stim=rec_native(c_id1), recovered_side="both",
                           expected="about_same"))
        # catch2: identical short
        trials.append(dict(trial_id="t_%s_catchS" % code, type="catch",
                           catch_kind="identical_short", pair_id=None,
                           severity="catch", duration="short", ytid=c_id2["ytid"],
                           caption=c_id2["caption"], A_stim=rec_short(c_id2),
                           B_stim=rec_short(c_id2), recovered_side="both",
                           expected="about_same"))
        # catch3: matched (own caption) vs unrelated (foreign caption), both recovered native
        m_flag = bool(prng.integers(0, 2))
        matched, unrelated = rec_native(c_mA), rec_native(c_mB)
        A_stim, B_stim, matched_side = (matched, unrelated, "A") if m_flag else (unrelated, matched, "B")
        trials.append(dict(trial_id="t_%s_catchM" % code, type="catch",
                           catch_kind="matched_vs_unrelated", pair_id=None,
                           severity="catch", duration="native", ytid=c_mA["ytid"],
                           caption=c_mA["caption"], A_stim=A_stim, B_stim=B_stim,
                           recovered_side="na", matched_side=matched_side,
                           expected="prefer_%s" % matched_side))

        # ---- constrained ordering ----
        order = order_trials(trials, prng)

        # ---- build public + private, register audio hashes ----
        pub_trials, priv_trials = [], []
        for k, t in enumerate(order):
            pub_id = "%s_%02d" % (code, k + 1)   # opaque; no severity/duration/ytid
            A_hn = register(code, t["trial_id"], "A", t["A_stim"])
            B_hn = register(code, t["trial_id"], "B", t["B_stim"])
            pub_trials.append(dict(trial_id=pub_id, type=t["type"],
                                   prompt_text=t["caption"],
                                   audio_A="audio/" + A_hn, audio_B="audio/" + B_hn))
            pt = dict(t); pt["public_trial_id"] = pub_id
            pt["audio_A_hash"] = A_hn; pt["audio_B_hash"] = B_hn
            pt["A_sha256"] = stim[t["A_stim"]]["sha256"] if t["A_stim"] in stim else None
            pt["B_sha256"] = stim[t["B_stim"]]["sha256"] if t["B_stim"] in stim else None
            priv_trials.append(pt)
            if t["type"] == "experimental":
                ab_global["total"] += 1
                if t["recovered_side"] == "A":
                    ab_global["recovered_as_A"] += 1

        # level-check clip: a recovered native from another participant (neutral)
        lvl_stim = rec_native(pick[6])
        lvl_hn = register(code, "levelcheck", "L", lvl_stim)

        assignment_hash = hashlib.sha256(json.dumps(
            [(t["trial_id"], t["A_stim"], t["B_stim"]) for t in order],
            sort_keys=True).encode()).hexdigest()

        pub_all[code] = dict(study_version=STUDY_VERSION,
                             protocol_hash=protocol_hash,
                             assignment_hash=assignment_hash,
                             participant_code=code,
                             level_check_audio="audio/" + lvl_hn,
                             n_trials=len(pub_trials), trials=pub_trials)
        priv_all[code] = dict(participant_code=code, assignment_hash=assignment_hash,
                              level_check_stim=lvl_stim, level_check_hash=lvl_hn,
                              trials=priv_trials)

    # ---- design record ----
    n_by = {c: {"sev1_prompts": len(s1_assign[c]),
                "sev1_trials": 2 * len(s1_assign[c]),
                "sev2_trials": len(s2_assign[c]),
                "catch": N_CATCH,
                "total": 2 * len(s1_assign[c]) + len(s2_assign[c]) + N_CATCH}
            for c in codes}
    design = {
        "artifact": "listening_study_design", "study_version": STUDY_VERSION,
        "protocol_doc": "docs/listening_study_protocol.md", "protocol_sha256": protocol_hash,
        "design": "D1", "n_listeners": N_LISTENERS,
        "sev1_prompts_total": 80, "sev1_raters_per_prompt": 1, "sev1_both_durations": True,
        "sev2_selected": SEV2_N, "sev2_select_namespace": NS_SEV2,
        "sev2_selected_ytids": [q["ytid"] for q in sev2_sel],
        "catch_per_listener": N_CATCH, "min_pair_separation_target": MIN_SEP,
        "trials_by_listener": n_by,
        "ab_counterbalance_global": {**ab_global,
            "recovered_as_A_frac": round(ab_global["recovered_as_A"] / ab_global["total"], 4)},
        "loudness": inv["normalization_design"],
        "inventory_sha256": inv["self_sha256"],
        "power_ref": "configs/research/listening_study_power.json",
    }

    # canonical public hash across design + all public manifests (protocol-independent)
    pub_payload = json.dumps({"design": design, "public": pub_all},
                             sort_keys=True, default=str)
    design["public_bundle_sha256"] = hashlib.sha256(pub_payload.encode()).hexdigest()

    if args.check:
        ok = True
        old = json.load(open(DESIGN)) if os.path.exists(DESIGN) else {}
        same = old.get("public_bundle_sha256") == design["public_bundle_sha256"]
        print("CHECK public_bundle_sha256", "PASS" if same else "FAIL",
              design["public_bundle_sha256"][:16], "vs", str(old.get("public_bundle_sha256"))[:16])
        sys.exit(0 if same else 2)

    with open(DESIGN, "w") as f:
        json.dump(design, f, indent=2, sort_keys=True, default=str)
    priv = {"artifact": "listening_study_assignments_private",
            "WARNING": "UNBLINDING KEY — do not deploy publicly; gitignored.",
            "study_version": STUDY_VERSION, "salt": salt,
            "audio_render_map": hashes_map, "participants": priv_all,
            "public_bundle_sha256": design["public_bundle_sha256"]}
    with open(PRIV, "w") as f:
        json.dump(priv, f, indent=2, sort_keys=True, default=str)
    os.makedirs(PUB_DIR, exist_ok=True)
    for c in codes:
        with open(os.path.join(PUB_DIR, c + ".json"), "w") as f:
            json.dump(pub_all[c], f, indent=2, sort_keys=True, default=str)

    print("WROTE design + private key + %d public manifests" % len(codes))
    print("trials_by_listener:", json.dumps(n_by))
    print("AB global recovered-as-A frac:", design["ab_counterbalance_global"]["recovered_as_A_frac"])
    print("audio render files:", len(hashes_map))
    print("public_bundle_sha256:", design["public_bundle_sha256"])


def order_trials(trials, prng):
    n = len(trials)
    pair_ids = [t.get("pair_id") for t in trials]
    catch_idx = set(i for i, t in enumerate(trials) if t["type"] == "catch")

    def ok(perm, min_sep):
        pos = {}
        for pos_i, ti in enumerate(perm):
            pid = pair_ids[ti]
            if pid is not None:
                if pid in pos and (pos_i - pos[pid]) < min_sep:
                    return False
                pos[pid] = pos_i
        # catch constraints: not in first 2 slots, no two catch adjacent
        cpos = [i for i, ti in enumerate(perm) if ti in catch_idx]
        if any(p < 2 for p in cpos):
            return False
        cpos.sort()
        if any(cpos[k + 1] - cpos[k] == 1 for k in range(len(cpos) - 1)):
            return False
        return True

    for target in (MIN_SEP, MIN_SEP_FALLBACK):
        for _ in range(ORDER_TRIES):
            perm = list(prng.permutation(n))
            if ok(perm, target):
                return [trials[i] for i in perm]
    # last resort: any permutation satisfying adjacency (sep>=2)
    for _ in range(ORDER_TRIES):
        perm = list(prng.permutation(n))
        if ok(perm, 2):
            return [trials[i] for i in perm]
    raise RuntimeError("could not order trials")


if __name__ == "__main__":
    main()
