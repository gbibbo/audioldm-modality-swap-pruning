#!/usr/bin/env python3
"""Frozen-contract tests for RECOVERY-REVERSAL-V1 (selection, seeds, scorer, verdict).

V1  SELECTION-DET     selection_order_key ordering + select_prompts deterministic & salt-bound.
V2  EXCLUSIONS        apply_exclusions removes train/music64/kim44, records counts, order-correct.
V3  CAPTION-DET       choose_caption deterministic, canonical UTF-8 order, index consistent.
V4  GEN-SEED          generation_seed deterministic; r0!=r1; identical across all 3 backbones.
V5  CARDINALITY       score groups = exactly 192/system, canonical order; !=192 refused.
V6  SCORED-GUARD      assert_scored_cardinality refuses !=192 cosines (anti-chunking).
V7  MALFORMED-MANIFEST corrupt/short groups raise.
V8  R_AC-CASES        primary_verdict: strong+ -> PASS; null -> not PASS.
V9  INTERACTION       interaction_ci: strongly-negative music -> lo(I)>0; determinism.
V10 DENSE-INERT       dense cannot change PASS or R_AC.
V11 SESOI-BOUNDARY    point just below SESOI fails cond1; just above can pass.
V12 HC-NO-PASS        secondary_hc_verdict carries no PASS and cannot change primary.
V13 WAVEFORM-NO-PASS  waveform clip_stats are pure descriptors, absent from the verdict path.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python tests/research/test_reversal_v1.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/research")
from research_pruning.eval.reversal import (  # noqa: E402
    BACKBONES_V1, N_PROMPTS_V1, SELECTION_SALT_V1, SESOI, apply_exclusions, choose_caption,
    generation_seed, interaction_ci, primary_verdict, secondary_hc_verdict, select_prompts,
    selection_order_key)
import hashlib  # noqa: E402


def v1_selection_det():
    uni = [f"yt{i:04d}" for i in range(500)]
    a = select_prompts(uni, 96); b = select_prompts(list(reversed(uni)), 96)
    order_ok = a == b == sorted(set(uni), key=selection_order_key)[:96]
    # salt-bound: the key uses the frozen salt
    exp = hashlib.sha256(f"{SELECTION_SALT_V1}|yt0001".encode()).hexdigest()
    salt_ok = selection_order_key("yt0001") == exp
    uniq_ok = len(set(a)) == 96
    print(f"  V1 selection deterministic={order_ok} salt_bound={salt_ok} unique={uniq_ok}")
    return order_ok and salt_ok and uniq_ok


def v2_exclusions():
    uni = [f"yt{i:04d}" for i in range(200)]
    train = {"yt0000", "yt0001"}; music = {"yt0002"}; kim = {"yt0003"}
    elig, counts = apply_exclusions(uni, train, music, kim)
    removed = set(uni) - set(elig)
    ok = (removed == {"yt0000", "yt0001", "yt0002", "yt0003"}
          and counts["after_not_in_train"] == 198 and counts["after_not_in_music64"] == 197
          and counts["after_not_in_kim44"] == 196)
    print(f"  V2 exclusions removed={sorted(removed)} counts={counts} ok={ok}")
    return ok


def v3_caption_det():
    caps = ["b caption", "a caption", "a caption", "c caption"]
    a = choose_caption("ytX", caps); b = choose_caption("ytX", list(reversed(caps)))
    uniq_sorted = sorted(set(caps), key=lambda c: c.encode("utf-8"))
    idx_ok = uniq_sorted[a["chosen_caption_index"]] == a["caption"]
    ok = a == b and a["n_captions"] == 3 and idx_ok
    print(f"  V3 caption deterministic={a==b} n={a['n_captions']} index_ok={idx_ok}")
    return ok


def v4_gen_seed():
    s0 = generation_seed("ytZ", 0); s1 = generation_seed("ytZ", 1)
    det = generation_seed("ytZ", 0) == s0
    # "same across backbones" == seed depends only on (ytid, replicate), no backbone term
    across = len({generation_seed("ytZ", 0) for _ in BACKBONES_V1}) == 1
    ok = det and s0 != s1 and across
    print(f"  V4 gen_seed det={det} r0!=r1={s0!=s1} same_across_backbones={across}")
    return ok


def _mk_groups(n_items=192):
    import reversal_v1_score as S
    prompts = [{"prompt_index": p, "caption": f"c{p}"} for p in range(96)]
    roots = {bk: "/root" for bk in BACKBONES_V1}
    return S.build_groups({"prompts": prompts}, roots)


def v5_cardinality():
    import reversal_v1_score as S
    g = _mk_groups()
    sizes = [len(x["items"]) for x in g["groups"]]
    ok = sizes == [192, 192, 192]
    # non-192 refused
    raised = False
    try:
        S.build_groups({"prompts": [{"prompt_index": 0, "caption": "c"}]}, {bk: "/r" for bk in BACKBONES_V1})
    except SystemExit:
        raised = True
    print(f"  V5 sizes={sizes} refuse_short={raised}")
    return ok and raised


def v6_scored_guard():
    import reversal_v1_score as S
    good = {"results": [{"name": bk, "cosines": [0.1] * 192} for bk in BACKBONES_V1]}
    bad = {"results": [{"name": "dense_ema", "cosines": [0.1] * 64}]}
    S.assert_scored_cardinality(good)  # no raise
    raised = False
    try:
        S.assert_scored_cardinality(bad)
    except SystemExit:
        raised = True
    print(f"  V6 refuse_wrong_cardinality={raised}")
    return raised


def v7_malformed():
    import reversal_v1_verdict as V
    raised = False
    try:
        V.compute({"recovered": [0.1] * 100, "pruned": [0.1] * 192, "dense": [0.1] * 192})
    except SystemExit:
        raised = True
    print(f"  V7 malformed scores raise={raised}")
    return raised


def _music(seed=0):
    return -0.09 + 0.10 * np.random.default_rng(seed).standard_normal(64)


def v8_rac_cases():
    rng = np.random.default_rng(1); music = _music()
    rec = 0.25 + 0.05 * rng.standard_normal((96, 2)); pru = 0.15 + 0.05 * rng.standard_normal((96, 2))
    v = primary_verdict(rec, pru, music)
    rec0 = 0.15 + 0.05 * rng.standard_normal((96, 2)); pru0 = 0.15 + 0.05 * rng.standard_normal((96, 2))
    v0 = primary_verdict(rec0, pru0, music)
    ok = v["PASS"] is True and v0["PASS"] is False
    print(f"  V8 strong_pass={v['PASS']} null_fail={not v0['PASS']} R_AC={v['R_AC']['point']:.3f}")
    return ok


def v9_interaction():
    ac = np.full(96, 0.03)
    a = interaction_ci(ac, _music()); b = interaction_ci(ac, _music())
    det = a == b
    pos = a["lo"] > 0
    print(f"  V9 interaction lo(I)={a['lo']:.3f} pos={pos} det={det}")
    return pos and det


def v10_dense_inert():
    rng = np.random.default_rng(2); music = _music()
    rec = 0.25 + 0.05 * rng.standard_normal((96, 2)); pru = 0.15 + 0.05 * rng.standard_normal((96, 2))
    v = primary_verdict(rec, pru, music)
    for dense in (np.full((96, 2), 0.9), np.full((96, 2), -0.9), np.zeros((96, 2))):
        vd = primary_verdict(rec, pru, music, dense=dense)
        if vd["PASS"] != v["PASS"] or vd["R_AC"] != v["R_AC"] or vd["I"] != v["I"]:
            print("  V10 FAIL: dense changed the gate"); return False
    print("  V10 dense inert across extreme dense values = True")
    return True


def v11_sesoi_boundary():
    music = _music()
    # near-zero-variance so point ~ mean and lo just tracks point
    below = primary_verdict(np.full((96, 2), 0.024), np.zeros((96, 2)), music)
    above = primary_verdict(np.full((96, 2), 0.026), np.zeros((96, 2)), music)
    ok = (below["PASS_conditions"]["R_AC_point_ge_SESOI"] is False
          and above["PASS_conditions"]["R_AC_point_ge_SESOI"] is True)
    print(f"  V11 SESOI boundary: 0.024<{SESOI}->{below['PASS_conditions']['R_AC_point_ge_SESOI']}, "
          f"0.026>={SESOI}->{above['PASS_conditions']['R_AC_point_ge_SESOI']}")
    return ok


def v12_hc_no_pass():
    rng = np.random.default_rng(3)
    rec = 0.2 + 0.05 * rng.standard_normal((96, 2)); pru = 0.1 + 0.05 * rng.standard_normal((96, 2))
    s = secondary_hc_verdict(rec, pru, _music())
    ok = "PASS" not in s and "R_AC_HC" in s and "I_HC" in s
    print(f"  V12 HC secondary has no PASS key={ok}")
    return ok


def v13_waveform_no_pass():
    import reversal_waveform_panel as W
    st = W.clip_stats(np.linspace(-0.5, 0.5, 4096).astype(np.float64), 16000)
    keys_ok = set(st) == {"rms", "peak", "near_clip_frac", "crest_db", "spectral_centroid_hz"}
    # verdict path never imports waveform metrics
    import reversal_v1_verdict as V
    src = open(V.__file__).read()
    isolated = "waveform" not in src and "clip_stats" not in src
    print(f"  V13 waveform pure descriptors={keys_ok} absent_from_verdict={isolated}")
    return keys_ok and isolated


def main():
    checks = [v1_selection_det, v2_exclusions, v3_caption_det, v4_gen_seed, v5_cardinality,
              v6_scored_guard, v7_malformed, v8_rac_cases, v9_interaction, v10_dense_inert,
              v11_sesoi_boundary, v12_hc_no_pass, v13_waveform_no_pass]
    res = [c() for c in checks]
    ok = all(res)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(res)}/{len(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
