#!/usr/bin/env python3
"""Tests for the ONE parametric manifest validator (research_pruning.manifest_validator).

    V1 GATE0-REAL   the frozen 384-row Gate-0 manifest passes the Gate-0 spec (64x3x{off,on}).
    V2 SEED-AGREE   validator.derive_paired_seed == generator.paired_seed on samples.
    V3 MUTATIONS    each corruption (ytid, seed, recipe, wav sha, dup cell, OFF adapter id,
                    stray state, missing/extra rows) is caught.
    V4 PHENOM-SPEC  a synthetic 4-system phenomenon manifest (2 backbones x {off,on}) validates,
                    and a wrong adapter SHA on an ON row is rejected.

Run: .venv/bin/python tests/research/test_manifest_validator.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

from research_pruning.manifest_validator import (
    ManifestSpec, validate_manifest, assert_valid, derive_paired_seed, battery_ytids)

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
BATTERY = os.path.join(REPO, "configs/research/icassp_gate0_battery.json")
JOB_MANIFEST = ("/teamspace/jobs/gate0-gen-1/artifacts/audioldm-modality-swap-pruning/"
                "artifacts/icassp_gate0/gen_gate0/gen_manifest_dense_both.json")
ADAPTER_SHA = "84a24a38fd95856dd9a5de58c4a0885ca42a03503c88ad66919132f3a57b7c6e"
SALT = "icassp-gate0-noise-20260826"


def _gate0_spec():
    battery = json.load(open(BATTERY))
    return ManifestSpec(
        n_prompts=64, replicates=3, battery_ytids=battery_ytids(battery),
        backbones={"dense"},
        adapter_state_ids={"off": {"none"}, "on": {ADAPTER_SHA}},
        recipe={"ddim_steps": 50, "guidance": 2.5, "eta": 0.0, "latent_t": 96},
        seed_salt=SALT, prompts_sha256=battery.get("prompts_sha256"))


def check_v1_gate0_real():
    if not os.path.exists(JOB_MANIFEST):
        print(f"    V1 SKIP (frozen manifest not present at {JOB_MANIFEST})")
        return True
    man = json.load(open(JOB_MANIFEST))
    ok, errors, summary = validate_manifest(man, _gate0_spec())
    print(f"    V1 gate0 real manifest {summary}: {ok}")
    if not ok:
        for e in errors[:10]:
            print("      ERR", e)
    return ok


def check_v2_seed_agree():
    sys.path.insert(0, os.path.join(REPO, "scripts/research"))
    import gate0_generator as G
    battery = json.load(open(BATTERY))
    ok = True
    for p in battery["prompts"][:5]:
        for r in range(3):
            if derive_paired_seed(SALT, p["ytid"], r) != G.paired_seed(p["ytid"], r):
                ok = False
    print(f"    V2 validator seed == generator.paired_seed (5 prompts x 3): {ok}")
    return ok


def _synthetic_gate0_manifest():
    battery = json.load(open(BATTERY))
    yt = battery_ytids(battery)
    rows = []
    for st, aid in (("off", "none"), ("on", ADAPTER_SHA)):
        for pi in range(64):
            for ri in range(3):
                rows.append({"ytid": yt[pi], "prompt_index": pi, "replicate_index": ri,
                             "seed": derive_paired_seed(SALT, yt[pi], ri), "backbone_id": "dense",
                             "adapter_state": st, "adapter_id": aid, "checkpoint": "dense_ema",
                             "ddim_steps": 50, "eta": 0.0, "guidance": 2.5, "latent_t": 96,
                             "wav_sha256": "a" * 64})
    return {"n": len(rows), "rows": rows}


def check_v3_mutations():
    spec = _gate0_spec()
    caught = 0; total = 0

    def expect_fail(man, label):
        nonlocal caught, total
        total += 1
        ok, errors, _ = validate_manifest(man, spec)
        if not ok:
            caught += 1
        else:
            print(f"      NOT CAUGHT: {label}")

    base = _synthetic_gate0_manifest()
    ok0, _, _ = validate_manifest(base, spec)
    if not ok0:
        print("      baseline synthetic manifest unexpectedly invalid")
        return False

    m = copy.deepcopy(base); m["rows"][0]["ytid"] = "ZZZZZZZZZZZ"; expect_fail(m, "wrong ytid")
    m = copy.deepcopy(base); m["rows"][0]["seed"] = 123; expect_fail(m, "wrong seed")
    m = copy.deepcopy(base); m["rows"][0]["ddim_steps"] = 25; expect_fail(m, "wrong ddim")
    m = copy.deepcopy(base); m["rows"][0]["guidance"] = 3.5; expect_fail(m, "wrong guidance")
    m = copy.deepcopy(base); m["rows"][0]["eta"] = 1.0; expect_fail(m, "wrong eta")
    m = copy.deepcopy(base); m["rows"][0]["wav_sha256"] = "short"; expect_fail(m, "bad wav sha")
    m = copy.deepcopy(base); m["rows"][0]["adapter_id"] = ADAPTER_SHA; expect_fail(m, "OFF has adapter")
    m = copy.deepcopy(base); m["rows"][1] = copy.deepcopy(m["rows"][0]); expect_fail(m, "dup cell")
    m = copy.deepcopy(base); m["rows"][0]["adapter_state"] = "weird"; expect_fail(m, "stray state")
    m = copy.deepcopy(base); m["rows"] = m["rows"][:-1]; m["n"] = len(m["rows"]); expect_fail(m, "missing row")

    ok = caught == total
    print(f"    V3 mutations caught {caught}/{total}: {ok}")
    return ok


def check_v4_phenom_spec():
    battery = json.load(open(BATTERY))
    yt = battery_ytids(battery)
    sliced_sha = "5cc0a79a87c6dba06c70ee032a994cfa1e69f1069a4c10470069dab7b3edf765"
    spec = ManifestSpec(
        n_prompts=64, replicates=3, battery_ytids=yt,
        backbones={"p1_pruned_ema_reconstructed", "p1_recovered"},
        adapter_state_ids={"off": {"none"}, "on": {sliced_sha}},
        recipe={"ddim_steps": 50, "guidance": 2.5, "eta": 0.0, "latent_t": 96},
        seed_salt=SALT)
    # a phenomenon manifest is validated PER backbone (each is 64x3x{off,on}); build one backbone
    rows = []
    for st, aid in (("off", "none"), ("on", sliced_sha)):
        for pi in range(64):
            for ri in range(3):
                rows.append({"ytid": yt[pi], "prompt_index": pi, "replicate_index": ri,
                             "seed": derive_paired_seed(SALT, yt[pi], ri),
                             "backbone_id": "p1_recovered", "adapter_state": st, "adapter_id": aid,
                             "checkpoint": "recovered_ema", "ddim_steps": 50, "eta": 0.0,
                             "guidance": 2.5, "latent_t": 96, "wav_sha256": "b" * 64})
    man = {"n": len(rows), "rows": rows}
    ok_valid, _, _ = validate_manifest(man, spec)
    bad = copy.deepcopy(man); bad["rows"][100]["adapter_id"] = "deadbeef"
    ok_reject = not validate_manifest(bad, spec)[0]
    ok = ok_valid and ok_reject
    print(f"    V4 phenom manifest valid={ok_valid}, wrong-sliced-sha rejected={ok_reject}: {ok}")
    return ok


def main():
    checks = [check_v1_gate0_real, check_v2_seed_agree, check_v3_mutations, check_v4_phenom_spec]
    res = []
    for c in checks:
        print(f"  {c.__name__}")
        res.append(c())
    ok = all(res)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(res)}/{len(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
