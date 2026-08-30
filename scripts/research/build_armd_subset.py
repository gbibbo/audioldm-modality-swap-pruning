#!/usr/bin/env python3
"""Instantiate the frozen 80-ytid subset for OP-DURATION-DISCRIMINATOR-1 (Arm D). CPU, 0 cr.

Deterministic, outcome-blind: sha256(f"{SUBSET_SALT}|YTID|{ytid}") ascending, first 80 of the frozen
96 V1.1 ytids. Reuses each ytid's V1.1 replicate-0 integer generation seed. No outcome/caption/label
filtering. Run AFTER the protocol freeze (docs/op_duration_discriminator_1.md).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/build_armd_subset.py
"""
from __future__ import annotations
import hashlib, json, os, sys

V1_MANIFEST = "configs/research/reversal_v1_1_audiocaps_manifest.json"
PROTOCOL = "docs/op_duration_discriminator_1.md"
OUT = "configs/research/op_duration_discriminator_1_subset.json"
SUBSET_SALT = "OP-DURATION-DISCRIMINATOR-1|SUBSET|2026-08-30"
N_SUBSET = 80


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sel_key(ytid):
    return hashlib.sha256(f"{SUBSET_SALT}|YTID|{ytid}".encode("utf-8")).hexdigest()


def main():
    man = json.load(open(V1_MANIFEST))
    prompts = man["prompts"]
    by_ytid = {p["ytid"]: p for p in prompts}
    assert len(by_ytid) == 96, f"expected 96 unique ytids, got {len(by_ytid)}"

    ordered = sorted(by_ytid, key=sel_key)          # ascending by selection hash
    chosen = ordered[:N_SUBSET]
    assert len(chosen) == N_SUBSET

    rows = []
    for new_idx, ytid in enumerate(chosen):
        p = by_ytid[ytid]
        seed_r0 = p["generation_seeds"][0]           # reuse frozen V1.1 r0 seed
        rows.append({
            "subset_prompt_index": new_idx,          # 0..79, ALT generation order
            "v1_1_prompt_index": p["prompt_index"],  # original index -> control WAV name p{idx}_r0
            "ytid": ytid,
            "caption": p["caption"],
            "selection_key": sel_key(ytid),
            "generation_seed_r0": seed_r0,
        })

    payload = {
        "artifact": "op_duration_discriminator_1_subset",
        "status": "frozen 80-ytid subset (outcome-blind); ALT=10.24s r0, control=V1.1 3.84s r0",
        "subset_salt": SUBSET_SALT,
        "n_subset": N_SUBSET,
        "parent_v1_1_manifest": V1_MANIFEST,
        "parent_v1_1_manifest_sha256": sha_file(V1_MANIFEST),
        "protocol_doc": PROTOCOL,
        "protocol_doc_sha256": sha_file(PROTOCOL),
        "selection_rule": "sha256(SUBSET_SALT|YTID|ytid) ascending, first 80; no outcome/caption/label filter",
        "seed_policy": "reuse V1.1 generation_seeds[0]; same integer seed across OPs; x_T shapes differ "
                       "(1,8,96,16) vs (1,8,256,16) so x_T is NOT identical across durations",
        "prompts": rows,
    }
    # subset identity = hash over the ordered ytid list (order matters for reproducibility)
    payload["subset_sha256"] = hashlib.sha256(
        json.dumps([r["ytid"] for r in rows], ensure_ascii=False).encode()).hexdigest()
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "manifest_sha256"},
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    json.dump(payload, open(OUT, "w"), indent=2, ensure_ascii=False)

    # sanity checks
    assert len({r["ytid"] for r in rows}) == N_SUBSET, "duplicate ytid"
    assert len({r["v1_1_prompt_index"] for r in rows}) == N_SUBSET, "duplicate v1_1 index"
    print(f"wrote {OUT}")
    print(f"subset_sha256   = {payload['subset_sha256']}")
    print(f"manifest_sha256 = {payload['manifest_sha256']}")
    print(f"protocol_sha256 = {payload['protocol_doc_sha256'][:16]}  parent_manifest = {payload['parent_v1_1_manifest_sha256'][:16]}")
    print(f"first 3 ytids: {[r['ytid'] for r in rows[:3]]}  ... last: {rows[-1]['ytid']}")
    print(f"seed sample: p0 ytid {rows[0]['ytid']} seed_r0 {rows[0]['generation_seed_r0']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
