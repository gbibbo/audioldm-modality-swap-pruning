#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — independent AudioCaps 192-ytid manifest (CPU, 0 cr).

Genuinely independent prospective battery: from the canonical 964-ytid AudioCaps-test universe, EXCLUDE
the 96 V1.1 ytids (already scientifically observed) plus the standard train/music64/kim44 exclusions,
then select 192 NEW ytids by a NEW deterministic salt. One caption per ytid by the V1.1 canonical
five-row/duplicate-preserving convention under a NEW caption salt. No outcome/content/label filtering.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_select_audiocaps.py [--check]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import reversal_v1_select_audiocaps as V10
from research_pruning.eval.reversal import (apply_exclusions, canonical_caption_rows_v11,
                                            derive_paired_seed)

SELECTION_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|AUDIOCAPS|2026-08-30"
CAPTION_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|CAPTION|2026-08-30"
GENERATION_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30"
N = 192
N_REPLICATES = 1                      # r0 only (matched design)
CAPTION_ROWS_EXPECTED = 5
V11_MANIFEST = "configs/research/reversal_v1_1_audiocaps_manifest.json"
OUT = "configs/research/xsev_audiocaps_manifest.json"


def sel_key(ytid):
    return hashlib.sha256(f"{SELECTION_SALT}|YTID|{ytid}".encode("utf-8")).hexdigest()


def choose_caption(ytid, caption_rows):
    rows = canonical_caption_rows_v11(caption_rows)          # multiset, UTF-8 sorted, dup-preserving
    n = len(rows)
    h = hashlib.sha256(f"{CAPTION_SALT}|CAPTION|{ytid}".encode("utf-8")).digest()
    idx = int.from_bytes(h[:8], "big") % n
    return {"caption": rows[idx], "n_caption_rows": n, "chosen_caption_index": idx,
            "caption_hash_hex": h.hex()}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build():
    caps_by_ytid, universe, _ = V10.load_universe()
    train = {r["youtube_id"] for r in csv.DictReader(open(V10.TRAIN_CSV))}
    music64 = {p["ytid"] for p in json.load(open(V10.BATTERY))["prompts"]}
    kim44 = set(json.load(open(V10.KIM_SOURCE))["source_ytids"])
    v11_96 = {p["ytid"] for p in json.load(open(V11_MANIFEST))["prompts"]}
    eligible, counts = apply_exclusions(universe, train, music64, kim44)
    eligible = [y for y in eligible if y not in v11_96]        # NEW: exclude the observed V1.1 96
    if len(eligible) < N:
        raise SystemExit(f"STOP: only {len(eligible)} eligible < {N}")

    selected = sorted(eligible, key=sel_key)[:N]
    bad = [(y, len(caps_by_ytid[y])) for y in selected if len(caps_by_ytid[y]) != CAPTION_ROWS_EXPECTED]
    if bad:
        raise SystemExit(f"STOP: {len(bad)} selected ytids have caption-row count != {CAPTION_ROWS_EXPECTED}: {bad[:5]}")

    prompts = []
    for i, yt in enumerate(selected):
        cap = choose_caption(yt, caps_by_ytid[yt])
        prompts.append({
            "prompt_index": i, "ytid": yt, "caption": cap["caption"],
            "selection_key": sel_key(yt),
            "caption_row_count": cap["n_caption_rows"], "chosen_caption_index": cap["chosen_caption_index"],
            "caption_hash_hex": cap["caption_hash_hex"],
            "generation_seeds": [derive_paired_seed(GENERATION_SALT, yt, r) for r in range(N_REPLICATES)],
        })
    yts = [p["ytid"] for p in prompts]
    assert len(prompts) == N and len(set(yts)) == N
    assert not (set(yts) & v11_96), "overlap with V1.1 96"
    payload = {
        "artifact": "xsev_audiocaps_manifest", "experiment": "RECOVERY-CROSS-SEVERITY-REP-1",
        "selection_salt": SELECTION_SALT, "caption_salt": CAPTION_SALT, "generation_salt": GENERATION_SALT,
        "n": N, "n_replicates": N_REPLICATES,
        "exclusions": {"v1_1_96": len(v11_96), "train": len(train), "music64": len(music64),
                       "kim44": len(kim44), "eligible_after": len(eligible)},
        "provenance": {"universe": len(universe), "v1_1_manifest_sha256": sha_file(V11_MANIFEST),
                       "train_csv_sha256": sha_file(V10.TRAIN_CSV), "battery_sha256": sha_file(V10.BATTERY),
                       "kim_source_sha256": sha_file(V10.KIM_SOURCE)},
        "prompts": prompts,
    }
    payload["prompts_sha256"] = hashlib.sha256(json.dumps(yts, ensure_ascii=False).encode()).hexdigest()
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "manifest_sha256"},
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return payload


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    p = build()
    if a.check:
        old = json.load(open(OUT))
        same = old.get("manifest_sha256") == p["manifest_sha256"]
        print("DETERMINISM", "OK" if same else "MISMATCH", "| sha", p["manifest_sha256"][:16])
        return 0 if same else 1
    json.dump(p, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {OUT} | n={p['n']} | prompts_sha256 {p['prompts_sha256'][:16]} | manifest_sha256 {p['manifest_sha256'][:16]}")
    print(f"eligible_after_exclusions={p['exclusions']['eligible_after']} | first3 {[q['ytid'] for q in p['prompts'][:3]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
