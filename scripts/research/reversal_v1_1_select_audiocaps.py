#!/usr/bin/env python3
"""FROZEN RECOVERY-REVERSAL-V1.1 AudioCaps selection (CPU-only; run AFTER the V1.1 amendment commit).

Administrative pre-data correction of the V1.0 deterministic hashing rules to the supervisor
specification (docs/recovery_reversal_v1_1.md). Reuses the V1.0 canonical universe + captions loader
byte-for-byte (reversal_v1_select_audiocaps.load_universe) and the shared exclusion filter; ONLY the
ytid-hash namespace and caption-selection algorithm change:

  * ytid order:  selection_key_v11 = sha256(SELECTION_SALT|YTID|ytid)   (ascending, first 96)
  * caption:     Convention B — n = number of caption ROWS (multiset, duplicates PRESERVED, NO dedup);
                 canonical order = rows sorted UTF-8 bytewise; caption text NOT hashed;
                 caption_index = int.from_bytes(sha256(SELECTION_SALT|CAPTION|ytid)[:8],"big") % n

STOP (non-zero exit, no manifest) if any selected ytid has a caption-row count other than 5, or if
the universe cannot be reproducibly established. Writes a NEW manifest; NEVER overwrites V1.0.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/reversal_v1_1_select_audiocaps.py \
        --out configs/research/reversal_v1_1_audiocaps_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/research")
import reversal_v1_select_audiocaps as V10  # reuse universe/caption loader + source paths
from research_pruning.eval.reversal import (  # noqa: E402
    BACKBONES_V1, CAPTION_ROWS_EXPECTED, GENERATION_SALT_V1, N_PROMPTS_V1, N_REPLICATES_V1,
    SELECTION_SALT_V1, apply_exclusions, canonical_caption_rows_v11, choose_caption_v11,
    generation_seed, select_prompts_v11, selection_key_v11)

AMENDMENT_DOC = "docs/recovery_reversal_v1_1.md"
AMENDMENT_SHA_SIDECAR = "docs/recovery_reversal_v1_1.md.sha256"
V10_MANIFEST = "configs/research/reversal_v1_audiocaps_manifest.json"


def _sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _sha256_obj(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def build(out_path: str) -> dict:
    caps_by_ytid, universe, discrepancy = V10.load_universe()
    train = {r["youtube_id"] for r in csv.DictReader(open(V10.TRAIN_CSV))}
    music64 = {p["ytid"] for p in json.load(open(V10.BATTERY))["prompts"]}
    kim44 = set(json.load(open(V10.KIM_SOURCE))["source_ytids"])
    eligible, counts = apply_exclusions(universe, train, music64, kim44)
    if len(eligible) < N_PROMPTS_V1:
        raise SystemExit(f"STOP: only {len(eligible)} eligible < {N_PROMPTS_V1}")

    selected = select_prompts_v11(eligible, N_PROMPTS_V1)

    # STOP if any selected ytid does not have exactly the expected caption-row count
    bad = [(y, len(caps_by_ytid[y])) for y in selected if len(caps_by_ytid[y]) != CAPTION_ROWS_EXPECTED]
    if bad:
        raise SystemExit(f"STOP: {len(bad)} selected ytids have caption-row count != "
                         f"{CAPTION_ROWS_EXPECTED}: {bad[:5]}")

    prompts = []
    for i, yt in enumerate(selected):
        cap = choose_caption_v11(yt, caps_by_ytid[yt])
        seeds = [generation_seed(yt, r) for r in range(N_REPLICATES_V1)]
        prompts.append({
            "prompt_index": i, "ytid": yt, "caption": cap["caption"],
            "selection_key": selection_key_v11(yt),
            "caption_row_count": cap["n_caption_rows"], "chosen_caption_index": cap["chosen_caption_index"],
            "caption_hash_hex": cap["caption_hash_hex"],
            "caption_order": "5 caption ROWS (multiset, duplicates preserved) sorted UTF-8 bytewise",
            "replicate_indices": list(range(N_REPLICATES_V1)), "generation_seeds": seeds,
        })

    yt_all = [p["ytid"] for p in prompts]
    assert len(prompts) == N_PROMPTS_V1 and len(set(yt_all)) == N_PROMPTS_V1
    assert all(p["caption_row_count"] == CAPTION_ROWS_EXPECTED for p in prompts)
    assert all(len(p["generation_seeds"]) == N_REPLICATES_V1 for p in prompts)
    assert all(p["generation_seeds"][0] != p["generation_seeds"][1] for p in prompts)
    assert not (set(yt_all) & train) and not (set(yt_all) & music64) and not (set(yt_all) & kim44)

    amend_sha = _sha256_file(AMENDMENT_DOC) if os.path.exists(AMENDMENT_DOC) else None
    manifest = {
        "artifact": "reversal_v1_1_audiocaps_manifest",
        "status": "FROZEN battery instantiation (post V1.1 pre-data amendment)",
        "amendment": AMENDMENT_DOC, "amendment_sha256": amend_sha,
        "parent_amendment_git_sha": _git_sha(),
        "supersedes_v1_0_manifest": {"path": V10_MANIFEST,
                                     "sha256": _sha256_file(V10_MANIFEST) if os.path.exists(V10_MANIFEST) else None},
        "SELECTION_SALT": SELECTION_SALT_V1, "GENERATION_SALT": GENERATION_SALT_V1,
        "selection_rule": "sort eligible unique ytids by sha256(SELECTION_SALT|YTID|ytid) ascending, first 96",
        "caption_rule": "Convention B: n=5 caption ROWS (multiset, dups preserved, no dedup) sorted UTF-8; "
                        "index=int.from_bytes(sha256(SELECTION_SALT|CAPTION|ytid)[:8],'big')%n; caption text NOT hashed",
        "generation_seed_rule": "int.from_bytes(sha256(GENERATION_SALT|ytid|r)[:8],'big'); shared across backbones",
        "backbones": list(BACKBONES_V1), "n_prompts": N_PROMPTS_V1, "n_replicates": N_REPLICATES_V1,
        "n_wavs_per_system": N_PROMPTS_V1 * N_REPLICATES_V1,
        "n_wavs_total": N_PROMPTS_V1 * N_REPLICATES_V1 * len(BACKBONES_V1),
        "csv_json_discrepancy": discrepancy, "exclusion_counts": counts,
        "exclusions": ["in canonical AudioCaps test", "not in AudioCaps train",
                       "not in frozen 64 music battery", "not in 44 Kim training-source ytids"],
        "sources": {"test_label_sha256": _sha256_file(V10.TEST_LABEL), "test_csv_sha256": _sha256_file(V10.TEST_CSV),
                    "train_csv_sha256": _sha256_file(V10.TRAIN_CSV), "battery_sha256": _sha256_file(V10.BATTERY),
                    "kim_source_sha256": _sha256_file(V10.KIM_SOURCE)},
        "git_sha": _git_sha(), "prompts": prompts,
    }
    manifest["manifest_sha256"] = _sha256_obj({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    json.dump(manifest, open(out_path, "w"), indent=1, ensure_ascii=False)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/research/reversal_v1_1_audiocaps_manifest.json")
    args = ap.parse_args()
    m = build(args.out)
    print(json.dumps({"exclusion_counts": m["exclusion_counts"], "n_prompts": m["n_prompts"],
                      "all_caption_row_counts_5": all(p["caption_row_count"] == 5 for p in m["prompts"]),
                      "amendment_sha256": m["amendment_sha256"],
                      "manifest_sha256": m["manifest_sha256"]}, indent=2))
    print("V1.1 AudioCaps manifest written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
