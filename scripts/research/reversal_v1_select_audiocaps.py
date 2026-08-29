#!/usr/bin/env python3
"""FROZEN RECOVERY-REVERSAL-V1 AudioCaps battery selection (CPU-only; run AFTER the freeze commit).

Deterministically instantiates the 96-prompt AudioCaps-test battery under the frozen contract
(docs/recovery_reversal_v1.md). Chronology is binding: the prereg freeze commit MUST already exist.

Canonical eligible universe = the audited AudioCaps TEST label JSON
(data/dataset/metadata/audiocaps/datafiles/audiocaps_test_label.json), NOT the larger test.csv.
The csv/json row discrepancy is DOCUMENTED (counts), not silently resolved by switching universe.

Frozen exclusions (in order; candidate counts recorded after each):
  1. ytid in canonical AudioCaps TEST universe
  2. ytid NOT in AudioCaps TRAIN
  3. ytid NOT in the frozen 64 music-battery ytids (configs/research/icassp_gate0_battery.json)
  4. ytid NOT in the 44 Kim training-source ytids (configs/research/kim193_source_ytids.json)
No semantic / Music / caption-length / difficulty filtering. No manual replacement.

Selection: sort eligible unique ytids by selection_order_key = sha256(SELECTION_SALT|ytid), take
the first 96. Caption: per ytid, choose_caption (UTF-8 canonical order, argmin sha256(SALT|ytid|cap)).
Generation seeds: generation_seed(ytid, r) for r in {0,1} (shared across all three backbones).

STOP (non-zero exit, no manifest written) if the universe cannot be reproducibly established, if
fewer than 96 eligible ytids remain, or if any ytid's caption set is ambiguous/empty.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/reversal_v1_select_audiocaps.py \
        --out configs/research/reversal_v1_audiocaps_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import (  # noqa: E402
    BACKBONES_V1, GENERATION_SALT_V1, N_PROMPTS_V1, N_REPLICATES_V1, SELECTION_SALT_V1,
    apply_exclusions, choose_caption, generation_seed, select_prompts, selection_order_key)

META = "data/dataset/metadata/audiocaps"
TEST_LABEL = f"{META}/datafiles/audiocaps_test_label.json"
TEST_CSV = f"{META}/test.csv"
TRAIN_CSV = f"{META}/train.csv"
BATTERY = "configs/research/icassp_gate0_battery.json"
KIM_SOURCE = "configs/research/kim193_source_ytids.json"
_YT_RE = re.compile(r"([A-Za-z0-9_-]{11})")


def _sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _sha256_obj(o):
    return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def _extract_ytid(wav_field: str) -> str:
    """Extract the 11-char YouTube id from a label-json wav path (AudioSet 'Y<ytid>.wav' convention).

    Basename is Y + 11-char ytid + .wav (12 chars before extension). Strip the leading 'Y'.
    Every extracted ytid is cross-validated against test.csv youtube_id in load_universe().
    """
    base = os.path.basename(wav_field)
    if base.endswith(".wav"):
        base = base[:-4]
    if base.startswith("Y") and len(base) == 12:
        base = base[1:]
    if not _YT_RE.fullmatch(base):
        raise SystemExit(f"cannot extract 11-char ytid from label wav field {wav_field!r} -> {base!r}")
    return base


def load_universe():
    """Canonical 964-ytid test universe from the label JSON; document csv/json discrepancy."""
    label = json.load(open(TEST_LABEL))
    rows = label["data"] if isinstance(label, dict) and "data" in label else label
    csv_rows = list(csv.DictReader(open(TEST_CSV)))
    csv_ytids = {r["youtube_id"] for r in csv_rows}
    # captions per ytid come from the CANONICAL label JSON (self-consistent with the universe)
    caps_by_ytid: dict[str, list] = {}
    for r in rows:
        yt = _extract_ytid(r["wav"])
        caps_by_ytid.setdefault(yt, []).append(r["caption"])
    universe = sorted(caps_by_ytid)
    # cross-validate every canonical ytid is a real AudioCaps test youtube_id
    unknown = [y for y in universe if y not in csv_ytids]
    discrepancy = {
        "test_csv_rows": len(csv_rows), "test_csv_unique_ytids": len(csv_ytids),
        "test_label_json_rows": len(rows), "test_label_json_unique_ytids": len(universe),
        "label_ytids_not_in_csv": len(unknown),
        "explanation": "label JSON is the audited canonical eval universe (one row per test caption "
                       "actually scored); test.csv is the raw AudioCaps release (more caption rows). "
                       "V1 eligible universe = label-JSON unique ytids.",
    }
    if unknown:
        raise SystemExit(f"STOP: {len(unknown)} canonical label ytids absent from test.csv — "
                         f"universe not reproducibly established: {unknown[:5]}")
    return caps_by_ytid, universe, discrepancy


def build(out_path: str) -> dict:
    caps_by_ytid, universe, discrepancy = load_universe()
    train_ytids = {r["youtube_id"] for r in csv.DictReader(open(TRAIN_CSV))}
    music64 = {p["ytid"] for p in json.load(open(BATTERY))["prompts"]}
    kim44 = set(json.load(open(KIM_SOURCE))["source_ytids"])

    eligible, counts = apply_exclusions(universe, train_ytids, music64, kim44)
    if len(eligible) < N_PROMPTS_V1:
        raise SystemExit(f"STOP: only {len(eligible)} eligible ytids < {N_PROMPTS_V1}")

    selected = select_prompts(eligible, N_PROMPTS_V1)

    prompts = []
    for i, yt in enumerate(selected):
        cap = choose_caption(yt, caps_by_ytid[yt])  # raises on empty; STOP on ambiguity upstream
        seeds = [generation_seed(yt, r) for r in range(N_REPLICATES_V1)]
        prompts.append({
            "prompt_index": i, "ytid": yt, "caption": cap["caption"],
            "selection_key": selection_order_key(yt),
            "caption_candidate_count": cap["n_captions"], "chosen_caption_index": cap["chosen_caption_index"],
            "caption_key": cap["caption_key"], "caption_order": "unique captions, UTF-8 bytewise ascending",
            "replicate_indices": list(range(N_REPLICATES_V1)),
            "generation_seeds": seeds,
        })

    # frozen assertions
    yt_all = [p["ytid"] for p in prompts]
    assert len(prompts) == N_PROMPTS_V1, "prompt count"
    assert len(set(yt_all)) == N_PROMPTS_V1, "ytid uniqueness"
    assert all(len(p["generation_seeds"]) == N_REPLICATES_V1 for p in prompts), "seed count"
    assert all(p["generation_seeds"][0] != p["generation_seeds"][1] for p in prompts), "replicate seeds distinct"
    assert not (set(yt_all) & train_ytids), "train overlap"
    assert not (set(yt_all) & music64), "music64 overlap"
    assert not (set(yt_all) & kim44), "kim44 overlap"

    manifest = {
        "artifact": "reversal_v1_audiocaps_manifest",
        "status": "FROZEN battery instantiation (post-prereg-freeze)",
        "contract": "docs/recovery_reversal_v1.md",
        "SELECTION_SALT": SELECTION_SALT_V1, "GENERATION_SALT": GENERATION_SALT_V1,
        "selection_rule": "sort eligible unique ytids by sha256(SELECTION_SALT|ytid) ascending, first 96",
        "caption_rule": "per ytid: unique captions UTF-8 bytewise; choose argmin sha256(SELECTION_SALT|ytid|caption)",
        "generation_seed_rule": "int.from_bytes(sha256(GENERATION_SALT|ytid|r)[:8],'big'); shared across backbones",
        "backbones": list(BACKBONES_V1),
        "n_prompts": N_PROMPTS_V1, "n_replicates": N_REPLICATES_V1,
        "n_wavs_per_system": N_PROMPTS_V1 * N_REPLICATES_V1, "n_wavs_total": N_PROMPTS_V1 * N_REPLICATES_V1 * len(BACKBONES_V1),
        "csv_json_discrepancy": discrepancy,
        "exclusion_counts": counts,
        "exclusions": ["in canonical AudioCaps test", "not in AudioCaps train",
                       "not in frozen 64 music battery", "not in 44 Kim training-source ytids"],
        "sources": {"test_label_sha256": _sha256_file(TEST_LABEL), "test_csv_sha256": _sha256_file(TEST_CSV),
                    "train_csv_sha256": _sha256_file(TRAIN_CSV), "battery_sha256": _sha256_file(BATTERY),
                    "kim_source_sha256": _sha256_file(KIM_SOURCE)},
        "git_sha": _git_sha(),
        "prompts": prompts,
    }
    manifest["manifest_sha256"] = _sha256_obj({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    json.dump(manifest, open(out_path, "w"), indent=1, ensure_ascii=False)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/research/reversal_v1_audiocaps_manifest.json")
    args = ap.parse_args()
    m = build(args.out)
    print(json.dumps({"exclusion_counts": m["exclusion_counts"],
                      "csv_json_discrepancy": m["csv_json_discrepancy"],
                      "n_prompts": m["n_prompts"], "manifest_sha256": m["manifest_sha256"]}, indent=2))
    print("V1 AudioCaps manifest written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
