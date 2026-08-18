#!/usr/bin/env python3
"""Task 5: define a validation split DISJOINT from the evaluation (test) set.

Upstream `dataset_root.json` maps `val` to the SAME file as `test`
(`audiocaps_test_nonrepeat_subset_0.json`, 964 items), so any tuning or model
selection against `val` contaminates the test set (M0 finding 9.1). The five
`audiocaps_test_nonrepeat_subset_{0..4}.json` files are reorderings of the SAME
964 test items (verified: subset_1 ∩ subset_0 = 964), so they are NOT a disjoint
source.

This script adopts the upstream AudioCaps validation set
(`datafiles/audiocaps_val_label.json`) as the disjoint split, after proving by
**wav file id** (not index) that it is disjoint from BOTH the test set and the
training manifest:

    val ∩ test = 0   and   val ∩ train = 0

It writes a self-contained manifest `configs/research/val_split_disjoint.json`
(never modifying upstream `dataset_root.json`) and records the proof + sha256 to
`artifacts/m3_pilot/val_split_check.json`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

META = "data/dataset/metadata/audiocaps"
TEST = f"{META}/testset_subset/audiocaps_test_nonrepeat_subset_0.json"
TRAIN = f"{META}/datafiles/audiocaps_train_label.json"
VAL_SRC = f"{META}/datafiles/audiocaps_val_label.json"
MANIFEST = "configs/research/val_split_disjoint.json"
OUT = "artifacts/m3_pilot/val_split_check.json"


def load_items(path):
    d = json.load(open(path))
    return d["data"] if isinstance(d, dict) and "data" in d else d


def wav_ids(items):
    return [x["wav"] for x in items]


def main() -> int:
    os.makedirs("configs/research", exist_ok=True)
    os.makedirs("artifacts/m3_pilot", exist_ok=True)

    test = set(wav_ids(load_items(TEST)))
    train = set(wav_ids(load_items(TRAIN)))
    val_items = load_items(VAL_SRC)
    val = set(wav_ids(val_items))

    inter_test = sorted(val & test)
    inter_train = sorted(val & train)
    disjoint = (len(inter_test) == 0 and len(inter_train) == 0)

    # Deterministic manifest: sorted wav ids + captions, with provenance.
    by_wav = {}
    for x in val_items:
        by_wav.setdefault(x["wav"], x.get("caption", ""))
    manifest = {
        "name": "audiocaps_val_disjoint",
        "purpose": "validation split disjoint from the test/evaluation set (M0 finding 9.1)",
        "source_metadata": VAL_SRC,
        "n_items": len(by_wav),
        "disjoint_from_test_by_wav_id": len(inter_test) == 0,
        "disjoint_from_train_by_wav_id": len(inter_train) == 0,
        "test_source": TEST,
        "train_source": TRAIN,
        "items": [{"wav": w, "caption": by_wav[w]} for w in sorted(by_wav)],
    }
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    manifest_bytes = open(MANIFEST, "rb").read()
    sha = hashlib.sha256(manifest_bytes).hexdigest()

    result = {
        "test_items": len(test),
        "train_items": len(train),
        "val_items": len(val),
        "val_inter_test": len(inter_test),
        "val_inter_train": len(inter_train),
        "disjoint": disjoint,
        "manifest_path": MANIFEST,
        "manifest_sha256": sha,
        "manifest_bytes": len(manifest_bytes),
    }
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nVAL SPLIT: {'DISJOINT — OK' if disjoint else 'NOT DISJOINT — FAIL'}")
    return 0 if disjoint else 1


if __name__ == "__main__":
    sys.exit(main())
