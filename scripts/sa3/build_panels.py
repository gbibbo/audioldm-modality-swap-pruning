#!/usr/bin/env python3
"""Build the SA3 analysis prompt panels (protocol section 2.1, 2.3; overnight mandate Phase 0).

Three panels, pairwise DISJOINT by source wav (youtube_id), drawn deterministically from the
AudioCaps *test* captions (ARC eval set, arXiv 2505.08175 section 3.3):

  * P_smoke  -- engineering only (2-4 prompts). NEVER enters S1, margins, N_main, n_u, or any
               scientific result. Kept in a separate file and tagged role="smoke".
  * P_pilot  -- the ONLY data used to size/calibrate the main experiment (S1, N_main, n_u,
               margins, dense seed streams). Reported as pilot; no section-8 decision read.
  * P_main   -- reserve pool; the frozen main panel is the deterministic size-N_main prefix.

Determinism: one caption per source wav (lowest audiocap_id); caption-length terciles by rank
(exact thirds); a single random.Random(MASTER_SEED) stream permutes each tercile; per-tercile
allocation is fixed. `--check` rebuilds and asserts every output file is byte-identical.

Run:   .venv-sa3/bin/python scripts/sa3/build_panels.py --write
Check: .venv-sa3/bin/python scripts/sa3/build_panels.py --check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys

MASTER_SEED = 20260818
TEST_CSV = "data/dataset/metadata/audiocaps/test.csv"
OUT_DIR = "configs/sa3"
SECONDS_TOTAL = 10  # protocol section 2.1: AudioCaps clip length; fixed for the whole panel

# Panel sizes. Disjoint by source wav; total 4+128+256 = 388 << 975 available.
SIZES = {"smoke": 4, "pilot": 128, "main": 256}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def split_count(total: int, k: int) -> list:
    """Split `total` into k parts as evenly as possible; remainder to the lower-index parts."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def load_pool():
    rows = list(csv.DictReader(open(TEST_CSV)))
    # one caption per source wav (youtube_id): the lowest audiocap_id, deterministic
    by_src = {}
    for r in rows:
        yt = r["youtube_id"]
        aid = int(r["audiocap_id"])
        if yt not in by_src or aid < int(by_src[yt]["audiocap_id"]):
            by_src[yt] = r
    pool = []
    for yt, r in by_src.items():
        cap = r["caption"].strip()
        pool.append({
            "source_wav": yt,
            "audiocap_id": r["audiocap_id"],
            "start_time": int(r["start_time"]),
            "caption": cap,
            "nwords": len(cap.split()),
        })
    # canonical base order independent of dict iteration
    pool.sort(key=lambda d: int(d["audiocap_id"]))
    return pool, len(rows)


def assign_terciles(pool):
    """Exact thirds by caption word-length rank; ties broken by audiocap_id."""
    order = sorted(range(len(pool)), key=lambda i: (pool[i]["nwords"], int(pool[i]["audiocap_id"])))
    n = len(order)
    b1, b2 = n // 3, 2 * n // 3
    tercile_of = {}
    for rank, idx in enumerate(order):
        t = 0 if rank < b1 else (1 if rank < b2 else 2)
        tercile_of[idx] = t
    boundaries = (pool[order[b1]]["nwords"], pool[order[b2]]["nwords"])
    tercs = {0: [], 1: [], 2: []}
    for idx in range(len(pool)):
        d = dict(pool[idx])
        d["tercile"] = tercile_of[idx]
        tercs[tercile_of[idx]].append(d)
    return tercs, boundaries


def build(pool, csv_rows, csv_sha):
    tercs, boundaries = assign_terciles(pool)
    rng = random.Random(MASTER_SEED)
    # permute each tercile with the single shared stream (order 0,1,2 fixed)
    for t in (0, 1, 2):
        rng.shuffle(tercs[t])
    # per-tercile allocation counts for each panel, fixed order smoke<pilot<main
    alloc = {name: split_count(SIZES[name], 3) for name in ("smoke", "pilot", "main")}
    panels = {name: [] for name in SIZES}
    cursor = {0: 0, 1: 0, 2: 0}
    for name in ("smoke", "pilot", "main"):
        for t in (0, 1, 2):
            k = alloc[name][t]
            chunk = tercs[t][cursor[t]:cursor[t] + k]
            cursor[t] += k
            panels[name].extend(chunk)
    # each panel sorted canonically for a stable file
    for name in panels:
        panels[name].sort(key=lambda d: int(d["audiocap_id"]))
    out = {}
    for name, items in panels.items():
        obj = {
            "name": f"panel_{name}",
            "role": name,
            "master_seed": MASTER_SEED,
            "seconds_total": SECONDS_TOTAL,
            "size": len(items),
            "source_csv": TEST_CSV,
            "source_csv_sha256": csv_sha,
            "source_csv_rows": csv_rows,
            "tercile_word_boundaries": list(boundaries),
            "tercile_counts": {str(t): sum(1 for d in items if d["tercile"] == t) for t in (0, 1, 2)},
            "note": ("engineering only; excluded from S1/margins/N_main/n_u and all scientific results"
                     if name == "smoke" else
                     ("pilot: sizes/calibrates the main experiment; no section-8 decision" if name == "pilot"
                      else "reserve pool; frozen main panel = deterministic size-N_main prefix")),
            "items": items,
        }
        out[name] = obj
    return out, panels


def serialize(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (args.write or args.check):
        ap.error("pass --write or --check")

    csv_sha = sha256_file(TEST_CSV)
    pool, csv_rows = load_pool()
    out, panels = build(pool, csv_rows, csv_sha)

    # disjointness + validity assertions (always)
    srcs = {name: set(d["source_wav"] for d in panels[name]) for name in panels}
    for a in ("smoke", "pilot", "main"):
        assert len(srcs[a]) == SIZES[a], (a, len(srcs[a]), SIZES[a])
    assert not (srcs["smoke"] & srcs["pilot"]), "smoke/pilot share a source wav"
    assert not (srcs["smoke"] & srcs["main"]), "smoke/main share a source wav"
    assert not (srcs["pilot"] & srcs["main"]), "pilot/main share a source wav"

    index = {}
    rc = 0
    for name in ("smoke", "pilot", "main"):
        path = os.path.join(OUT_DIR, f"panel_{name}.json")
        data = serialize(out[name])
        digest = sha256_bytes(data)
        index[f"panel_{name}.json"] = {"sha256": digest, "size": out[name]["size"]}
        if args.check:
            if not os.path.exists(path):
                print(f"MISSING {path}"); rc = 1; continue
            cur = open(path, "rb").read()
            same = (cur == data)
            print(f"{'OK  ' if same else 'DIFF'} {path}  sha256={digest}")
            if not same:
                rc = 1
        else:
            with open(path, "w") as fh:
                fh.write(data.decode("utf-8"))
            print(f"WROTE {path}  size={out[name]['size']}  sha256={digest}")

    # panel index file
    idx_obj = {"master_seed": MASTER_SEED, "source_csv_sha256": csv_sha, "panels": index,
               "disjoint_by_source_wav": True}
    idx_path = os.path.join(OUT_DIR, "panels_index.json")
    idx_data = serialize(idx_obj)
    if args.check:
        cur = open(idx_path, "rb").read() if os.path.exists(idx_path) else b""
        same = (cur == idx_data)
        print(f"{'OK  ' if same else 'DIFF'} {idx_path}")
        if not same:
            rc = 1
    else:
        with open(idx_path, "w") as fh:
            fh.write(idx_data.decode("utf-8"))
        print(f"WROTE {idx_path}")
    print("ALL DISJOINT + SIZES OK" if rc == 0 else "MISMATCH")
    return rc


if __name__ == "__main__":
    sys.exit(main())
