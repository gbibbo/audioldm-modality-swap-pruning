#!/usr/bin/env python3
"""Build the ENRICHED 512-slot calibration manifest for the Tier-0 Gate B' saliency job.

Plan v4 §7: "Gate B' saliency on enriched pool 512 ex × K=5 × 2 draws with per-slot
storage". The enriched pool is the Q2 calibration partition = 256 NATURAL (the frozen M3B
calibration slots, reused verbatim, ranks 0–255) + 256 TAIL-enriched (the Q2
`data_partition.json` calibration_tail wavs, ranks 256–511). The natural block keeps its
frozen timesteps; the tail block gets its own timestep draws from a distinct sub-seed so
the two blocks do not collide. Schema matches `calibration_manifest.json` (E, K, strata,
slots with wav/caption/t_paired/t_p1), so `m3b_saliency.py --manifest` consumes it directly.

CPU-only, deterministic; reads only frozen inputs (the M3B manifest + the Q2 partition).
No model, no pruned generation.

    .venv/bin/python scripts/research/build_enriched_calibration_manifest.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NATURAL = os.path.join(ROOT, "configs/research/calibration_manifest.json")
PARTITION = os.path.join(ROOT, "configs/research/data_partition.json")
OUT = os.path.join(ROOT, "configs/research/calibration_manifest_enriched.json")

MASTER_SEED = 20260818
TAIL_TIMESTEP_SUBSEED = MASTER_SEED + 1001   # distinct from natural's MASTER_SEED+1
TIMESTEPS = 1000
K = 5


def build():
    natural = json.load(open(NATURAL))
    strata = natural["strata"]
    assert natural["K"] == K and len(strata) == K

    nat_slots = [dict(s) for s in natural["slots"]]     # ranks 0..255, verbatim
    n_nat = len(nat_slots)

    tail_recs = json.load(open(PARTITION))["calibration_tail"]
    tgen = np.random.default_rng(TAIL_TIMESTEP_SUBSEED)  # example-major, then stratum
    tail_slots = []
    for i, rec in enumerate(tail_recs):
        t_paired, t_p1 = [], []
        for (lo, hi) in strata:
            t_paired.append(int(tgen.integers(lo, hi)))
            t_p1.append([int(tgen.integers(lo, hi)), int(tgen.integers(lo, hi))])
        tail_slots.append({
            "rank": n_nat + i,
            "wav": rec["wav"],
            "caption": rec["caption"],
            "t_paired": t_paired,
            "t_p1": t_p1,
        })

    slots = nat_slots + tail_slots
    manifest = {
        "name": "gate_b_prime_enriched_calibration_slots",
        "purpose": "Tier-0 Gate B' P1 per-slot saliency storage on the enriched pool (plan §7)",
        "pre_registration_of": "plan v4 §6 Gate B' (null-split overlap)",
        "E": len(slots), "K": K, "TIMESTEPS": TIMESTEPS, "strata": strata,
        "master_seed": MASTER_SEED, "tail_timestep_subseed": TAIL_TIMESTEP_SUBSEED,
        "caption_rule": "first-caption-in-source-order (natural: from M3B manifest; tail: from Q2 partition)",
        "composition": {"natural": n_nat, "tail": len(tail_slots),
                        "natural_source": "configs/research/calibration_manifest.json (verbatim, ranks 0..%d)" % (n_nat - 1),
                        "tail_source": "configs/research/data_partition.json calibration_tail (ranks %d..%d)" % (n_nat, len(slots) - 1)},
        "noise_policy": natural.get("noise_policy", ""),
        "timestep_generator": "natural: as M3B; tail: np.random.default_rng(MASTER_SEED+1001), example-major then stratum",
        "slots": slots,
    }
    return manifest


def dump(manifest):
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
        fh.write("\n")


def sha_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    m = build()
    # disjointness sanity: all wav ids unique
    wavs = [os.path.splitext(os.path.basename(s["wav"]))[0] for s in m["slots"]]
    assert len(set(wavs)) == len(wavs), "duplicate wav in enriched manifest"
    print(f"E={m['E']} (natural {m['composition']['natural']} + tail {m['composition']['tail']}), "
          f"K={m['K']}, unique wavs={len(set(wavs))}")

    if args.check:
        tmp = json.dumps(m, sort_keys=True, indent=2) + "\n"
        disk = open(OUT).read()
        same = hashlib.sha256(tmp.encode()).hexdigest() == hashlib.sha256(disk.encode()).hexdigest()
        print("DETERMINISM:", "PASS" if same else "FAIL")
        return 0 if same else 1

    dump(m)
    sha = sha_of_file(OUT)
    print(f"wrote {OUT}")
    print(f"sha256: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
