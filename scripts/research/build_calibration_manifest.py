#!/usr/bin/env python3
"""Build the FROZEN M3B calibration slot manifest (pre-registration).

This materialises the slot recipe that `docs/pilot_protocol.md` already
pre-registers, into an explicit, hashable manifest, so the saliency run draws
EXACTLY these `(example, stratum, timestep)` slots and nothing can be selected
post hoc. It is pure CPU / deterministic: it reads only the AudioCaps **train**
manifest (wav ids + captions) and seeded RNGs. It never opens the base model, the
L1 checkpoint, the dataset audio, or any GPU — so building it cannot contaminate
the pre-registration.

Frozen recipe (from `docs/pilot_protocol.md`, DECISION-M3B-002/003):
  * Example pool  : AudioCaps train (`audiocaps_train_label.json`), disjoint from
                    test/val by wav id (verified in build_val_split.py).
  * E = 256 base examples, drawn by a seeded permutation of the sorted unique wav
                    ids (master seed).
  * Caption rule  : first caption in source-file order (`dict.setdefault`).
  * K = 5 equal-width timestep strata over [0, 1000):
                    [0,200) [200,400) [400,600) [600,800) [800,1000).
  * P2/P3 slots   : 1 timestep per (example, stratum)  -> K   per example.
  * P1 slots      : 2 timesteps per (example, stratum) -> 2K  per example
                    (matches P2/P3 per-stratum weight; B2 in the protocol).
  * Master seed   : 20260818.  Noise eps ~ N(0, I) is regenerated at run time from
                    a per-slot seed derived from the master seed (policy recorded
                    below); only the timesteps are pinned here because only they
                    could otherwise be chosen after seeing results.

Output:
  configs/research/calibration_manifest.json   (tracked; the frozen pre-registration)
  artifacts/m3_pilot/calibration_manifest_check.json  (sha256 + summary, gitignored)

The sha256 printed here is what goes into `docs/pilot_protocol.md` ->
"Calibration manifest SHA256" at freeze time.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

META = "data/dataset/metadata/audiocaps"
TRAIN = f"{META}/datafiles/audiocaps_train_label.json"
MANIFEST = "configs/research/calibration_manifest.json"
OUT = "artifacts/m3_pilot/calibration_manifest_check.json"

# --- frozen recipe constants (mirror docs/pilot_protocol.md) ---------------- #
E = 256
K = 5
TIMESTEPS = 1000
STRATA = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
MASTER_SEED = 20260818
CAPTION_RULE = "first-caption-in-source-order (dict.setdefault)"
NOISE_POLICY = (
    "eps ~ N(0, I) per slot, regenerated at run time from a generator seeded "
    "deterministically as (MASTER_SEED, rank, stratum_index, draw_index); only "
    "timesteps are pinned in this manifest."
)


def load_items(path):
    d = json.load(open(path))
    return d["data"] if isinstance(d, dict) and "data" in d else d


def main() -> int:
    assert len(STRATA) == K
    for (lo, hi) in STRATA:
        assert hi - lo == TIMESTEPS // K, "strata must be equal width"
    assert STRATA[0][0] == 0 and STRATA[-1][1] == TIMESTEPS

    os.makedirs("configs/research", exist_ok=True)
    os.makedirs("artifacts/m3_pilot", exist_ok=True)

    items = load_items(TRAIN)

    # First-caption rule, preserving source order for the caption itself.
    by_wav: dict[str, str] = {}
    for x in items:
        by_wav.setdefault(x["wav"], x.get("caption", ""))

    wav_sorted = sorted(by_wav)  # deterministic base order
    if len(wav_sorted) < E:
        print(f"FAIL: only {len(wav_sorted)} unique wavs, need E={E}", file=sys.stderr)
        return 1

    # Seeded permutation for example selection (sub-seed = MASTER_SEED).
    perm = np.random.default_rng(MASTER_SEED).permutation(len(wav_sorted))
    chosen = [wav_sorted[i] for i in perm[:E]]  # draw order == permuted order

    # Timestep draws: an independent generator (sub-seed = MASTER_SEED + 1),
    # consumed example-major then stratum, so the sequence is fully reproducible.
    tgen = np.random.default_rng(MASTER_SEED + 1)
    slots = []
    for rank, wav in enumerate(chosen):
        t_paired = []  # 1 per stratum (P2/P3 shared audio+text timestep)
        t_p1 = []      # 2 per stratum (P1 text-only)
        for (lo, hi) in STRATA:
            t_paired.append(int(tgen.integers(lo, hi)))
            t_p1.append([int(tgen.integers(lo, hi)), int(tgen.integers(lo, hi))])
        slots.append({
            "rank": rank,
            "wav": wav,
            "caption": by_wav[wav],
            "t_paired": t_paired,   # len K ; used by P2 (audio+text) and P3
            "t_p1": t_p1,           # len K x 2 ; used by P1 (text-only)
        })

    manifest = {
        "name": "m3b_calibration_slots",
        "purpose": "frozen pre-registered saliency calibration slots (M3B, RQ2)",
        "pre_registration_of": "docs/pilot_protocol.md (DECISION-M3B-002/003)",
        "example_pool": TRAIN,
        "E": E,
        "K": K,
        "timesteps": TIMESTEPS,
        "strata": [list(s) for s in STRATA],
        "master_seed": MASTER_SEED,
        "example_selection": "np.random.default_rng(MASTER_SEED).permutation(sorted(unique_wav))[:E]",
        "timestep_generator": "np.random.default_rng(MASTER_SEED + 1), consumed example-major then stratum",
        "caption_rule": CAPTION_RULE,
        "noise_policy": NOISE_POLICY,
        "grad_eval_budget": {
            "B_slots": E * K,
            "P1_text_evals": 2 * E * K,
            "P2P3_shared_evals": 2 * E * K,
            "note": "P1 == P2 == P3 == 2*E*K gradient evals (matched budget, protocol B1/B2); P0 is data-free (0).",
        },
        "slots": slots,
    }

    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    manifest_bytes = open(MANIFEST, "rb").read()
    sha = hashlib.sha256(manifest_bytes).hexdigest()

    # Sanity summary: per-stratum slot coverage (should be exactly E in each).
    per_stratum_paired = [0] * K
    per_stratum_p1 = [0] * K
    for s in slots:
        for k, (lo, hi) in enumerate(STRATA):
            assert lo <= s["t_paired"][k] < hi
            per_stratum_paired[k] += 1
            for tt in s["t_p1"][k]:
                assert lo <= tt < hi
            per_stratum_p1[k] += 2

    check = {
        "manifest_path": MANIFEST,
        "manifest_sha256": sha,
        "manifest_bytes": len(manifest_bytes),
        "n_slots": len(slots),
        "unique_wavs_selected": len(set(s["wav"] for s in slots)),
        "per_stratum_paired_slots": per_stratum_paired,
        "per_stratum_p1_units": per_stratum_p1,
        "first_wav": slots[0]["wav"],
        "first_caption": slots[0]["caption"],
        "all_from_train": True,
    }
    with open(OUT, "w") as fh:
        json.dump(check, fh, indent=2)

    print(json.dumps(check, indent=2))
    ok = (
        len(slots) == E
        and len(set(s["wav"] for s in slots)) == E
        and all(c == E for c in per_stratum_paired)
        and all(c == 2 * E for c in per_stratum_p1)
    )
    print(f"\nCALIBRATION MANIFEST: {'OK' if ok else 'FAIL'}  sha256={sha}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
