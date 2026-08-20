#!/usr/bin/env python3
"""Freeze the plan-v4 pre-registration manifests (CPU queue Q2). Deterministic.

Reads ONLY frozen inputs (AudioCaps train labels/captions, the AudioSet ontology, the
class-label CSV, and the already-frozen M3B calibration manifest). Writes nothing that
depends on a model, a checkpoint, or any pruned generation — this runs entirely before
any pruned-model output is inspected, as pre-registration requires.

Outputs (all under configs/research/, sha256 printed for the ledger):
  event_synonyms_strict.json     comma-alias map + minimal plural morphology
  event_synonyms_expanded.json   strict + a small reviewed manual block (sensitivity only)
  event_family_map.json          mid -> AudioSet top-level family (from the ontology)
  event_set.json                 E*: n_labelled>=N_min AND n_requested>=n_min (per tier)
  event_covariates.json          per-event exposures (audio, calibration-caption)
  data_partition.json            calibration(256 nat + 256 tail) / mechanism(50x20) /
                                 holdout(500), pairwise-disjoint at source-wav level
  sentinel_panel.json            20 events x 15 prompts, stratified by exposure x family
  prompts_heterogeneity_screen.json  200 stratified prompts (Tier 0 screen)
  seed_table.json                master seed, per-prompt paired noise seeds, RAND masks

Run: .venv/bin/python scripts/research/build_v4_manifests.py [--check]
`--check` rebuilds in memory and compares sha256 to the on-disk files (determinism gate).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------- frozen constants
MASTER_SEED = 20260818
N_MIN = 200                       # DECISION-V4-04: AudioCaps-train clips per event
N_MIN_PROMPTS = {"tier0": 10, "tier1": 20}
K_RAND = 20                       # DECISION-V4-04: Gate E exact rank test
RAND_SEEDS = list(range(MASTER_SEED, MASTER_SEED + K_RAND))   # 20260818..837 (as M3A)
FAD_SEEDS = [MASTER_SEED, MASTER_SEED + 1, MASTER_SEED + 2]   # 3-seed FAD/robustness panel

CALIB_NATURAL = 256               # DECISION-V4-07
CALIB_TAIL = 256
MECH_EVENTS, MECH_PROMPTS = 50, 20
SENTINEL_EVENTS, SENTINEL_PROMPTS = 20, 15
HOLDOUT_PROMPTS = 500
SCREEN_PROMPTS = 200

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = os.path.join(ROOT, "data/dataset/metadata/audiocaps")
CSV_PATH = os.path.join(META, "class_labels_indices.csv")
TRAIN_JSON = os.path.join(META, "datafiles/audiocaps_train_label.json")
ONTOLOGY = os.path.join(ROOT, "configs/research/audioset_ontology.json")
CALIB_MANIFEST = os.path.join(ROOT, "configs/research/calibration_manifest.json")
OUT_DIR = os.path.join(ROOT, "configs/research")

# Expanded-map manual additions (reviewed once; SENSITIVITY ONLY, never in the primary
# tail block). Keyed by mid. Conservative: high-value families the strict comma-aliases
# under-count because captions use verbs, not the AudioSet noun. No LLM was used.
EXPANDED_MANUAL = {
    "/m/09x0r": ["talk", "talks", "talking", "speak", "speaks", "speaking", "voice", "voices"],  # Speech
    "/m/03kmc9": ["siren", "sirens", "wailing"],            # Siren
    "/m/032s66": ["gunshot", "gunshots", "gunfire", "shot", "shots", "shooting"],  # Gunshot, gunfire
    "/m/014zdl": ["explode", "explodes", "exploding", "blast", "blasts"],  # Explosion
    "/m/0ngt1": ["thunder", "thundering", "thunderstorm"], # Thunder
    "/m/07pp8cl": ["ding", "dinging"],                     # (example verb form)
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(obj, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2)
        fh.write("\n")
    return path


def plural(w: str) -> str:
    if re.search(r"[sxz]$|ch$|sh$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    return w + "s"


def strict_aliases(display_name: str):
    """Comma-separated official aliases (lowercased) + naive plural of single-word aliases.

    Conservative reading of plan §4 'minimal morphology': plurals of single-word aliases
    only; verb forms are deferred to the expanded map to avoid false positives.
    """
    al = []
    for piece in display_name.split(","):
        p = piece.strip().lower()
        if not p:
            continue
        if p not in al:
            al.append(p)
        if " " not in p and "(" not in p:
            pl = plural(p)
            if pl not in al:
                al.append(pl)
    return al


def wav_id(wav_path: str) -> str:
    return os.path.splitext(os.path.basename(wav_path))[0]


def build_family_map(mids):
    onto = json.load(open(ONTOLOGY))
    by_id = {e["id"]: e for e in onto}
    parents = defaultdict(set)
    child_ids_all = set()
    for e in onto:
        for c in e.get("child_ids", []):
            parents[c].add(e["id"])
            child_ids_all.add(c)
    roots = [e["id"] for e in onto if e["id"] not in child_ids_all]
    root_name = {r: by_id[r]["name"] for r in roots}

    def family_of(mid):
        # BFS upward to a root; deterministic (sorted) tie-break.
        seen, frontier = set(), [mid]
        found = []
        while frontier:
            nxt = []
            for x in frontier:
                if x in root_name:
                    found.append(x)
                for par in sorted(parents.get(x, ())):
                    if par not in seen:
                        seen.add(par)
                        nxt.append(par)
            frontier = nxt
        found = sorted(found, key=lambda r: root_name[r])
        return root_name[found[0]] if found else "UNKNOWN"

    return {m: family_of(m) for m in mids}


def rng_for(*parts):
    """Deterministic sub-RNG seeded from MASTER_SEED and a label, order-independent of BLAS."""
    key = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha256(f"{MASTER_SEED}|{key}".encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def draw_disjoint(event_pools, events, per_event, used, order_rng_label):
    """Round-robin draw `per_event` unused wav_ids per event; returns {event: [wav_ids]}."""
    out = {}
    for e in events:
        pool = [w for w in event_pools[e] if w not in used]
        rng = rng_for(order_rng_label, e)
        pool = list(np.array(pool)[rng.permutation(len(pool))]) if pool else []
        take = pool[:per_event]
        for w in take:
            used.add(w)
        out[e] = take
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify determinism against on-disk hashes")
    args = ap.parse_args()

    # ---- load classes
    mid2name, mid2idx = {}, {}
    for r in csv.DictReader(open(CSV_PATH)):
        mid2name[r["mid"]] = r["display_name"]
        mid2idx[r["mid"]] = int(r["index"])
    classes = set(mid2name)

    strict = {m: strict_aliases(mid2name[m]) for m in classes}
    patterns = {m: [re.compile(r"\b" + re.escape(a) + r"\b") for a in strict[m]] for m in classes}

    # ---- load train clips, compute per-clip requested events (STRICT map)
    data = json.load(open(TRAIN_JSON))["data"]
    n_labelled, n_requested = Counter(), Counter()
    event_clips = defaultdict(list)          # event -> [wav_id] where requested (sorted, unique)
    clip_caption, clip_path = {}, {}
    seen_wav = set()
    for e in data:
        wid = wav_id(e["wav"])
        if wid in seen_wav:
            continue                          # one prompt per source wav
        seen_wav.add(wid)
        clip_caption[wid] = e["caption"]
        clip_path[wid] = e["wav"]
        mids = [m for m in e["labels"].split(",") if m in classes]
        cap = e["caption"].lower()
        for m in set(mids):
            n_labelled[m] += 1
        for m in set(mids):
            if any(p.search(cap) for p in patterns[m]):
                n_requested[m] += 1
                event_clips[m].append(wid)
    for m in event_clips:
        event_clips[m] = sorted(event_clips[m])

    # ---- E* per tier
    def estar(nmin_prompts):
        return sorted(m for m in classes
                      if n_labelled[m] >= N_MIN and n_requested[m] >= nmin_prompts)
    estar_t0, estar_t1 = estar(N_MIN_PROMPTS["tier0"]), estar(N_MIN_PROMPTS["tier1"])

    family = build_family_map(classes)

    # ---- synonym maps
    strict_obj = {
        "map": "strict", "source": "class_labels_indices.csv comma aliases + plural morphology",
        "morphology": "plural of single-word aliases only (plan §4 minimal); verb forms in expanded",
        "n_min": N_MIN, "events": {
            m: {"index": mid2idx[m], "display_name": mid2name[m], "aliases": strict[m]}
            for m in sorted(classes)}
    }
    expanded_obj = {
        "map": "expanded", "role": "SENSITIVITY ONLY — never in the primary tail block",
        "base": "strict", "manual_reviewed": True, "llm_used": False,
        "events": {
            m: {"index": mid2idx[m], "display_name": mid2name[m],
                "aliases": sorted(set(strict[m]) | set(EXPANDED_MANUAL.get(m, [])))}
            for m in sorted(classes)}
    }
    family_obj = {m: family[m] for m in sorted(classes)}

    # ---- event set
    event_set_obj = {
        "N_min": N_MIN, "n_min": N_MIN_PROMPTS,
        "counts": {"E_star_tier0": len(estar_t0), "E_star_tier1": len(estar_t1),
                   "n_requested_ge_200": sum(1 for m in classes if n_requested[m] >= 200)},
        "tier0": estar_t0, "tier1": estar_t1,
        "events": {m: {"display_name": mid2name[m], "index": mid2idx[m],
                       "family": family[m],
                       "n_labelled": int(n_labelled[m]), "n_requested": int(n_requested[m]),
                       "in_tier0": m in set(estar_t0), "in_tier1": m in set(estar_t1)}
                   for m in sorted(set(estar_t0) | set(estar_t1))},
    }

    # ================= partition (draw order fixes disjointness) =================
    used = set()

    # (1a) natural calibration = the frozen M3B manifest wavs
    calib = json.load(open(CALIB_MANIFEST))
    natural = []
    for slot in calib["slots"]:
        w = wav_id(slot["wav"])
        if w not in used:
            natural.append(w); used.add(w)
    natural = natural[:CALIB_NATURAL]

    # (1b) tail-enriched calibration: lower-exposure half of E*(tier1), round-robin
    tail_events = sorted(estar_t1, key=lambda m: (n_labelled[m], m))[:len(estar_t1) // 2]
    tail_pool = {e: event_clips[e] for e in tail_events}
    tail_draw = []
    ei = 0
    rngs = {e: rng_for("tail", e) for e in tail_events}
    perm = {e: list(np.array(event_clips[e])[rngs[e].permutation(len(event_clips[e]))])
            for e in tail_events}
    cursor = {e: 0 for e in tail_events}
    while len(tail_draw) < CALIB_TAIL and tail_events:
        progressed = False
        for e in tail_events:
            while cursor[e] < len(perm[e]):
                w = perm[e][cursor[e]]; cursor[e] += 1
                if w not in used:
                    tail_draw.append(w); used.add(w); progressed = True
                    break
            if len(tail_draw) >= CALIB_TAIL:
                break
        if not progressed:
            break
    calib_pool = natural + tail_draw

    # (2) mechanism set: 50 events stratified across the exposure range of E*(tier1)
    ranked = sorted(estar_t1, key=lambda m: (n_labelled[m], m))
    idxs = np.linspace(0, len(ranked) - 1, MECH_EVENTS).round().astype(int)
    mech_events = []
    for i in idxs:
        m = ranked[int(i)]
        if m not in mech_events:
            mech_events.append(m)
    # top up if rounding collided
    j = 0
    while len(mech_events) < MECH_EVENTS and j < len(ranked):
        if ranked[j] not in mech_events:
            mech_events.append(ranked[j])
        j += 1
    mech_events = sorted(mech_events)
    mech_draw = draw_disjoint(event_clips, mech_events, MECH_PROMPTS, used, "mechanism")

    # (3) sentinel panel: 20 events stratified by exposure x family
    by_family = defaultdict(list)
    for m in estar_t1:
        by_family[family[m]].append(m)
    fams = sorted(by_family)
    sentinel_events = []
    # round-robin across families, within family pick spread by exposure
    fam_sorted = {f: sorted(by_family[f], key=lambda m: (n_labelled[m], m)) for f in fams}
    fam_cursor = {f: 0 for f in fams}
    while len(sentinel_events) < SENTINEL_EVENTS:
        progressed = False
        for f in fams:
            lst = fam_sorted[f]
            # stride through the family's exposure-sorted list
            while fam_cursor[f] < len(lst):
                m = lst[fam_cursor[f]]; fam_cursor[f] += 1
                if m not in mech_events and m not in sentinel_events:
                    sentinel_events.append(m); progressed = True
                    break
            if len(sentinel_events) >= SENTINEL_EVENTS:
                break
        if not progressed:
            break
    sentinel_events = sorted(sentinel_events)
    sentinel_draw = draw_disjoint(event_clips, sentinel_events, SENTINEL_PROMPTS, used, "sentinel")

    # (4) intervention holdout: 500 prompts across E*(tier1), disjoint from all above
    holdout = []
    ho_events = sorted(estar_t1)
    ho_perm = {e: list(np.array(event_clips[e])[rng_for("holdout", e).permutation(len(event_clips[e]))])
               for e in ho_events}
    ho_cursor = {e: 0 for e in ho_events}
    while len(holdout) < HOLDOUT_PROMPTS:
        progressed = False
        for e in ho_events:
            while ho_cursor[e] < len(ho_perm[e]):
                w = ho_perm[e][ho_cursor[e]]; ho_cursor[e] += 1
                if w not in used:
                    holdout.append(w); used.add(w); progressed = True
                    break
            if len(holdout) >= HOLDOUT_PROMPTS:
                break
        if not progressed:
            break
    holdout = sorted(holdout)

    # (5) Tier-0 heterogeneity screen: 200 prompts across E*(tier0), disjoint from all above
    screen = []
    sc_events = sorted(estar_t0)
    sc_perm = {e: list(np.array(event_clips[e])[rng_for("screen", e).permutation(len(event_clips[e]))])
               for e in sc_events}
    sc_cursor = {e: 0 for e in sc_events}
    while len(screen) < SCREEN_PROMPTS:
        progressed = False
        for e in sc_events:
            while sc_cursor[e] < len(sc_perm[e]):
                w = sc_perm[e][sc_cursor[e]]; sc_cursor[e] += 1
                if w not in used:
                    screen.append(w); used.add(w); progressed = True
                    break
            if len(screen) >= SCREEN_PROMPTS:
                break
        if not progressed:
            break
    screen = sorted(screen)

    # requested-events per wav (cheap: invert event_clips)
    wav_reqs = defaultdict(list)
    for e, ws in event_clips.items():
        for w in ws:
            wav_reqs[w].append(e)
    for w in wav_reqs:
        wav_reqs[w] = sorted(wav_reqs[w])

    def recs(wids):
        return [{"wav_id": w, "wav": clip_path[w], "caption": clip_caption[w],
                 "requested_events": wav_reqs.get(w, [])} for w in wids]

    # ---- calibration-caption exposure = n_requested within the calibration pool
    calib_set = set(calib_pool)
    calib_caption_exp = Counter()
    for w in calib_set:
        for e in wav_reqs.get(w, []):
            calib_caption_exp[e] += 1

    covariates_obj = {
        "note": "Exposures only (CPU-derivable). Acoustic + guidance covariates are added by "
                "later jobs per plan §3; frozen spec lives in the plan. AudioSet-unbalanced "
                "exposure not fetched -> null (sensitivity only).",
        "calibration_pool_size": len(calib_pool),
        "events": {m: {
            "display_name": mid2name[m], "family": family[m],
            "n_labelled": int(n_labelled[m]),
            "log_audio_exposure": float(np.log(n_labelled[m])),
            "n_requested_total": int(n_requested[m]),
            "n_requested_in_calibration_pool": int(calib_caption_exp[m]),
            "log_calibration_caption_exposure": float(np.log1p(calib_caption_exp[m])),
            "log_audioset_unbalanced_exposure": None,
        } for m in sorted(estar_t1)},
    }

    partition_obj = {
        "disjoint_at": "source-wav id (youtube id)", "master_seed": MASTER_SEED,
        "draw_order": ["calibration_natural", "calibration_tail", "mechanism", "sentinel", "holdout"],
        "sizes": {"calibration_natural": len(natural), "calibration_tail": len(tail_draw),
                  "calibration_total": len(calib_pool),
                  "mechanism_events": len(mech_events),
                  "mechanism_prompts": sum(len(v) for v in mech_draw.values()),
                  "holdout": len(holdout)},
        "calibration_natural_source": "configs/research/calibration_manifest.json (M3B, DECISION-M3B-002/003)",
        "calibration_natural": natural,
        "calibration_tail_events": tail_events,
        "calibration_tail": recs(sorted(tail_draw)),
        "mechanism_events": mech_events,
        "mechanism": {e: recs(mech_draw[e]) for e in mech_events},
        "holdout_blinded": True,
        "holdout": recs(holdout),
    }

    sentinel_obj = {
        "events": sentinel_events, "prompts_per_event": SENTINEL_PROMPTS,
        "stratified_by": "exposure x AudioSet family",
        "families": {f: [m for m in sentinel_events if family[m] == f] for f in sorted(set(family[m] for m in sentinel_events))},
        "panel": {e: recs(sentinel_draw[e]) for e in sentinel_events},
        "actual_prompt_counts": {e: len(sentinel_draw[e]) for e in sentinel_events},
    }

    screen_obj = {
        "tier": 0, "n_prompts": len(screen), "systems": ["base", "P0-std", "P1-nat"],
        "stratified_across": "E* tier0 events",
        "prompts": recs(screen),
    }

    # ---- seed table (seed pairing: one noise seed per prompt, shared by all systems)
    all_prompt_wavs = sorted(set(calib_pool) | {w for v in mech_draw.values() for w in v}
                             | {w for v in sentinel_draw.values() for w in v}
                             | set(holdout) | set(screen))
    per_prompt_seed = {w: int(hashlib.sha256(f"noise|{MASTER_SEED}|{w}".encode()).hexdigest(), 16) % (2**31)
                       for w in all_prompt_wavs}
    seed_obj = {
        "master_seed": MASTER_SEED,
        "seed_pairing": "one initial-noise seed per prompt (wav_id), identical across all systems",
        "rand_mask_seeds": RAND_SEEDS, "K_rand": K_RAND,
        "fad_robustness_seeds": FAD_SEEDS,
        "per_prompt_noise_seed": per_prompt_seed,
    }

    outputs = [
        ("event_synonyms_strict.json", strict_obj),
        ("event_synonyms_expanded.json", expanded_obj),
        ("event_family_map.json", family_obj),
        ("event_set.json", event_set_obj),
        ("event_covariates.json", covariates_obj),
        ("data_partition.json", partition_obj),
        ("sentinel_panel.json", sentinel_obj),
        ("prompts_heterogeneity_screen.json", screen_obj),
        ("seed_table.json", seed_obj),
    ]

    # ---- disjointness assertions (fail loudly before writing)
    sets = {"calibration": set(calib_pool),
            "mechanism": {w for v in mech_draw.values() for w in v},
            "sentinel": {w for v in sentinel_draw.values() for w in v},
            "holdout": set(holdout), "screen": set(screen)}
    names = sorted(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            assert not inter, f"OVERLAP {names[i]} & {names[j]}: {len(inter)}"

    print(f"E* tier0={len(estar_t0)} tier1={len(estar_t1)}  n_req>=200={event_set_obj['counts']['n_requested_ge_200']}")
    print(f"calibration: {len(natural)} natural + {len(tail_draw)} tail = {len(calib_pool)}")
    print(f"mechanism: {len(mech_events)} events, {sum(len(v) for v in mech_draw.values())} prompts")
    print(f"sentinel: {len(sentinel_events)} events, {sum(len(v) for v in sentinel_draw.values())} prompts")
    print(f"holdout: {len(holdout)}  screen: {len(screen)}  (all pairwise-disjoint OK)")

    if args.check:
        ok = True
        for name, obj in outputs:
            tmp = json.dumps(obj, sort_keys=True, indent=2) + "\n"
            disk = open(os.path.join(OUT_DIR, name)).read()
            same = hashlib.sha256(tmp.encode()).hexdigest() == hashlib.sha256(disk.encode()).hexdigest()
            print(f"  {'OK ' if same else 'DIFF'} {name}")
            ok = ok and same
        print("DETERMINISM:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    print("\n==== SHA256 (record in the ledger) ====")
    manifest_index = {}
    for name, obj in outputs:
        path = dump(obj, name)
        h = sha256_file(path)
        manifest_index[name] = h
        print(f"  {h}  {name}")
    manifest_index["audioset_ontology.json"] = sha256_file(ONTOLOGY)
    print(f"  {manifest_index['audioset_ontology.json']}  audioset_ontology.json")
    dump({"generated_by": "scripts/research/build_v4_manifests.py", "master_seed": MASTER_SEED,
          "sha256": manifest_index}, "v4_manifests_index.json")
    print(f"  {sha256_file(os.path.join(OUT_DIR, 'v4_manifests_index.json'))}  v4_manifests_index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
