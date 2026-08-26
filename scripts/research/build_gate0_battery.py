#!/usr/bin/env python3
"""Freeze the ICASSP Gate-0 held-out evaluation battery: EXACTLY 64 MusicCaps prompts,
hip-hop/rap-filtered, DISJOINT from the Kim-193 training captions.

Pre-registration artifact (DECISION-V4-09 §battery; frozen BEFORE any GPU). Deterministic:
same inputs -> byte-identical manifest. `--check` verifies the committed manifest reproduces.

Frozen rule (v2, do not change after freeze):
  1. Source = MusicCaps public CSV (google/MusicCaps, CC-BY-SA-4.0), column `caption`, id `ytid`.
  2. hip-hop/rap filter: caption matches (case-insensitive, word-boundaried) FROZEN_KEYWORDS.
  3. Disjoint from Kim-193 by SOURCE YTID (PRIMARY): drop any candidate whose ytid is in the
     recovered Kim training-source ytid set (scripts/research/gate0_leakage_audit.py ->
     configs/research/kim193_source_ytids.json). Kim's captions are MusicCaps captions + a
     deterministic subgenre suffix, so the training source is recoverable to exact ytids.
  4. Caption disjointness (SECONDARY safeguard) vs Kim-193 unique captions by BOTH:
       - exact normalized match  (norm = lowercase, strip non-word, collapse whitespace);
       - normalized token Jaccard >= NEAR_DUP_JACCARD  (near-duplicate).
     Any candidate matching on ytid OR caption is DROPPED.
  5. Deterministic selection of exactly 64: order candidates by sha256(SEED_SALT + ytid),
     take the first 64. No RNG state, no hand curation.
  v2 leakage-fix (2026-08-26): v1 (sha 46ee4203…) relied on caption near-dup only and leaked ONE
  training-source ytid (ZnBvXFDWpWo) into the battery. v1 preserved at ..._v1_superseded.json.
Outputs configs/research/icassp_gate0_battery.json (tracked) + prints its sha256.
"""
import argparse, csv, hashlib, json, re, sys

MUSICCAPS_CSV = "artifacts/icassp_gate0/musiccaps-public.csv"          # gitignored source
KIM_CAPTIONS = "configs/research/kim193_captions_unique.json"          # tracked provenance
KIM_SOURCE_YTIDS = "configs/research/kim193_source_ytids.json"         # recovered training-source ytids
OUT = "configs/research/icassp_gate0_battery.json"

N_PROMPTS = 64
SEED_SALT = "icassp-gate0-battery-20260826"
NEAR_DUP_JACCARD = 0.85
FROZEN_KEYWORDS = [r"hip[\s-]?hop", r"rap", r"rapping", r"rapper", r"rappers",
                   r"trap beat", r"boom bap", r"gangsta"]
KEYWORD_RE = re.compile(r"\b(" + "|".join(FROZEN_KEYWORDS) + r")\b", re.IGNORECASE)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def toks(s):
    return set(norm(s).split())


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def order_key(ytid):
    return hashlib.sha256((SEED_SALT + "|" + ytid).encode()).hexdigest()


def build():
    kim = json.load(open(KIM_CAPTIONS))
    kim_caps = kim["unique_captions"]
    kim_norm = {norm(c) for c in kim_caps}
    kim_tok = [toks(c) for c in kim_caps]
    source_ytids = set(json.load(open(KIM_SOURCE_YTIDS))["source_ytids"])

    rows = list(csv.DictReader(open(MUSICCAPS_CSV)))
    # frozen filter
    filtered = [(r["ytid"], r["caption"]) for r in rows if KEYWORD_RE.search(r["caption"] or "")]

    # disjointness vs Kim: PRIMARY = source ytid; SECONDARY = caption exact/near-dup
    ytid_hits, exact_hits, neardup_hits, kept = [], [], [], []
    for ytid, cap in filtered:
        if ytid in source_ytids:                       # PRIMARY exclusion
            ytid_hits.append(ytid); continue
        n = norm(cap)
        if n in kim_norm:
            exact_hits.append(ytid); continue
        ct = toks(cap)
        if any(jaccard(ct, kt) >= NEAR_DUP_JACCARD for kt in kim_tok):
            neardup_hits.append(ytid); continue
        kept.append((ytid, cap))

    # dedup candidates among themselves by normalized caption (keep first by order_key)
    kept.sort(key=lambda x: order_key(x[0]))
    seen, deduped = set(), []
    for ytid, cap in kept:
        n = norm(cap)
        if n in seen:
            continue
        seen.add(n); deduped.append((ytid, cap))

    selected = deduped[:N_PROMPTS]
    sel_ytids = {y for y, _ in selected}
    disjoint_by_ytid = len(sel_ytids & source_ytids) == 0     # PRIMARY guarantee
    diag = {
        "musiccaps_rows": len(rows),
        "keyword_filtered": len(filtered),
        "n_recovered_source_ytids": len(source_ytids),
        "dropped_by_source_ytid": len(ytid_hits),
        "dropped_exact_dup_vs_kim": len(exact_hits),
        "dropped_neardup_vs_kim": len(neardup_hits),
        "candidates_after_disjoint_and_selfdedup": len(deduped),
        "selected": len(selected),
        "kim_unique_captions": len(kim_caps),
        "selected_disjoint_from_source_ytids": disjoint_by_ytid,
    }
    manifest = {
        "name": "icassp_gate0_held_out_battery",
        "version": 2,
        "decision": "DECISION-V4-09 (battery) + DECISION-V4-10 (clip length)",
        "changelog": "v2 (2026-08-26 leakage fix): exclude by RECOVERED SOURCE YTID (primary) + "
                     "caption exact/near-dup (secondary). v1 (sha 46ee4203) leaked 1 source ytid "
                     "(ZnBvXFDWpWo) via caption-only near-dup; preserved at ..._v1_superseded.json.",
        "n_prompts": len(selected),
        "n_seeds": 3,
        "source": "google/MusicCaps public CSV (CC-BY-SA-4.0)",
        "musiccaps_csv_sha256": hashlib.sha256(open(MUSICCAPS_CSV, "rb").read()).hexdigest(),
        "kim_captions_ref": {"file": KIM_CAPTIONS, "sha256": kim["captions_sha256"],
                             "n_unique": len(kim_caps)},
        "kim_source_ytids_ref": {"file": KIM_SOURCE_YTIDS, "n": len(source_ytids)},
        "rule": {
            "primary_exclusion": "MusicCaps ytid in recovered Kim training-source ytid set",
            "keywords": FROZEN_KEYWORDS, "near_dup_jaccard": NEAR_DUP_JACCARD,
            "seed_salt": SEED_SALT, "order": "sha256(seed_salt|ytid) ascending, first N",
            "normalization": "lowercase, strip non-word chars, collapse whitespace",
        },
        "diagnostics": diag,
        "prompts": [{"ytid": y, "caption": c} for y, c in selected],
    }
    return manifest, diag


def canonical(manifest):
    # sha over prompts only (order + content), stable across cosmetic header changes
    return hashlib.sha256(
        json.dumps(manifest["prompts"], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify committed manifest reproduces")
    args = ap.parse_args()

    manifest, diag = build()
    manifest["prompts_sha256"] = canonical(manifest)

    if len(manifest["prompts"]) != N_PROMPTS:
        print(f"FAIL: got {len(manifest['prompts'])} prompts, need exactly {N_PROMPTS}", file=sys.stderr)
        print("diagnostics:", json.dumps(diag, indent=2))
        return 2
    if not diag["selected_disjoint_from_source_ytids"]:
        print("FAIL: selected battery ytids are NOT disjoint from recovered Kim source ytids", file=sys.stderr)
        print("diagnostics:", json.dumps(diag, indent=2))
        return 3

    if args.check:
        old = json.load(open(OUT))
        same = old.get("prompts") == manifest["prompts"]
        print("CHECK:", "REPRODUCES (byte-identical prompts)" if same else "MISMATCH")
        print("committed prompts_sha256:", old.get("prompts_sha256"))
        print("recomputed prompts_sha256:", manifest["prompts_sha256"])
        return 0 if same else 1

    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print("diagnostics:", json.dumps(diag, indent=2))
    print("prompts_sha256:", manifest["prompts_sha256"])
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
