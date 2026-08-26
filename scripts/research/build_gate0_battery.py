#!/usr/bin/env python3
"""Freeze the ICASSP Gate-0 held-out evaluation battery: EXACTLY 64 MusicCaps prompts,
hip-hop/rap-filtered, DISJOINT from the Kim-193 training captions.

Pre-registration artifact (DECISION-V4-09 §battery; frozen BEFORE any GPU). Deterministic:
same inputs -> byte-identical manifest. `--check` verifies the committed manifest reproduces.

Frozen rule (do not change after freeze):
  1. Source = MusicCaps public CSV (google/MusicCaps, CC-BY-SA-4.0), column `caption`, id `ytid`.
  2. hip-hop/rap filter: caption matches (case-insensitive, word-boundaried) FROZEN_KEYWORDS.
  3. Disjoint from Kim-193 unique training captions by BOTH:
       - exact normalized match  (norm = lowercase, strip non-word, collapse whitespace);
       - normalized token Jaccard >= NEAR_DUP_JACCARD  (near-duplicate).
     Any candidate matching a Kim caption on either test is DROPPED.
  4. Deterministic selection of exactly 64: order candidates by sha256(SEED_SALT + ytid),
     take the first 64. No RNG state, no hand curation.
Outputs configs/research/icassp_gate0_battery.json (tracked) + prints its sha256.
"""
import argparse, csv, hashlib, json, re, sys

MUSICCAPS_CSV = "artifacts/icassp_gate0/musiccaps-public.csv"          # gitignored source
KIM_CAPTIONS = "configs/research/kim193_captions_unique.json"          # tracked provenance
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

    rows = list(csv.DictReader(open(MUSICCAPS_CSV)))
    # frozen filter
    filtered = [(r["ytid"], r["caption"]) for r in rows if KEYWORD_RE.search(r["caption"] or "")]

    # disjointness vs Kim
    exact_hits, neardup_hits, kept = [], [], []
    for ytid, cap in filtered:
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
    diag = {
        "musiccaps_rows": len(rows),
        "keyword_filtered": len(filtered),
        "dropped_exact_dup_vs_kim": len(exact_hits),
        "dropped_neardup_vs_kim": len(neardup_hits),
        "candidates_after_disjoint_and_selfdedup": len(deduped),
        "selected": len(selected),
        "kim_unique_captions": len(kim_caps),
    }
    manifest = {
        "name": "icassp_gate0_held_out_battery",
        "decision": "DECISION-V4-09 (battery) + DECISION-V4-10 (clip length)",
        "n_prompts": len(selected),
        "n_seeds": 3,
        "source": "google/MusicCaps public CSV (CC-BY-SA-4.0)",
        "musiccaps_csv_sha256": hashlib.sha256(open(MUSICCAPS_CSV, "rb").read()).hexdigest(),
        "kim_captions_ref": {"file": KIM_CAPTIONS, "sha256": kim["captions_sha256"],
                             "n_unique": len(kim_caps)},
        "rule": {
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
