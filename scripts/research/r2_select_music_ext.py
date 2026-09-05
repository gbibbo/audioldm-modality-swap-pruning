#!/usr/bin/env python3
"""REVIEWER2-FOLLOWUP E7 — extension of the severity-2 hip-hop battery (CPU, 0 cr).

Same eligibility rules as the frozen batteries (gate-0 keyword filter + Kim-source exclusion +
caption exact/near-dup(0.85) vs Kim + self-dedup), additionally excluding BOTH frozen 64-prompt
batteries (severity-1 gate-0 battery and the severity-2 xsev battery). Takes EVERY remaining eligible
prompt (no cherry-picking; the pool is exhausted), ordered by a new salt. One generation replicate.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/r2_select_music_ext.py [--check]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import build_gate0_battery as B
from research_pruning.eval.reversal import derive_paired_seed

MUSIC_SALT = "REVIEWER2-FOLLOWUP|MUSIC-EXT|2026-09-05"
GENERATION_SALT = "REVIEWER2-FOLLOWUP|GENERATION|2026-09-05"
FROZEN_SEV1 = "configs/research/icassp_gate0_battery.json"
FROZEN_SEV2 = "configs/research/xsev_music_manifest.json"
OUT = "configs/research/r2_music_ext_manifest.json"


def order_key(ytid):
    return hashlib.sha256((MUSIC_SALT + "|" + ytid).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build():
    kim = json.load(open(B.KIM_CAPTIONS)); kim_caps = kim["unique_captions"]
    kim_norm = {B.norm(c) for c in kim_caps}; kim_tok = [B.toks(c) for c in kim_caps]
    source_ytids = set(json.load(open(B.KIM_SOURCE_YTIDS))["source_ytids"])
    frozen1 = {p["ytid"] for p in json.load(open(FROZEN_SEV1))["prompts"]}
    frozen2 = {p["ytid"] for p in json.load(open(FROZEN_SEV2))["prompts"]}
    rows = list(csv.DictReader(open(B.MUSICCAPS_CSV)))
    filtered = [(r["ytid"], r["caption"]) for r in rows if B.KEYWORD_RE.search(r["caption"] or "")]
    kept = []
    d = {"dropped_source_ytid": 0, "dropped_exact_dup": 0, "dropped_neardup": 0, "dropped_frozen_sev1_64": 0, "dropped_frozen_sev2_64": 0}
    for ytid, cap in filtered:
        if ytid in source_ytids: d["dropped_source_ytid"] += 1; continue
        if ytid in frozen1: d["dropped_frozen_sev1_64"] += 1; continue
        if ytid in frozen2: d["dropped_frozen_sev2_64"] += 1; continue
        n = B.norm(cap)
        if n in kim_norm: d["dropped_exact_dup"] += 1; continue
        ct = B.toks(cap)
        if any(B.jaccard(ct, kt) >= B.NEAR_DUP_JACCARD for kt in kim_tok): d["dropped_neardup"] += 1; continue
        kept.append((ytid, cap))
    kept.sort(key=lambda x: order_key(x[0]))
    seen, deduped = set(), []
    for ytid, cap in kept:
        n = B.norm(cap)
        if n in seen: continue
        seen.add(n); deduped.append((ytid, cap))
    selected = deduped                                # ALL remaining eligible prompts
    sel = {y for y, _ in selected}
    assert not (sel & source_ytids) and not (sel & frozen1) and not (sel & frozen2), "leakage"
    prompts = [{"prompt_index": i, "ytid": y, "caption": c, "order_key": order_key(y),
                "generation_seeds": [derive_paired_seed(GENERATION_SALT, y, 0)]}
               for i, (y, c) in enumerate(selected)]
    payload = {"artifact": "r2_music_ext_manifest", "experiment": "REVIEWER2-FOLLOWUP (E7)",
               "music_salt": MUSIC_SALT, "generation_salt": GENERATION_SALT, "n_prompts": len(prompts), "n_seeds": 1,
               "eligibility": "IDENTICAL to the frozen batteries (keyword+kim-source+exact/neardup0.85+selfdedup); "
                              "additionally exclude the frozen severity-1 64 and severity-2 64; ALL remaining eligible prompts; new salt; no rule loosening",
               "diagnostics": {**d, "keyword_filtered": len(filtered), "candidates_after": len(deduped)},
               "provenance": {"musiccaps_csv_sha256": sha_file(B.MUSICCAPS_CSV), "frozen_sev1_sha256": sha_file(FROZEN_SEV1),
                              "frozen_sev2_sha256": sha_file(FROZEN_SEV2), "kim_source_sha256": sha_file(B.KIM_SOURCE_YTIDS)},
               "prompts": prompts}
    yts = [p["ytid"] for p in prompts]
    payload["prompts_sha256"] = hashlib.sha256(json.dumps(yts, ensure_ascii=False).encode()).hexdigest()
    payload["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in payload.items() if k != "manifest_sha256"},
                                                           ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return payload


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    p = build()
    if a.check:
        old = json.load(open(OUT)); same = old.get("manifest_sha256") == p["manifest_sha256"]
        print("DETERMINISM", "OK" if same else "MISMATCH", "| sha", p["manifest_sha256"][:16]); return 0 if same else 1
    json.dump(p, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {OUT} | n={p['n_prompts']} | diagnostics {p['diagnostics']} | manifest_sha256 {p['manifest_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
