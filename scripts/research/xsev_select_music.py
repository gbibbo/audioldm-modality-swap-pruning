#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — independent 64-prompt music battery (CPU, 0 cr).

New hip-hop/rap MusicCaps battery, DISJOINT from the frozen severity-1 music-64 and from the Kim-193
training source, using the IDENTICAL eligibility rules as the gate-0 battery (keyword filter + Kim
source-ytid exclusion + caption exact/near-dup(0.85) + self-dedup) — only the selection salt changes
and the frozen 64 are additionally excluded. 3 generation replicates. NO rule loosening.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_select_music.py [--check]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import build_gate0_battery as B                      # reuse norm/toks/jaccard/KEYWORD_RE/paths
from research_pruning.eval.reversal import derive_paired_seed

MUSIC_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|MUSIC|2026-08-30"
GENERATION_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30"
N = 64
N_SEEDS = 3
FROZEN_BATTERY = "configs/research/icassp_gate0_battery.json"
OUT = "configs/research/xsev_music_manifest.json"


def order_key(ytid):
    return hashlib.sha256((MUSIC_SALT + "|" + ytid).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build():
    kim = json.load(open(B.KIM_CAPTIONS)); kim_caps = kim["unique_captions"]
    kim_norm = {B.norm(c) for c in kim_caps}; kim_tok = [B.toks(c) for c in kim_caps]
    source_ytids = set(json.load(open(B.KIM_SOURCE_YTIDS))["source_ytids"])
    frozen64 = {p["ytid"] for p in json.load(open(FROZEN_BATTERY))["prompts"]}

    rows = list(csv.DictReader(open(B.MUSICCAPS_CSV)))
    filtered = [(r["ytid"], r["caption"]) for r in rows if B.KEYWORD_RE.search(r["caption"] or "")]
    kept = []
    d = {"dropped_source_ytid": 0, "dropped_exact_dup": 0, "dropped_neardup": 0, "dropped_frozen64": 0}
    for ytid, cap in filtered:
        if ytid in source_ytids: d["dropped_source_ytid"] += 1; continue
        if ytid in frozen64: d["dropped_frozen64"] += 1; continue       # NEW: exclude frozen severity-1 64
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
    if len(deduped) < N:
        raise SystemExit(f"STOP: only {len(deduped)} eligible < {N} (no rule loosening allowed)")
    selected = deduped[:N]
    sel = {y for y, _ in selected}
    assert not (sel & source_ytids) and not (sel & frozen64), "leakage"

    prompts = [{"prompt_index": i, "ytid": y, "caption": c, "order_key": order_key(y),
                "generation_seeds": [derive_paired_seed(GENERATION_SALT, y, r) for r in range(N_SEEDS)]}
               for i, (y, c) in enumerate(selected)]
    payload = {
        "artifact": "xsev_music_manifest", "experiment": "RECOVERY-CROSS-SEVERITY-REP-1",
        "music_salt": MUSIC_SALT, "generation_salt": GENERATION_SALT, "n_prompts": N, "n_seeds": N_SEEDS,
        "eligibility": "IDENTICAL to gate0 battery (keyword+kim-source+exact/neardup0.85+selfdedup); "
                       "additionally exclude frozen severity-1 64; new salt; no rule loosening",
        "diagnostics": {**d, "keyword_filtered": len(filtered), "candidates_after": len(deduped)},
        "provenance": {"musiccaps_csv_sha256": sha_file(B.MUSICCAPS_CSV),
                       "frozen_battery_sha256": sha_file(FROZEN_BATTERY),
                       "kim_source_sha256": sha_file(B.KIM_SOURCE_YTIDS)},
        "prompts": prompts,
    }
    yts = [p["ytid"] for p in prompts]
    payload["prompts_sha256"] = hashlib.sha256(json.dumps(yts, ensure_ascii=False).encode()).hexdigest()
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "manifest_sha256"},
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return payload


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    p = build()
    if a.check:
        old = json.load(open(OUT)); same = old.get("manifest_sha256") == p["manifest_sha256"]
        print("DETERMINISM", "OK" if same else "MISMATCH", "| sha", p["manifest_sha256"][:16]); return 0 if same else 1
    json.dump(p, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {OUT} | n={p['n_prompts']} | candidates {p['diagnostics']['candidates_after']} | "
          f"prompts_sha256 {p['prompts_sha256'][:16]} | manifest_sha256 {p['manifest_sha256'][:16]}")
    print("diagnostics:", p["diagnostics"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
