#!/usr/bin/env python3
"""REVIEWER2-FOLLOWUP E5 — Clotho evaluation-split battery (CPU, 0 cr).

96 clips of the Clotho v2.1 evaluation split (Zenodo 4783391, `clotho_captions_evaluation.csv`) chosen by
seeded hash of the file name (NO content filtering; outcome-blind), one of the five captions per clip
chosen by a second seeded hash. Clotho audio is Freesound, disjoint from AudioCaps (YouTube) by source.
One generation replicate; CRN seed = derive_paired_seed(GENERATION_SALT, file_name, 0).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/r2_select_clotho.py [--check]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import derive_paired_seed

CSV = "artifacts/clotho/clotho_captions_evaluation.csv"
CSV_SHA256_EXPECTED = "0e116233909a57449572b2d67d4c7a2f7df5b7a88c918f3697e61f684e664e84"
SELECT_SALT = "REVIEWER2-FOLLOWUP|CLOTHO-SELECT|2026-09-05"
CAPTION_SALT = "REVIEWER2-FOLLOWUP|CLOTHO-CAPTION|2026-09-05"
GENERATION_SALT = "REVIEWER2-FOLLOWUP|GENERATION|2026-09-05"
N = 96
OUT = "configs/research/r2_clotho_manifest.json"


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build():
    sha = hashlib.sha256(open(CSV, "rb").read()).hexdigest()
    if sha != CSV_SHA256_EXPECTED:
        raise SystemExit(f"STOP: Clotho captions CSV sha256 {sha} != expected {CSV_SHA256_EXPECTED}")
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rows.sort(key=lambda r: h(SELECT_SALT + "|" + r["file_name"]))
    sel = rows[:N]
    prompts = []
    for i, r in enumerate(sel):
        ci = 1 + int(h(CAPTION_SALT + "|" + r["file_name"]), 16) % 5
        cap = r[f"caption_{ci}"].strip()
        prompts.append({"prompt_index": i, "clip_id": r["file_name"], "caption": cap, "chosen_caption_index": ci,
                        "selection_key": h(SELECT_SALT + "|" + r["file_name"]),
                        "generation_seeds": [derive_paired_seed(GENERATION_SALT, r["file_name"], 0)]})
    payload = {"artifact": "r2_clotho_manifest", "experiment": "REVIEWER2-FOLLOWUP (E5)", "source": "Clotho v2.1 evaluation split, Zenodo 4783391",
               "csv": CSV, "csv_sha256": sha, "universe": len(rows), "n_prompts": N, "n_seeds": 1,
               "select_salt": SELECT_SALT, "caption_salt": CAPTION_SALT, "generation_salt": GENERATION_SALT,
               "rule": "sort file_name by sha256(select_salt|file_name), first N; caption index 1 + sha256(caption_salt|file_name) mod 5; no filtering",
               "caption_words": {"median": sorted(len(p["caption"].split()) for p in prompts)[N // 2]},
               "prompts": prompts}
    payload["prompts_sha256"] = h(json.dumps([p["clip_id"] for p in prompts], ensure_ascii=False))
    payload["manifest_sha256"] = h(json.dumps({k: v for k, v in payload.items() if k != "manifest_sha256"}, ensure_ascii=False, sort_keys=True))
    return payload


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    p = build()
    if a.check:
        old = json.load(open(OUT)); same = old.get("manifest_sha256") == p["manifest_sha256"]
        print("DETERMINISM", "OK" if same else "MISMATCH", "| sha", p["manifest_sha256"][:16]); return 0 if same else 1
    json.dump(p, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {OUT} | n={p['n_prompts']} of {p['universe']} | caption words median {p['caption_words']['median']} | manifest_sha256 {p['manifest_sha256'][:16]}")
    print("first:", p["prompts"][0]["clip_id"], "|", p["prompts"][0]["caption"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
