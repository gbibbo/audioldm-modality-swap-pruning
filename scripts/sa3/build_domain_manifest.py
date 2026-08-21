#!/usr/bin/env python3
"""Freeze a per-domain CC0 data manifest for the RQ2 real-LoRA validation (protocol §4, §5.2).

Step-0 tooling (CPU, 0 cr). Scans a directory of `<id>.wav` + `<id>.txt` (caption) [+ optional
`<id>.meta.json` = {source_url, freesound_id, license, uploader}] and freezes:

  * per-clip record: sha256, duration_s, sample_rate, caption, CC0 license proof;
  * a DETERMINISTIC seeded 80/20 train_L / eval_L split (min 5 eval clips, §5.2 constants);
  * a held-out `prompts_L` list (captions NOT used in training — disjointness enforced);
  * a manifest sha256 over the sorted (id, sha256) pairs, for the ledger.

Hard rules enforced (raise, do not warn):
  * every clip is CC0 (license == "CC0"); non-CC0 or unknown-license clips are rejected;
  * target sample rate 44100 Hz;
  * eval_L ∩ train_L = ∅; prompts_L captions ∩ train_L captions = ∅  (no eval/prompt leakage);
  * at least MIN_EVAL held-out clips and at least N_MIN_CLIP total.

The manifest freezes WHICH clips/splits/prompts; audio and generated wavs are gitignored — only the
manifest JSON (ids + hashes + splits + prompts) is tracked. This tool sources NOTHING from the
internet; it operates on a directory you have already curated (or on --selftest synthetic data).

Usage:
  .venv-sa3/bin/python scripts/sa3/build_domain_manifest.py --dir data/sa3/adapters/impact_percussion \
       --domain impact_percussion --seed 20260821 --prompts-file prompts_impact.txt \
       --out configs/sa3/adapters/impact_percussion.manifest.json
  .venv-sa3/bin/python scripts/sa3/build_domain_manifest.py --selftest
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, random, sys

MIN_EVAL = 5          # §5.2 pre-registered constant
EVAL_FRAC = 0.20      # §5.2 pre-registered constant
N_MIN_CLIP = 20       # §4 pre-registered minimum per domain
TARGET_SR = 44100     # §4


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _audio_info(path: str):
    import soundfile as sf
    info = sf.info(path)
    return int(info.samplerate), float(info.frames) / float(info.samplerate)


def build_clip_records(dir_: str) -> list:
    """One record per `<id>.wav` with a matching `<id>.txt`. Enforces CC0 + 44.1 kHz."""
    records = []
    for wav in sorted(glob.glob(os.path.join(dir_, "*.wav"))):
        cid = os.path.splitext(os.path.basename(wav))[0]
        txt = os.path.join(dir_, f"{cid}.txt")
        if not os.path.exists(txt):
            raise FileNotFoundError(f"missing caption for {cid}: {txt}")
        caption = open(txt).read().strip()
        if not caption:
            raise ValueError(f"empty caption for {cid}")
        meta_path = os.path.join(dir_, f"{cid}.meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        license_ = str(meta.get("license", "UNKNOWN")).upper().replace("-", "").replace(" ", "")
        if license_ not in ("CC0", "CC01_0", "CC010"):
            raise ValueError(f"clip {cid} license is {meta.get('license')!r}, not CC0 -- rejected (§4)")
        sr, dur = _audio_info(wav)
        if sr != TARGET_SR:
            raise ValueError(f"clip {cid} sample rate {sr} != {TARGET_SR} Hz (§4)")
        records.append({
            "id": cid,
            "sha256": sha256_file(wav),
            "duration_s": round(dur, 4),
            "sample_rate": sr,
            "caption": caption,
            "license": "CC0",
            "source_url": meta.get("source_url"),
            "freesound_id": meta.get("freesound_id"),
        })
    return records


def deterministic_split(ids: list, seed: int, eval_frac: float = EVAL_FRAC, min_eval: int = MIN_EVAL):
    """Seeded shuffle then split. eval_L = max(min_eval, round(eval_frac*n)) held-out clips."""
    n = len(ids)
    n_eval = max(min_eval, round(eval_frac * n))
    if n_eval >= n:
        raise ValueError(f"n={n} too small for min_eval={min_eval} + a non-empty train split")
    order = list(ids)
    random.Random(seed).shuffle(order)
    eval_ids = sorted(order[:n_eval])
    train_ids = sorted(order[n_eval:])
    return train_ids, eval_ids


def build_manifest(domain: str, records: list, prompts_L: list, seed: int,
                   source: str = "freesound-cc0") -> dict:
    if len(records) < N_MIN_CLIP:
        raise ValueError(f"{len(records)} clips < N_MIN_CLIP={N_MIN_CLIP} for domain {domain} (§4)")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate clip ids")
    train_ids, eval_ids = deterministic_split(ids, seed)
    # leakage guards
    if set(train_ids) & set(eval_ids):
        raise AssertionError("train_L ∩ eval_L != ∅")
    by_id = {r["id"]: r for r in records}
    train_caps = {by_id[i]["caption"] for i in train_ids}
    leak = [p for p in prompts_L if p.strip() in train_caps]
    if leak:
        raise AssertionError(f"prompts_L leak into train captions: {leak[:3]}")
    if not prompts_L:
        raise ValueError("prompts_L must be non-empty (held-out domain prompts, §5.2)")
    manifest_sha = hashlib.sha256(
        json.dumps([(r["id"], r["sha256"]) for r in sorted(records, key=lambda r: r["id"])],
                   sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema": "sa3-rq2-domain-manifest/1",
        "domain": domain,
        "source": source,
        "sample_rate_target": TARGET_SR,
        "split_seed": seed,
        "constants": {"eval_frac": EVAL_FRAC, "min_eval": MIN_EVAL, "n_min_clip": N_MIN_CLIP},
        "n_clips": len(records),
        "clips": sorted(records, key=lambda r: r["id"]),
        "split": {"train_L": train_ids, "eval_L": eval_ids},
        "prompts_L": [p.strip() for p in prompts_L],
        "manifest_sha256": manifest_sha,
        "leakage_checks": {"train_eval_disjoint": True, "prompts_not_in_train": True,
                           "eval_never_selects_checkpoints": "enforced by convention (§5.2)"},
    }


def selftest() -> int:
    import numpy as np, soundfile as sf
    sc = os.environ.get("SCRATCH", "/tmp")
    d = os.path.join(sc, "sa3_domain_manifest_selftest"); os.makedirs(d, exist_ok=True)
    # 22 synthetic 44.1kHz clips + captions + CC0 meta
    rng = np.random.default_rng(0)
    n = 22
    for i in range(n):
        cid = f"clip{i:04d}"
        t = np.linspace(0, 1.0, TARGET_SR, endpoint=False)
        x = (0.3 * np.sin(2 * np.pi * (200 + 10 * i) * t)).astype("float32")
        sf.write(os.path.join(d, f"{cid}.wav"), x, TARGET_SR)
        open(os.path.join(d, f"{cid}.txt"), "w").write(f"a training impact sound number {i}")
        json.dump({"license": "CC0", "source_url": f"https://freesound.org/s/{1000+i}/",
                   "freesound_id": 1000 + i}, open(os.path.join(d, f"{cid}.meta.json"), "w"))
    recs = build_clip_records(d)
    prompts = ["a held-out metallic impact", "a wooden clap", "a distant hit"]
    man = build_manifest("impact_percussion_selftest", recs, prompts, seed=20260821)
    # assertions
    ok = man["n_clips"] == n
    ok = ok and len(man["split"]["eval_L"]) >= MIN_EVAL
    ok = ok and set(man["split"]["train_L"]) & set(man["split"]["eval_L"]) == set()
    ok = ok and len(man["split"]["train_L"]) + len(man["split"]["eval_L"]) == n
    # determinism: rebuilding gives the identical manifest sha + identical split
    man2 = build_manifest("impact_percussion_selftest", build_clip_records(d), prompts, seed=20260821)
    ok = ok and man2["manifest_sha256"] == man["manifest_sha256"]
    ok = ok and man2["split"] == man["split"]
    # a different seed changes the split but not the clip-hash manifest sha
    man3 = build_manifest("impact_percussion_selftest", recs, prompts, seed=1)
    ok = ok and man3["manifest_sha256"] == man["manifest_sha256"] and man3["split"] != man["split"]
    # leakage guard fires: feed a caption that is genuinely in train_L
    by_id = {r["id"]: r for r in recs}
    train_caption = by_id[man["split"]["train_L"][0]]["caption"]
    leaked = False
    try:
        build_manifest("x", recs, [train_caption], seed=20260821)
    except AssertionError:
        leaked = True
    ok = ok and leaked
    # non-CC0 rejection fires
    rejected = False
    json.dump({"license": "CC-BY"}, open(os.path.join(d, "clip0000.meta.json"), "w"))
    try:
        build_clip_records(d)
    except ValueError:
        rejected = True
    ok = ok and rejected
    print(json.dumps({"n_clips": man["n_clips"], "train": len(man["split"]["train_L"]),
                      "eval": len(man["split"]["eval_L"]), "sha": man["manifest_sha256"][:12],
                      "determinism": man2["manifest_sha256"] == man["manifest_sha256"],
                      "leakage_guard": leaked, "cc0_guard": rejected}, indent=2))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--domain"); ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--prompts-file", help="one held-out prompt per line")
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.dir and a.domain and a.out and a.prompts_file):
        ap.error("--dir --domain --out --prompts-file are required (or use --selftest)")
    recs = build_clip_records(a.dir)
    prompts = [ln.strip() for ln in open(a.prompts_file) if ln.strip()]
    man = build_manifest(a.domain, recs, prompts, seed=a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(man, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}: {man['n_clips']} clips, train={len(man['split']['train_L'])} "
          f"eval={len(man['split']['eval_L'])} prompts_L={len(man['prompts_L'])} sha={man['manifest_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
