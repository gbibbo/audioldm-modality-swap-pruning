#!/usr/bin/env python3
"""Freeze a per-domain CC0 data manifest for the RQ2 real-LoRA validation (protocol §4, §5.2;
selection rule frozen in `docs/sa3/freesound_selection_spec.md`, rc1.1 patch 5).

Step-0 tooling (CPU, 0 cr). Scans a directory of `<id>.wav` + REQUIRED `<id>.meta.json` (Freesound
metadata: {license, name, tags, permalink/source_url, id/freesound_id}) and freezes:

  * per-clip record: 44.1k-mono sha256, file sha256, native duration/SR/channels, `resampled` flag,
    a DETERMINISTICALLY-DERIVED caption (spec §4 — never hand-written), CC0 license proof;
  * a DETERMINISTIC seeded 80/20 train_L / eval_L split (min 5 eval clips, §5.2 constants);
  * a held-out `prompts_L` list (captions NOT used in training — disjointness enforced);
  * a manifest sha256 over the sorted (id, audio_sha) pairs, for the ledger.

Hard rules enforced (raise or deterministically skip, never manual judgement):
  * every clip is CC0 ("Creative Commons 0"); non-CC0 rejected;
  * ANY native SR accepted and deterministically resampled to 44100 Hz (spec §3) — native 44.1 is
    NOT required (that would bias/shrink the pool); original SR/channels recorded;
  * duration ∈ [DUR_MIN, DUR_MAX], non-silent (peak/RMS), de-duplicated by 44.1k-mono sha (spec §3);
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
import argparse, glob, hashlib, json, os, random, re, sys

MIN_EVAL = 5          # §5.2 pre-registered constant
EVAL_FRAC = 0.20      # §5.2 pre-registered constant
N_MIN_CLIP = 20       # §4 pre-registered minimum per domain
TARGET_SR = 44100     # resample TARGET (not a native-SR filter; freesound_selection_spec §3)
DUR_MIN = 0.3         # spec §3 (s)
DUR_MAX = 12.0        # spec §3 (s)
PEAK_MIN = 1e-4       # spec §3 silence guard
RMS_MIN = 1e-3        # spec §3 silence guard

_DOMAIN_NOUN = {"impact_percussion": "impact", "water_liquid": "water", "mechanical": "machine",
                "animal": "animal", "ambience": "ambience"}


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _audio_load_44k_mono(path: str):
    """Deterministic load: native probe + resample-to-44.1-kHz-mono (spec §3). Returns
    (wav44_mono float32, native_sr, native_channels, native_duration_s)."""
    import numpy as np, soundfile as sf, librosa
    info = sf.info(path)
    native_sr = int(info.samplerate)
    native_ch = int(info.channels)
    native_dur = float(info.frames) / float(native_sr) if native_sr else 0.0
    w44, _ = librosa.load(path, sr=TARGET_SR, mono=True)   # deterministic resample + downmix
    return w44.astype("float32"), native_sr, native_ch, native_dur


def derive_caption(meta: dict, domain_noun: str) -> str:
    """FROZEN deterministic caption (spec §4) — never hand-written. From Freesound name + domain tags."""
    name = str(meta.get("name", ""))
    name = os.path.splitext(name)[0].replace("_", " ").replace("-", " ").lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = " ".join(name.split()[:12]).strip()
    tags = [t for t in (meta.get("domain_tags") or meta.get("tags") or [])][:3]
    base = f"a sound of {name}" if name else f"a sound of {domain_noun}"
    return base + (" (" + ", ".join(tags) + ")" if tags else "")


def build_clip_records(dir_: str, domain: str = "") -> list:
    """One record per `<id>.wav` with a required `<id>.meta.json` (Freesound metadata). CC0-only;
    resample-tolerant (any native SR); duration/silence/channel filters (spec §3); caption is
    DERIVED deterministically from metadata (spec §4), never read from a hand-written file."""
    import numpy as np
    domain_noun = _DOMAIN_NOUN.get(domain, domain or "sound")
    records, seen_audio_sha = [], set()
    for wav in sorted(glob.glob(os.path.join(dir_, "*.wav"))):
        cid = os.path.splitext(os.path.basename(wav))[0]
        meta_path = os.path.join(dir_, f"{cid}.meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"missing metadata for {cid}: {meta_path} (captions derive from it)")
        meta = json.load(open(meta_path))
        license_ = str(meta.get("license", "UNKNOWN")).upper().replace("-", "").replace(" ", "")
        if license_ not in ("CC0", "CC01_0", "CC010", "CREATIVECOMMONS0"):
            raise ValueError(f"clip {cid} license is {meta.get('license')!r}, not CC0 -- rejected (spec §1)")
        w44, native_sr, native_ch, native_dur = _audio_load_44k_mono(wav)
        # spec §3 filters (deterministic)
        if not (DUR_MIN <= native_dur <= DUR_MAX):
            print(f"  [skip] {cid}: duration {native_dur:.2f}s outside [{DUR_MIN},{DUR_MAX}]")
            continue
        peak = float(np.abs(w44).max()) if w44.size else 0.0
        rms = float(np.sqrt(np.mean(w44 ** 2))) if w44.size else 0.0
        if peak < PEAK_MIN or rms < RMS_MIN:
            print(f"  [skip] {cid}: silent/dead (peak {peak:.2e} rms {rms:.2e})")
            continue
        audio_sha = hashlib.sha256(w44.tobytes()).hexdigest()   # 44.1k-mono sha (spec §3 dedup / §6)
        if audio_sha in seen_audio_sha:
            print(f"  [skip] {cid}: duplicate audio (sha {audio_sha[:12]})")
            continue
        seen_audio_sha.add(audio_sha)
        records.append({
            "id": cid,
            "audio_sha256_44k_mono": audio_sha,
            "file_sha256": sha256_file(wav),
            "duration_s": round(native_dur, 4),
            "original_sample_rate": native_sr,
            "original_channels": native_ch,
            "resampled": native_sr != TARGET_SR,
            "resample_target_sr": TARGET_SR,
            "caption": derive_caption(meta, domain_noun),
            "license": "CC0",
            "source_url": meta.get("source_url") or meta.get("permalink"),
            "freesound_id": meta.get("freesound_id") or meta.get("id"),
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
        json.dumps([(r["id"], r["audio_sha256_44k_mono"]) for r in sorted(records, key=lambda r: r["id"])],
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
    d = os.path.join(sc, "sa3_domain_manifest_selftest")
    os.system(f"rm -rf {d}"); os.makedirs(d, exist_ok=True)
    dom = "impact_percussion"
    n_valid = 22
    # 22 valid clips: vary native SR (half at 22050 -> exercises deterministic resample) + CC0 meta;
    # captions DERIVE from meta (no .txt). Plus one too-short and one silent clip that must be SKIPPED.
    for i in range(n_valid):
        cid = f"clip{i:04d}"
        native_sr = 22050 if i % 2 else TARGET_SR
        t = np.linspace(0, 1.0, native_sr, endpoint=False)
        x = (0.3 * np.sin(2 * np.pi * (200 + 10 * i) * t)).astype("float32")
        sf.write(os.path.join(d, f"{cid}.wav"), x, native_sr)
        json.dump({"license": "Creative Commons 0", "name": f"Impact_Hit {i}.wav",
                   "tags": ["impact", "hit", "metal"], "domain_tags": ["impact", "hit"],
                   "permalink": f"https://freesound.org/s/{1000+i}/", "id": 1000 + i},
                  open(os.path.join(d, f"{cid}.meta.json"), "w"))
    # too-short (0.1s) -> skipped by DUR_MIN
    sf.write(os.path.join(d, "short.wav"), (0.3 * np.sin(np.linspace(0, 0.1, int(0.1 * TARGET_SR)))).astype("float32"), TARGET_SR)
    json.dump({"license": "CC0", "name": "tiny", "tags": ["impact"]}, open(os.path.join(d, "short.meta.json"), "w"))
    # silent (1s of ~zeros) -> skipped by RMS/peak
    sf.write(os.path.join(d, "silent.wav"), np.zeros(TARGET_SR, "float32"), TARGET_SR)
    json.dump({"license": "CC0", "name": "silence", "tags": ["impact"]}, open(os.path.join(d, "silent.meta.json"), "w"))

    recs = build_clip_records(d, domain=dom)
    filters_ok = len(recs) == n_valid and all(r["id"].startswith("clip") for r in recs)   # short/silent dropped
    resample_ok = any(r["resampled"] for r in recs) and any(not r["resampled"] for r in recs)
    caption_ok = all(r["caption"].startswith("a sound of impact hit") for r in recs)       # derived, not hand-written
    prompts = ["a held-out metallic impact", "a wooden clap", "a distant hit"]
    man = build_manifest(dom, recs, prompts, seed=20260821)
    ok = filters_ok and resample_ok and caption_ok
    ok = ok and man["n_clips"] == n_valid
    ok = ok and len(man["split"]["eval_L"]) >= MIN_EVAL
    ok = ok and set(man["split"]["train_L"]) & set(man["split"]["eval_L"]) == set()
    ok = ok and len(man["split"]["train_L"]) + len(man["split"]["eval_L"]) == n_valid
    # determinism: rebuild -> identical manifest sha + split
    man2 = build_manifest(dom, build_clip_records(d, domain=dom), prompts, seed=20260821)
    ok = ok and man2["manifest_sha256"] == man["manifest_sha256"] and man2["split"] == man["split"]
    # different seed changes the split but not the clip-hash manifest sha
    man3 = build_manifest(dom, recs, prompts, seed=1)
    ok = ok and man3["manifest_sha256"] == man["manifest_sha256"] and man3["split"] != man["split"]
    # leakage guard fires: feed a caption genuinely in train_L
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
    json.dump({"license": "CC-BY", "name": "x", "tags": ["impact"]}, open(os.path.join(d, "clip0000.meta.json"), "w"))
    try:
        build_clip_records(d, domain=dom)
    except ValueError:
        rejected = True
    ok = ok and rejected
    print(json.dumps({"n_clips": man["n_clips"], "train": len(man["split"]["train_L"]),
                      "eval": len(man["split"]["eval_L"]), "sha": man["manifest_sha256"][:12],
                      "filters_dropped_short_silent": filters_ok, "resample_tolerant": resample_ok,
                      "caption_derived": caption_ok,
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
    recs = build_clip_records(a.dir, domain=a.domain)
    prompts = [ln.strip() for ln in open(a.prompts_file) if ln.strip()]
    man = build_manifest(a.domain, recs, prompts, seed=a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(man, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}: {man['n_clips']} clips, train={len(man['split']['train_L'])} "
          f"eval={len(man['split']['eval_L'])} prompts_L={len(man['prompts_L'])} sha={man['manifest_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
