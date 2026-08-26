#!/usr/bin/env python3
"""Freeze a per-domain CC0 data manifest for the RQ2 real-LoRA validation (protocol §4, §5.2;
selection rule frozen in `docs/sa3/freesound_selection_spec.md`).

Step-0 tooling (CPU, 0 cr). Reads a directory of `<id>.wav` + REQUIRED `<id>.meta.json` produced by
`fetch_freesound_domain.py` (which already streamed candidates in downloads_desc and applied EVERY
§3 filter — duration + silence + 44.1k-mono dedup — with backfill, so the clips here are the final
accepted set). This tool then freezes:

  * per-clip record: 44.1k-mono sha256, file sha256, source-original vs fetched SR/channels,
    acquisition representation, a DETERMINISTICALLY-DERIVED caption (spec §4), CC0 proof;
  * a DETERMINISTIC seeded 80/20 train_L / eval_L split (min 5 eval, §5.2);
  * `prompts_L` DERIVED AUTOMATICALLY from eval_L captions (never hand-written) + the generic
    domain prompt, with train-caption collisions removed (no leakage);
  * a manifest sha256 over the sorted (id, audio_sha) pairs.

`clip_accept()` is the SINGLE source of the §3 audio filters (imported by the fetcher too), so
sourcing and manifesting can never diverge. This tool sources NOTHING from the internet.

Usage:
  .venv-metrics/bin/python scripts/sa3/build_domain_manifest.py --dir data/sa3/adapters/impact_percussion \
       --domain impact_percussion --out configs/sa3/adapters/impact_percussion.manifest.json
  .venv-metrics/bin/python scripts/sa3/build_domain_manifest.py --selftest
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, random, re, sys

MIN_EVAL = 5          # §5.2
EVAL_FRAC = 0.20      # §5.2
N_MIN_CLIP = 20       # §4
TARGET_SR = 44100     # resample target (spec §3)
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
    (wav44_mono float32, fetched_sr, fetched_channels, fetched_duration_s)."""
    import soundfile as sf, librosa
    info = sf.info(path)
    fsr = int(info.samplerate)
    fch = int(info.channels)
    fdur = float(info.frames) / float(fsr) if fsr else 0.0
    w44, _ = librosa.load(path, sr=TARGET_SR, mono=True)   # deterministic resample + downmix
    return w44.astype("float32"), fsr, fch, fdur


def clip_accept(w44, native_duration, seen_shas):
    """SINGLE source of the §3 audio accept/reject rule (used by BOTH the fetcher and this tool).
    Returns (accepted: bool, reason: str, audio_sha256_44k_mono: str|None). Deterministic."""
    import numpy as np
    if not (DUR_MIN <= float(native_duration) <= DUR_MAX):
        return False, f"duration {float(native_duration):.3f}s outside [{DUR_MIN},{DUR_MAX}]", None
    if w44 is None or getattr(w44, "size", 0) == 0:
        return False, "empty/undecodable", None
    peak = float(np.abs(w44).max())
    rms = float(np.sqrt(np.mean(w44 ** 2)))
    if peak < PEAK_MIN or rms < RMS_MIN:
        return False, f"silent (peak {peak:.2e} rms {rms:.2e})", None
    sha = hashlib.sha256(w44.tobytes()).hexdigest()
    if sha in seen_shas:
        return False, f"duplicate audio {sha[:12]}", sha
    return True, "accepted", sha


def _is_cc0(license_field) -> bool:
    lic = str(license_field or "").lower()
    return ("creative commons 0" in lic) or ("publicdomain/zero" in lic) or \
           lic.upper().replace("-", "").replace(" ", "") in ("CC0", "CC01_0", "CC010", "CREATIVECOMMONS0")


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
    """One record per `<id>.wav` + required `<id>.meta.json`. CC0-only; §3 filters via clip_accept;
    caption DERIVED from metadata (spec §4). Records source-original (API) vs fetched (decoded)
    SR/channels separately (spec §6) — the fetched representation is the preview, NOT the original."""
    domain_noun = _DOMAIN_NOUN.get(domain, domain or "sound")
    records, seen_sha = [], set()
    for wav in sorted(glob.glob(os.path.join(dir_, "*.wav"))):
        cid = os.path.splitext(os.path.basename(wav))[0]
        meta_path = os.path.join(dir_, f"{cid}.meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"missing metadata for {cid}: {meta_path} (captions derive from it)")
        meta = json.load(open(meta_path))
        if not _is_cc0(meta.get("license")):
            raise ValueError(f"clip {cid} license {meta.get('license')!r} not CC0 (spec §1)")
        w44, file_sr, file_ch, file_dur = _audio_load_44k_mono(wav)
        # source-original (API) vs fetched (decoded preview) — read from meta; the saved .wav is a
        # 44.1k-mono training artifact, NOT the fetched representation. Fall back to file for selftest.
        fetched_sr = meta.get("fetched_samplerate") or file_sr
        fetched_ch = meta.get("fetched_channels") or file_ch
        src_sr = meta.get("source_original_samplerate") or fetched_sr
        src_ch = meta.get("source_original_channels") or fetched_ch
        src_dur = float(meta.get("source_original_duration") or file_dur)
        ok, reason, sha = clip_accept(w44, src_dur, seen_sha)
        if not ok:
            print(f"  [skip] {cid}: {reason}")
            continue
        seen_sha.add(sha)
        records.append({
            "id": cid,
            "audio_sha256_44k_mono": sha,
            "file_sha256": sha256_file(wav),
            "duration_s": round(src_dur, 4),
            "source_original_sample_rate": src_sr,
            "source_original_channels": src_ch,
            "fetched_sample_rate": fetched_sr,
            "fetched_channels": fetched_ch,
            "resampled_to_44k": fetched_sr != TARGET_SR,
            "acquisition_representation": meta.get("acquisition_representation", "local"),
            "caption": derive_caption(meta, domain_noun),
            "license": "CC0",
            "source_url": meta.get("source_url") or meta.get("permalink"),
            "freesound_id": meta.get("freesound_id") or meta.get("id"),
        })
    return records


def deterministic_split(ids: list, seed: int, eval_frac: float = EVAL_FRAC, min_eval: int = MIN_EVAL):
    n = len(ids)
    n_eval = max(min_eval, round(eval_frac * n))
    if n_eval >= n:
        raise ValueError(f"n={n} too small for min_eval={min_eval} + a non-empty train split")
    order = list(ids)
    random.Random(seed).shuffle(order)
    return sorted(order[n_eval:]), sorted(order[:n_eval])   # train, eval


def derive_prompts_L(records: list, train_ids: list, eval_ids: list, domain: str) -> list:
    """AUTO prompts_L (spec §4): eval_L captions (held-out) minus any that collide with train
    captions, deduped in deterministic order, plus the generic domain prompt. No hand editing."""
    by_id = {r["id"]: r for r in records}
    train_caps = {by_id[i]["caption"] for i in train_ids}
    prompts, seen = [], set()
    for i in sorted(eval_ids):
        c = by_id[i]["caption"]
        if c in train_caps or c in seen:
            continue
        seen.add(c); prompts.append(c)
    generic = f"a {_DOMAIN_NOUN.get(domain, domain)} sound"
    if generic not in seen:
        prompts.append(generic)
    return prompts


def build_manifest(domain: str, records: list, seed: int, prompts_L=None,
                   source: str = "freesound-cc0", sourcing_record: dict = None,
                   eval_frac: float = EVAL_FRAC) -> dict:
    if len(records) < N_MIN_CLIP:
        raise ValueError(f"{len(records)} clips < N_MIN_CLIP={N_MIN_CLIP} for domain {domain} (§4)")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate clip ids")
    train_ids, eval_ids = deterministic_split(ids, seed, eval_frac=eval_frac)
    if set(train_ids) & set(eval_ids):
        raise AssertionError("train_L ∩ eval_L != ∅")
    auto = prompts_L is None
    if auto:
        prompts_L = derive_prompts_L(records, train_ids, eval_ids, domain)
    if not prompts_L:
        raise ValueError("prompts_L empty (held-out domain prompts, §5.2)")
    by_id = {r["id"]: r for r in records}
    train_caps = {by_id[i]["caption"] for i in train_ids}
    leak = [p for p in prompts_L if p.strip() in train_caps]
    if leak:
        raise AssertionError(f"prompts_L leak into train captions: {leak[:3]}")
    manifest_sha = hashlib.sha256(
        json.dumps([(r["id"], r["audio_sha256_44k_mono"]) for r in sorted(records, key=lambda r: r["id"])],
                   sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema": "sa3-rq2-domain-manifest/2",
        "domain": domain,
        "source": source,
        "acquisition_representation": (sourcing_record or {}).get("acquisition_representation", "preview-hq-mp3"),
        "sample_rate_target": TARGET_SR,
        "split_seed": seed,
        "constants": {"eval_frac": eval_frac, "min_eval": MIN_EVAL, "n_min_clip": N_MIN_CLIP,
                      "dur_min": DUR_MIN, "dur_max": DUR_MAX},
        "n_clips": len(records),
        "clips": sorted(records, key=lambda r: r["id"]),
        "split": {"train_L": train_ids, "eval_L": eval_ids},
        "prompts_L": [p.strip() for p in prompts_L],
        "prompts_L_auto_derived": auto,
        "manifest_sha256": manifest_sha,
        "sourcing_record_sha256": (sourcing_record or {}).get("_self_sha256"),
        "leakage_checks": {"train_eval_disjoint": True, "prompts_not_in_train": True,
                           "prompts_auto_from_eval": auto,
                           "eval_never_selects_checkpoints": "enforced by convention (§5.2)"},
    }


def selftest() -> int:
    import numpy as np, soundfile as sf
    sc = os.environ.get("SCRATCH", "/tmp")
    d = os.path.join(sc, "sa3_domain_manifest_selftest")
    os.system(f"rm -rf {d}"); os.makedirs(d, exist_ok=True)
    dom = "impact_percussion"
    n_valid = 22
    for i in range(n_valid):
        cid = f"clip{i:04d}"
        native_sr = 22050 if i % 2 else TARGET_SR
        t = np.linspace(0, 1.0, native_sr, endpoint=False)
        x = (0.3 * np.sin(2 * np.pi * (200 + 10 * i) * t)).astype("float32")
        sf.write(os.path.join(d, f"{cid}.wav"), x, native_sr)
        json.dump({"license": "http://creativecommons.org/publicdomain/zero/1.0/",
                   "name": f"Impact_Hit {i}.wav", "tags": ["impact", "hit", "metal"],
                   "domain_tags": ["impact", "hit"], "permalink": f"https://freesound.org/s/{1000+i}/",
                   "id": 1000 + i, "acquisition_representation": "preview-hq-mp3",
                   "source_original_samplerate": native_sr, "source_original_channels": 1,
                   "source_original_duration": 1.0},
                  open(os.path.join(d, f"{cid}.meta.json"), "w"))
    sf.write(os.path.join(d, "short.wav"), (0.3 * np.sin(np.linspace(0, 0.1, int(0.1 * TARGET_SR)))).astype("float32"), TARGET_SR)
    json.dump({"license": "CC0", "name": "tiny", "tags": ["impact"], "source_original_duration": 0.1},
              open(os.path.join(d, "short.meta.json"), "w"))
    sf.write(os.path.join(d, "silent.wav"), np.zeros(TARGET_SR, "float32"), TARGET_SR)
    json.dump({"license": "CC0", "name": "silence", "tags": ["impact"], "source_original_duration": 1.0},
              open(os.path.join(d, "silent.meta.json"), "w"))

    recs = build_clip_records(d, domain=dom)
    filters_ok = len(recs) == n_valid and all(r["id"].startswith("clip") for r in recs)
    resample_ok = any(r["resampled_to_44k"] for r in recs) and any(not r["resampled_to_44k"] for r in recs)
    caption_ok = all(r["caption"].startswith("a sound of impact hit") for r in recs)
    prov_ok = all(("source_original_sample_rate" in r and "fetched_sample_rate" in r
                   and r["acquisition_representation"] == "preview-hq-mp3") for r in recs)
    man = build_manifest(dom, recs, seed=20260821)                       # AUTO prompts
    train_caps = {r["caption"] for r in recs if r["id"] in man["split"]["train_L"]}
    auto_ok = (man["prompts_L_auto_derived"] and len(man["prompts_L"]) >= 1
               and any(p.startswith("a impact sound") for p in man["prompts_L"])
               and not (set(man["prompts_L"]) & train_caps))                # no leakage
    ok = filters_ok and resample_ok and caption_ok and prov_ok and auto_ok
    ok = ok and man["n_clips"] == n_valid and len(man["split"]["eval_L"]) >= MIN_EVAL
    ok = ok and set(man["split"]["train_L"]) & set(man["split"]["eval_L"]) == set()
    man2 = build_manifest(dom, build_clip_records(d, domain=dom), seed=20260821)
    ok = ok and man2["manifest_sha256"] == man["manifest_sha256"] and man2["split"] == man["split"]
    man3 = build_manifest(dom, recs, seed=1)
    ok = ok and man3["manifest_sha256"] == man["manifest_sha256"] and man3["split"] != man["split"]
    # explicit-leak guard still fires
    by_id = {r["id"]: r for r in recs}
    train_caption = by_id[man["split"]["train_L"][0]]["caption"]
    leaked = False
    try:
        build_manifest("x", recs, seed=20260821, prompts_L=[train_caption])
    except AssertionError:
        leaked = True
    ok = ok and leaked
    rejected = False
    json.dump({"license": "http://creativecommons.org/licenses/by/4.0/", "name": "x", "tags": ["impact"]},
              open(os.path.join(d, "clip0000.meta.json"), "w"))
    try:
        build_clip_records(d, domain=dom)
    except ValueError:
        rejected = True
    ok = ok and rejected
    print(json.dumps({"n_clips": man["n_clips"], "train": len(man["split"]["train_L"]),
                      "eval": len(man["split"]["eval_L"]), "prompts_L": len(man["prompts_L"]),
                      "auto_prompts": auto_ok, "provenance": prov_ok, "resample_tolerant": resample_ok,
                      "caption_derived": caption_ok, "determinism": man2["manifest_sha256"] == man["manifest_sha256"],
                      "leakage_guard": leaked, "cc0_guard": rejected}, indent=2))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir"); ap.add_argument("--domain"); ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-frac", type=float, default=EVAL_FRAC,
                    help=f"eval_L fraction (default {EVAL_FRAC}; RQ2b mechanical uses 0.4 -> 96/64 of 160)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.dir and a.domain and a.out):
        ap.error("--dir --domain --out are required (or use --selftest); prompts_L are AUTO-derived")
    sr_path = os.path.join(a.dir, "sourcing_record.json")
    sourcing = json.load(open(sr_path)) if os.path.exists(sr_path) else None
    recs = build_clip_records(a.dir, domain=a.domain)
    man = build_manifest(a.domain, recs, seed=a.seed, sourcing_record=sourcing,
                         eval_frac=a.eval_frac)   # AUTO prompts
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(man, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}: {man['n_clips']} clips, train={len(man['split']['train_L'])} "
          f"eval={len(man['split']['eval_L'])} prompts_L={len(man['prompts_L'])} (auto={man['prompts_L_auto_derived']}) "
          f"sha={man['manifest_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
