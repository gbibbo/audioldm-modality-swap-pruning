#!/usr/bin/env python3
"""Deterministically STREAM-source one CC0 domain from Freesound per the FROZEN selection rule
(`docs/sa3/freesound_selection_spec.md`). Writes the final ACCEPTED `<id>.wav` + `<id>.meta.json`
plus a `sourcing_record.json` (full provenance) that `build_domain_manifest.py` then freezes.

Final-N discipline (spec §2–§3): candidates are streamed in `downloads_desc`, each is downloaded +
decoded, and the FULL §3 filter set — duration + silence/corruption + 44.1k-mono dedup — is applied
via `build_domain_manifest.clip_accept` (ONE source of truth). The stream continues PAST candidate
#40 when some are rejected, until `N_TAKE` are finally accepted or candidates are exhausted. So the
final N is after all filters, with backfill. `--dry-list` prints only the first N *metadata
candidates* (pre-download; label makes clear it is not the final accepted selection).

Acquisition representation is FROZEN to Freesound **`preview-hq-mp3`** (token-authorised; no OAuth /
original-download branch for this experiment). Provenance records API-side
`source_original_samplerate/channels/duration` SEPARATELY from the actual `fetched_samplerate/
channels` of the decoded preview — the fetched preview is never called "original".

Run:  .venv-metrics/bin/python scripts/sa3/fetch_freesound_domain.py --domain impact_percussion \
          --out data/sa3/adapters/impact_percussion --token-file <path> [--dry-list]
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, urllib.parse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_domain_manifest as BDM   # clip_accept + §3 constants (single source of truth)

API = "https://freesound.org/apiv2"
SEARCH_PATH = "/search/"
ACQ_REPR = "preview-hq-mp3"   # FROZEN acquisition representation (spec erratum rc1.3)

DOMAINS = {
    "impact_percussion": {"tags": ["impact", "hit", "percussion", "clap", "knock"]},
    "water_liquid":      {"tags": ["water", "splash", "liquid", "drip", "pour"]},
    "mechanical":        {"tags": ["machine", "motor", "mechanical", "engine"]},
    "animal":            {"tags": ["animal", "bird", "dog", "cat"]},
    "ambience":          {"tags": ["ambience", "atmosphere", "ambient"]},
}
LICENSE_FILTER = 'license:"Creative Commons 0"'
SORT = "downloads_desc"
N_TAKE = 40
N_MIN = 20
FIELDS = "id,name,tags,license,previews,duration,samplerate,channels,url"


def _is_cc0(license_field) -> bool:
    lic = str(license_field or "").lower()
    return ("creative commons 0" in lic) or ("publicdomain/zero" in lic)


def build_filter(domain: str) -> str:
    """FROZEN Solr filter: CC0 AND (any required domain tag). OR belongs in the filter, not query."""
    return f'{LICENSE_FILTER} tag:({" OR ".join(DOMAINS[domain]["tags"])})'


def build_search_params(domain: str, page_size: int = 150) -> dict:
    return {"query": "", "filter": build_filter(domain), "sort": SORT, "fields": FIELDS, "page_size": page_size}


def search_url(domain: str, page_size: int = 150, base: str = API) -> str:
    return base + SEARCH_PATH + "?" + urllib.parse.urlencode(build_search_params(domain, page_size))


def _token(a):
    tok = a.token or os.environ.get("FREESOUND_TOKEN")
    if not tok and a.token_file and os.path.exists(a.token_file):
        tok = open(a.token_file).read().strip()
    if not tok:
        raise SystemExit("no Freesound token: set $FREESOUND_TOKEN or --token/--token-file")
    return tok


def _get_page(url, token):
    """Return (json_dict, response_page_sha256) — the sha is provenance (§6)."""
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def stream_candidates(domain, token, prov):
    """Yield metadata-prefiltered candidates in downloads_desc order (CC0 + tag + API-duration in
    range + has preview-hq-mp3), recording each response page's sha256 into prov['pages']."""
    tagset = set(t.lower() for t in DOMAINS[domain]["tags"])
    url = search_url(domain, page_size=150)
    rank, seen = 0, set()
    while url:
        data, psha = _get_page(url, token)
        prov["pages"].append({"page": len(prov["pages"]), "sha256": psha,
                              "count": len(data.get("results", [])), "has_next": bool(data.get("next"))})
        for r in data.get("results", []):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            if not _is_cc0(r.get("license")):
                continue
            if not (tagset & set(t.lower() for t in (r.get("tags") or []))):
                continue
            if not (BDM.DUR_MIN <= float(r.get("duration", 0)) <= BDM.DUR_MAX):
                continue
            prev = (r.get("previews") or {}).get(ACQ_REPR)
            if not prev:
                continue
            yield {"rank": rank, "id": r["id"], "name": r.get("name", ""), "tags": r.get("tags", []) or [],
                   "domain": domain, "preview_url": prev, "permalink": r.get("url"),
                   "source_original_samplerate": r.get("samplerate"),
                   "source_original_channels": r.get("channels"),
                   "source_original_duration": r.get("duration")}
            rank += 1
        url = data.get("next")
        time.sleep(0.2)


def stream_accept(candidates, decode_fn, want, seen_shas=None):
    """PURE + testable. Iterate candidates in order; decode each via decode_fn(cand)->w44 (or None on
    failure); apply BDM.clip_accept (duration+silence+dedup); ACCEPT until `want` accepted or the
    stream is exhausted — BACKFILLING past rejected candidates. Returns (accepted, decisions)."""
    seen = set(seen_shas or [])
    accepted, decisions = [], []
    for cand in candidates:
        if len(accepted) >= want:
            break
        w44 = decode_fn(cand)
        ok, reason, sha = BDM.clip_accept(w44, cand.get("source_original_duration") or 0.0, seen)
        decisions.append({"rank": cand.get("rank"), "id": cand.get("id"), "accepted": bool(ok),
                          "reason": reason, "audio_sha256_44k_mono": sha})
        if ok:
            seen.add(sha)
            c = dict(cand); c["_audio_sha"] = sha
            accepted.append(c)
    return accepted, decisions


def download_decode(cand, token):
    """Download the preview-hq-mp3 and decode to (w44_mono float32, fetched_sr, fetched_channels)."""
    import tempfile, soundfile as sf, librosa
    req = urllib.request.Request(cand["preview_url"], headers={"Authorization": f"Token {token}"})
    data = urllib.request.urlopen(req, timeout=120).read()
    tf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False); tf.write(data); tf.close()
    try:
        info = sf.info(tf.name)
        fetched_sr, fetched_ch = int(info.samplerate), int(info.channels)
        w44 = librosa.load(tf.name, sr=BDM.TARGET_SR, mono=True)[0].astype("float32")
    finally:
        os.remove(tf.name)
    return w44, fetched_sr, fetched_ch


def source_domain(domain, token, out_dir, want, prov):
    """Stream + accept-with-backfill + write accepted <id>.wav/.meta.json. Returns accepted list."""
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    stash = {}

    def decode_fn(cand):
        try:
            w44, fsr, fch = download_decode(cand, token)
            stash[cand["id"]] = (w44, fsr, fch)
            return w44
        except Exception as e:
            prov["errors"].append({"id": cand["id"], "error": f"{type(e).__name__}: {e}"})
            return None

    accepted, decisions = stream_accept(stream_candidates(domain, token, prov), decode_fn, want)
    prov["candidates"] = decisions
    for c in accepted:
        w44, fsr, fch = stash[c["id"]]
        cid = f"fs{c['id']}"
        sf.write(os.path.join(out_dir, f"{cid}.wav"), w44, BDM.TARGET_SR)
        meta = {
            "license": "Creative Commons 0", "name": c["name"], "tags": c["tags"],
            "domain_tags": [t for t in DOMAINS[domain]["tags"] if t in set(x.lower() for x in c["tags"])],
            "permalink": c["permalink"], "id": c["id"], "freesound_id": c["id"],
            "acquisition_representation": ACQ_REPR,
            "source_original_samplerate": c["source_original_samplerate"],
            "source_original_channels": c["source_original_channels"],
            "source_original_duration": c["source_original_duration"],
            "fetched_samplerate": fsr, "fetched_channels": fch,
            "audio_sha256_44k_mono": c["_audio_sha"], "candidate_rank": c["rank"],
        }
        json.dump(meta, open(os.path.join(out_dir, f"{cid}.meta.json"), "w"), indent=2)
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--token", default=None); ap.add_argument("--token-file", default=None)
    ap.add_argument("--n-take", type=int, default=N_TAKE)
    ap.add_argument("--dry-list", action="store_true",
                    help="metadata candidates only (pre-download; NOT the final accepted selection)")
    a = ap.parse_args()
    token = _token(a)
    prov = {"domain": a.domain, "endpoint": API + SEARCH_PATH, "query": "", "filter": build_filter(a.domain),
            "sort": SORT, "page_size": 150, "acquisition_representation": ACQ_REPR, "n_take": a.n_take,
            "git_commit": subprocess.getoutput("git rev-parse HEAD"),
            "pages": [], "candidates": [], "errors": []}

    if a.dry_list:
        cands = []
        for c in stream_candidates(a.domain, token, prov):
            cands.append(c)
            if len(cands) >= a.n_take:
                break
        print(f"[freesound] {a.domain}: {len(cands)} METADATA candidates (pre-download; final selection "
              f"additionally applies silence/corruption/dedup after decode — NOT final N)")
        for c in cands:
            print(f"  #{c['rank']:02d} fs{c['id']}  dur={float(c['source_original_duration']):.2f}s  "
                  f"src_sr={c['source_original_samplerate']}  tags={c['tags'][:4]}  name={c['name'][:40]!r}")
        return 0

    os.makedirs(a.out, exist_ok=True)
    accepted = source_domain(a.domain, token, a.out, a.n_take, prov)
    prov["n_accepted"] = len(accepted)
    prov["n_rejected"] = sum(1 for d in prov["candidates"] if not d["accepted"])
    prov["_self_sha256"] = hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()
    json.dump(prov, open(os.path.join(a.out, "sourcing_record.json"), "w"), indent=2)
    status = "OK" if len(accepted) >= N_MIN else f"BELOW N_MIN={N_MIN} — domain FAILS (use frozen fallback)"
    print(f"[freesound] {a.domain}: {len(accepted)} ACCEPTED (of {len(prov['candidates'])} streamed, "
          f"{prov['n_rejected']} rejected) -> {a.out}  [{status}]")
    print(f"[freesound] provenance: {a.out}/sourcing_record.json ; next: build_domain_manifest.py "
          f"--dir {a.out} --domain {a.domain} (run in .venv-metrics)")
    return 0 if len(accepted) >= N_MIN else 1


if __name__ == "__main__":
    sys.exit(main())
