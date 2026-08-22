#!/usr/bin/env python3
"""Deterministically source one CC0 domain from Freesound per the FROZEN selection rule
(`docs/sa3/freesound_selection_spec.md`, rc1.1 patch 5). Writes a curated dir of `<id>.wav` +
`<id>.meta.json` that `build_domain_manifest.py` then freezes. NO manual clip picking.

Auth: needs a Freesound API token in `$FREESOUND_TOKEN` (or --token / --token-file). The API key
(token) authorises search + preview download; original-quality download needs OAuth2 (--use-original
with $FREESOUND_OAUTH). Everything is resampled to 44.1 kHz mono at manifest time anyway (spec §3).

Determinism: the frozen query + `sort=downloads_desc` + take-first-N give a reproducible selection;
no randomness, no human choice. `--dry-list` does search-only (cheap) and prints exactly which clips
WOULD be taken, for validation before downloading.

Run:  FREESOUND_TOKEN=... .venv-metrics/bin/python scripts/sa3/fetch_freesound_domain.py \
          --domain impact_percussion --out data/sa3/adapters/impact_percussion [--dry-list]
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.parse, urllib.request

API = "https://freesound.org/apiv2"
SEARCH_PATH = "/search/"   # rc1.2: current endpoint; `/search/text/` is equivalent (both return 200,
                           # identical results) — the 0-results bug was the query semantics, not the path.

# FROZEN per spec §1-§3 (do not edit without a ledger entry). Selection is by DOMAIN TAGS only.
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
DUR_MIN, DUR_MAX = 0.3, 12.0
FIELDS = "id,name,tags,license,previews,duration,samplerate,channels,url,download"


def _is_cc0(license_field) -> bool:
    """rc1.2 fix: the API returns `license` as a URL (e.g. http://creativecommons.org/publicdomain/
    zero/1.0/), NOT the string 'Creative Commons 0'. Accept both forms. (The Solr filter already
    enforces CC0; this is a defensive re-check that must not use the wrong format.)"""
    lic = str(license_field or "").lower()
    return ("creative commons 0" in lic) or ("publicdomain/zero" in lic)


def build_filter(domain: str) -> str:
    """FROZEN Solr filter: CC0 AND (any required domain tag). rc1.2 fix — the OR belongs in the
    `filter`, NOT the `query` (Freesound treats `query` terms as mandatory-AND by default, which is
    why the first authenticated dry-list returned 0). See docs/sa3/freesound_selection_spec.md."""
    tags = " OR ".join(DOMAINS[domain]["tags"])
    return f'{LICENSE_FILTER} tag:({tags})'


def build_search_params(domain: str, page_size: int = 150) -> dict:
    """Deterministic tag-based selection: empty query, tags+license in the filter, downloads_desc."""
    return {"query": "", "filter": build_filter(domain), "sort": SORT,
            "fields": FIELDS, "page_size": page_size}


def search_url(domain: str, page_size: int = 150, base: str = API) -> str:
    return base + SEARCH_PATH + "?" + urllib.parse.urlencode(build_search_params(domain, page_size))


def _token(a):
    tok = a.token or os.environ.get("FREESOUND_TOKEN")
    if not tok and a.token_file and os.path.exists(a.token_file):
        tok = open(a.token_file).read().strip()
    if not tok:
        raise SystemExit("no Freesound token: set $FREESOUND_TOKEN or --token/--token-file "
                         "(search+preview need the API token; original download needs OAuth2)")
    return tok


def _get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def search_domain(domain, token, want):
    """Return the FIRST `want` qualifying results in the frozen sort order (deterministic)."""
    spec = DOMAINS[domain]
    tagset = set(t.lower() for t in spec["tags"])
    url = search_url(domain, page_size=150)
    kept, seen = [], set()
    while url and len(kept) < want:
        data = _get(url, token)
        for r in data.get("results", []):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rtags = set(t.lower() for t in (r.get("tags") or []))
            if not (tagset & rtags):
                continue
            if not (DUR_MIN <= float(r.get("duration", 0)) <= DUR_MAX):
                continue
            if not _is_cc0(r.get("license")):
                continue
            kept.append(r)
            if len(kept) >= want:
                break
        url = data.get("next")
        time.sleep(0.2)
    return kept


def download_clip(r, out_dir, token, use_original=False, oauth=None):
    """Download the preview (token) or original (OAuth2) and save as `<id>.wav` (resample happens at
    manifest time). Returns the saved wav path."""
    import numpy as np, soundfile as sf, librosa
    cid = f"fs{r['id']}"
    if use_original:
        if not oauth:
            raise SystemExit("--use-original needs $FREESOUND_OAUTH (OAuth2 access token)")
        src = r["download"]; hdr = {"Authorization": f"Bearer {oauth}"}
    else:
        src = r["previews"]["preview-hq-mp3"]; hdr = {"Authorization": f"Token {token}"}
    raw = os.path.join(out_dir, f"{cid}.src")
    req = urllib.request.Request(src, headers=hdr)
    with urllib.request.urlopen(req, timeout=120) as resp, open(raw, "wb") as fh:
        fh.write(resp.read())
    w, sr = librosa.load(raw, sr=None, mono=True)     # decode as-is; manifest resamples to 44.1
    wav = os.path.join(out_dir, f"{cid}.wav")
    sf.write(wav, w.astype("float32"), sr)
    os.remove(raw)
    meta = {"license": "Creative Commons 0", "name": r.get("name", ""), "tags": r.get("tags", []),
            "domain_tags": [t for t in DOMAINS[r["_domain"]]["tags"] if t in set(x.lower() for x in (r.get("tags") or []))],
            "permalink": r.get("url"), "id": r["id"], "freesound_id": r["id"],
            "original_samplerate": r.get("samplerate"), "original_channels": r.get("channels"),
            "native_duration": r.get("duration"), "download_kind": "original" if use_original else "preview-hq-mp3"}
    json.dump(meta, open(os.path.join(out_dir, f"{cid}.meta.json"), "w"), indent=2)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAINS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--token", default=None); ap.add_argument("--token-file", default=None)
    ap.add_argument("--n-take", type=int, default=N_TAKE)
    ap.add_argument("--use-original", action="store_true")
    ap.add_argument("--dry-list", action="store_true", help="search only; print the frozen selection, download nothing")
    a = ap.parse_args()
    token = _token(a)
    results = search_domain(a.domain, token, a.n_take)
    for r in results:
        r["_domain"] = a.domain
    print(f"[freesound] {a.domain}: {len(results)} qualifying clips (frozen query/sort/filters)")
    if len(results) < N_MIN:
        print(f"[freesound] WARNING: < N_MIN={N_MIN} — domain FAILS, use the frozen fallback order (spec §2)")
    if a.dry_list:
        for r in results:
            print(f"  fs{r['id']}  dur={r.get('duration'):.2f}s  sr={r.get('samplerate')}  "
                  f"tags={r.get('tags')[:4]}  name={r.get('name')[:40]!r}")
        return 0
    os.makedirs(a.out, exist_ok=True)
    oauth = os.environ.get("FREESOUND_OAUTH")
    n = 0
    for r in results:
        try:
            download_clip(r, a.out, token, use_original=a.use_original, oauth=oauth)
            n += 1
        except Exception as e:
            print(f"  [skip] fs{r['id']}: {type(e).__name__}: {e}")
    print(f"[freesound] wrote {n} clips to {a.out}; next: build_domain_manifest.py --dir {a.out} "
          f"--domain {a.domain} (run in .venv-metrics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
