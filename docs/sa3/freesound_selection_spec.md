# Freesound intra-domain selection — FROZEN pre-registration (2026-08-21 18:52, rc1.1 patch 5)

> **ERRATUM rc1.2 (2026-08-22, API-semantics repair — pre-data, not dataset tuning).** The first
> authenticated `--dry-list impact_percussion` returned **0 qualifying clips**. Cause: the frozen
> **`query`** strings below (`impact OR hit OR …`) don't match current Freesound semantics — `query`
> terms are **mandatory-AND** by default, so no clip matched all of them. `OR` belongs in the Solr
> **`filter`**, not `query`. **Repair (the frozen INTENT is unchanged — CC0 AND any-one required
> tag):** send `query=""` and
> `filter=license:"Creative Commons 0" tag:(impact OR hit OR percussion OR clap OR knock)`.
> Empirically (token-authenticated) this returns 27 938 candidates for `impact_percussion`. **The
> `query` column in §1 is SUPERSEDED by this tag-filter; the `required tags` column is authoritative
> and unchanged.** Endpoint note: both `/apiv2/search/` and `/apiv2/search/text/` return http 200 with
> **identical** results — the endpoint was **not** the cause; the code uses `/apiv2/search/`. **No clip
> IDs were observed or selected in the failed first attempt.** Everything else (downloads_desc, N=40,
> N_min=20, duration, silence, dedup, captions, split, resample, 10 s crop) is unchanged. Unit-tested:
> `tests/sa3/test_freesound_url.py`.
>
> **Second API-semantics fix (same erratum):** after the query fix, the dry-list still returned 0 —
> the API returns the **`license` field as a URL** (`http://creativecommons.org/publicdomain/zero/
> 1.0/`), not the string "Creative Commons 0", so the defensive client-side license re-check dropped
> every clip. Fixed with `_is_cc0()` (accepts the CC0 URL **and** string forms; the Solr `filter`
> already guarantees CC0). **Validated:** `impact_percussion --dry-list` now returns **40 qualifying
> clips** (≥ N_min=20) — coherent in-domain (impacts/hits/knocks/claps/crashes/drops), mixed native
> SR (32/44.1/48/96 kHz) handled by the deterministic resample-to-44.1. No fallback domain needed.
> Ledger SA3-FREESOUND-APIFIX-001.
>
> **rc1.3 (2026-08-22, pre-data implementation patch — before any download).** Three
> implementation/spec alignments (Gabriel's review), none of which changes the frozen INTENT:
> **(a) Final N=40 is counted AFTER every §3 filter.** Sourcing now STREAMS candidates in
> `downloads_desc`, downloads+decodes each, and applies duration + silence/corruption + 44.1k-mono
> dedup via a single `clip_accept()` (shared by fetcher and manifest), **continuing past candidate
> #40 (backfill) until 40 are finally accepted or candidates are exhausted**. `--dry-list` shows only
> *metadata candidates* (pre-download), explicitly labelled — not the final selection.
> **(b) `prompts_L` is 100% automatic** — after the deterministic split, `prompts_L =`
> captions(`eval_L`) (train-collisions removed) `+` the generic domain prompt (`"a {noun} sound"`),
> via frozen `derive_caption()`. The manual `--prompts-file` is removed for scientific manifests.
> **(c) Sourcing provenance is persisted** to `sourcing_record.json`: endpoint, exact
> query/filter/sort/page_size, per-response-page SHA256, per-candidate rank/id/decision/reason/
> audio-sha, git commit, and acquisition representation. **(d) Acquisition representation FROZEN to
> Freesound `preview-hq-mp3`** (token-only; no OAuth/original branch this experiment). Per clip the
> manifest records API-side `source_original_samplerate/channels/duration` SEPARATELY from the
> decoded preview's `fetched_samplerate/channels` — the preview is never labelled "original". All
> adapters use exactly this representation. Manifest schema bumped to `/2`. Unit-tested:
> `tests/sa3/{test_sourcing_stream.py (backfill/auto-prompts/provenance), test_freesound_url.py}` +
> `build_domain_manifest.py --selftest`. Ledger SA3-FREESOUND-STREAM-002.

**Written BEFORE any Freesound page is opened (rule S4).** Fixes every degree of freedom in turning a
domain name into a trained-adapter dataset, so no clip, caption, or split is chosen after seeing
results. Domains and fallback order were frozen in `rq2_validation_protocol.md` rc1
(`impact/percussion`, `water/liquid`; fallback `mechanical → animal → ambience`). This document
freezes the *selection within a domain*. Nothing here is scientific tuning — it is a deterministic
recipe; the sourcing script `scripts/sa3/fetch_freesound_domain.py` MUST implement exactly this and
record its provenance.

## 1. Query / tags (exact, per domain)

Freesound text search + tag filter, `license=Creative Commons 0` ONLY (API `filter=license:"Creative
Commons 0"`). Query strings are frozen here:

| domain | query | required tags (any-of) |
| --- | --- | --- |
| `impact_percussion` | `impact OR hit OR percussion OR clap` | `impact, hit, percussion, clap, knock` |
| `water_liquid` | `water OR splash OR liquid OR drip` | `water, splash, liquid, drip, pour` |
| `mechanical` (fallback) | `machine OR motor OR mechanical OR engine` | `machine, motor, mechanical, engine` |
| `animal` (fallback) | `animal OR bird OR dog OR cat` | `animal, bird, dog, cat` |
| `ambience` (fallback) | `ambience OR atmosphere OR room tone` | `ambience, atmosphere, ambient` |

A candidate qualifies iff its Freesound `license` is exactly `Creative Commons 0` AND at least one of
the domain's required tags is in its tag list.

## 2. Ordering and N (deterministic, no manual picking)

* **Sort key (frozen):** Freesound API `sort=downloads_desc` (most-downloaded first) — a fixed,
  reproducible order that does not depend on our judgement of "good for training".
* **Take the first `N_TAKE = 40` qualifying clips** after the filters in §3, in that sort order. If
  fewer than `N_MIN_CLIP = 20` qualify, the domain FAILS and we move to the next fallback domain (do
  not relax the filters). If between 20 and 40 qualify, take all of them.
* **No manual selection of "the best" clips** — the first N in the frozen sort order, full stop.

## 3. Per-clip filters (deterministic, applied in this order)

1. **Duration:** keep only `DUR_MIN = 0.3 s ≤ native_duration ≤ DUR_MAX = 12.0 s` (SFX-scale; the
   12 s cap is above the 10 s training crop so nothing is truncated pre-crop by a hair).
2. **Channels:** any channel count accepted; **downmix to mono deterministically** (mean of
   channels) for hashing/analysis; training uses `force_channels="stereo"` (upstream) which
   duplicates mono → stereo deterministically.
3. **Sample rate:** **accept any native SR; deterministically resample to 44 100 Hz** at load time
   (`librosa.load(sr=44100)`, the same resampler the dataset uses). Do NOT require native 44.1 kHz —
   that would bias/shrink the pool. Record `original_sample_rate` and `resampled=True/False`.
4. **Silence / corruption:** reject if the file fails to decode, or if peak `|x| < 1e-4`, or if RMS
   over the whole clip `< 1e-3` (dead/near-silent). Deterministic thresholds, frozen here.
5. **De-duplication:** reject a clip whose 44.1 kHz-mono sha256 equals one already accepted.

## 4. Caption derivation (deterministic — never hand-optimised)

`caption = derive_caption(meta)` = a fixed transformation of the Freesound metadata, NOT a
human-written or human-tuned prompt:

```
name  = meta["name"] with the file extension stripped, underscores/hyphens -> spaces, lowercased,
        collapsed whitespace, non-alphanumeric (except spaces) removed, truncated to 12 words.
tags  = the first 3 domain-required tags present in meta["tags"], in the §1 table order.
caption = "a sound of {name}" if name is non-empty else "a sound of {domain-noun}",
          then append " (" + ", ".join(tags) + ")" if any tags.
```

Held-out `prompts_L` are generated by the SAME transform applied to the **eval_L** clips' metadata
(so prompts are in-domain but come from held-out clips, never used in training), plus the domain's
generic prompt `"a {domain-noun} sound"`. `prompts_L` is disjoint from train captions by
construction (eval_L ∩ train_L = ∅) and is re-checked by `build_domain_manifest.py`.

## 5. Split and training constants (frozen)

* **Split:** deterministic seeded 80/20 `train_L` / `eval_L`, `min_eval = 5`, `split_seed = 20260821`
  (`build_domain_manifest.py`; §5.2 constants).
* **Training crop / duration:** `10.0 s` for controls AND ecological adapters (matches the analysis
  panel `SECONDS = 10` and the generation length used for `ΔT_L`). Frozen; the training wrapper
  default is 10.0.
* **Adapter recipe (frozen, §5.1/§5.2):** standard `lora`, rank 16, alpha 16, backbone (`transformer.
  layers`) for ecological / single block for controls; `1000` steps (upstream quick-start); LoRA
  params fp32; T4 base precision fp16 / Trainer `16-mixed`.

## 6. Provenance the sourcing script must record

Per clip: Freesound id, permalink URL, `license` string ("Creative Commons 0"), `name`, `tags`,
`original_sample_rate`, `original_channels`, `native_duration`, the 44.1 kHz-mono sha256, and the
derived caption. Per domain: the exact query, sort, `N_TAKE`, the frozen thresholds above, the API
response page hashes, and the git commit. All of this is frozen in the manifest
(`build_domain_manifest.py`) before `train_control_loras.py` runs. Audio and generated wavs are
gitignored; only manifests/hashes/splits/captions are tracked.
