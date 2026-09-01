# Listening Study — Deployment Guide

## Hosting route (FINAL, 2026-09-01): GitHub Pages

Delivery-route history: **EAL Sharing** rejected (team-only Google login → blocks anonymous external
panel + breaks anonymity); **Claude Artifacts** built and published but rejected as the delivery
mechanism (hard to share with external participants; claude.ai access/telemetry uncertainty; manual
JSON return inferior to the working receiver). **Final route = GitHub Pages**, serving the
self-contained static pages that POST automatically to the Apps Script receiver.

**Build & deploy (reproducible):**
```
# 1. build self-contained pages (live endpoint injected from local config.js):
OPENBLAS_CORETYPE=Haswell .venv-loudness/bin/python scripts/research/build_listening_deploy.py
# 2. stage the gh-pages tree with fresh opaque slugs + local-only slug map:
OPENBLAS_CORETYPE=Haswell .venv-loudness/bin/python scripts/research/deploy_gh_pages.py
# 3. push the orphan gh-pages branch from the staging dir (plumbing; does not touch main):
STAGE="$PWD/artifacts/gh_pages_stage"; IDX=$(mktemp -u)
GIT_INDEX_FILE="$IDX" git -C "$STAGE" --git-dir="$PWD/.git" --work-tree="$STAGE" add -A -f .
TREE=$(GIT_INDEX_FILE="$IDX" git write-tree)
COMMIT=$(printf "gh-pages deploy\n" | git commit-tree "$TREE")   # orphan (no -p)
git branch -f gh-pages "$COMMIT"; git push -f -u origin gh-pages
# Pages: source = gh-pages, folder = / (already enabled).
```
Re-running step 2 mints NEW slugs; deploy the staging content and slug map from the SAME run.

**Structure on gh-pages** (orphan branch, deployment material only — no repo source/history):
`.nojekyll`, `robots.txt` (Disallow: /), `index.html` ("Not found."), `s/<32-hex-slug>/index.html` ×6.
Each participant page: self-contained (inline CSS/JS, embedded blinded manifest, lossless FLAC audio),
`<meta name="robots" content="noindex,nofollow,noarchive">`, auto-POST to the receiver, Download/Copy
fallback. Opaque 128-bit slugs; the `participant_code → slug/URL` map is **local-only**
(`configs/research/listening_deploy_slugs.local.json`, gitignored) — participant URLs are NOT committed
to any branch or doc; Claude returns them to Gabriel in chat.

Base URL: `https://gbibbo.github.io/audioldm-modality-swap-pruning/` (public, no login).

**Collection = automatic** POST to the Apps Script receiver (external POST is allowed on GitHub Pages),
with Download-JSON / Copy-to-clipboard fallback. The recipient email stays server-side in the receiver.

**Remaining pre-launch item:** in-browser QA (audio playback, answer-lock, replay, an actual browser
POST reaching the receiver) — not runnable from the CLI. All HTTP/source QA has passed remotely and the
receiver was verified live via a synthetic POST.

The published Claude Artifacts (superseded) remain private in Gabriel's gallery and are not used.

---


Static bundle for the blinded expert listening study. Target host:
`https://sharing.edgeaudiolabs.com/`. Desktop Chrome/Firefox/Safari + headphones.

**Version: `LSTUDY-2026-08-31-v1.2`** (amendments `docs/listening_study_protocol_v1_1_amendment.md`,
`…_v1_2_amendment.md`). Design **D3** (80 sev-1 prompts × both durations single-rated + 18 bridge
prompts double-rated; no sev-2 human arm). Inference target = **fixed panel** (six listeners). **Primary
estimator = unique-prompt** (ratings averaged within prompt; each of the 80 prompts counts once; bridge
prompts NOT double-weighted; unique-prompt bootstrap, B=10000). v1.1 listener-stratified + pooled +
leave-one-listener-out + bridge agreement are non-gating sensitivities. Matched-vs-unrelated catch uses
**real AudioCaps references**. Answers are **locked until both clips play in full**.

**Receiver Script Properties to set (exact values):**
```
EXPECT_STUDY     = LSTUDY-2026-08-31-v1.2
EXPECT_PROTOCOL  = fd53a5babda774efc8186a2601e2366f8e59f7ad8739ae5648a4f328a075719d
RECIPIENTS       = <organiser email(s), comma-separated>   # server-side only, never in the bundle
```

## What is committed vs. local-only

| Artifact | Location | Committed? |
|---|---|---|
| Client app | `listening_study/{index.html,app.js,style.css,config.js}` | yes |
| Blinded per-participant manifests | `listening_study/public_manifests/P0{1..6}.json` | yes |
| Protocol / design / power / inventory / validation | `docs/`, `configs/research/` | yes |
| Email receiver template | `receiver/google_apps_script/` | yes |
| **Loudness-normalized audio** | `listening_study/audio/*.wav` | **NO — gitignored** (generated audio) |
| **Unblinding key** (A/B→system map, salt) | `configs/research/listening_study_assignments_private.json` | **NO — gitignored** |
| **Bundle manifest** (hash→source map) | `configs/research/listening_study_bundle_manifest.json` | **NO — gitignored** (unblinding) |
| Real-ref catch pool (metadata) | `configs/research/listening_study_realref_pool.json` | yes (ytid/caption/labels/sha; no unblinding) |
| Real-ref staged 16 kHz clips | `artifacts/listening_study/real_refs/*.wav` | **NO — gitignored** (artifacts/) |
| Power sim v2 / analyzer / pair audit | `configs/research/listening_study_power_v2.json`, `scripts/research/{listening_power_sim_v2,listening_analyze,listening_loudness_pair_audit}.py` | yes |

The private key and bundle manifest are unblinding artifacts and must never reach the
public repo or the deployed bundle. The GitHub remote is public.

## Build order (freeze order §15)

1. Freeze + commit the protocol, design, public manifests, code (this is the freeze commit).
2. **Then** build the deployed audio from the frozen private key:
   ```
   OPENBLAS_CORETYPE=Haswell .venv-loudness/bin/python scripts/research/build_listening_bundle.py
   ```
   Writes `listening_study/audio/*.wav` (−36 LUFS listening copies) + the local bundle manifest.
   The applied gain reaches −36 LUFS on the source measurement exactly; on re-measurement ~430/434
   copies sit within ±1 dB of −36, and ~4 near-silent failed-pruned clips drift a few dB (down to
   −40.7 LUFS) because the BS.1770 absolute gate is not scale-invariant. This is expected, is not a
   rule violation (single fixed gain, no limiting), and does not favour recovered.
3. Run internal QA:
   ```
   OPENBLAS_CORETYPE=Haswell .venv-loudness/bin/python scripts/research/listening_study_validate.py
   ```
   All checks (incl. audio paths + loudness) must pass.
4. Return to Gabriel for the participant-launch GO. No real participant before the freeze commit.

## Deploy

1. Configure the receiver: see `receiver/google_apps_script/README.md`. Set `RECIPIENTS`
   (server-side), deploy the web app, copy the `/exec` URL.
2. Set `RESULTS_ENDPOINT` in `listening_study/config.js` to that URL **at deploy time**
   (do not commit a live endpoint to the public repo unless intended).
3. Upload the `listening_study/` directory (including `audio/`, excluding nothing public) to the host.
4. Sanity-test each participant link before sending.

## Participant links

```
https://sharing.edgeaudiolabs.com/?p=P01
https://sharing.edgeaudiolabs.com/?p=P02
https://sharing.edgeaudiolabs.com/?p=P03
https://sharing.edgeaudiolabs.com/?p=P04
https://sharing.edgeaudiolabs.com/?p=P05
https://sharing.edgeaudiolabs.com/?p=P06
```

Each link loads only that participant's frozen assignment. 35–37 trials, ≤ 20 min.

## Testing before recruitment (synthetic only)

* Open a participant link locally (`python -m http.server` in `listening_study/`) and walk a few trials.
* POST a synthetic payload to the endpoint (curl) and confirm the organiser email arrives with the JSON
  attached; verify no IP/user-agent is included.
* Disconnect the endpoint and confirm the **Download** and **Copy to clipboard** fallbacks work.
* Confirm no console errors; confirm A and B never expose model identity in DOM, network, or filenames.

## Results

The client POSTs a minimal JSON (no PII) to `RESULTS_ENDPOINT`; the receiver emails it. If the POST
cannot be confirmed, the participant is shown Download/Copy fallbacks so responses are never lost.
De-duplicate on `submission_uuid`. Analysis decodes A/B→system and severity/duration via the private
key, then runs the frozen §5–§6 endpoints. No human data is analyzed this turn.

## Ethics

We minimize data collection and burden. We do **not** assert that ethics approval is unnecessary; the
corresponding author/institution determines any ethics/exemption requirement before recruitment.
