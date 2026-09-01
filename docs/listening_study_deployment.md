# Listening Study — Deployment Guide

## Hosting route (decided 2026-09-01): Claude Artifacts

**EAL Sharing (`sharing.edgeaudiolabs.com`) was REJECTED** for this study: its landing page states
*"Team only. Sign in with your authorized Google account."* It is a login-gated internal tool, so
(a) external panelists have no authorized EAL Google accounts, and (b) Google-OAuth login would break
the anonymity the frozen consent promises. Not usable for an anonymous external panel.

**Chosen route: six private Claude Artifacts (one per participant)**, built by
`scripts/research/build_listening_artifact.py` (self-contained: inline CSS/JS, embedded blinded
manifest, all audio as lossless FLAC `data:` URIs; each < 16 MB; no external requests). Published
private; Gabriel shares each from the artifact's share menu.

Participant links (private until shared):
```
P01  https://claude.ai/code/artifact/2840cebd-99a8-450a-888a-4b2545c73441
P02  https://claude.ai/code/artifact/22504abe-1abe-453e-9561-eb04882c31df
P03  https://claude.ai/code/artifact/0a43bdee-adaf-49d9-99fc-3974e5a386f6
P04  https://claude.ai/code/artifact/52a53898-ad16-4f4a-ba6c-18a189dd798c
P05  https://claude.ai/code/artifact/01f7094b-1e31-492d-98b7-852fc51bf21d
P06  https://claude.ai/code/artifact/88cf21d4-9859-4a17-bbe4-da30523df80d
```

**Collection under the artifact route = MANUAL.** The artifact CSP blocks external `fetch`, so the
live Apps Script receiver (verified working, see below) is NOT reachable from inside an artifact. On
submit the page shows Download (via the `downloads` capability — `<a download>` is inert in artifacts)
and Copy-to-clipboard; the participant sends the JSON to the organiser. The Apps Script receiver
remains available only if the study is instead hosted on a plain static site (see the alternative).

**Two pre-launch items requiring Gabriel's decision/confirmation:**
1. **Human browser QA** (cannot be done from the CLI): open one artifact and confirm FLAC audio plays
   in full, the answer-lock unlocks only after both clips finish, one replay works, and Download/Copy work.
2. **Privacy/consent**: the frozen consent says "no cookies or analytics." claude.ai frames artifacts
   and may record view-level engagement telemetry, and viewers likely need claude.ai access. If that
   conflicts with the consent text, either add a pre-participant consent note (supervisor decision — do
   NOT change consent silently) or host on a static no-analytics site instead.

**Alternative (automatic collection, no login):** the six self-contained files from
`scripts/research/build_listening_deploy.py` on a plain static host (e.g. GitHub Pages) — external
POST is allowed there, so the live Apps Script receiver collects automatically and no login is needed.

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
