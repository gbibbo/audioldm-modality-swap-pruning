# Listening Study — Deployment Guide

Static bundle for the blinded expert listening study. Target host:
`https://sharing.edgeaudiolabs.com/`. Desktop Chrome/Firefox/Safari + headphones.

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
