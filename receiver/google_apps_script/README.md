# Listening-study results receiver (Google Apps Script)

A minimal serverless endpoint that receives the anonymous results JSON from the
static listening-study client and emails it to the study organiser. No database,
no personal metadata (no IP, user-agent, or location) is read or stored.

## Why not `mailto:` or client-side email
A static page cannot reliably send email. `mailto:` is fragile and is **not** used.
This receiver is the configurable `RESULTS_ENDPOINT` the client POSTs to. The
client always also offers an offline fallback (download JSON / copy to clipboard),
so responses are never lost even if the endpoint is down.

## Setup
1. Create a new Apps Script project at <https://script.google.com>.
2. Paste `Code.gs`.
3. Project Settings → **Script properties**, add:
   - `RECIPIENTS` = organiser email(s), comma-separated (kept server-side; never in the public bundle).
   - `EXPECT_STUDY` = the frozen `study_version` (optional, rejects mismatches).
   - `EXPECT_PROTOCOL` = the frozen `protocol_hash` (optional, rejects mismatches).
4. Deploy → **New deployment** → **Web app**: execute as *Me*, access *Anyone*.
5. Copy the `/exec` URL into `listening_study/config.js` → `RESULTS_ENDPOINT`
   **at deploy time only** (do not commit a live endpoint to the public repo unless intended).
6. Test with a synthetic payload before recruiting (see `docs/listening_study_deployment.md`).

## Payload contract (only these fields; no personal data)
```
study_version, protocol_hash, participant_code, assignment_hash,
submission_uuid, client_started_ts, client_completed_ts, total_ms,
responses[ {trial_id, type, relevance, quality, plays_A, plays_B,
            shown_ts, responded_ts, dwell_ms} ]
```
`submission_uuid` lets you de-duplicate accidental double submissions.

## Alternative receivers
Any HTTPS endpoint honouring the same contract works (e.g. a Cloud Function or a
Cloudflare Worker). Keep recipient addresses and secrets server-side, out of the
committed client bundle.
