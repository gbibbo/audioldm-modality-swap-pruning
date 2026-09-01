# Human Listening Study — FORMAL CLOSURE (CANCELLED PRE-LAUNCH)

**Decision date:** 2026-09-01. **Status: CANCELLED / VOID for scientific inference.**

```
PARTICIPANTS LAUNCHED:      0
HUMAN RESPONSES COLLECTED:  0
HUMAN ANALYSIS:             NONE
```

## What happened

A blinded expert listening study was **designed and frozen** through three pre-participant versions —
v1 (`aa906c2`), v1.1 (`a6c46a3`, D3 + fixed-panel + real-reference catch), and v1.2 (`7019ea5`,
unique-prompt estimator) — and a delivery platform was prepared (Apps Script receiver; a Claude-Artifact
route; finally a GitHub-Pages route). It was **NEVER launched**: no participant was ever contacted for
scientific data collection, and **zero human responses were collected**. All verification used synthetic
inputs only.

## Why it was cancelled

Arshdeep Singh (co-author/reviewer) did not approve running a listening test because it may require
ethics / institutional approvals. The supervisor decision was to **cancel rather than seek or assume an
exemption**. The cancellation happened **before recruitment and before any human response existed**.

## Consequences (binding)

* The listening study **must not appear as evidence in the paper**. No human claim may be made from it.
* All listening-study endpoints, power analyses (`listening_study_power*.json`), assignments
  (`listening_study_design.json`, private key), the analysis estimator (`listening_analyze.py`), and the
  protocol/amendments (`docs/listening_study_protocol*.md`) are **VOID for scientific inference**. They
  remain in git history **only as provenance** and process documentation.
* The **Apps Script receiver**, the **Claude-Artifact** pages, and the **GitHub-Pages participant route**
  are **operationally abandoned**. They are not deployed as an active study. No participant URLs are
  active study links. (The abandoned participant `gh-pages` slug pages were replaced by the public
  audio-examples site; the Apps Script `/exec` endpoint is no longer used and is not committed to `main`.)
* Frozen protocol and history are **not deleted and not rewritten**.

## Replacement deliverable

A public, anonymous, **non-interactive** companion webpage with representative generated audio examples
(no forms, no consent, no participant codes, no receiver, no cookies, no analytics, no POST, no login).
See `docs/listening_study_deployment.md` history and `configs/research/public_audio_examples_manifest.json`;
built by `scripts/research/build_public_audio_examples.py`; hosted on GitHub Pages.

## Provenance pointers (retained, void for inference)

`docs/listening_study_protocol.md`, `…_v1_1_amendment.md`, `…_v1_2_amendment.md`,
`configs/research/listening_study_{design,inventory,power,power_v2,power_v3,validation,realref_pool}.json`,
`scripts/research/listening_*.py`, `scripts/research/build_listening_*.py`, ledger entries
LISTENING-STUDY-PREP / -PRELAUNCH-AUDIT / -ESTIMATOR-FIX-V1.2 / -DEPLOY-ARTIFACT / -DEPLOY-GHPAGES.
