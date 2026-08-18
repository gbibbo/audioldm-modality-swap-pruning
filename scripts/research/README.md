# `scripts/research/`

Reproducible research entrypoints.

Present:

* `fetch_public_artifacts.sh` — M0 public-artifact fetch from Zenodo with md5
  verification against the published record checksums. Public network + CPU
  only; downloads no private artifact and runs no model.

Pending, per `docs/master_plan_v3.md`: M2 conditioning instrumentation, the
single reproducible GPU benchmark that records all Section 7.2 variables, and
the M3 pilot driver. None of these exist yet.
