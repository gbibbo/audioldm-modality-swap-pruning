# Listening Study — PRE-PARTICIPANT AMENDMENT v1.1 → v1.2

**Study version:** `LSTUDY-2026-08-31-v1.2` (amends v1.1 `LSTUDY-2026-08-31-v1.1`; v1 and v1.1 remain
immutable in git history at `aa906c2`/`95350fd`/`a6c46a3`). **Detected:** final statistical pre-launch
correction (2026-09-01 00:24 MVD). **No human responses existed**; no participant had started; manuscript
frozen; no GPU; no new generation. **Nothing about participant assignments, audio, prompt selection,
response scale, catches, loudness, or UI content changes** — only the statistical estimator, the
documentation, and the version/protocol/public-bundle hashes.

Append-only provenance: v1.2 `protocol_hash = sha256(v1_protocol ++ v1.1_amendment ++ this_amendment)`.
The governing inference below supersedes the v1.1 §A4 estimator.

---

## Issue: bridge double-weighting + mild Type-I inflation

The project principle is **prompt = statistical unit**. D3 has 80 unique sev-1 prompts, 62 rated by one
listener and **18 bridge prompts rated by two listeners**. The v1.1 estimator (`listener-stratified`)
placed every listener rating into listener strata, so the 18 bridge prompts contributed **twice** to
H1/H2 relative to the other 62 — not the intended estimand (the bridge ratings were added for
information + inter-rater reliability, **not** extra scientific weight). The v1.1 stratified bootstrap
was also mildly liberal under the fixed-panel null (Type-I ≈ 0.039–0.043 for a nominal 0.025). Better-
calibrated prompt-oriented alternatives exist, so this is corrected before launch.

## New PRIMARY estimand — unique-prompt weighting

For every unique sev-1 ytid `i` and duration `d` (positive = recovered):
```
recovered_score(trial) = v if recovered is side B else -v      (v in {-2..+2})
H_{i,d} = mean over ALL assigned raters of recovered_score      (non-bridge: 1 rating; bridge: mean of 2)
A_native = (1/80) * sum_i H_{i,native}                          (H1)
J_H      = (1/80) * sum_i (H_{i,native} - H_{i,short})          (H2, only if H1 passes)
```
Every prompt contributes **exactly once**. **Bootstrap = unique-prompt bootstrap**: resample the 80
unique prompt records (each already rater-averaged) with replacement, B = 10000, namespace
`LISTENING-STUDY|HUMAN-BOOTSTRAP|V1.2|2026-08-31`, 95% percentile. **Gates:** H1 PASS iff
`lower95(A_native) > 0`; H2 (only if H1) PASS iff `lower95(J_H) > 0`. Prompt is the ONLY inferential
sampling unit. Frozen in `scripts/research/listening_analyze.py` (fails closed unless it sees 80 unique
prompts, all with both durations, 18 two-rater + 62 one-rater).

## Fixed-panel target retained

Estimand: *mean prompt-level preference estimated from the assigned members of this fixed six-listener
panel over the frozen 80-prompt battery.* No generalization to an expert-listener population (the
naive-generalization null inflates Type-I to 0.18–0.23 for every estimator). Wording: "in our
six-listener expert panel".

## Calibration + power (exact deployed D3 estimator)

`scripts/research/listening_power_sim_v3.py` (`configs/research/listening_study_power_v3.json`,
self-sha `02d5c2b6…`) implements D3 exactly (80 unique prompts, 62×1 + 18×2 raters, average within
prompt, then unique-prompt bootstrap) and compares estimators:

| | fixed-panel Type-I (H1, nominal .025) | J_H hierarchical null (H1∧H2) | power H1 anchor / conserv | power H1∧H2 anchor / conserv |
|---|---|---|---|---|
| **unique-prompt (PRIMARY)** | **0.019** | **0.026** | 0.982 / 0.816 | 0.641 / 0.256 |
| listener-stratified (v1.1) | 0.039 | 0.038 | 0.996 / 0.909 | 0.725 / 0.342 |
| pooled-all-ratings | 0.021 | 0.023 | 0.987 / 0.833 | 0.694 / 0.272 |

The unique-prompt estimator is **well-calibrated** (H1 Type-I 0.019, J-null 0.026, both ≤ nominal) and
is the **correct estimand** (no bridge over-weighting). It cedes a small amount of power to the v1.1
stratified estimator — but that "gain" is partly an artifact of the stratified estimator's Type-I
inflation and bridge double-weighting. Per the fixed criterion (**calibration + correct estimand over a
small power gain**) the unique-prompt estimator is adopted. Type-I 0.019 is comfortably below the
~0.03–0.035 pre-registered STOP threshold; H1 power ≥ 0.82 (conservative) / 0.98 (anchor); H2 remains
the harder endpoint (0.64 anchor / 0.26 conservative), a null H2 is inconclusive not a clean negative.

## Bridge role (unchanged design, reused two ways)

* **Primary:** the two bridge ratings are averaged WITHIN prompt before H1/H2 — no double weight.
* **Secondary reliability (non-gating):** signed inter-rater agreement, exact/within-1-category
  agreement, disagreement distribution on the 18 bridge prompts × 2 durations. No reliability statistic
  gates the primary result; no new participant-exclusion rule is added.

## Sensitivities (descriptive, non-gating; cannot rescue a failed gate)

v1.1 listener-stratified estimator; pooled-all-ratings estimator; **leave-one-listener-out** (drop a
listener's ratings, retain a prompt if ≥1 assigned rating remains, report resulting N, recompute the
unique-prompt estimator, **no imputation**); bridge agreement. All reported alongside the primary; none
can rescue a failed H1/H2.

## Documentation fix

`receiver/google_apps_script/README.md` payload contract updated to the actual v1.2 payload (removed the
stale `type` field; added `completed_A/completed_B`). Receiver code was already independent of `type`.

## Regenerated artifacts

Only `study_version` and `protocol_hash` change in the public manifests; **assignment_hash, trial order,
A/B sides, and audio bytes are unchanged** (the assignment salt and render map are version-independent).
`public_bundle_sha256` changes because it covers the version/protocol fields. The audio bundle is
rebuilt and verified byte-identical (copy SHA256 unchanged). Validation and deployment docs updated.
Private key remains gitignored.
