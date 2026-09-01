# Listening Study — PRE-PARTICIPANT AMENDMENT v1 → v1.1

**Study version:** `LSTUDY-2026-08-31-v1.1` (amends `LSTUDY-2026-08-31-v1`, frozen at commit `aa906c2`).
**Detected by:** final pre-launch audit (2026-08-31, 23:40 MVD). **No human responses existed** at the
time of these corrections; **no participant had started**; the manuscript remained frozen; no GPU, no
new generation. v1 artifacts remain immutable provenance in git history (`aa906c2`/`95350fd`); this
document + the regenerated artifacts supersede them for launch. Base protocol
`docs/listening_study_protocol.md` (v1) is unchanged and read together with this amendment; the v1.1
`protocol_hash = sha256(v1_protocol_bytes ++ this_amendment_bytes)`.

The governing statistical inference below (fixed-panel estimand, listener-stratified estimator, H1/H2
gates) supersedes the corresponding v1 §2/§5 text where they differ.

---

## A1. Persistent-rater power model (was: independent per-observation noise)

**Why:** v1's power sim drew `sigma_r·N(0,1)` independently per prompt×duration, ignoring that only
SIX listeners each rate ~35 trials, so persistent listener heterogeneity and within-listener
correlation were unmodeled.
**New:** `scripts/research/listening_power_sim_v2.py` models prompt×duration latent, **listener
persistent intercept**, **listener×duration interaction**, prompt×listener residual, correlated
short/native, a listener side-A bias (A/B counterbalanced), and ordinal discretization; runs null +
positive; both bootstraps; `configs/research/listening_study_power_v2.json` (self-sha `f5ce4bd7…`).
**Consequence:** the design/inference decisions below are grounded in the realistic structure.

## A2. Design D1 → **D3 (partially crossed)**

**Old (v1):** D1 — 80 sev-1 prompts × 1 rater × both durations (160) + 36 sev-2 native × 1 rater.
**New (v1.1):** **D3** — 80 sev-1 prompts × 1 primary rater × both durations (160) **+ 18 outcome-blind
hash-selected bridge prompts each given a SECOND rater at both durations (36)**; **NO severity-2 human
arm**. Bridge selection namespace `LISTENING-STUDY|BRIDGE-SELECT|2026-08-31` (IDs only).
**Why:** the scientific priority order is sev1 `A_native` > sev1 `J_H` > inter-rater interpretability >
sev2 corroboration, and sev2 is expendable if the trials better serve the primary. D3 strictly adds
sev-1 information and yields real inter-rater data (directly addressing the persistent-rater threat),
at the cost of the expendable sev2 human arm (already corroborated by the automatic multi-metric +
FineLAP frame-level cross-severity evidence). Simulation confirms **D3 dominates D1 and D2 on power at
every effect size** (fixed-panel, stratified):

| Effect | H1 D1 / D2 / **D3** | H1∧H2 D1 / D2 / **D3** |
|---|---|---|
| anchor μ_n=0.5 | 0.993 / 0.975 / **0.998** | 0.664 / 0.642 / **0.743** |
| conservative μ_n=0.35 | 0.868 / 0.825 / **0.901** | 0.283 / 0.295 / **0.343** |
| stressed heterogeneity | 0.989 / 0.972 / **0.994** | 0.652 / 0.618 / **0.722** |

D2 additionally shows an inflated stratified Type-I (0.056) from tiny per-listener strata. **Consequence:**
better primary power, plus inter-rater reliability + robustness checks; human evidence is now sev-1 only.

## A3. Inference target = **FIXED PANEL** (was: unspecified / implicitly generalizing)

**New:** the estimand is the **average blinded preference across THESE six expert listeners** over the
frozen prompt population. Listener sampling uncertainty is **not** estimated (six listeners cannot
support it). Maximum wording later: *"in our six-listener expert panel"* — never *"human listeners
generally"*. **Why:** simulation shows that treating the panel as generalizing (uncentered listener
tendencies) inflates Type-I to **0.15–0.23** for every design and both bootstraps; conditioning on the
fixed panel (centered null) is calibrated. **Consequence:** honest, defensible claim scope.

## A4. Primary bootstrap = **listener-stratified prompt bootstrap** (pooled = sensitivity)

**Old (v1):** pooled prompt bootstrap, all 80 prompt observations exchangeable.
**New (v1.1):** point estimator = **equal-listener-weight mean**; CI = **listener-stratified prompt
bootstrap** (resample prompts *within* each listener, equal listener weight), B = 10000, seed namespace
`LISTENING-STUDY|HUMAN-BOOTSTRAP|2026-08-31`, percentile 95%. The **pooled** prompt bootstrap is
retained as a **pre-specified sensitivity** (no gate). Frozen estimator in
`scripts/research/listening_analyze.py` (self-test: fixed-panel Type-I 0.020, power 1.0 at μ=0.5).
**Why:** the stratified estimator is the fixed-panel estimand (equal listener weight, robust to unequal
prompt counts). Calibration note (documented, not hidden): under the fixed-panel centered null the
stratified bootstrap is **mildly liberal (~0.035–0.041)** while pooled is conservative (~0.014–0.018);
both control the J_H (H2) gate near nominal (~0.03). Borderline H1/H2 results will be read with this in
mind and the pooled sensitivity reported alongside.

**Frozen gates (unchanged form; scale unchanged: signed −2..+2, positive = recovered):**
* **H1:** `A_native = mean_L mean_{i∈L,native} recovered_score`; PASS iff `lower95(A_native) > 0`.
* **H2 (only if H1):** per listener `J^L = mean_i(native−short)` over both-duration prompts;
  `J_H = mean_L J^L`; PASS iff `lower95(J_H) > 0`.
* Pre-specified, no gate: leave-one-listener-out (A_native, J_H); bridge inter-rater agreement; pooled
  sensitivity. No re-encoding, no threshold tuning.

## A5. Public manifests: remove catch/type leakage

**Old (v1):** public manifests exposed `"type": "catch"`.
**New (v1.1):** public trials contain ONLY `trial_id, prompt_text, audio_A, audio_B`. Catch identity,
severity, duration, and system live solely in the private key; the submitted payload does not carry
`type`. Verified by static scan (no `catch|recovered|pruned|dense|realref|_alt10s` outside prompt text).
**Why:** unnecessary information leakage. **Consequence:** the browser/participant cannot identify catch
trials.

## A6. Matched-vs-unrelated catch: generated clip → **real AudioCaps references**

**Old (v1):** matched-vs-unrelated attention catch used a generated recovered-native clip, which can
itself fail to realize its caption (a preference for the "unrelated" clip could be scientifically
defensible), so it was weak for a gross-failure rule.
**New (v1.1):** matched = a **real** AudioCaps clip under its own human caption; unrelated = a **real**
AudioCaps clip with **disjoint** AudioSet labels, under the matched caption; side randomized. Pool built
outcome-independently (`scripts/research/build_realref_catch_pool.py`; 48 clips resampled 16 kHz, padded
to native length, all peak-safe at −36 LUFS; `configs/research/listening_study_realref_pool.json`).
**Why:** a real clip genuinely realizes its caption, giving a robust attention control.
**Consequence:** this catch **may** enter the gross-failure criterion (below); the two A/A identical
catches are retained.

**Frozen gross-failure criterion (v1.1, before recruitment):** a rater is flagged grossly unreliable
iff **both** identical catches are answered with |relevance| = 2 (a large A/B difference between
identical audio) **OR** the real-reference matched-vs-unrelated catch is answered toward the *unrelated*
side with magnitude ≥ 1. All six raters are retained in the primary analysis regardless; one flagged
rater triggers only a single pre-specified leave-that-rater-out **sensitivity** analysis. No rater is
excluded merely for disagreeing with a generated clip.

## A7. Client: require both clips played in full before answering

**Old (v1):** responses could be given without listening.
**New (v1.1):** on every trial the relevance and quality controls are **locked until both A and B have
played to completion** at least once; one optional replay per side remains. Playback completion and
counts are recorded in the payload (`completed_A`, `completed_B`, `plays_A`, `plays_B`).
**Why:** ensures judgments are based on hearing both clips. **Consequence:** the mandatory single full
pass of both clips is already inside the ≤20-min budget (see below).

## A8. Loudness-outlier pair audit (adds evidence for "does not favour recovered")

**New:** `scripts/research/listening_loudness_pair_audit.py` computes, per experimental pair, the
recovered−pruned integrated-loudness difference of the LISTENING COPIES by stratum, and STOPs if the
signed mean exceeds ±0.5 dB in any stratum. See `configs/research/listening_loudness_pair_audit.json`
for the frozen numbers and verdict. **Why:** the v1 "does not favour recovered" statement must be
demonstrated, not assumed.

---

## Regenerated artifacts (v1.1)

Public manifests, assignment hashes, public bundle hash, validation, and deployment docs are
regenerated. The private key remains gitignored. Time budget re-verified after A7. v1 remains immutable
provenance at `aa906c2`/`95350fd`.
