# Blinded Expert Listening Study — FROZEN PROTOCOL

**Study version:** `LSTUDY-2026-08-31-v1`
**Type:** post-result, prospectively frozen perceptual **corroboration**. NOT an independent
preregistration of the original discovery. Frozen BEFORE any human response is collected.
**Status at freeze:** MANUSCRIPT FROZEN · HUMAN PANEL PREPARED, NOT LAUNCHED · GPU CLOSED.

This document fixes the endpoints, gates, design, response scale, randomization, loudness
procedure, catch rules, seeds, and analysis BEFORE recruitment. No human outcome may change any
rule here. All generation is reused byte-for-byte; **no new WAV is generated** and **no GPU** is used.

---

## 1. Question and evidence hierarchy

The automatic pipeline (CLAP + Human-CLAP + KL + PANN + FD/FAD + FineLAP frame-level) established a
recovered-over-pruned **native-duration** semantic advantage, prospectively replicated at a second
pruning severity, and a positive **duration interaction** (advantage larger at 10.24 s than 3.84 s).
This study asks whether that advantage and its temporal amplification are **perceptually observable**
to blinded expert listeners. It is corroboration, not rediscovery.

**Primary human target = the severity-1 (Arm-D lineage) temporal interaction**, chosen (before any
human data) because severity 1 has perceptual headroom: the automatic recovered−pruned contrast was
≈ null at 3.84 s and positive at 10.24 s. Severity 2 shows a strong recovered advantage at BOTH
durations (higher risk of a perceptual ceiling for an interaction endpoint) and is therefore used only
as a SECONDARY native-duration corroboration (§6). This allocation is fixed and may not be revised
after human responses exist.

---

## 2. Panel, design, and workload (D1)

Six audio-experienced listeners (`P01…P06`), anonymous. Hard burden budget ≤ 15–20 min / listener.

The design was selected by a pre-data CPU Monte-Carlo power simulation
(`scripts/research/listening_power_sim.py`; `configs/research/listening_study_power.json`), which
compared, under a matched 160-judgment severity-1 budget:

* **D1** — 80 sev-1 prompts, **1 rater/prompt**, both durations (maximize prompt diversity);
* **D2** — 40 sev-1 prompts, **2 raters/prompt**, both durations (rater replication).

Because prompt is the statistical unit and inference is a paired **prompt** bootstrap, between-prompt
variance dominates and **D1 has strictly higher power than D2 in every simulated scenario and on both
endpoints**. **D1 is adopted.**

Selected power (D1, central dispersion σ_prompt=0.6, σ_rater=0.8 on the −2..+2 scale):

| Endpoint | MDE₈₀ | anchor power (μ_native=0.5) | conservative |
|---|---|---|---|
| `A_native` (H1) | μ_native ≈ **0.35** | 0.99 | 0.84 (μ=0.35) |
| `J_H` (H1∧H2) | Δ ≈ **0.45** | 0.63 | 0.26 (Δ=0.20) |

Honest reading fixed here: **H1 (native advantage) is well powered**; **H2 (`J_H`, the human duration
interaction) is powered only for a fairly large interaction (Δ≥0.45)** — a null H2 is *inconclusive*,
not a clean negative. Severity-2 secondary `A_native_2` at N=36 single-rater has power 0.80 at μ=0.5
and 0.99 at the large sev-2 effect.

Per-listener workload (D1): sev-1 = 2 × (13 or 14) trials; sev-2 = 6; catch = 3; **total 35–37**.
Native trials require ≈ 20.5 s of audio before a response, short ≈ 7.7 s. Estimated completion time is
verified in `scripts/research/listening_study_validate.py` to be ≤ 20 min.

---

## 3. Stimuli (existing audio only)

Severity 1 (Arm-D 80 ytids): `p1_recovered` and `p1_pruned_ema_reconstructed` at **3.84 s** (V1.1 job)
and **10.24 s** (Arm-D job). Within-duration pairs share the common x_T seed (verified). Severity 2:
`recovered2` and `pruned2_A` at 10.24 s (xsev job), from the 110 outcome-blind FineLAP-eligible prompts.
Every source WAV, its SHA256, sample rate, duration, integrated loudness and peak are frozen in
`configs/research/listening_study_inventory.json` (self-sha `34f39ae7…`; 540 stimuli, 0 problems).

---

## 4. Response scale (comparative)

Every trial shows the text prompt and two clips **A** and **B** (identity hidden). Two questions:

1. **Text relevance (PRIMARY):** *Which clip better matches the text prompt?*
2. **Overall acoustic quality (SECONDARY / descriptive, no gate):** *Which has better sound quality?*

Both on the same five-level comparative scale:
`A much better · A slightly better · About the same · B slightly better · B much better`.
After unblinding, encode as a signed score `{-2,-1,0,+1,+2}` with **positive = RECOVERED preferred**
(the A/B→system mapping is in the private key). Quality has **no gate** and cannot rescue relevance.
No post-hoc change of encoding is permitted.

---

## 5. Primary human estimands (severity 1)

Per prompt/duration, signed relevance preference `H_short_i`, `H_native_i` (recovered positive; average
raters within prompt if the design ever has >1 rater/prompt — D1 has 1). Prompt is the unit. Paired
**prompt** bootstrap, **B = 10000**, percentile 95%, frozen seed namespace
`LISTENING-STUDY|HUMAN-BOOTSTRAP|2026-08-31` (resolved to a PCG64 seed at analysis time; no human data
touched before then). No individual click is treated as an independent observation.

* **Gate H1:** `A_native = mean_i H_native_i`; **PASS iff `lower95(A_native) > 0`.**
* **Gate H2 (only if H1 passes):** `J_H = mean_i (H_native_i − H_short_i)`; **PASS iff `lower95(J_H) > 0`.**

Pre-specified sensitivity (no gate role): leave-one-rater-out; rater-stratified descriptive; all six
raters retained (see §9). No threshold tuning, no encoding change.

**Claim ladder (fixed):**
* H1 ∧ H2 pass → *"The native-duration recovered advantage and its temporal amplification are
  corroborated by blinded human relevance judgments in the severity-1 lineage."*
* H1 pass, H2 fail → *"The native recovered advantage is perceptually corroborated; the human duration
  interaction remains unresolved."*
* H1 fail → *"The small expert panel is inconclusive for the native semantic advantage."*
* Preference favours pruned → report disagreement with the automatic metrics, without rescue.

---

## 6. Severity-2 secondary

`A_native_2 = mean signed human relevance preference` (recovered2 − pruned2_A) at 10.24 s over **36**
prompts, **outcome-blind** hash-selected from the 110 FineLAP-eligible sev-2 prompts using ONLY frozen
identifiers + the committed salt `LISTENING-STUDY|SEV2-SELECT|2026-08-31` (no CLAP/FineLAP/HC/PANN/KL
value influences selection). SECONDARY only: does the large native recovered advantage at stronger
pruning appear in blinded human judgments? Severity-2 `J_H` is **not** co-primary (the pre-data power
analysis gives no compelling reason to promote it, and severity-2 has ceiling risk for an interaction).

---

## 7. Loudness control (listening copies only)

Human relevance must not be driven by loudness (V1.1 showed recovered is markedly louder than pruned:
RMS 0.128 vs 0.026). Listening copies are loudness-matched; **original scientific WAVs are never
modified**. Procedure (frozen, ITU-R BS.1770-4 via `pyloudnorm`):

* single fixed integrated-loudness target **−36.0 LUFS**, one fixed gain per stimulus; **no limiting,
  no compression, no condition-specific target, no post-outcome adjustment**;
* peak-safety: post-gain **sample peak ≤ −1.0 dBFS**.

**−23 LUFS (the initial proposal) is infeasible** for this material: the clips are quiet in integrated
loudness but carry high transient peaks (crest up to ~34 dB, driven by near-silent failed-pruned
generations that must NOT be excluded, since loudness correlates with the pruned-failure effect). The
most binding stimulus fixes the feasible target at ≤ −35.06 LUFS for a −1 dBFS ceiling; **−36 LUFS is
the frozen conservative choice → 0 unsafe over all 540 stimuli** (`listening_study_inventory.json`,
`feasible_all=true`). Absolute level is set by the participant's fixed comfortable volume; only the
matched *relative* loudness matters. Original and listening-copy SHA256 are recorded by the bundle
builder. The manuscript, if authorized, must state that the comparison used loudness-matched copies.

---

## 8. Randomization / blinding

`scripts/research/build_listening_assignments.py` (frozen namespace
`LISTENING-STUDY|ASSIGN|2026-08-31`) produces six fixed assignments with:

* model identity, severity, and duration hidden from the participant (present only in the private key);
* randomized A/B side per trial, recovered-as-A counterbalanced globally **and** ~within each
  participant × stratum (sev1-native, sev1-short, sev2-native);
* trial order randomized; the short and native versions of a sev-1 prompt go to the **same** rater,
  are **non-adjacent**, and are separated by **≥ 5 intervening trials** where feasible (verified);
* anonymized hashed audio filenames (secret salt in the private key); no visible filename, URL, label,
  or caption contains `recovered`, `pruned`, `dense`, or a severity name.

Artifacts: **PUBLIC** blinded manifests `listening_study/public_manifests/P0{1..6}.json`;
**PRIVATE** unblinding key `configs/research/listening_study_assignments_private.json`
(salt + A/B→system map; **gitignored, never deployed**).

---

## 9. Catch / reliability trials (3 per listener)

Outcome-independent, built from prompts the listener never sees experimentally:

1. **identical-audio (native)** — A and B are the same recovered-native clip → expect *About the same*
   (answered on both scales; covers the identical-relevance and identical-quality checks);
2. **identical-audio (short)** — a second identical pair (different clip) for reliability;
3. **matched-vs-unrelated (native)** — a clip under its own caption vs a clip from a very different
   caption, both recovered-native, side randomized → expect a strong preference for the matched side.
   Constructed from caption↔generation metadata only (no generated outcome score).

**Frozen retention rule:** all six raters are retained in the primary analysis unless the study is
incomplete or technically invalid. Catch performance is reported descriptively. A single **gross-failure**
sensitivity analysis (re-running primary endpoints excluding one rater) is permitted **only** if that
rater meets the pre-frozen gross-failure criterion: **both** identical trials answered with |score| = 2
(claiming a large A/B difference between identical audio) **OR** the matched-vs-unrelated trial answered
with a preference for the *unrelated* side of magnitude ≥ 1. One catch failure never auto-deletes a
rater. The criterion is frozen here, before recruitment.

---

## 10–13. Platform, delivery, privacy (summary; full detail in deployment doc)

Static bundle `listening_study/` (desktop Chrome/Firefox/Safari; headphones). Participant opens
`…/?p=P0X` and receives only their frozen assignment. One optional replay per sample; keyboard
shortcuts; progress indicator; no autoplay; no cookies/analytics; trial timing recorded for QC; no
IP/location/fingerprinting. Results are POSTed to a configurable HTTPS endpoint (Apps Script receiver
`receiver/google_apps_script/`) that emails the anonymous JSON; recipient configured server-side, never
in the client. Payload = study/protocol identifiers, participant code, assignment hash, responses,
timings, catch responses, completion timestamp, client UUID — **no name/email/IP**. Offline fallback
(download JSON / copy to clipboard) is always available; `mailto:` is never the only method. Minimal
consent screen (anonymous, ~15–20 min, voluntary, may stop, 18+, no personal/health data). We do NOT
claim ethics approval is unnecessary — the corresponding author/institution determines any
ethics/exemption requirement before recruitment.

---

## 14–15. Validation and freeze order

Before any real listener: `scripts/research/listening_study_validate.py` verifies audio paths, hashes,
A/B masking, no identity leakage, per-listener counts, A/B balance, duration pairing, repeated-prompt
separation, catch trials, loudness normalization, result JSON shape, endpoint (synthetic only), offline
fallback, and estimated completion time. Freeze chronology: power sim → design → this protocol →
loudness → assignments → manifests → **freeze protocol + hashes → commit/push → THEN build the
loudness-normalized stimulus bundle from the frozen manifests → internal QA → return for GO**. No real
participant begins before the freeze commit. This turn stops at **READY FOR PARTICIPANT LAUNCH**.
