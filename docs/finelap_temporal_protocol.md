# Part A — FineLAP Temporal-Semantic Recovery Profile — FROZEN PROTOCOL

**STATUS: PROSPECTIVELY FROZEN POST-RESULT DIAGNOSTIC. Frozen BEFORE any FineLAP score on the
generated battery was inspected.** CPU only, 0 GPU, 0 new generation. This analysis is motivated by
already-observed global duration interactions (Arm-D, xsev CASE C) and is therefore an explanatory
**post-result** diagnostic, NOT an independent preregistered confirmation of the original hypothesis
(see §A5). Authorized by supervisor reopening of commit `f18d148` (2026-08-31).

Chronology guard: the endpoints, gates, windows, seed, eligibility, and outcome branches in this
document are fixed here; the FineLAP frame scores on the recovered/pruned generations are computed
only AFTER this document and the eligibility manifests are committed.

---

## A0. Exact FineLAP time-axis — RESOLVED (PASS)

Audit: `scripts/research/finelap_geometry_audit.py`; result `artifacts/finelap_temporal/geometry_audit.json`
(VERDICT PASS). Model: pinned local `AndreasXi/FineLAP`, weights sha256
`13b9646c9f9d48513c0145bed75e654179e83f0fd8d49ed4ffc5d6b8f3353fb4`. Established by code inspection AND
empirical verification on CPU (torch 1.13.1):

1. **Waveform samples consumed** — a 10.24-s clip at 16 kHz produces **1022 Kaldi fbank frames**
   (`frame_shift=10 ms`, 25-ms Hanning window, `snip_edges=True`). Our AudioLDM native WAVs are
   163 872 samples → also 1022 mel frames (empirically confirmed). Only the final ~5–7 ms
   (< one hop, Kaldi snip-edges tail) is uncovered.
2. **Drop/pad** — `load_audio` forces `target_len=1024`: 1022 < 1024 → **2 zero-frames appended**
   (~20 ms of silence at the end). **No truncation, no meaningful content dropped.**
3. **Feature (mel) frame count before the encoder** — **1024** (1022 real + 2 pad).
4. **Dense output frame count** — **64** (empirically `get_dense_audio_embeds` → shape (1, 64, D)).
   EAT patch grid = 1024/16 × 128/16 = 64 (time) × 8 (freq) = 512 patches (+1 CLS); `encode_audio`
   drops CLS, then `reshape(B, 64, 8, D).mean(dim=2)`.
5. **Stride / frame-center mapping** — `PatchEmbed_new` conv (k=s=16) then `flatten(2)` is
   **time-major, freq-minor** (patch index `p = h·8 + w`); empirically proven by perturbing time-block
   5 and observing exactly patches 40–47 change. Therefore `reshape(64, 8).mean` averages the **8
   frequency bands per time patch** → 64 **clean time frames**. Each frame spans 16 mel-frames =
   **0.16 s**. Frame `i` covers `[0.16·i, 0.16·(i+1))`; center ≈ `0.16·i + 0.08 s`.
6. **0.15625 vs 0.16** — 0.15625 = 10.0/64 assumes a 10.0-s clip and is **NOT applicable**. The real
   preprocessing pads to 1024 mel frames (10.24 s) → **0.16 s/frame**, exact for our 10.24-s WAVs
   (10.24 = 64 × 0.16).
7. **Contextualization** — the EAT encoder is 12 `AltBlock`s of **full bidirectional self-attention,
   no causal/window mask**; empirically, perturbing ONLY the late audio changes early-frame
   embeddings. **Every frame embedding depends on the entire temporal sequence.** Frame scores are
   therefore *contextualized frame-level audio–phrase grounding evidence*, NOT independent local
   activations.

**Consequence for A2:** 3.84 / 0.16 = **24 exactly** → the early/late boundary lands on an exact patch
boundary; no frame straddles it. **EARLY = frames 0–23 (24 frames), LATE = frames 24–63 (40 frames).**

**Permitted terminology (frozen):** "frame-level semantic evidence", "frame-level audio–phrase
similarity", "grounding score". **Forbidden:** "calibrated event probability", "causal local
activation", "ground-truth event presence probability". `tau = 0.5` is the natural sigmoid/BCE decision
threshold (model trained with frame-level BCE grounding) but calibration is NOT assumed.

---

## A1. Outcome-blind eligibility manifests — FROZEN

Built by `scripts/research/build_finelap_eligibility.py` (deterministic; `--check` reproduces
`manifest_sha256`). Uses ONLY: frozen generation caption (selection manifest), frozen AudioSet
ground-truth labels (`audiocaps_test_label.json`; all 80+192 ytids present), and the frozen strict
caption→event alias rule (`configs/research/event_synonyms_strict.json`, 527 classes, identical to
`build_v4_manifests.py`). **Zero outcome-dependent filtering.**

Rule: `requested_events(prompt) = { m ∈ ground_truth_labels(ytid) : some strict alias of m matches the
caption via \b word-boundary, case-insensitive }`; `eligible ⇔ ≥1 requested event`. Independent unit =
prompt; event-level quantities are averaged **within prompt** before inference. FineLAP scoring phrase
(frozen) = `display_name.split(",")[0].strip()`.

| Severity | Battery | Considered | **Eligible prompts** | Excluded | **Event occurrences** | manifest_sha256 |
|---|---|---:|---:|---:|---:|---|
| sev-1 | Arm-D 80 @10.24 s | 80 | **49** | 31 | **63** | `1e3dc770…` |
| sev-2 | xsev 192 @10.24 s | 192 | **110** | 82 | **131** | `b543b481…` |

Independently reproduces the design-review feasibility counts (49/63; 110/131) exactly, without
forcing them. Exclusion reason (all excluded prompts): no ground-truth AudioSet label matched by any
strict caption alias. Manifests: `configs/research/finelap_eligibility_sev{1,2}.json`. All 428 native
WAVs (sev-1 49×{recovered, pruned_A}; sev-2 110×{recovered, pruned_A, pruned_B}) exist and match their
generation-manifest SHA256 (0 mismatch, 0 missing).

---

## A2. Primary temporal endpoint — FROZEN

Windows (A0): EARLY = frames 0–23, LATE = frames 24–63. Baselines: **sev-2 primary = pruned2_A**;
**pruned2_B = seam sensitivity only** (§A3); sev-1 = Arm-D pruned (`p1_pruned_ema_reconstructed`).

Per eligible prompt `p`, over eligible events `e` and window `W`, with recovered `rec` and pruned
`pru`:

```
mean_W(sys, p) = mean_e [ mean_{t∈W} score(sys, p, e, t) ]
D_early(p) = mean_EARLY(rec, p) − mean_EARLY(pru, p)
D_late(p)  = mean_LATE(rec, p)  − mean_LATE(pru, p)
T(p)       = D_late(p) − D_early(p)
```

Severity estimands = mean over eligible prompts: `T_s`, `D_early_s`, `D_late_s`. Inference = **paired
prompt bootstrap**, resampling the eligible-prompt set with replacement, **B = 10000**, percentile 95%
CI, frozen seed namespace `FINELAP-TEMPORAL-RECOVERY|BOOTSTRAP|2026-08-31` → **PCG64(1698610719)**.

* **PRIMARY diagnostic statistic = `T_2`** (severity 2, pruned2_A). **Gate: `lower_CI95(T_2) > 0`.**
  Severity 2 is primary because it has the larger outcome-blind eligible n (110) and is the
  independently replicated severity where the duration interaction J was already prospectively
  resolved. This avoids inventing two co-primary tests after the fact.
* **Cross-severity replication support = `T_1` point > 0** (report full CI). Severity 1 is a
  pre-specified **directional** support condition; it is NOT called statistically confirmed if its CI
  crosses zero (eligible n=49).
* **INTERPRETATION GUARD.** A positive `T` means only that the recovered−pruned FineLAP contrast is
  larger late than early **within the same long generation**. It does NOT by itself establish a
  positive late semantic advantage. To promote wording such as "late temporal semantic gain" /
  "late recovered advantage", ALSO require **`D_late_2 > 0`** (report its CI; stronger if
  `lower_CI95(D_late_2) > 0`). If `T_2 > 0` arises because `D_early_2` is strongly negative while
  `D_late_2` is merely less negative, state exactly that and do NOT call it semantic recovery.
* Do NOT claim that cropping a 10.24-s generation reproduces the separately generated 3.84-s operating
  point.

---

## A3. Secondary endpoints — FROZEN (descriptive/explanatory only; no fishing)

Per system/prompt, averaged within prompt over eligible events, on the full 64-frame window:

1. **semantic mass** = mean frame score over all 64 frames;
2. **occupancy** = fraction of frames with score ≥ `tau=0.5`;
3. **quarter coverage** = over four fixed 2.56-s quarters (frames 0–15, 16–31, 32–47, 48–63), the
   proportion containing ≥1 frame with score ≥ 0.5;
4. **peak evidence** = maximum frame score;
5. **D_early and D_late component contrasts** (from A2);
6. **sev-2 pruned2_B seam sensitivity** — repeat `T_2, D_early_2, D_late_2` as `recovered2 − pruned2_B`;
   report only. **B never rescues a failed A′/pruned2_A conclusion.**
7. **sev-1 dense and real-reference distributions** as DESCRIPTIVE anchors only, IF available with
   clean provenance.

`tau=0.5` is the sigmoid/BCE decision geometry, frozen before these outputs existed; it is NOT a
calibrated probability threshold. No threshold sweeps are added after seeing results.

Coverage/persistence interpretation rule: call the gain **coverage/persistence-dominant** only if
occupancy or quarter coverage improves while peak evidence is approximately unchanged or materially
smaller in effect; otherwise report a generic frame-level semantic gain. No onset/timing-accuracy claim
is permitted (AudioCaps captions do not specify target timestamps).

---

## A4. Outcome branches — FROZEN BEFORE SCORING

* **A1 — Late allocation supported.** Require `lower95(T_2) > 0` AND `T_1 point > 0` AND
  `D_late_2 point > 0`. Stronger wording only if `lower95(D_late_2) > 0`. Allowed statement: *"FineLAP
  evidence indicates that the recovered-minus-pruned semantic contrast is disproportionately expressed
  later in the long generation, with the same directional pattern at both pruning severities."* NOT a
  causal-mechanism claim.
* **A2 — Frame-level gain but no late redistribution.** If full-window semantic mass / occupancy
  improves but `T_2` fails: *"recovery improves FineLAP-localized requested-event evidence, but the
  long-duration advantage is not explained by preferential late allocation."*
* **A3 — FineLAP null / disagreement.** If a clip-level recovery advantage exists but the frame-level
  endpoints do not: *"the native-duration advantage is not explained by localized requested-event
  evidence under FineLAP."* A scientifically valid negative. **No new temporal metric may be introduced
  to rescue A3.**

---

## A5. Validity / novelty boundary — FROZEN

FineLAP is a legitimate temporal grounding evaluator (official implementation exposes frame-level
audio–phrase similarities; trained with frame-level grounding supervision). However this analysis is a
**prospectively frozen POST-RESULT diagnostic**, motivated by already-observed global duration
interactions. **Do NOT describe it as an independent preregistered confirmation of the original
hypothesis.** If positive, it may become an explanatory main-paper analysis, but the chronology
(post-result) must remain explicit. Mechanism attribution remains blocked (the dense-FT control is
unavailable): no "pruning/recovery causes …" language.

---

## Execution

Score existing native WAVs with FineLAP on CPU (`scripts/research/finelap_temporal_score.py`), compute
the frozen statistics (`scripts/research/finelap_temporal_verdict.py`), persist
`configs/research/finelap_temporal_result.json`. No GPU, no new generation, no manuscript work.
