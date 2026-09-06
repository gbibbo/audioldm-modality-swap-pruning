# Scientific report — response to the second review (methodological weaknesses)

**Purpose.** Every new number, with its 95% interval and provenance, organized by the reviewer's seven points, so the
manuscript prose can be written from it. Numbers formatted directly from the committed artifacts (verifiable, not
retyped). **Nothing in this report is prose for the paper; it is the evidence layer.**

**Scope of the new work (this round).** Two experimental additions — the pruned symmetric control (`longft`) and the
dense 2×2 (`denseft_short`, `denseft_native`) — plus two CPU re-analyses (severity-1 heterogeneity, hip-hop dense
anchors at full coverage). The earlier follow-ups the review was reacting to (E3, E1c, E5, E6, E7, E8, B) are quoted
where a point needs them, marked *[already reported]*.

**Conventions.** CLAP = fused CLAP cosine, `laion/clap-htsat-fused` rev `365dea6e`; unit = prompt; percentile bootstrap
B=10⁴; intervals are 95% CIs. R(d) = CLAP(fine-tuned) − CLAP(base) at requested duration d; J = R(10.24 s) − R(3.84 s);
ΔJ = J(trained@10.24) − J(trained@3.84), paired per prompt. SESOI = 0.025.

**Compute / provenance.** Three T4 fine-tune+eval jobs (`r2-longft`, `r2-denseft-s`, `r2-denseft-n`) + CPU scoring.
Protocol `docs/reviewer2_followup_ext.md` (frozen + sha256 before any output). Results: `configs/research/r2_EXT2x2_result.json`,
`configs/research/r2_posthoc_pooled_anchors.json`, `configs/research/r2_{E3,E1c,E7,E8,B}_result.json`. The campaign hit
`OUT_OF_FUNDS` at the 20-cr ceiling; `denseft_short` at 10.24 s is n=170 of 192 (22 WAVs uncompleted; result already
resolved). Full audit of trustworthiness in `docs/reviewer2_followup_ext_results.md` (§Audit).

---

## Point 1 — the 3.84 s intervention is not a clean specialization test; add the symmetric 20 000-step control at 10.24 s

**What was asked.** Fine-tune the same pruned checkpoint for the same 20 000 steps at **10.24 s** and compare J(long FT)
against J(short FT); until that control exists, soften "contradicting" to "does not support".

**What we ran (NEW).** `longft` = the severity-2 pruned checkpoint P fine-tuned 20 000 steps at 10.24 s (identical recipe
to E3 except the training duration), evaluated against P with common noise on the frozen 192 AudioCaps prompts.

| Checkpoint (base = P) | R(3.84 s) | R(10.24 s) | J |
|---|---|---|---|
| `shortft` (trained @3.84 s) *[= E3, already reported]* | +0.009 [−0.006, +0.024] | +0.075 [+0.053, +0.097] | **+0.065 [+0.044, +0.087]** |
| `longft` (trained @10.24 s) **(NEW)** | +0.017 [+0.001, +0.033] | +0.048 [+0.025, +0.070] | **+0.031 [+0.010, +0.052]** |

**Primary contrast: ΔJ = J(long FT) − J(short FT) = −0.035 [−0.064, −0.004]** (n=192, resolved negative).

Raw levels (CLAP): P 0.015 → 0.055; shortft 0.024 → 0.129; longft 0.032 → 0.103 (at 3.84 → 10.24 s).

**Reading (factual).** Both checkpoints recover more at 10.24 s than at 3.84 s (both J resolved > 0). Training at the
longer duration did **not** move the gain toward that duration; the interaction is if anything *smaller* for the
10.24 s-trained checkpoint (ΔJ resolved < 0). Specialization predicts ΔJ > 0; the data give ΔJ < 0. The reviewer's
alternative reading — "20 000 steps cannot adapt short generation, so the gain appears at 10.24 s only because that is
the one duration the U-Net handles" — is not supported: `longft`, trained at 10.24 s where the base model works, still
gains only +0.048 at 10.24 s and a *resolved* +0.017 at 3.84 s. Neither checkpoint concentrates its gain at its own
training duration. → The symmetric control **supports** the operating-point account over training-duration
specialization; the "does not support specialization" statement is now backed by a paired control, not one arm.

**Supporting reference *[already reported]*.** A public dense text-fine-tuned checkpoint (B) also gains more at 10.24 s:
G_tf(3.84)=−0.022 [−0.061, +0.017], G_tf(10.24)=+0.091 [+0.042, +0.141], J_tf=+0.113 [+0.051, +0.173] — duration-dependence
of a fine-tuning gain is not unique to post-pruning recovery (reference, not a matched control).

---

## Point 2 — no dense model at the same budget; build the 2×2 (dense and P × train@3.84 and @10.24)

**What was asked.** Since the paper now trains, run a reduced-scale 2×2: dense and pruned, each 20 000 steps at 3.84 and
10.24 s, to get the paired design the paper lacked.

**What we ran (NEW).** Dense AudioLDM-M-Full fine-tuned 20 000 steps at each duration (`denseft_short`, `denseft_native`),
evaluated against the dense baseline (CLAP 0.207 at 3.84 s, 0.354 at 10.24 s; XSEV-DENSE-192-CONTROL, CRN-paired).

| Checkpoint (base = dense) | level 3.84 / 10.24 | G(3.84 s) | G(10.24 s) | J |
|---|---|---|---|---|
| `denseft_short` (train @3.84 s) | 0.025 / 0.121 | −0.182 [−0.207, −0.157] | −0.235 [−0.271, −0.199] | −0.051 [−0.090, −0.013] |
| `denseft_native` (train @10.24 s) | 0.041 / 0.131 | −0.166 [−0.190, −0.141] | −0.223 [−0.255, −0.191] | −0.057 [−0.091, −0.023] |

**ΔJ_dense = J(dense long) − J(dense short) = −0.005 [−0.034, +0.024]** (n=170; within SESOI → training-duration-independent).

**Reading (factual, this is a NEGATIVE / limitation).** The 20 000-step full-parameter dense fine-tune **degrades** the
dense model (CLAP 0.354 → 0.131 at 10.24 s), so every G is strongly negative and the fine-tuned dense checkpoints land at
≈ the same low profile as the fine-tuned *pruned* ones. Audit (`…_ext_results.md` §Audit): the training loss is stable
(mean(last 50) 0.184–0.210, like the pruned runs), so this is not a diverged optimizer; it is the pre-declared
raw-weight-vs-EMA effect (the released baseline is EMA-averaged; our export is raw) plus zero AudioCaps headroom
(M-Full was already AudioCaps-fine-tuned 0.25 M steps). **Consequence: the dense 2×2 does not give a clean "dense
recovers similarly" analogue and must be reported as a limitation, not as the missing paired design.** It does show no
training-duration specialization in the dense model either (ΔJ_dense ≈ 0), consistent with Point 1. The matched dense
control at the released 10⁶-step budget (Singh's deleted checkpoint) remains unavailable.

**Cross-arm.** pruned-minus-dense duration response: trained@3.84 s +0.011 [−0.008, +0.029] (n=170); trained@10.24 s
−0.019 [−0.044, +0.005] (n=192) — statistically similar, unsurprising once both collapse to the same fine-tuning attractor.

---

## Point 3 — the manuscript narrates its own revision history

Presentation, no new number. The pre-registered-vs-follow-up distinction should be stated as design, not as rebuttal
chronology; no prior title should be referenced. (Handled in prose; nothing to compute.)

---

## Point 4 — "plateaus at 15.36 s" overstates the evidence

**Number *[already reported, E1c]*.** D4 = R(15.36 s) − R(10.24 s) = **+0.021 [−0.023, +0.067]** (n=96). The interval admits
a +0.067 step (≈ 40% of R). The supported statement is **"recovery does not clearly increase beyond 10.24 s"** (it rules
out a sharp peak at the training duration; it does not establish a flat plateau). More prompts cannot bound this within
±SESOI at feasible n (≈310 prompts needed), so this is a wording correction, not an experiment.

---

## Point 5 — severity-1 heterogeneity deserves reporting

**What we computed (NEW, CPU post-hoc; frozen n=80 stays the pre-specified primary).** `r2_posthoc_pooled_anchors.json`.

| Prompt set | J (severity-1 duration interaction) |
|---|---|
| original Arm-D 80 *[already reported]* | +0.044 [−0.000, +0.088] (n=80) |
| new 96 (E8) | +0.169 [+0.115, +0.222] (n=96) |
| **difference (new − original), unpaired** | **+0.124 [+0.058, +0.194]** |
| pooled n=176 *[already reported]* | +0.112 [+0.076, +0.149] |

**Reading.** The two subsets differ by a resolved +0.124, larger than the original effect. Both are outcome-blind
seeded-hash draws (original = 80 of the V1.1 96; new = first 96 of the severity-2 192 manifest, a disjoint pool with a
different salt and a 5-caption-row rule); same checkpoints, sampler, scorer, hardware. The pooled value should be
reported *with* this between-set heterogeneity, both subsets shown.

---

## Point 6 — hip-hop dense anchors cover only 64 of 127 prompts (+ PANNs/refs, presentation)

**What we computed (NEW, CPU post-hoc; the E7 job had generated dense on the 63 extension prompts too).**
`r2_posthoc_pooled_anchors.json`, pooled over all 127 hip-hop prompts.

| Duration | recovery R (n=127) *[already reported]* | dense above chance A_dense (n=127) | ρ_dense pooled (n=127) | ρ_dense frozen-64 / ext-63 |
|---|---|---|---|---|
| 3.84 s | +0.026 [+0.007, +0.044] | +0.110 [+0.091, +0.128] | **+0.106 [+0.031, +0.177]** | 0.038 / 0.172 |
| 10.24 s | +0.027 [+0.003, +0.050] | +0.108 [+0.087, +0.130] | **+0.119 [+0.015, +0.215]** | 0.020 / 0.241 |

**Reading.** The battery discriminates for the dense model at full coverage (A_dense lower bounds > SESOI). Recovery
closes **≈ 11–12% of the dense gap pooled over 127 prompts** (not the "≈ 2%" of the pre-specified 64-prompt subset; the
extension prompts behave differently, ρ ≈ 0.17–0.24). The hip-hop gain stays small and an order of magnitude below
AudioCaps; the floor objection is answered at full coverage. (PANNs-capture definition and uncited references are
presentation fixes, no number.)

---

## Point 7 — is 3.84 s out-of-distribution for the U-Net regardless of fine-tuning?

Hypothesis H_OP: short generation is out of distribution for the base U-Net irrespective of fine-tuning, and fine-tuning
gains express only where the base model works.

* Prediction "the base model's own alignment is degraded at 3.84 s beyond the scorer's crop effect" — **not supported as
  stated** *[already reported, XSEV-DENSE-192 + floor-ceiling]*: the dense model's floor-corrected duration response is
  +0.142 [+0.111, +0.172], statistically the same as real audio's crop response +0.150 [+0.133, +0.166]; the dense model
  keeps 74% of its above-chance alignment at 3.84 s vs 81% at 10.24 s (mild). Under CLAP the base model is not "broken"
  at 3.84 s.
* Prediction "every fine-tune gains ≈ 0 at 3.84 s" — **supported so far**: shortft +0.009, longft +0.017 (small),
  dense text-FT −0.022; released recovery +0.085 (small vs +0.244 at 10.24 s).
* The discriminating prediction (ΔJ ≈ 0 if H_OP, ΔJ > 0 if specialization) is delivered by Point 1: ΔJ_pruned = −0.035,
  which rejects specialization and is consistent with the weaker H_OP form (gains express at the longer operating point;
  the base model's own alignment does not explain the difference).

---

## Summary

| # | Reviewer point | New evidence | Key number | Status |
|---|---|---|---|---|
| 1 | symmetric 10.24 s control | `longft` (NEW) | ΔJ = −0.035 [−0.064, −0.004] | **addressed**: supports operating-point, rejects specialization |
| 2 | dense 2×2 | `denseft_{short,native}` (NEW) | dense fine-tune degrades (G ≈ −0.2); ΔJ_dense = −0.005 [−0.034, +0.024] | **negative/limitation** (raw-vs-EMA + no headroom); matched dense control still unavailable |
| 3 | revision narrative | — | — | presentation |
| 4 | "plateaus" | E1c *[reported]* | D4 = +0.021 [−0.023, +0.067] | wording: "does not clearly increase beyond 10.24 s" |
| 5 | severity-1 heterogeneity | post-hoc (NEW) | diff +0.124 [+0.058, +0.194]; pooled +0.112 [+0.076, +0.149] | **addressed**: report both subsets |
| 6 | hip-hop anchors 64→127 | post-hoc (NEW) | ρ_dense 0.106 / 0.119 (n=127); A_dense +0.11 | **addressed**: full coverage, ≈11–12% of dense gap |
| 7 | out-of-distribution hypothesis | XSEV-DENSE-192 *[reported]* + Point 1 | dense dur. response +0.142 ≈ real +0.150 | formulated; discriminating datum = ΔJ_pruned < 0 |
