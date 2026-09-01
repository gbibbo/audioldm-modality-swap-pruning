# Final Scientific Story — Source of Truth (pre-manuscript freeze)

**Date:** 2026-09-01. **Status:** FROZEN specification for the future manuscript. **Not manuscript prose.**
No experiment, GPU, scoring, audio, or example-selection change was made to produce this document. Every
number is read from a committed frozen artifact (cited inline); none is copied from chat.
**Framing = evaluation of post-pruning fine-tuning.** Mechanistic attribution is BLOCKED (§Limitations).

Terminology (fixed): **pruned checkpoint** = `p1` L1-pruned AudioLDM-M (sev-1 `(1,2,3,1)`; sev-2 `(1,2,1,1)`);
**post-fine-tuning checkpoint** = the pruned checkpoint after the recovery fine-tuning stage (Singh's public
`l1_p1_finetuned…`); at sev-2 the primary pruned baseline is **pruned2_A**, **pruned2_B** = 3-tensor seam
sensitivity only. Do not write "recovered … model" in the manuscript body (public-page term is "post-fine-tuning").

---

## 1. Core story (tightened against the frozen evidence)

1. A single aggregate evaluation operating point does **not** characterize post-pruning fine-tuning: its
   benefit over the pruned checkpoint depends on **evaluation context** and **temporal operating point**,
   and is **heterogeneous across prompts**.
2. The original **domain-specialization / recovery-reversal** hypothesis — that fine-tuning trades in-domain
   for out-of-domain alignment — was tested prospectively at the controlled 3.84 s operating point and is a
   **pre-registered NEGATIVE** (`R_AC` ≈ 0, gate FALSE). This reframed the work from a mechanism claim to an
   **evaluation** contribution.
3. A **temporal operating-point interaction** `J` (native-minus-short recovered−pruned advantage) is positive
   at both pruning severities: **borderline at severity 1** (95% CI narrowly includes 0) and **prospectively
   confirmed at the stronger severity 2** (`lo95(J) > 0`), robust to the seam convention.
4. At severity 2, post-fine-tuning yields a **large native-duration semantic advantage** over pruned
   (`R_native` +0.244, CI excludes 0); the specific **severity-1 music-negative sign pattern does NOT
   replicate** (`R_music` ≈ 0 at severity 2) → **CASE C** (context + duration dependence replicate; the exact
   sign pattern does not).
5. **FineLAP** (an independent, non-CLAP frame-level grounding evaluator) corroborates a **large, temporally
   broad** increase in frame-level requested-event grounding after fine-tuning at native scale (mass,
   occupancy, quarter-coverage, peak; CIs exclude 0; both severities; seam-robust), **while rejecting** the
   simpler "back-loaded late" explanation (`T_2` ≈ 0, gate FAILS; `T_1` not > 0).
6. **Sample-level outcomes are heterogeneous**: post-fine-tuning does not improve every prompt (illustrated,
   not evidenced, by the public examples; e.g. one prompt where pruned scores higher on both CLAP and FineLAP).

Every sentence above is bounded by the negatives and limitations in §4–§6.

---

## 2. Numerical source of truth (exact, with provenance)

CLAP = `laion/clap-htsat-fused` rev `365dea6e…`; bootstrap B=10000; unit = prompt (ytid). "CI" = 95% percentile.

### Severity 1 — AudioCaps, `configs/research/op_duration_discriminator_1_result.json` (sha `9c92552a…`; Arm-D 80 ytids, r0)
| Quantity | Point | 95% CI | Note |
|---|---:|---|---|
| R_short (3.84 s, recovered−pruned) | +0.0076 | [−0.0235, +0.0388] | near zero |
| R_native (10.24 s) | +0.0518 | [+0.0089, +0.0933] | CI excludes 0 |
| **J = R_native − R_short** | **+0.0442** | **[−0.0008, +0.0872]** | **95% CI narrowly includes 0 → gate NOT passed (directional)** |
| means | pruned 0.104 / rec 0.111 (3.84); pruned 0.253 / rec 0.304 (10.24) | | |

### Severity 1 — music (3.84 s), `configs/research/reversal_v1_r_music_clap.json`
| R_music (recovered−pruned) | −0.0941 | [−0.1241, −0.0646] | music-negative at severity 1 |

### Severity 1 — Human-CLAP (corroborative, no gate), same op_duration result
| R_short +0.044 [−0.002,+0.093] · R_native +0.119 [+0.063,+0.175] · J +0.075 [+0.012,+0.137] | | | HC J CI excludes 0 |

### Severity 1 — dense reference (SECONDARY; Arm-D 80 @10.24 s), `xsev_result.json:DENSE_CONTROL`
C_dense 0.352 / C_pruned 0.253 / C_post-FT 0.304; G_pruned (dense−pruned) +0.0994 [+0.0576,+0.1399];
G_post-FT (dense−post-FT) +0.0476 [−0.00035,+0.0957]. **No "restored to dense" claim** (CI includes 0, not a TOST).

### Severity 2 — AudioCaps + music, `configs/research/xsev_result.json` (sha `02e3bd11…`; PCG64 20260831)
| Quantity | Point | 95% CI | Gate |
|---|---:|---|---|
| R_native (10.24 s) | +0.2443 | [+0.2145, +0.2729] | H_native **TRUE** |
| R_short (3.84 s) | +0.0849 | [+0.0659, +0.1051] | short-equivalence (±0.025, 90% CI [0.069,0.102]) **FALSE** |
| R_music (3.84 s) | +0.0092 | [−0.0134, +0.0319] | H_music **FALSE** (≈ null) |
| **K** (context) | **+0.2351** | [+0.1974, +0.2719] | **PASS** |
| **J** (duration) | **+0.1594** | [+0.1309, +0.1873] | **PASS** |
| means | rec_native 0.299 / pruned_native 0.055; rec_short 0.0995 / pruned_short 0.0146; rec_music 0.0141 / pruned_music 0.0049 | | **CASE C** |

### Severity 2 — seam sensitivity B′ (report robustness only), `xsev_result.json:SENSITIVITY_B` + `seam_robustness`
K_B +0.2382 [0.201,0.275]; J_B +0.1615 [0.132,0.190]. **K seam-robust, J seam-robust, sign-pattern NOT robust
(fails under both A′ and B′).** B′ never rescues a failed A′ conclusion.

### Severity 2 — Human-CLAP (corroborative), `configs/research/xsev_hc_secondary.json`
R_native +0.3745 [0.340,0.408]; R_short +0.1894 [0.162,0.217]; **R_music −0.0365 [−0.068,−0.005]** (HC shows a
small music-negative that the **primary CLAP does not** — HC is secondary, no gate; music sign-pattern remains not established); K +0.4111 [0.363,0.457].

### Severity 2 — AudioCaps native secondaries, `configs/research/xsev_secondary_metrics.json`
| Metric | Value | Inference class |
|---|---|---|
| KL (`R_KL` = KL_pruned − KL_post-FT, lower KL better) | +2.224 | **[1.926, 2.530] paired ytid bootstrap — CI excludes 0** |
| PANN top-10 **captured-label count** (`R_cap`, a count, can exceed 1) | +0.859 | **[0.698, 1.021] paired — CI excludes 0** |
| FD (PANN-2048 Fréchet) | post-FT 48.7 vs pruned 107.2 | **descriptive, distribution-level, NO CI** |
| FAD (VGGish) | post-FT 6.92 vs pruned 27.4 | **descriptive, distribution-level, NO CI** |

### Severity 1 — Arm-D native secondaries, `configs/research/op_duration_discriminator_1_secondary.json`
KL R_native +0.582 [0.195,1.003]; PANN R_native +0.3625 [0.1125,0.625] (both CI-exclude-0 at native);
FD/FAD descriptive (FAD alt pruned 12.25 vs post-FT 5.41).

### FineLAP frame-level grounding, `configs/research/finelap_temporal_result.json` (sha `ce5519c8…`; native only; post-result diagnostic)
Windows: early frames 0–23, late 24–63 (0.16 s/frame), τ=0.5.
| Quantity | Point | 95% CI | Reading |
|---|---:|---|---|
| **T_2 = D_late2 − D_early2** (sev-2 primary) | **−0.0017** | [−0.0236, +0.0201] | **gate `lo95>0` FAILS — no late redistribution** |
| D_early2 | +0.2746 | [+0.2191, +0.3336] | post-FT ≫ pruned early |
| D_late2 | +0.2729 | [+0.2160, +0.3327] | post-FT ≫ pruned late (≈ early) |
| T_1 (sev-1 directional) | −0.0203 | [−0.0532, +0.0112] | point NOT > 0 |
| seam T_2^B (pruned2_B) | −0.0011 | [−0.0229, +0.0207] | seam-robust null |
FineLAP secondaries sev-2 (post-FT − pruned, CIs exclude 0): semantic mass +0.2735, occupancy +0.2680,
quarter-coverage +0.3761, peak +0.4065. **Interpretation boundary:** frame-level *grounding evidence*, NOT
calibrated event probability, NOT local causal activation, NOT perceptual quality.

### Pre-registered NEGATIVE (severity-1, controlled 3.84 s), `configs/research/reversal_v1_1_result.json` (sha `cad7c454…`)
C_dense 0.204 / C_pruned 0.100 / C_post-FT 0.098; **R_AC = −0.0024 [−0.0267, +0.0214]**; I = +0.0917
[+0.0535, +0.1311]; **PASS = FALSE** (recovery-reversal / domain-specialization NOT supported at the
controlled operating point).

---

## 3. Evidence-class classification (per result)

| Result | Class |
|---|---|
| Severity-2 CLAP CASE-C (K, J, R_native, R_short, R_music) | **prospectively frozen primary** (protocol `19c50cc3` frozen before scoring) |
| Severity-2 seam B′ robustness | prospectively frozen primary (sensitivity arm) |
| Severity-1 duration interaction J (op_duration Arm-D) | prospectively specified follow-up; **J directional (CI includes 0)** |
| Severity-1 pre-registered reversal NEGATIVE (V1.1) | **prospectively frozen primary NEGATIVE** |
| Human-CLAP contrasts (sev-1 & sev-2) | prospectively frozen **secondary**, corroborative, no gate |
| KL / PANN captured-label count | **pre-specified secondary; implemented AFTER the primary result** (metric named in frozen protocol §7; driver code written post-result) — inferential (paired CI) |
| FD / FAD | pre-specified secondary; **descriptive distribution-level only (no CI)** |
| FineLAP T/D + frame secondaries | **prospectively frozen POST-RESULT diagnostic** — NOT an independent preregistered confirmation |
| Cross-severity magnitude/"severity dependence" | **exploratory / post-hoc, confounded** (different experiments, prompt sets, n, checkpoints) — never promote |
| Public audio examples | descriptive, outcome-independent selection; **illustrative only, not evidence** |

---

## 4. Negative results (must survive into the paper)

* Severity-1 **strict recovery-reversal / domain-specialization hypothesis FAILED** (V1.1, R_AC ≈ 0, PASS=FALSE).
* Severity-1 primary **temporal J narrowly missed its gate** (J +0.044, 95% CI [−0.0008, +0.087]).
* Severity-2 specific **native-positive / music-negative sign pattern FAILED** (H_music FALSE; the defining CASE-C failure).
* Severity-2 **short-duration equivalence FAILED** (R_short +0.085, lo95>0; the advantage is present, not absent, at 3.84 s).
* **FineLAP late-redistribution hypothesis FAILED** (T_2 ≈ 0 seam-robust; T_1 not > 0).
* **No demonstrated restoration to dense** (G_post-FT CI includes 0; no TOST).
* **No dense fine-tuned control available** (Singh's checkpoint deleted; §Limitations).
* **No human listening study** (cancelled pre-launch; 0 participants, 0 data).
* **Sample-level post-fine-tuning failures exist** (heterogeneity); the public Example 4 (sev-1, 10.24 s:
  ΔCLAP −0.275, FineLAP pruned > post-FT) is a **legitimate illustration only — NOT elevated to evidence**.

---

## 5. Limitations (complete)

1. **Mechanistic attribution BLOCKED.** The ideal control separating *generic AudioCaps fine-tuning* from
   *pruning-associated post-fine-tuning behavior* is unavailable: Arshdeep confirmed the matched dense
   fine-tuned checkpoint was deleted (storage). The official public `audioldm-m-text-ft` is **not** an
   equivalent substitute (different data incl. MusicCaps, unspecified recipe/steps, dense start). Therefore
   the surviving positive result is an **evaluation** finding, not a causal/mechanism claim. This is a
   limitation, **not** a falsification of the evaluation result.
2. **Single primary scorer family.** Primary = CLAP; Human-CLAP is CLAP-derived. FineLAP (EAT+RoBERTa) is a
   genuinely different family and corroborates the frame-level gain, but there is no human evaluation.
3. **Off-recipe operating point vs Singh** (DDIM 50 vs 200, guidance 2.5, single generation, 3.84/10.24 s) —
   internal validity intact (both systems scored identically); external validity to the published recipe untested.
4. **Cross-severity comparison is exploratory/confounded** (two experiments, prompt sets, n, checkpoints).
5. **FineLAP is a post-result diagnostic** and provides grounding evidence, not calibrated probability or
   perceptual quality; native-duration only; eligibility AudioCaps-only.
6. **Sample-level heterogeneity** — population-level advantages are not monotonic per prompt.
7. **Public examples** are illustrative, deterministically selected, not human evidence; severity-1 music
   audio and a fully-matched dense reference were not retained on disk (omitted from the demo).

---

## 6. Forbidden claims (registry) + safe replacements

**FORBIDDEN:** "fine-tuning restores the pruned model"; "post-fine-tuning consistently improves samples";
"post-fine-tuning restores dense performance"; "recovery causes specialization"; "pruning causes the domain
dependence"; "fine-tuning causes OOD degradation"; "the post-fine-tuning advantage disappears at 3.84 s"
(FALSE — R_short lo95>0 at sev-2); "the improvement occurs because later events are generated" (FineLAP
rejects); "severity 2 is universally better/worse"; "FineLAP measures local causal activations"; "FineLAP
measures perceptual audio quality"; "human listeners confirmed the results"; and, for any individual sample,
"recovered/post-fine-tuning is better".

**SAFE:** pruned checkpoint · post-fine-tuning checkpoint · post-pruning fine-tuning · native-duration
semantic advantage · temporal operating-point dependence/interaction · context dependence · frame-level
grounding evidence · sample-level heterogeneity · prospectively frozen · pre-registered negative ·
post-result diagnostic · "in our (non-human) automatic evaluation".

---

## 7. Contribution

**One sentence:** We show that post-pruning fine-tuning of a text-to-audio diffusion model cannot be
characterized by a single aggregate evaluation point — its advantage over the pruned checkpoint is
context- and temporal-operating-point-dependent and prompt-heterogeneous — and we establish this with a
prospectively frozen cross-severity evaluation, independent frame-level corroboration, and explicit negatives.

**Three contributions:**
* A prospectively-frozen, seam-robust **evaluation protocol** exposing context (K) and temporal (J) dependence
  of post-pruning fine-tuning, with a **cross-severity prospective replication** (CASE C) and honest negatives.
* **Independent frame-level corroboration** (FineLAP, non-CLAP) of the native-duration grounding gain, plus a
  clean prospectively-frozen negative on the "late-redistribution" mechanism (T_2 ≈ 0).
* A **methodological caution**: single-operating-point / single-metric evaluation misranks post-pruning
  fine-tuning; sample-level outcomes are heterogeneous.

**One-sentence limitation:** Mechanistic attribution is blocked (no matched dense fine-tuned control) and
there is no human evaluation; the claims are about *evaluation*, not causal mechanism or perceptual quality.

---

## 8. Minimal ICASSP figure/table plan (do NOT build yet)

* **Figure 1 (main):** duration × system interaction — two panels (severity 1 with the dense@10.24 s
  reference; severity 2 pruned2_A vs post-FT), R_short vs R_native with 95% CIs and J annotated. (An internal
  mockup already exists at `docs/figures/`.)
* **Table 1 (core):** the CASE-C contrasts and cross-severity replication — R_native, R_short, R_music, K, J
  with 95% CIs and gate outcomes (sev-2 primary; sev-1 directional), plus the pre-registered V1.1 negative row.
* **Optional Figure 2 / compact panel (only if space):** forest plot of recovered−pruned across the six
  context×severity rows (already mocked internally) OR a FineLAP grounding-vs-duration strip. Prefer folding
  FineLAP + secondaries into Table 1 footnotes to save space.
* Public-audio URL `https://gbibbo.github.io/audioldm-modality-swap-pruning/` as a caption/footnote only.

---

## 9. Adversarial ICASSP review — five strongest objections

1. **No human evaluation.** *Have:* multi-metric automatic (CLAP+HC+KL+PANN+FD/FAD+FineLAP), prospective,
   cross-severity. *Cannot claim:* perceptual quality or human preference. *Survivable:* yes, if framed as an
   automatic-evaluation contribution and human eval listed as future work (the designed study was cancelled
   pre-ethics; 0 data).
2. **Generic-fine-tuning confound / missing dense-FT control.** *Have:* the explicit limitation; the surviving
   claim is evaluation, not mechanism. *Cannot claim:* pruning-specific causation. *Survivable:* yes as a
   bounded evaluation claim; this is the single biggest reviewer risk and must be stated up-front.
3. **Reliance on CLAP.** *Have:* FineLAP (different family) + KL/PANN (paired CI) corroborate at native scale.
   *Cannot claim:* metric-independent perceptual truth. *Survivable:* yes.
4. **Unusual duration/generation configuration vs Singh (DDIM50, 3.84/10.24 s).** *Have:* identical evaluation
   of both systems (internal validity), and DDIM200 judged low marginal value. *Cannot claim:* reproduction of
   Singh's absolute FAD or external validity to the published recipe. *Survivable:* yes as a limitation.
5. **Post-result diagnostics + sample-level heterogeneity look like fishing.** *Have:* explicit chronology
   (protocol froze before scoring; FineLAP labelled post-result; secondaries labelled pre-specified/
   implemented-after), and the negatives are reported. *Cannot claim:* FineLAP as independent preregistration,
   or heterogeneity as a positive result. *Survivable:* yes, because the primary CASE-C result is prospectively
   frozen and the diagnostics are honestly labelled.

**No experiment is proposed to answer these — the experimental phase is closed.**

---

## 10. Manuscript-readiness verdict

**READY FOR MANUSCRIPT.** No unresolved scientific or provenance contradiction remains: the CLAP absolute-value
discrepancy was a chat transcription error (resolved, `docs/public_examples_score_provenance_resolution.md`);
all numbers trace to committed frozen artifacts; the active project state is reconciled; the negatives,
limitations, forbidden claims, and evidence classes are frozen here and in `configs/research/final_claim_registry.json`.
**The next operation may be manuscript drafting ONLY after explicit Gabriel GO.** Do not reopen experiments.
