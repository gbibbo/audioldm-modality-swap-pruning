# Master Plan v3 - Modality-Swap-Aware Structured Pruning + Parameter-Efficient Recovery for AudioLDM

> **SUPERSEDED 2026-08-20 by `docs/master_plan_v4.md` (DECISION-V4-00, ledger). Kept for
> history — do NOT execute from this file.** v3's RQs and M3–M7 are replaced by v4; v3's
> provenance, Git discipline, CPU-Studio/GPU-Job policy, frozen SHAs, and "negative results
> are valid" carry over unchanged into v4.

**Status:** Execution in progress. Lightning AI access, Remote-SSH, GitHub authentication, and remote Claude Code are operational. GPU-hour totals remain deliberately parameterized until the first real GPU benchmark is completed.  
**Plan date:** 2026-08-17 (operational update of v3)  
**Primary short target:** ICASSP 2027, full-paper deadline 2026-09-16.  
**Longer fallback/extension:** DCASE++ / broader efficiency line discussed with Arshdeep.  
**Core scientific objective:** Determine whether structured pruning damages AudioLDM in a modality-dependent way, test whether paired audio-text saliency improves pruning at a matched structural and calibration budget, and determine how much residual damage a fixed parameter-efficient recovery configuration can restore.

---

# 0. Project contract

The project must end with explicit answers to three research questions, not merely with trained models.

The project has two coupled outputs:

1. **Engineering:** a reproducible LoRA-based recovery stack for pruned AudioLDM.
2. **Scientific:** a controlled, falsifiable study of modality-swapped conditioning under structured pruning.

LoRA is **not** the novelty. The paper must use the more precise term **parameter-efficient recovery** whenever biases and GroupNorm affine parameters are also trainable. Trainable parameters must be reported separately as:

- LoRA low-rank parameters;
- trainable biases;
- trainable GroupNorm affine parameters;
- total trainable parameters.

The project follows four rules:

1. no expensive recovery before the modality-swap hypothesis passes its low-cost gates;
2. no claim of cross-modal benefit without an equally data-aware text-only baseline;
3. no final FAD claim without generation-seed uncertainty on the key comparison;
4. no paper claim is written until its corresponding branch in the final claims matrix is resolved.

## 0.1 Current execution state as of 2026-08-17

The infrastructure uncertainty assumed by the original v3 has been partially resolved:

- Lightning AI account verification is complete and the Studio is accessible.
- VS Code Remote-SSH access to the Studio is working at `/teamspace/studios/this_studio`.
- GitHub CLI in the Studio is authenticated as `gbibbo` using HTTPS for Git operations.
- Claude Code is installed in the remote Studio.
- The Studio is currently being used in CPU mode; the first real GPU benchmark has **not** yet been run.
- The existing local research repository state must be synchronized into the Lightning workspace and re-validated there before GPU work begins.

Therefore, Lightning availability is no longer a project risk. The immediate infrastructure risk is now **measured GPU suitability/cost**, not account access.

---

# 1. Scientific framing

## 1.1 Baseline substrate

Use:

- AudioLDM-M-Full;
- the original `haoheliu/AudioLDM-training-finetuning` codebase;
- the public `Arshdeep-Singh-Boparai/PruningAudioLDM` implementation;
- AudioCaps;
- the `(1,2,3,1)` structural budget as the only mandatory pruning architecture for the short paper;
- FAD, KL, and PANNs semantic-event analysis.

Freeze public code at:

```text
AudioLDM-training-finetuning:
702a638d023b008a2d9a45cdf1e1f4fcdc590dfc

PruningAudioLDM:
6f65f628fabc4ad27770753698fc81944e820f9f
```

The `(1,2,1,1)` configuration is outside the ICASSP core scope.

## 1.2 Candidate novelty

The candidate claim is deliberately narrow:

> **Preserving modality-swapped conditioning under structured pruning of AudioLDM.**

The novelty is not generic cross-modal pruning, generic conditioning-aware pruning, Taylor pruning, or pruning + LoRA.

The project studies whether the AudioLDM conditioning swap between CLAP audio and CLAP text is selectively damaged by structural pruning, and whether saliency calibrated under both conditioning modalities makes better structural pruning decisions than:

- L1 magnitude pruning;
- an equally data-aware text-only Taylor baseline.

---

# 2. Final research questions

## RQ1 - Phenomenon

**Does structured magnitude pruning introduce modality-dependent degradation in AudioLDM beyond what is expected from generic pruning damage at comparable denoiser degradation?**

## RQ2 - Pruning criterion

**At the same architecture, structural budget, and gradient-evaluation budget, does paired audio-text saliency preserve text-conditioned generation better than L1 and a faithful text-only Taylor criterion?**

## RQ3 - Recovery

**How much residual degradation can a fixed parameter-efficient recovery configuration restore, what is its compute/memory cost relative to the available full-fine-tuning reference, and which semantic event families remain vulnerable?**

---

# 3. Diagnostic definitions

For the same example, noisy latent `z_t`, diffusion timestep `t`, and noise realization, define:

\[
\epsilon_{F,a},\; \epsilon_{F,t}
\]

as predictions from the full model under audio and text conditioning, and:

\[
\epsilon_{P,a},\; \epsilon_{P,t}
\]

as predictions from a pruned model.

Define pruning errors:

\[
E_a = \epsilon_{P,a} - \epsilon_{F,a}
\]

\[
E_t = \epsilon_{P,t} - \epsilon_{F,t}
\]

Generic pruning damage:

\[
D_{gen}=\frac{1}{2}(\|E_a\|+\|E_t\|)
\]

Modality-dependent damage:

\[
D_{mod}=\|E_a-E_t\|
\]

Normalized diagnostic:

\[
R_{mod}=\frac{\|E_a-E_t\|}{\|E_a\|+\|E_t\|+\epsilon}
\]

`D_mod` and `R_mod` are diagnostics only. They are **not** the pruning loss.

This formulation avoids the earlier degeneracy and does not require audio-conditioned and text-conditioned predictions to be identical.

---

# 4. Pruning criteria

Introduce a channel gate `g_c` and use ordinary first-order Taylor saliency under the diffusion loss.

Text-conditioned saliency:

\[
S_c^t=\mathbb{E}\left|g_c\frac{\partial L_{diff}(e_t)}{\partial g_c}\right|
\]

Audio-conditioned saliency:

\[
S_c^a=\mathbb{E}\left|g_c\frac{\partial L_{diff}(e_a)}{\partial g_c}\right|
\]

Normalize within each prunable layer before combining modalities.

## P0 - L1 magnitude

Historical, data-free baseline.

## P1 - Text-only Taylor

A faithful text-conditioned Taylor baseline, implemented as closely as the architecture permits to the timestep aggregation logic of Diff-Pruning.

P1 is mandatory. Any cross-modal claim fails if the paired criteria do not improve on a correctly implemented P1.

## P2 - Paired-mean Taylor

\[
S_c^{mean}=\frac{\widetilde S_c^a+\widetilde S_c^t}{2}
\]

Tests whether simply using paired modalities helps.

## P3 - Swap-robust Taylor

Initial candidate:

\[
S_c^{swap}=\max(\widetilde S_c^a,\widetilde S_c^t)
\]

Tests a worst-modality preservation rule: do not remove a channel that is important under either conditioning mode.

P3 is not protected as a claim. If P2 matches or beats P3, the contribution becomes paired-modality calibration rather than the `max` rule.

---

# 5. Calibration-budget contract

The phrase "same calibration budget" is defined operationally as the **same number of gradient evaluations**, not merely the same number of AudioCaps items.

Let one gradient-evaluation unit be one forward + backward saliency evaluation for one conditioning modality at one pre-specified `(example, noise, timestep)` slot.

Let the paired calibration use `B` base slots.

Then:

- **P2/P3:** `B` text evaluations + `B` audio evaluations = `2B` gradient evaluations.
- **P1:** `2B` text evaluations = `2B` gradient evaluations.

To avoid giving P1 redundant duplicate gradients, P1 uses two pre-registered timestep/noise draws per base example where possible. P2/P3 use one paired audio/text evaluation on each base slot. The exact slot construction is frozen in `docs/pilot_protocol.md` before any saliency results are inspected.

P2 and P3 share the same computed `S_a` and `S_t`; no duplicate calibration compute is spent merely to change the aggregation rule.

Report pruning/calibration cost explicitly:

- calibration GPU-hours;
- peak VRAM;
- number of forward/backward evaluations;
- wall time;
- dataset examples used.

This cost is part of the efficiency comparison because P0 is data-free.

---

# 6. Timestep protocol

Before M3 begins, freeze:

- the list or strata of calibration timesteps;
- the number of samples per stratum;
- the primary aggregation statistic across `t`.

Primary diagnostic aggregation:

> equal-weight mean across pre-registered timestep strata, followed by averaging across examples.

Timestep-specific curves are secondary analyses only. The primary conclusion must not change by choosing a favorable timestep after inspecting results.

---

# 7. Compute strategy and hard project gate

## 7.1 Compute environments

**Primary cloud environment is confirmed: Lightning AI.** Account verification, Studio access, VS Code Remote-SSH, GitHub authentication, and remote Claude Code are operational.

The current Studio state is CPU-only for development. The next compute task is to select an available Lightning GPU and run the benchmark defined in Section 7.2.

Fallback if Lightning GPU availability, VRAM, or projected cost is unsuitable: activate a paid cloud GPU provider rather than blocking the project, provided the projected short-paper compute fits the user's approximately US$50 total discretionary cloud budget or an institutional cluster becomes available.

An institutional KCL/UCL/Surrey allocation remains desirable for long recovery runs but is not allowed to block M0-M3.

## 7.2 Benchmark variables

The first usable GPU session must measure:

```text
GPU_MODEL
VRAM_GB
TRAIN_SEC_PER_STEP
SALiency_SEC_PER_GRAD_EVAL_OR_BATCH
FORWARD_SEC_PER_DIAGNOSTIC_BATCH
GEN_SEC_PER_CLIP_OR_BATCH
GEN_BATCH_SIZE
PEAK_TRAIN_VRAM_GB
PEAK_SALIENCY_VRAM_GB
PEAK_GENERATION_VRAM_GB
```

No fixed GPU-hour estimate is invented before these values exist.

## 7.3 Compute formulas

Let:

- `Ttrain` = measured training seconds/step;
- `Tsal` = measured seconds per saliency gradient-evaluation batch;
- `Tfwd` = measured seconds per diagnostic forward batch;
- `Tgen` = measured seconds per generated evaluation clip equivalent;
- `Neval` = frozen test-manifest size;
- `Bcal` = calibration gradient-evaluation units;
- `Krand` = random-pruning null masks;
- `Sfinal` = final generation seeds on the confirmatory comparison.

Approximate core costs:

```text
M1 smoke training:
500 * Ttrain / 3600

M3 saliency:
(total saliency gradient batches) * Tsal / 3600

M3 random-control diagnostics:
Krand * paired diagnostic batches * Tfwd / 3600

M4 one full generation set:
Neval * Tgen / 3600

M5 100k-step recovery per model:
100000 * Ttrain / 3600
```

After the benchmark, `docs/compute_budget.md` must contain numeric GPU-hours and monetary cost for every milestone.

## 7.4 Compute decision gate CG

**Target decision date: 2026-08-18.**

ICASSP remains the active target only if by the end of 2026-08-18:

1. **[SATISFIED]** a usable cloud environment is confirmed: Lightning AI;
2. M1 benchmark has produced real throughput/VRAM numbers on the selected Lightning GPU;
3. the projected cost of the ICASSP core program fits available Lightning credits plus the approximately US$50 discretionary cloud budget and/or institutional compute;
4. projected completion leaves the paper-writing milestone starting no later than 2026-09-05.

If any condition fails, the project continues scientifically but switches from deadline-driven ICASSP execution to the longer DCASE++ / efficiency plan.

A failed compute gate is a schedule decision, not a scientific failure.

---

# 8. Project milestones

Every milestone ends with a tangible artifact and a written decision.

---

# M0 - Public baseline reproduction

**Status on 2026-08-17:** in progress. Repository/bootstrap work and CPU-side scaffold/tests have already started locally; the next action is to synchronize that state into Lightning and verify it remotely.  
**Revised completion target:** 2026-08-18  
**GPU dependence:** none for repository/environment setup; minimal GPU later for generation smoke test.

## Tasks

- create the research repository from frozen AudioLDM upstream history;
- preserve `upstream-frozen` at commit `702a638d...`;
- record PruningAudioLDM reference commit `6f65f628...`;
- create reproducible environment metadata;
- download/prepare AudioLDM checkpoints and AudioCaps when storage is available;
- reconstruct or download the L1 `(1,2,3,1)` pre-recovery checkpoint;
- freeze checkpoint and manifest hashes;
- run model-loading and generation smoke tests once GPU exists;
- verify the FAD/KL pipeline;
- reproduce/implement the PANNs top-k semantic pipeline.

## Full-FT checkpoint strength gate

During M0, search public releases/Zenodo for a final recovered full-FT `(1,2,3,1)` checkpoint.

If a checkpoint that can be evaluated in our pipeline is still unavailable after the public-artifact search is completed on **2026-08-18**, request the final full-FT checkpoint from Arshdeep immediately.

Interpretation:

- **checkpoint available:** RQ3 can compare LoRA and full FT in the exact same generation/evaluation pipeline;
- **checkpoint unavailable:** RQ3 is downgraded to a published-reference comparison. Do **not** claim an exact percentage of full-FT recovery from cross-pipeline numbers.

Intermediate full-FT checkpoints/logs are useful but not mandatory.

## Deliverable M0

`artifacts/m0_baseline_reproduction/`

- frozen SHAs;
- environment report;
- dataset/checkpoint manifests;
- reconstruction commands;
- public/private artifact inventory;
- smoke-test outputs when GPU becomes available.

## Acceptance

M0 passes when the base and pruned architectures can be reconstructed deterministically and the remaining unavailable artifacts are explicitly listed.

---

# M1 - Parameter-efficient recovery v1

**Status on 2026-08-17:** CPU scaffold exists and its initial tests have passed locally. Remaining work is upstream integration, remote re-validation in Lightning, and the real GPU smoke benchmark.  
**Revised completion target:** 2026-08-18 for integration + benchmark inputs; GPU acceptance may extend only if a platform-side GPU issue occurs.  
**GPU dependence:** CPU implementation/tests first; GPU smoke benchmark last.

## Engineering decisions

Use the original AudioLDM training codebase, not Diffusers.

Default recovery:

```yaml
lora:
  rank: 8
  alpha: 16
  dropout: 0.0
  scope: full_unet

auxiliary_trainables:
  bias: true
  groupnorm_affine: true
```

Eligible diffusion-U-Net targets:

- `nn.Linear`;
- `nn.Conv2d` with supported grouping;
- attention q/k/v/out projections;
- ResBlock embedding projections;
- time/FiLM-related linear projections;
- residual/down/middle/up/output convolutions.

Exclude CLAP, VAE, vocoder, and PANNs.

## Optimizer contract

The upstream `configure_optimizers()` currently collects the whole diffusion model and conditional-stage parameters. The research hook must instead build parameter groups only from intended trainables.

Report three parameter groups separately:

1. LoRA;
2. biases;
3. GroupNorm affine parameters.

Initial LR values remain provisional until the smoke benchmark.

## EMA contract

The upstream EMA is constructed during model initialization and tracks parameters that are trainable at that moment. Its validation `store()` path can copy all parameters passed to it.

For PEFT recovery:

- do not allow EMA to create a full frozen-U-Net shadow accidentally;
- initialize/rebuild EMA only after PEFT injection/freezing, or disable it until an adapter-safe EMA path is validated;
- validation store/copy/restore must operate only on the tracked parameter set.

## Lifecycle

Required:

- adapter save/load;
- full resume state;
- merge/unmerge;
- numerical equivalence after merge;
- module/parameter report;
- CPU unit tests.

## Deliverable M1

- LoRA/recovery module;
- selection/config abstraction;
- adapter state lifecycle;
- CPU tests;
- integration notes/patch points;
- GPU benchmark report once compute is available.

## Acceptance

CPU acceptance:

- Linear LoRA merge/unmerge tests pass;
- Conv2d LoRA merge/unmerge tests pass on supported convs;
- injector freezes base parameters correctly;
- auxiliary trainables are separately reported;
- adapter-only state saves and reloads.

GPU acceptance:

- several hundred real AudioLDM optimization steps run stably;
- peak VRAM and step time are recorded;
- resume succeeds.

---

# M2 - Conditioning-path validation

**Revised target window:** 2026-08-17 to 2026-08-18  
**GPU dependence:** small.

## Objective

Prove that audio-conditioned and text-conditioned paths are instrumented correctly before interpreting any cross-modal result.

## Tasks

- expose CLAP audio conditioning;
- expose CLAP text conditioning;
- verify embedding dimensions;
- verify normalization/scaling conventions;
- verify that both routes feed the same intended FiLM conditioning interface;
- inspect embedding norm distributions;
- hold `z_t`, `t`, and noise fixed;
- confirm deterministic repeated predictions.

## Deliverable M2

`docs/condition_swap_validation.md`

plus executable tests and diagnostic plots.

## Fail condition

If normalization or code-path differences cannot be resolved, M3 is blocked. No cross-modal conclusion may be drawn.

---

# M3 - Falsifiable no-training pilot

**Target window:** 2026-08-18 to 2026-08-20  
**GPU dependence:** modest.

M3 contains two independent gates.

## M3A - Gate A: modality-dependent damage against a matched random null

The null is **not numerical noise**.

Generate `Krand = 20` random structured pruning masks at the same `(1,2,3,1)` architecture/budget. For each random mask and L1 compute aggregated:

- `D_gen`;
- `D_mod`;
- `R_mod`.

Because random pruning may have much larger generic damage than L1, do not compare raw `R_mod` blindly.

Primary matched-null statistic:

1. fit the relationship between `R_mod` and `D_gen` across random controls;
2. estimate expected random `R_mod` at the observed `D_gen` of L1;
3. define

\[
\Delta_{swap}=R_{mod}^{L1}-E[R_{mod}^{random}\mid D_{gen}^{L1}]
\]

4. bootstrap over evaluation examples and random masks.

### Gate A PASS

Pass only if:

- the 95% bootstrap CI of `Delta_swap` is above zero; and
- the standardized residual is at least `0.5` random-control SD.

The `0.5 SD` threshold is a project-level practical-effect gate, not a universal scientific constant.

### Gate A FAIL

If L1 is explainable by the random-pruning `R_mod` vs `D_gen` relationship, conclude:

> the observed modality-dependent difference is not specific enough to support the condition-swap pruning hypothesis at this budget.

Stop P2/P3 recovery work and reassess the novelty.

## M3B - Gate B: audio/text saliency disagreement

Compute `S_a` and `S_t` using the frozen calibration protocol.

Primary comparison is at the actual prune tail:

- overlap@k;
- Jaccard of prune sets;
- cutoff-rank disagreement;
- layer-wise analysis in the affected blocks.

Global Spearman is secondary.

### Pre-registered Gate B PASS

Pass if both conditions hold:

1. weighted prune-set overlap across targeted layers is `<= 0.80`; and
2. at least two key prunable layers show overlap `<= 0.70`.

These are project go/no-go thresholds, not claims of universal effect size.

### Gate B FAIL

If the modalities select essentially the same channels, conclude:

> paired-modality saliency is operationally redundant at the target pruning ratio.

Do not spend recovery compute on P2/P3.

## Deliverable M3

`artifacts/m3_pilot/`

- frozen sample/timestep/seed protocol;
- random masks;
- `D_gen`, `D_mod`, `R_mod` results;
- matched-null model;
- bootstrap outputs;
- `S_a`/`S_t` overlap tables;
- explicit Gate A and Gate B decisions.

---

# M4 - Matched pre-recovery pruning experiment

**Target window:** 2026-08-20 to 2026-08-24  
**GPU dependence:** calibration + generation.

Run only if M3 passes.

## Systems

| ID | Criterion | Role |
|---|---|---|
| P0 | L1 | historical/data-free baseline |
| P1 | text-only Taylor | equally data-aware unimodal baseline |
| P2 | paired mean | bimodal calibration control |
| P3 | paired max | swap-robust candidate |

All four produce exactly the same `(1,2,3,1)` architecture.

## Matched conditions

- same base checkpoint;
- same target channels per layer;
- same parameter and MAC budget;
- same frozen dataset version;
- P1/P2/P3 same total number of gradient evaluations;
- same generation protocol;
- same evaluation manifest.

## External evaluation

Primary evidence:

- FAD;
- KL;
- PANNs semantic/event preservation.

A CLAP-derived metric may be secondary but may not be the sole evidence for a CLAP-informed method.

## FAD seed protocol

To control compute:

### Screening phase

Run one pre-registered generation seed for P0-P3.

Purpose:

- catch large failures;
- identify the best paired method;
- avoid tripling generation for dominated candidates.

### Confirmatory phase

For the key comparison only, use **three total generation seeds**:

- P0;
- P1;
- the best of P2/P3.

If P2 and P3 are statistically/operationally indistinguishable after screening, include both only if the compute budget permits; otherwise choose the method according to the pre-registered primary metric hierarchy and document the choice.

Report between-seed FAD variability. Clip bootstrap alone is not treated as full generation uncertainty.

## Pruning-cost report

For P0-P3 report:

- calibration GPU-hours;
- peak calibration VRAM;
- number of gradient evaluations;
- pruning/checkpoint creation time;
- storage.

P0's data-free cost advantage must remain visible.

## Complete M4 decision tree

### 4A: `P3 > P2 > P1 > P0`

Strongest result. Both paired modality and robust max aggregation add value.

### 4B: `P3 ~= P2 > P1 > P0`

Paired-modality calibration helps; max aggregation is not uniquely supported.

### 4C: `P1 ~= P2 ~= P3 > P0`

Data-aware Taylor helps; cross-modal information does not.

### 4D: `P0 >= P1` and `best(P2,P3) > P0`

Text-only Taylor is not superior to L1, but paired-modality calibration produces a real gain over both. This is still potentially strong evidence for modality information, provided P1 faithfully implements the text-only baseline and the confirmatory seeds support the result.

### 4E: `P0 >= P1` and `best(P2,P3) <= P0`

Neither data-aware text-only nor paired-modality pruning improves on the cheap L1 baseline. The new pruning-method claim fails.

### 4F: `P1 ~= P2 ~= P3 ~= P0`

Criterion choice has little measurable downstream effect at this budget.

### 4G: paired method improves PANNs but FAD/KL tie

Conclude semantic/event preservation benefit without claiming global distributional-quality improvement.

### 4H: `P3 < P1` or `P3 < P2`

Reject the max rule. Retain P2 only if independently supported.

## Deliverable M4

- P0-P3 manifests/checkpoints;
- saliency/pruning cost table;
- one-seed screening results;
- confirmatory three-seed key comparison;
- FAD/KL/PANNs table;
- explicit selected method for M5;
- written branch conclusion.

---

# M5 - Fixed parameter-efficient recovery

**Target window:** 2026-08-24 to 2026-09-03  
**GPU dependence:** dominant compute stage.

## Recovery variants

Prioritize exactly three:

1. P0 L1;
2. P1 text-only Taylor;
3. best supported paired-modality method from M4.

Recover both P2 and P3 only if M4 leaves their distinction scientifically unresolved and compute allows.

## Fixed recovery configuration

Same for every variant:

- LoRA rank 8;
- alpha 16;
- full U-Net target scope;
- same Linear/Conv2d policy;
- same auxiliary bias/GroupNorm policy;
- same optimizer groups;
- same LR schedule;
- same batch size/effective batch;
- same precision;
- same seed policy;
- same AudioCaps training manifest.

No rank sweep in the short-paper core.

## Training checkpoints

Save:

- 25k;
- 50k;
- 100k;
- 200k only for informative runs if compute permits.

## Evaluation economy

At 25k and 50k:

- use a fixed diagnostic generation subset;
- compute cheap semantic/quality diagnostics;
- do not run a full three-seed FAD campaign.

At 100k:

- run full evaluation for all three main variants;
- use one screening generation seed first;
- run three-seed confirmatory generation only on the final key comparison if the compute budget permits.

Extend to 200k only when the 100k trajectory indicates that the conclusion could materially change.

## M5 decision tree

### 5A - Near-full recovery

Parameter-efficient recovery approaches the same-pipeline full-FT checkpoint while using far fewer trainable parameters/optimizer state.

### 5B - Partial recovery, pruning ordering persists

Better pruning reduces damage that fixed PEFT cannot fully repair. This is a strong result.

### 5C - Recovery erases pruning differences

Criterion matters before recovery, but PEFT dominates the final quality under this budget.

### 5D - Recovery fails similarly

The tested structural damage exceeds the recovery capacity of the fixed PEFT setup.

### 5E - Paired criterion helps only after recovery

Report an observed pruning-recovery interaction. Do not retrospectively claim the pruning method was designed for LoRA recoverability.

## Deliverable M5

- adapters and resume checkpoints;
- recovery curves;
- full metric table at final selected step;
- trainable-parameter breakdown;
- peak VRAM;
- seconds/step;
- GPU-hours;
- adapter/full checkpoint sizes;
- same-pipeline full-FT comparison if checkpoint available.

---

# M6 - Semantic and efficiency closure

**Target window:** 2026-09-02 to 2026-09-04.

## Semantic analysis

For each event family report:

- support/sample count;
- pre-pruning change;
- recovery trajectory;
- uncertainty;
- criterion comparison.

Treat class analysis as exploratory unless effects are stable and support is sufficient.

Do not infer that modality-dependent pruning causally explains individual classes merely from correlation.

## Efficiency analysis

Report separately:

### Pruning/calibration efficiency

- gradient evaluations;
- calibration GPU-hours;
- calibration peak VRAM;
- time to produce pruning manifest/checkpoint.

### Recovery efficiency

- LoRA parameters;
- bias parameters;
- GroupNorm affine parameters;
- total trainable parameters;
- peak VRAM;
- seconds/step;
- total GPU-hours;
- adapter size;
- full checkpoint size.

### Inference efficiency

- total parameters;
- MACs under common counter;
- generation throughput where measured.

## Deliverable M6

`artifacts/m6_analysis/`

- semantic tables/plots;
- support/uncertainty tables;
- pruning-cost table;
- recovery-efficiency table;
- quality-efficiency plots.

---

# M7 - Scientific closure

**Target date:** 2026-09-04.

Produce a one-page answer sheet.

## RQ1

Exactly one:

- supported;
- not supported;
- inconclusive due to instrumentation/statistical power.

## RQ2

Exactly one:

- paired modality adds value beyond P1;
- paired modality helps but max does not;
- only data-aware Taylor helps;
- L1 remains best/competitive;
- no criterion difference;
- paired method is worse.

## RQ3

Exactly one:

- PEFT recovers most damage;
- PEFT partially recovers with a clear ceiling;
- PEFT erases criterion differences;
- PEFT fails at this pruning budget;
- recovery depends on pruning criterion.

## Claims matrix

Every abstract/conclusion claim is marked:

- supported;
- rejected;
- exploratory;
- unavailable because required artifact was missing.

No rejected/exploratory claim is promoted to a headline claim.

---

# M8 - Paper writing and coauthor iteration

This milestone was missing from v2 and is mandatory.

## M8A - Paper skeleton

**Target:** 2026-09-01.

Before training is fully finished, create the ICASSP manuscript skeleton with:

- Introduction;
- Related Work;
- Method;
- Experimental Setup;
- pre-reserved Results tables/figures;
- Limitations.

Do not wait for M7 to start writing stable sections.

## M8B - Results-complete first draft

**Target:** 2026-09-07.

Insert frozen M7 claims and all core tables.

## M8C - Coauthor review

**Target:** 2026-09-08 to 2026-09-12.

Send to Arshdeep and other confirmed coauthors for scientific review.

## M8D - Final revision

**Target:** 2026-09-13 to 2026-09-15.

- resolve comments;
- verify every reported number against raw outputs;
- final reproducibility check;
- formatting/submission validation.

## Submission target

**2026-09-16.**

If core M5/M7 conclusions are not frozen in time for M8B, do not submit an underpowered paper merely to meet the date. Continue under the longer research plan.

---

# 9. Schedule and critical path

| Date | Critical output |
|---|---|
| Aug 13-17 | **Completed/started:** local repo bootstrap, frozen references, LoRA CPU scaffold/tests |
| Aug 17 | **Completed:** Lightning verification, Remote-SSH, GitHub `gbibbo` authentication, remote Claude Code |
| Aug 17-18 | synchronize repo to Lightning; finish M0; integrate/re-validate M1; complete M2 instrumentation |
| Aug 17-18 | first Lightning GPU benchmark + numeric compute budget |
| Aug 18 | Compute Gate CG |
| Aug 18-20 | M3 pilot and GO/NO-GO |
| Aug 20-24 | M4 pre-recovery P0-P3 |
| Aug 24-Sep 3 | M5 fixed recovery |
| Sep 1 | paper skeleton already exists |
| Sep 2-4 | M6 analysis + M7 scientific closure |
| Sep 7 | results-complete first draft |
| Sep 8-12 | coauthor review |
| Sep 13-15 | final revision |
| Sep 16 | ICASSP submission target |

## Hard schedule warnings

- If Compute Gate CG fails on Aug 18, activate the longer-plan schedule.
- If M3 has no GO decision by Aug 20, ICASSP scope is at risk.
- If M4 is not scientifically closed by Aug 24, do not add any ablation.
- If the three core 100k recovery runs cannot finish by Sep 3, freeze the strongest defensible result and decide whether it supports the short paper; otherwise move to the longer plan.
- No experiment added after Sep 5 may displace paper verification/review unless it is required to resolve a fatal ambiguity.

---

# 10. GPU budget table template

**Confirmed provider for the first benchmark:** Lightning AI.  
**GPU model:** `TBD_SELECTED_ON_LIGHTNING`  
**Benchmark status:** pending.

The benchmark fills the following table. Until then, GPU-hour cells remain `TBD_MEASURED` rather than guessed.

| Milestone | GPU task | Runs/units | Formula | GPU-hours | Cost |
|---|---|---:|---|---:|---:|
| M0 | generation/eval smoke | small | measured | TBD | TBD |
| M1 | 500-step real smoke | 1 | `500*Ttrain/3600` | TBD | TBD |
| M2 | paired conditioning diagnostics | frozen | `batches*Tfwd/3600` | TBD | TBD |
| M3A | L1 + random-null diagnostics | 20 random + L1 | measured formula | TBD | TBD |
| M3B | saliency | fixed `Bcal` | `grad_batches*Tsal/3600` | TBD | TBD |
| M4 | P0-P3 screening generation | 4 sets | `4*Neval*Tgen/3600` | TBD | TBD |
| M4 | confirmatory extra seeds | key 3 models | measured | TBD | TBD |
| M5 | 100k recovery | 3 runs | `3*100000*Ttrain/3600` | TBD | TBD |
| M5 | diagnostic evals | 25k/50k | subset only | TBD | TBD |
| M5 | final full eval | 100k | 3 models + confirmatory | TBD | TBD |
| M6 | analysis classifiers | as needed | measured | TBD | TBD |

After benchmark, add:

- cloud price per GPU-hour;
- storage cost if any;
- free credits remaining;
- paid amount forecast;
- institutional compute assumed;
- 20% contingency.

Project discretionary cloud spend should remain within the user's approximately US$50 total ceiling unless explicitly re-approved.

---

# 11. Public-first artifact policy

Do not ask Arshdeep for items already public or deterministically reconstructable.

First obtain independently:

- original AudioLDM repo/checkpoint;
- public AudioCaps preparation;
- public pruning repo/scripts/indices;
- L1 `(1,2,3,1)` reconstruction;
- public FAD/KL tools;
- documented generation configuration;
- PANNs implementation.

Ask Arshdeep only for a concrete missing artifact.

The most scientifically valuable possible request is the **final full-FT recovered checkpoint** if it cannot be found publicly, because same-pipeline evaluation materially strengthens RQ3.

Institutional cluster access is requested only if the Lightning benchmark shows that long recovery runs would exceed the available cloud budget or deadline margin.

---

# 12. Reproducibility contract

Every reported number maps to:

- Git commit;
- resolved config;
- base checkpoint SHA256;
- pruning manifest;
- dataset manifest;
- calibration slot list;
- timestep list;
- random seed;
- generation seed;
- adapter/checkpoint SHA256;
- raw metric output;
- GPU model and runtime.

Failed/stopped runs remain in the experiment ledger.

No large checkpoint is committed to Git.

---

# 13. Repository architecture

One research repository based on the frozen AudioLDM upstream history:

```text
audioldm_train/                 upstream code, minimal surgical patches
audioldm_peft/                  LoRA/recovery implementation
research_pruning/
    diagnostics/
    taylor/
    paired_modality/
configs/
    research/
scripts/
    research/
tests/
    research/
docs/
    master_plan_v3.md
    pilot_protocol.md
    compute_budget.md
    experiment_ledger.md
    claims_matrix.md
artifacts/                       ignored by Git
_external/                       ignored reference clones
```

Keep upstream modifications minimal and reviewable with:

```bash
git diff upstream-frozen -- audioldm_train/
```

## 13.1 Immediate execution queue for Claude Code

Once this plan is present in the Lightning research repository, Claude should work in this order:

1. verify repository branch, remotes, and the two frozen reference SHAs;
2. reproduce the existing CPU test result in the Lightning environment before changing code;
3. inspect and integrate the LoRA/PEFT hooks into the real AudioLDM `configure_optimizers()`, EMA lifecycle, checkpoint/resume path, and model-loading path;
4. complete the remaining M0 public-artifact inventory and record exact hashes/commands;
5. implement and validate M2 audio/text conditioning instrumentation;
6. prepare a single reproducible GPU benchmark command/config that records all Section 7.2 variables;
7. run no M3 scientific experiment until the benchmark has populated `docs/compute_budget.md` and Compute Gate CG is explicitly resolved.

Claude must preserve the scientific gates and must not simplify P1/P2/P3, change calibration budgets, or add new experiments without recording the change in `docs/experiment_ledger.md`.

---

# 14. Final deliverables

## Engineering

- reproducible research repo;
- frozen upstream/reference SHAs;
- LoRA Linear/Conv2d adapter implementation or validated backend;
- configurable injection/selection;
- bias/GN auxiliary controls;
- PEFT-safe EMA behavior;
- save/load/merge/resume;
- CPU tests;
- GPU benchmark and runtime profile.

## M0/M2 diagnosis

- baseline reproduction pack;
- condition-path validation report;
- public/private artifact inventory.

## M3 pilot

- matched random null;
- Gate A decision;
- audio/text saliency overlap;
- Gate B decision.

## M4 pruning

- P0-P3 manifests/checkpoints;
- pruning calibration-cost table;
- pre-recovery FAD/KL/PANNs;
- generation-seed uncertainty on key comparison;
- explicit M4 branch conclusion.

## M5 recovery

- fixed recovery adapters/checkpoints;
- recovery curves;
- parameter breakdown;
- VRAM/time/GPU-hours;
- same-pipeline full-FT comparison if available;
- explicit M5 branch conclusion.

## M6/M7 closure

- semantic uncertainty/support analysis;
- efficiency tables;
- RQ answer sheet;
- claims matrix;
- limitations and rejected hypotheses.

## M8 paper

- manuscript skeleton;
- results-complete draft;
- coauthor-reviewed final draft;
- submission package if the evidence is strong enough.

---

# 15. Definition of done

Training completion is not project completion.

The project is done when we can answer, with recorded evidence:

1. **Does pruning create modality-dependent damage beyond a random-pruning null at comparable generic damage?**
2. **Do audio and text conditioning actually select different channels at the target pruning tail?**
3. **Does paired-modality saliency beat a faithful text-only Taylor baseline at matched gradient-evaluation budget?**
4. **Does the specific max aggregation add anything beyond a paired mean?**
5. **What is the calibration cost of the new pruning method relative to data-free L1?**
6. **How much can the fixed parameter-efficient recovery restore?**
7. **Does the pruning criterion still matter after recovery?**
8. **Which semantic event families remain vulnerable, with what uncertainty/support?**
9. **What actual GPU, memory, time, and storage efficiency is achieved?**
10. **Which claims are sufficiently supported to enter the paper?**

Every possible major result has an interpretation branch in M3-M5. A negative result is accepted as a conclusion; it is not silently converted into a new hypothesis after seeing the data.
