# Hostile review — round 1 (2026-08-20)

Ping-pong document between two reviewers (Reviewer A = literature-heavy external
reviewer; Reviewer B = Fable, repo-side auditor). Goal: find the literature niche
that is actually open, fix the methodology, and produce master plan v4.

Nothing in this file changes a scientific gate. Everything here is a proposal
until Gabriel records a DECISION in `docs/experiment_ledger.md`.

---

## 1. Points of agreement (both reviewers)

1. M3A / M3B are **rigorously executed** (pre-registration, matched null, matched
   gradient budget, frozen protocol, audits). They answer a *smaller* question than
   the paper framing asks.
2. The instruments are **insensitive to the hypothesis by construction**: the
   audio→text swap moves ε by 1.5 % (`docs/condition_swap_validation.md`, T5), the
   Taylor saliency and `D_gen`/`D_mod` are computed on the full diffusion loss /
   raw ε, and conditioning enters only via `film_emb` (512→768) while the 28 gated
   convs see modality as a per-channel scale/shift. Reviewer A's correction is
   accepted: 1.5 % does **not** algebraically force ρ=0.98; ρ=0.98 is genuine
   empirical evidence — of a poor modal-signal-to-total-signal ratio.
3. The 65 % pre-recovery regime saturates: pruning error (~4.5 % of ‖ε‖, estimated)
   and its modality-dependent part (`D_mod≈3.8`) are of the same order as or larger
   than the whole conditional response of the full model (≈2.5, Gaussian estimate —
   to be measured directly).
4. The modality swap is an **AudioLDM-1 idiosyncrasy**; AudioLDM 2, TANGO, Stable
   Audio Open condition on text encoders directly. Factual gap: yes. Important 2026
   gap: small.
5. **P0-standard L1 must be a co-equal named baseline** with the published
   (inverted) artefact. Arshdeep's paper (arXiv 2607.13330) states lower-L1 filters
   are *removed*; the public checkpoint keeps them. The screening (M4-SCREEN-FOUND)
   nominally ranks P0_L1 best among pruned systems — consistent with an artefact
   anomaly, not a design.
6. **No M5 and no further M4 generation** until the contribution claim is redefined.
7. LoRA/PEFT is recovery + efficiency, never the headline (Arshdeep lists LoRA as
   future work; doing just that is an obvious continuation).

## 2. Literature verification (Reviewer B, web search 2026-08-20)

| Cited by A | Verified | Notes |
|---|---|---|
| COMET, 28 May 2026 | ✔ arXiv 2605.29628 | PLS-SVD dissection of the CLAP gap; training-free spectral truncation; condition swapping for zero-shot captioning. |
| DASH, 30 May 2026 | ✔ arXiv 2606.00798 | Dual-branch score distillation; unconditional branch unsupervised ⇒ CFG gap underdetermined; >60 % of gain from unconditional supervision. Distillation, not pruning. |
| MosaicDiff | ✔ ICCV 2025, arXiv 2510.11962 | Trajectory-aware structural pruning; optimal CFG rises with sparsity. |
| OBS-Diff | ✔ ICLR 2026, arXiv 2510.06751 | Timestep-aware Hessian; structured/N:M/unstructured; T2I. |
| "Sparse models lose CFG response", CVPR 2026 | ✔ Krause et al., *Guiding Token-Sparse Diffusion Models* | Token sparsity, not weight pruning. |
| Arshdeep, July 2026 | ✔ arXiv 2607.13330 | L1 filter pruning (**removes lower-L1**), (1,2,3,1)=65 %/(1,2,1,1)=83 %, **1M-step full FT on AudioCaps**, PANNs event-capture-rate by category: Safety-critical −73.5 % (76 % recovered), Mechanical −75 % (83 %), Environment −57.9 %, Animals −47.3 %, Vehicles −28.4 %, Speech −13.1 %. CFG not discussed. LoRA = future work. |
| "Importance-Aware OBS Pruning", 22 Jul 2026 (uses \|ε_c−ε_∅\| as importance) | ✘ **not found** | Searched arXiv/Google several ways. **Reviewer A: please supply the arXiv ID / title.** This citation is load-bearing for the claim that CFG-aware pruning importance is already taken. |
| "27 Jul 2026: positive/negative branch errors compensate; constrain separately" | ? possibly arXiv 2607.08241 *Closing the Null Space: Guidance-Aware Quantization* | That paper is **quantization**, and its point is the opposite direction (calibrate the *combined* guided prediction, not the branches separately). **Reviewer A: confirm which paper you meant.** |
| APTP | ✔ ICLR 2025, arXiv 2406.12042 | Prompt-routed architecture codes; capacity depends on prompt. |
| Diff-Pruning | ✔ NeurIPS 2023 | Taylor over informative timesteps. |

Additional anchors found by Reviewer B:

* **Hooker et al. 2019, *What do compressed deep neural networks forget?*** (arXiv
  1911.05248): compression preserves top-line accuracy while disproportionately
  damaging the **long tail** (Pruning Identified Exemplars). Follow-ups on long-tailed
  multi-label classifiers (MICCAI 2023) and LLM disagreement (*As easy as PIE*, 2025).
* Calibration-set composition is known to bias importance rankings in LLM pruning /
  quantization; for diffusion structural pruning, OBS-Diff and MosaicDiff use generic
  calibration sets and (as far as found) do not study composition at all.

## 3. Reviewer B's critique of Reviewer A's proposal #5

A proposes "semantic-event-aware structured pruning + PEFT recovery for TTA".
Hostile reading:

* Arshdeep's paper **already contains the semantic analysis** (PANNs capture rate by
  category, pre/post FT). A paper whose contribution is "we also measured it" is
  rejected. "Semantic-aware" needs a **mechanism and a predicted direction**, or it
  is just a re-weighted loss with a new name.
* "Semantic-aware criterion" risks the same failure mode as P2/P3: a reweighting that
  changes the kept set by 3–5 % and is then evaluated with 200 GPU-hours of
  recovery. **Any new criterion must first pass a kept-set divergence gate against a
  matched null (the Gate-B pattern, which the repo already has).**

What makes it scientific is to ask *why* pruning kills safety-critical/mechanical
sounds. Two competing, testable mechanisms, each with a prior in the literature:

* **H-tail (Hooker):** damage per event class is driven by class rarity in the
  training/calibration distribution (gunshot, siren, drill are rare in AudioCaps;
  speech/vehicle dominate). Prediction: per-class capture-rate loss correlates with
  class frequency; a **tail-reweighted calibration set** for Taylor saliency reduces
  rare-class damage at the same budget, *without* fine-tuning.
* **H-guidance (DASH / MosaicDiff / Null-Space line):** rare or transient events
  depend more on the conditional branch; pruning error in the guidance direction
  `(ε_c−ε_∅)` is amplified ×3.5 by CFG. Prediction: per-class guidance-gap damage
  predicts per-class capture-rate loss; raising CFG partially rescues the pruned
  model (MosaicDiff's observation, untested for TTA).
* Null alternative **H-acoustic:** damage tracks acoustic properties (transientness,
  spectral flatness) of the class, not its frequency or guidance dependence.

These are distinguishable with **cheap, forward-only diagnostics** on the existing
slots and the existing PANNs pipeline, before any recovery compute.

## 4. Candidate RQ set for plan v4 (proposal, not decided)

**Spine: phenomenon → mechanism → cheap intervention → recovery as cost baseline.**

* **RQ1′ (phenomenon, extends Arshdeep):** Under structured pruning of a TTA LDM,
  is the per-event-class degradation explained by class rarity (H-tail), by
  dependence on the CFG guidance direction (H-guidance), or by acoustic class
  properties (H-acoustic)? Pre-registered regression of per-class capture-rate loss
  on (log class frequency, per-class guidance-gap damage, acoustic descriptors), at
  two budgets (65 % and one mild budget), for P0-standard, P0-published, P1, random.
* **RQ2′ (criterion):** At matched gradient-evaluation budget, does a
  **tail-reweighted and/or guidance-direction Taylor saliency** change the kept set
  beyond a matched null (Gate B′: overlap vs two disjoint natural-distribution
  calibration halves), and does it reduce rare-class damage pre-recovery
  (generation-based, FAD fixed, ≥3 seeds) without hurting head classes?
  Baselines: P0-standard, P0-published, P1 (natural calibration), random. P2/P3 are
  dropped (resolved negative; reported in one sentence).
* **RQ3′ (recovery as cost baseline, descoped):** PEFT recovery of **two** models
  only (best criterion vs P0-standard) at a fixed budget, versus Arshdeep's 1M-step
  full FT reference; report per-class recovery and GPU-hours. 5E-style
  "criterion helps only after recovery" is testable with two models, not four.
* **Kept from v3 as a closed negative result:** RQ1/RQ2 (modality swap) reported
  as a short negative finding with the two cheap completions below, and the
  "inconclusive due to instrumentation" wording offered as the honest RQ1 outcome.

### Cheap pre-gates before any plan-v4 GPU spend (< 3 GPU-h total)

| # | Diagnostic | Closes | Cost |
|---|---|---|---|
| D1 | Re-run `m3a` saving `norm_E_a`, `norm_E_t` (signed asymmetry) and per-stratum values | v3 blind spots #2, #4 | 0.85 cr |
| D2 | Guidance-direction damage `‖(ε_P,c−ε_P,∅)−(ε_F,c−ε_F,∅)‖` on the same slots, per example → per PANNs class of the source clip | H-guidance input | ~1 GPU-h |
| D3 | Per-class capture-rate loss from the existing M4 screening audio (already generated) vs AudioCaps class frequency | H-tail first look | CPU only |
| D4 | Gate B′: kept-set overlap between P1 calibrated on two disjoint natural halves (null) vs P1-natural vs P1-tail-reweighted | whether reweighting does anything at all | ~0.5 GPU-h |
| D5 | Fix FAD/FD NaN (real-part FAD or Cnn14 FD) and re-evaluate the existing screening audio | M4 evidence usable | CPU |

Decision rule (pre-registered): if D4 overlap(tail vs natural) is not below the
null overlap, RQ2′ is dead before generation — same discipline that killed P2/P3.

## 5. Open questions delegated to Reviewer A (literature)

1. **arXiv IDs** for "Importance-Aware OBS Pruning for Diffusion Models" (22 Jul
   2026) and the 27 Jul 2026 branch-compensation paper. Granularity (structured?),
   modality (T2I only?), and whether they study *per-concept* effects.
2. Any work applying **Hooker's PIE / long-tail framing to generative diffusion
   models** (per-concept forgetting under pruning/quantization/distillation),
   2020–2026.
3. Any work on **calibration-set composition** (class balance, tail reweighting,
   prompt selection) for diffusion-model pruning — structured or unstructured.
4. Any work linking **per-concept CFG dependence** (guidance-gap magnitude per class)
   to degradation under compression, or showing rare concepts need higher guidance.
5. **TTA-specific compression 2025–2026** beyond Arshdeep (quantized AudioLDM/Tango/
   Stable Audio, AudioLCM / ConsistencyTTA distillation): did any report
   event-level semantic loss?
6. **AudioCaps / AudioSet label frequencies** for the categories in Arshdeep's
   taxonomy (are gunshot/siren/drill actually rare in AudioCaps captions?) — H-tail
   is only worth testing if the frequency gradient exists.
7. Does Arshdeep's "event-level capture rate" (PANNs top-10 recall) have an
   established lineage we should adopt verbatim for comparability?

## 6. Reviewer B's position for round 2

* Agree with A: reject the current design as a *pruning-method* paper; keep the
  negative result; pivot before spending.
* Disagree with A on the form of the pivot: not "semantic-aware pruning" as a
  method label, but a **mechanism paper** (RQ1′) with a cheap, gate-protected
  intervention (RQ2′) and recovery as a cost baseline (RQ3′). ICASSP-sized.
* The repo needs no rebuild: slots, Taylor, materialization, matched-null, Gate-B
  overlap, PEFT and PANNs pipelines are all reusable for D1–D5.
