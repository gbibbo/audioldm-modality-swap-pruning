# Master plan v4 — DRAFT rc3 (2026-08-20)

**Status: DRAFT. `docs/master_plan_v3.md` remains the execution contract until Gabriel
records DECISION-V4-00 (adopt v4) in `docs/experiment_ledger.md`.** Everything v3 says
about provenance, Git discipline, the CPU-Studio/GPU-Job policy, frozen SHAs, and
"negative results are valid" carries over unchanged. This draft replaces v3 §1–§2
(framing, RQs), §4–§6 (criteria, budget contract, timestep protocol — kept as machinery,
re-purposed), and M3–M7; it is the product of the two-reviewer audit recorded in
`docs/review/2026-08-20_reframing_round{1,2,3,4}.md`.

**rc2 → rc3 changelog (round 4 — identification layer):** H-guidance becomes an event-specific counterfactual (`c_full` vs `c_without_e`), not a whole-caption guidance norm; H-acoustic descriptors become event-specific (FineLAP-masked, fallback = single-label-clip subset, never whole-clip descriptors on multi-event clips); H-tail split into audio exposure vs calibration-caption exposure (P0 vs P1 contrast); Gate E needs `K_rand = 20` minimum (exact test, p_min = 1/(K+1)), a balanced sentinel panel, and a CPU power simulation as a Tier-1 prerequisite; RQ3′ gets a **P1-placebo** control; mechanism set and intervention holdout are disjoint at source-wav level; Gate I gets non-inferiority margins, a mechanism-general target-set rule, and rules for zero/multiple winning mechanisms; Tier 1 honestly re-costed (≈45 credits). Measured this round: 61 events have ≥200 strictly-requested captions; 22 697 train clips have exactly one requested event; 9 637 (19.5 %) have a single AudioSet label; strict aliases under-count "Speech" (1 882 requested vs 20 561 labelled).

**rc1 → rc2 changelog (round 3):** Gate E moved out of Tier 0 (Tier 0 generates no RAND audio, so it cannot be evaluated there); Gate E null needs `K_rand ≥ 10` masks, not 2; Gate M re-specified as nested block likelihood-ratio tests; temporal occupancy measured with an evaluator independent of the outcome detector; two-level pre-registered synonym protocol derived from official AudioSet aliases; mild budget fixed from measured geometry `(1,2,3,4)` = −23.7 %; FAD/FD rule restated; seed-robustness check reuses the FAD 3-seed audio; Music-drift observation demoted to exploratory with an independent-evaluator control.

---

## 0. What changed and why (one paragraph)

v3 asked whether AudioLDM's CLAP audio→text conditioning swap is selectively damaged by
structured pruning (RQ1) and whether paired audio/text Taylor saliency prunes better
(RQ2). Both pre-registered gates failed cleanly (M3A: Δ_swap = 0.0007, CI ⊃ 0; M3B:
kept-set overlap 0.9475, ρ(S_a,S_t)=0.98 while ρ(S_a,L1)=0.57). The audit found the
instruments are insensitive to the hypothesis by construction (the swap is 1.5 % of the ε
signal; conditioning enters only via `film_emb`), the 65 % pre-recovery regime is
saturated, and the 2026 literature has narrowed the gap (COMET 2605.29628; DASH
2606.00798; OBS-Diff 2510.06751; MosaicDiff ICCV 2025; Importance-Aware OBS 2607.20048;
Singh et al. 2607.13330). What remains open, and is audio-specific, is **why structured
pruning of a TTA LDM forgets some sound events far more than others** (Singh et al.
Table 3: Safety-critical −73.5 %, Mechanical −75 %, Speech −13 %), whether that is
**exposure, guidance dependence, or acoustics**, and whether a **fixed-compute,
mechanism-informed calibration** mitigates it before any recovery.

## 1. Contribution claim (what the paper will try to show)

> Structured pruning of a text-to-audio latent diffusion model causes strongly
> event-dependent semantic forgetting that aggregate metrics (FAD/KL/IS) hide. We test
> whether the disparity is explained by event exposure, by dependence on the
> classifier-free-guidance direction, or by acoustic properties of the event, and whether
> a mechanism-informed, fixed-compute calibration of Taylor saliency reduces the
> disparity at matched budget. Parameter-efficient recovery is reported as a cost
> baseline, not as the contribution.

Explicit non-claims: not a new pruning algorithm; not "CFG-aware pruning" (Importance-
Aware OBS §4.4 and MosaicDiff own that principle); not a modality-swap result (v3
negative, reported in one paragraph).

## 2. Research questions

* **RQ1′ — Phenomenon.** Under structured channel pruning at a fixed architecture
  budget, is per-event semantic loss heterogeneous beyond what a random structured mask
  of matched generic damage produces, and is the heterogeneity stable across pruning
  criteria?
* **RQ2′ — Mechanism.** Is per-event loss predicted by (H-tail) the event's exposure in
  the calibration/fine-tuning distribution, (H-guidance) the event's dependence on the
  CFG guidance direction and the pruning error in that direction, or (H-acoustic)
  pre-registered acoustic descriptors of the event? Pre-registered model comparison; any
  of the three, or none, is an acceptable answer.
* **RQ3′ — Intervention.** At matched gradient-evaluation budget, does a calibration
  reweighting motivated by the winning mechanism (a) change the kept set beyond the
  calibration-sampling null (Gate B′) and (b) reduce tail-event loss pre-recovery without
  increasing head-event loss?
* **RQ4′ — Recovery as cost baseline (tiered, optional).** With a fixed
  parameter-efficient budget, how much of the event-level loss is restored for the best
  criterion vs P0-standard, versus Singh et al.'s 1M-step full fine-tune — with the
  caveat that their fine-tuned model (FAD 1.57) beats their unpruned model (FAD 3.95),
  i.e. full FT is also AudioCaps domain adaptation.
* **RQ-swap (closed negative, v3).** Reported as: no modality-specific damage
  detectable (Gate A), no modality-specific saliency (Gate B); D1 decides the wording
  "not supported" vs "not detectable at this budget".

## 3. Hypotheses, predictions, pre-registered covariates

| Hypothesis | Prediction | Covariate (pre-registered, computed before any generation is inspected) |
|---|---|---|
| H-tail | per-event loss ↓ with exposure | **Two exposures, pre-registered separately:** (a) *audio exposure* = `log n_labelled(event)` in AudioCaps-train (`audiocaps_train_label.json`, 49 502 clips; Speech 41.5 %, Siren 1.81 %, Gunshot 1.69 %, Drill 1.51 %, Explosion 0.47 %) — what the model learned; (b) *calibration-caption exposure* = `log n_requested(event)` within the calibration pool under the strict map — what Taylor calibration protects. **Identification:** P0 uses no calibration, P1 does; a dependence on (b) that appears only in P1 is a calibration-sampling mechanism, a dependence on (a) shared by P0 and P1 is learned-representation forgetting. Sensitivity: `log n_AudioSet-unbalanced(event)`. Strict-map caveat: "Speech" has 1 882 strictly requested captions vs 20 561 labelled (captions say *talks/talking*); the expanded map must cover this family or it is excluded from the tail block. |
| H-guidance | per-event loss ↑ with the event's *own* conditional contribution and with pruning error in that contribution | **Event-specific counterfactual, not the whole-caption guidance norm** (a caption "a dog barks while a siren sounds" must not credit the same `‖ε_c−ε_∅‖` to dog and siren). For occurrence `(p,e)`: `c_without_e` = caption with the strict-map span of `e` deleted by a frozen deletion rule; for single-requested-event captions (22 697 in train) `c_without_e` ≡ unconditional. `G_event,F(p,e) = ‖ε_F(c_full) − ε_F(c_without_e)‖`, `ΔG_event,P(p,e) = ‖[ε_P(c_full)−ε_P(c_without_e)] − [ε_F(c_full)−ε_F(c_without_e)]‖`, plus the relative form `ΔG/G_F`; second contrast `G_only,F = ‖ε_F(c_only_e) − ε_F(∅)‖` with `c_only_e` = the alias phrase alone. Same `z_t`, `t`, noise for every term; forward-only; text path already accepts arbitrary strings (`modality="text"`, `list[str]`). The whole-caption `G_F`/`ΔG_P` of rc2 are reported as secondary only. |
| H-acoustic | per-event loss tracks acoustic structure | onset strength, temporal occupancy, spectral flatness, spectral flux — **computed event-specifically**: inside the FineLAP frame mask for `e` on the reference clip (mask rule frozen), or, if FineLAP fails its smoke, **only on the single-AudioSet-label clip subset** (9 637 train clips, 19.5 %), where whole-clip descriptors are event-specific by construction. Whole-clip descriptors on multi-event clips are never attributed to an event. Fixed list, no post-hoc additions. **Occupancy must not be measured with the outcome detector (PANNs)** — circularity: an event PANNs handles poorly would look both low-occupancy and poorly captured. Primary estimator: frame-level text-audio grounding with an independent model (candidate **FineLAP**, arXiv 2604.01155, EAT + RoBERTa, code github.com/xiquan-li/FineLAP; public checkpoint **unconfirmed**), with the score→duration rule frozen before use; sensitivity: PANNs `Cnn14_DecisionLevelMax` framewise output. **If FineLAP fails a CPU validity smoke on known events, occupancy is dropped from the block rather than replaced at the last minute.** |

Descriptors and exposures are frozen in a manifest (`configs/research/event_covariates.json`, sha256 in the ledger) **before** any pruned-model generation is evaluated.

## 4. Definitions

* **Event unit.** An AudioSet label `e` (display name from `class_labels_indices.csv`).
  Event set `E*` = labels with ≥ `N_min` AudioCaps-train clips **and** ≥ `n_min`
  evaluation prompts after the requested-event filter. Proposed `N_min = 200`,
  `n_min = 10` (Tier 0) / `20` (Tier 1).
* **Requested event.** `e` is requested by prompt `p` iff `e` is among the AudioSet
  labels of the source clip **and** the caption contains `e`'s display name or a
  pre-registered synonym. **Two-level protocol, both frozen with hash before any pruned
  generation is inspected:** (i) *strict map* = the official AudioSet display name and its
  comma-separated aliases as shipped in `class_labels_indices.csv` (e.g. `Gunshot, gunfire`;
  `Police car (siren)`) plus minimal morphology (plurals, obvious verb forms); (ii)
  *expanded map* = manual additions reviewed once, used only as a sensitivity analysis.
  No LLM-generated synonyms. Files: `configs/research/event_synonyms_{strict,expanded}.json`.
  This removes the construct-validity problem of scoring labels the caption never asked for.
* **Capture.** Occurrence `(p, e)` is captured iff `e` ∈ PANNs (Cnn14) top-10 of the
  generated clip — "PANNs top-10 event recall, following Singh et al. (2026)". Secondary:
  CLAP score of the event phrase vs the clip. Optional (Tier 1, API-bound): ELSA
  (2606.17404; GPT-5.2 + SAM Audio + Human-CLAP).
* **Per-event loss.** `L_S(e) = recall_base(e) − recall_S(e)` for system `S`; the
  inferential unit is the occurrence, modelled as
  `captured ~ system × covariates + (1|event) + (1|prompt)` (mixed-effects logistic);
  the pre-registered tests are the `system × covariate` interactions. Family is a
  grouping level for reporting, never the regression unit.
* **Generic damage match.** As v3: random masks at the same architecture, `D_gen`
  matched by interpolation; `K_rand` per tier.
* **Data partition (frozen at source-wav level, hashes in the ledger).** (1) *Calibration
  pool* (natural + tail-enriched; train). (2) *Mechanism set*: prompts for Gate E / Gate M
  (train clips disjoint from the calibration pool). (3) *Intervention holdout*: prompts and
  source wavs disjoint from (1) and (2), **not inspected until the intervention criterion is
  frozen**; Gate I runs only there. All train captions were seen by the generator during
  pretraining — equally for every system, stated in the paper.
* **Matched gradient budget.** As v3 §5; every calibration variant uses the same number
  of gradient evaluations.

## 5. Systems and baselines

| ID | Definition | Role |
|---|---|---|
| base | AudioLDM-M-Full `(1,2,3,5)` | reference |
| **P0-std** | keep-highest-L1, `(1,2,3,1)` | **primary scientific baseline** (DECISION-V4-01 pending) |
| P0-pub | Singh et al. published artefact (keeps lowest-L1) | reproducibility control; "vs the published artefact" wording |
| P1-nat | text Taylor, natural calibration (M3B manifest) | data-aware baseline |
| RAND×k | random structured masks, seeds as M3A | matched null |
| P1-tail / P1-guid | P1 with tail-reweighted / guidance-weighted calibration | RQ3′ intervention, only if Gate B′ passes |
| mild budget | `channel_mult=(1,2,3,4)`: **measured 317.308 M params = −23.7 %** (vs `(1,2,3,3)` −42.5 %, `(1,2,3,2)` −56.2 %, `(1,2,3,1)` −65.0 %; CPU build, this session). Engineering prerequisite: parameterize `build_pruned_unet`/`materialize` (currently hardcoded to `(1,2,3,1)`) with a bit-exact regression test at `(1,2,3,1)` | saturation control; Tier 1 |

P2/P3 are dropped (resolved negative; one sentence in the paper).

## 6. Gates (pre-registered)

* **Heterogeneity screen (Tier 0, not a gate).** On 200 stratified prompts × {base, P0-std,
  P1-nat}: between-event dispersion of `L(e)` with prompt-bootstrap CI, and Spearman
  between `L_{P0-std}(e)` and `L_{P1-nat}(e)`. Its only role is to inform DECISION-V4-02
  (fund Tier 1 or not). **Tier 0 generates no RAND audio, so Gate E cannot be evaluated in
  Tier 0.**
* **Gate E (heterogeneity, RQ1′) — Tier 1 only.** The between-event variance of
  `L_{P0-std}(e)` exceeds the 95th percentile of the same statistic across **`K_rand ≥ 10`
  random structured masks** (20 if funded) matched on `D_gen`, **and** Spearman between
  `L_{P0-std}(e)` and `L_{P1-nat}(e)` ≥ 0.5 (the pattern is a property of pruning, not of
  one criterion). The between-mask percentile is computed on the **same matched prompt
  subset for every system** (RAND masks are generated on 300 prompts each; P0-std's
  statistic for the gate is computed on that same 300-prompt subset; the 1000-prompt set
  feeds Gate M). Prompt-bootstrap of two masks does **not** estimate between-mask
  variability and is not accepted. FAIL ⇒ the phenomenon is not specific; report as such
  and stop RQ2′/RQ3′.
* **Gate M (mechanism, RQ2′).** Three pre-registered covariate **blocks**: tail = {log
  AudioCaps exposure} (AudioSet exposure as sensitivity, not forced in jointly if
  collinear); guidance = {`G_F`, `ΔG_P`}; acoustic = {onset, occupancy, flatness, flux}.
  Nested likelihood-ratio tests of the full mixed model against full-minus-block, on the
  `system × block` interactions, α = 0.01 per block; a mechanism "wins" only by its
  **whole block**, never by one descriptor chosen after the fact. AIC reported
  descriptively. "None" is a valid outcome.
* **Gate B′ (intervention changes the mask).** Kept-set overlap(P1-nat, P1-variant) falls
  below the 5th percentile of the null distribution of overlap(P1-half_i, P1-half_j)
  over ≥ 1000 natural splits of matched size and budget, computed from stored per-slot
  saliency contributions. Applied to **both** P1-mechanism and P1-placebo. FAIL ⇒ RQ3′
  dead before any generation (same discipline that killed P2/P3).
* **Mechanism → intervention rules (frozen).** No block wins ⇒ RQ3′ is not run. One block
  wins ⇒ its variant. Several win ⇒ each variant must pass Gate B′; the variant of the
  block with the larger LRT statistic goes to generation first; a combined variant only if
  pre-specified here before Gate M is read.
* **P1-placebo (mandatory control for novelty).** Same calibration-pool size, same
  reweighting magnitude, same gradient budget, targeted at a set of events of the same
  cardinality that the winning mechanism predicts **non-vulnerable**. Importance-Aware OBS
  §4.4 already shows targeted calibration protects the targeted category; only
  `P1-mech > P1-placebo` shows the *mechanism* carries information about what to protect.
* **Gate I (intervention helps) — on the intervention holdout only.** Target set `T` =
  events predicted vulnerable by the winning mechanism under a rule frozen before the
  holdout is unblinded. PASS iff (i) target recall gain ≥ `δ_target` with 95 % CI excluding
  0; (ii) **non-inferiority** on non-target events: CI lower bound of the recall change
  > `−δ_harm`; (iii) FAD/FD/KL within pre-set margins vs P1-nat; (iv) `P1-mech − P1-placebo`
  on `T` > 0 with CI excluding 0. Proposed values (to be fixed in DECISION-V4-07 before
  unblinding): `δ_target = +5 pp`, `δ_harm = 2 pp`, FAD/FD relative +5 %, KL +0.05.

## 7. Compute — measured units and credit tiers

Measured (T4, `docs/compute_budget.md`, M0-006-TGEN): generation **8.44 s/clip at
S=50**; saliency **0.222 s/backward**; diagnostic forward **≈0.059 s/sample**; PEFT train
**1.67 s/step at batch 8**. Observed job price ≈ **1.05 credits/GPU-h**. Spent to date:
**4.205 credits** (14 jobs). Remaining spendable at CG-001 terms: **≈3–5 credits**.

| Tier | Content | Est. credits |
|---|---|---|
| **0 — closes v3 + screens v4** (fits current balance) | D1+D2 in one job: {base, P0-std, P0-pub, P1-nat, RAND×5} × 500 slots × {audio-cond, text-cond, uncond} forward (~0.35); Gate B′ saliency on enriched pool 512 ex × K=5 × 2 draws with per-slot storage (~0.5); event-level screening: 200 stratified prompts × {base, P0-std, P1-nat}, 1 seed (~1.5); smokes (~0.3) | **≈2.7** |
| **1a — RQ1′/RQ2′ confirmatory** | mechanism set: 50 events × 20 prompts × {base, P0-std, P0-pub, P1-nat, mild P0-std, mild P1} (6 000 clips, 1 seed); **Gate E sentinel panel** 20 events × 15 prompts (stratified by exposure × family) × **RAND×20** (6 000 clips); FAD guardrail 3 seeds × 300 clips × 3 systems (1 800 clips) — that audio is also scored for event recall as the seed-robustness check; counterfactual ε forwards for H-guidance are negligible | ≈33 |
| **1b — H-guidance CFG grid** | 3 CFG × {base, P0-std} × 300 prompts (one CFG value coincides with 1a) | ≈3 |
| **1c — RQ3′ generation** | intervention holdout 500 prompts × {base, P1-nat, P1-mech, P1-placebo} (2 000 clips; only if Gate B′ passes for P1-mech) | ≈5 |
| **Tier 1 total** | **≈41 + smokes/builds ≈ 45 credits.** Lever if unaffordable: sentinel panel 20 × 10 prompts (saves ≈5). **Tier 0 is screening only and cannot yield the paper; a positive Tier 0 is not to be stretched into an underpowered submission.** | |
| **2 — RQ4′ PEFT recovery** | 2 models × 25k steps + eval; optional unpruned-PEFT control | ≈25–35 |

Hard rules: no tier starts without Gabriel's written authorization in the ledger; the
2.0-credit reserve of DECISION-CG-001 stands; every GPU job is preceded by a CPU dry-run
and a GPU smoke; **FAD/FD NaN (F-eval-3) must be fixed before any new audio generation
that will be used scientifically** (it does not block D1/D2 or Gate B′, which use no FAD;
in practice it is fixed first because it is CPU work).

## 8. Milestones and dates (submission target 2026-09-16 retained, with v3's rule: do not submit underpowered)

| Milestone | Content | Target |
|---|---|---|
| M4-0 | Gabriel decisions (§10); fix FAD/FD; freeze event set, synonym maps, covariate manifest, data partition (calibration / mechanism / holdout), sentinel panel, prompt manifests (hashes in ledger) | 08-22 |
| M4-1 (Tier 0) | D1+D2 job; Gate B′ saliency job; screening generation; **heterogeneity screen read-out (not Gate E)**; Gate B′ verdict; FineLAP validity smoke (CPU) | 08-25 |
| M4-1b | **CPU power simulation** from Tier-0 rates: Gate E design must reach ≥ 80 % power at the pre-set minimum effect, else resize the sentinel panel before any Tier-1 spend | 08-26 |
| M4-2 | Tier decision on evidence; if Tier 1 funded: 1a (RAND×20 → Gate E; Gate M) / 1b / Gate B′ for mech + placebo / 1c on the holdout | 08-27 → 09-06 |
| M4-3 | Gate M model comparison; claims matrix; RQ-swap wording from D1 | 09-07 |
| M4-4 (optional Tier 2) | PEFT recovery of 2 models | 09-08 → 09-12 |
| M8′ | paper; coauthor review | 09-12 → 09-16 |

## 9. Claims matrix mapping (to be applied to `docs/claims_matrix.md` on adoption)

RQ1 → `rejected` (wording pending D1). RQ2a/RQ2b → `rejected` at the saliency stage
(M4 screening recorded, not promotable). New rows: RQ1′ (Gate E), RQ2′ (Gate M), RQ3′
(Gate B′ + Gate I), RQ4′ (`unavailable` until Tier 2 is funded), SEM folded into RQ1′,
EFF unchanged.

## 10. Decisions required from Gabriel before execution

* **DECISION-V4-00** adopt v4 (supersedes v3 RQs/M3–M7) — or reject and keep v3.
* **DECISION-V4-01** P0-standard becomes the primary scientific baseline; P0-published
  becomes the reproducibility control (amends DECISION-M3B-002/003).
* **DECISION-V4-02** credit tier authorized now (0 only / 0+1 / 0+1+2) and the balance
  figure the plan may assume.
* **DECISION-V4-03** ELSA: adopt as secondary metric (API cost, reproducibility caveat) or
  defer.
* **DECISION-V4-04** `N_min = 200`, `n_min = 10/20`, mild budget `(1,2,3,4)` (−23.7 %),
  RAND×5 for Tier-0 forward diagnostics, **`K_rand ≥ 10` for Gate E** (20 if funded).
* **DECISION-V4-05** keep RQ-swap in the paper as a one-paragraph negative result (yes /
  no).
* **DECISION-V4-06** occupancy estimator: FineLAP (conditional on a CPU validity smoke) or
  drop occupancy from the acoustic block; if FineLAP fails, H-acoustic is restricted to the
  single-label-clip subset.
* **DECISION-V4-07** partition sizes (calibration pool / mechanism set / holdout), sentinel
  panel composition, minimum detectable effect for the power simulation, and the Gate I
  margins (`δ_target`, `δ_harm`, FAD/FD/KL) — all frozen before the holdout is unblinded.

## 10b. Exploratory observations (never in the claims matrix unless promoted by a decision)

* **"Music" drift.** AudioCaps-train has 0 Music clips, yet PANNs top-1 = Music rises from
  9 (base) to 24 (P3) of 100 in the M4 screening. Rival explanation: AudioSet's hierarchical
  ontology lets PANNs back off to broad classes (Music/Speech) when audio is ambiguous —
  i.e. detector behaviour, not generator collapse. Promotion requires agreement of an
  independent evaluator (CLAP/FineLAP similarity to "music" vs the requested event) **and**
  the PANNs distribution on real reference audio as control.

## 11. Literature positioning (verified 2026-08-20; see round-1/2 docs for URLs)

Owned by prior work: Taylor/OBS/trajectory-aware diffusion pruning (Diff-Pruning 2023,
OBS-Diff ICLR 2026, MosaicDiff ICCV 2025); CFG-response importance and category-
targeted calibration for T2I (Importance-Aware OBS 2607.20048, §4.4, T2I only); CFG
branch fidelity in distillation/quantization (DASH 2606.00798; 2607.24731; 2607.08241);
L1 structured pruning of AudioLDM with a family-level PANNs analysis (Singh et al.
2607.13330); compression forgets the long tail in classifiers (Hooker et al. 2019).
Not found: event-level forgetting under TTA compression with a mechanism comparison;
exposure-vs-guidance-vs-acoustics attribution; calibration-composition effects for
structured pruning of an audio U-Net; any PIE-style analysis of generative compression.
