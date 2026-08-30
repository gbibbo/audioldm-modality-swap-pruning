# What Does Post-Pruning Recovery Restore in a Conditional Diffusion Model? A Controlled Paired Evaluation of AudioLDM

> **Current ICASSP-2027 draft (started 2026-08-30).** Supersedes `icassp2027_scenario_b_skeleton.md`
> (obsolete Scenario-B/adapter-survival thesis, preserved for provenance). Target: 4 pages + refs,
> deadline **2026-09-16**. Author: Gabriel Bibbó (finalize affiliation). Result-independent sections are
> written; the only data-dependent item is **Finding 3 (Arm D)** — two branches are pre-written; insert
> the observed numbers and keep the matching branch. **Framing: an evaluation-methodology study for
> post-pruning recovery in conditional diffusion models, with AudioLDM as a controlled case study — NOT
> an audit/attack of Singh et al.** The studied recovery artifact is the object of study, not an adversary.
>
> **Wording invariants (binding):** never assert *catastrophic forgetting*, *retraining from scratch*, or
> *functional replacement* as established mechanisms. Use "strong out-of-domain degradation," "large
> domain-dependent recovered-vs-pruned interaction," "large post-pruning recovery drift." Prefer "no
> statistically resolved advantage" / "approximately matched point estimates" over "equivalent" (no
> equivalence bounds were tested). Novelty stated as "we found no prior work…", never "first".

## One-sentence contribution
We introduce a controlled, paired evaluation framework for post-pruning *recovery* in conditional
diffusion models — using common generation noise and prompt-clustered inference — and show, for a
published recovered AudioLDM checkpoint, that extensive recovery fine-tuning produces **no
statistically resolved recovered-over-pruned advantage across six evaluation axes at a controlled
in-domain operating point**, while exhibiting a **large domain-dependent recovered-vs-pruned
interaction** (severe on held-out music) — evidence that "recovery" is multi-dimensional and
context-dependent rather than a recipe-invariant scalar.

## 1. Introduction (result-independent — DRAFT)
Structured pruning plus recovery fine-tuning is a standard route to cheaper generative diffusion models,
and "recovery" is typically summarized by the restoration of an aggregate benchmark number (e.g. FAD/KL
on an in-domain test set). It is rarely asked whether that restoration is **domain-robust,
evaluator-robust, inference-recipe-robust, or compatible with downstream parameter-efficient
modification**. We study this for text-to-audio (TTA) generation with AudioLDM, using a published
structurally-pruned checkpoint and its released recovery-fine-tuned counterpart.

Research question: **what properties are actually restored by post-pruning recovery fine-tuning in a
conditional diffusion model?** We answer with a controlled paired design (same prompt, common generation
noise across compared systems, prompt as the statistical unit) and a panel of complementary evaluators.

Contributions (stated narrowly):
1. A controlled **paired evaluation framework** for post-pruning recovery: common-random-number
   generation across compared systems, deterministic manifests, prompt-clustered bootstrap, paired
   contrasts instead of unpaired aggregate comparisons.
2. Evidence of a **large domain-dependent recovered-vs-pruned interaction** in AudioLDM (in-domain
   AudioCaps vs held-out music).
3. Evidence that **recovered ≈ pruned in-domain** under a controlled operating point across **six**
   independent evaluation axes (CLAP, Human-CLAP, PANN event-capture, KL, FAD, FD).
4. A prospectively-specified test of **temporal-scale sensitivity** (Arm D). *[Final wording of this
   contribution is fixed only after Arm D is observed — see §4.3.]*
5. Supporting analysis of **parameter drift** and **adapter-transfer** behavior, carefully bounded, plus
   reusable methodological machinery (deterministic mask-induced LoRA slicing).

## 2. Related work (compact — DRAFT)
**Diffusion pruning / compression & recovery.** Structural pruning of diffusion U-Nets with fine-tuning
recovery (Diff-Pruning, Fang et al., NeurIPS 2023 [2305.10924]; BK-SDM, Kim et al., ECCV 2024
[2305.15798]; 2ndMatch, CVPR 2026 [2506.05398]) evaluates recovery at a fixed sampler. TTA pruning:
Singh et al. [2607.13330] prune an AudioLDM-family model and recover with lightweight fine-tuning,
reporting FAD/KL and PANN top-10 event capture on the AudioCaps test set at 200 inference steps.
**Compression × inference regime.** Post-training quantization studies show compression error accumulates
along the reverse trajectory (PTQD, He et al., NeurIPS 2023 [2305.10657]) and that few-step models are
disproportionately sensitive to further compression (Q-Sched [2509.01624]; MixDQ [2405.17873]). We found
**no prior work** establishing, for structural pruning, an *evaluation-time* interaction between recovery
and sampler step-count/solver — an open question we frame as a limitation (§6), not a claim.
**Fine-tuning and OOD.** Fine-tuning on a narrow distribution can improve in-distribution metrics without
implying broader generalization; this motivates our domain-robustness axis. **TTA evaluation.** CLAPScore
correlates imperfectly with subjective alignment, motivating human-aligned CLAP variants; FAD is
encoder- and sample-size-sensitive. These limits motivate our multi-axis, paired treatment.

## 3. Method (result-independent — DRAFT; a core contribution)
### 3.1 Systems
Three systems with exact provenance: **dense** (AudioLDM-M-Full, itself AudioCaps-specialized by an
additional 0.25M-step fine-tune per the source), **pruned-only** (published `(1,2,3,1)` L1 structural
prune, U-Net 415.955M→145.674M params, −65.0%; verified pure prune-and-merge, never fine-tuned), and the
**published recovered** checkpoint (the pruned budget fine-tuned ~1M steps / ~40 epochs on AudioCaps,
Zenodo record, md5-verified). We do **not** re-train any system.

### 3.2 Paired evaluation design
For each prompt we generate with **common generation noise** shared across the compared systems (x_T a
deterministic function of (ytid, replicate) via a fixed salt), so system differences are not confounded
by sampling noise. Deterministic, hash-frozen manifests fix prompt selection and seeds. The **prompt
(ytid) is the independent statistical unit**; replicates are averaged within prompt before a
**prompt-clustered percentile bootstrap** (B=10000). We report **paired contrasts** (recovered−pruned per
prompt), never comparisons of unpaired aggregate scores. Scoring uses **frozen, revision-pinned**
models with a fixed batch/order/seed convention (part of the endpoint definition).

### 3.3 Domains
**In-domain:** AudioCaps test captions. **Out-of-domain:** a held-out music battery. An exposure audit
establishes that the recovery corpus is **not literally music-free** but has **near-zero exposure to the
tested music subdomain** (music <1% by ontology, ≈2% lexically, ~zero genre/hip-hop labels; the battery
is 100% hip-hop-style music captions). We therefore say "near-zero exposure to the tested subdomain,"
never "music-free."

### 3.4 Operating point
Controlled common operating point for the primary comparisons: **3.84 s clips, DDIM 50 steps, guidance
2.5, eta 0, fp32, single generation, EMA weights**. This is deliberately a single declared regime;
§3.6/§4.3 examine one factor (temporal extent). We note the published recovery numbers were obtained
under a different framework recipe (10.24 s, 200 steps; framework-default guidance 3.5 and best-of-3-by-
CLAP — see §6/reproducibility).

### 3.5 Metric panel and roles
Primary **CLAP** cosine (text–audio alignment). Diagnostic panel (roles differ; not statistically
equivalent): **Human-CLAP** (human-aligned CLAP variant; corroborative, **not** human listening),
**PANN top-10 event capture** (semantic events, directly tied to the recovery paper's analysis), **KL**
(PANN-logit divergence to the real clip; the recovery paper's metric), **FAD** (VGGish, distributional),
**FD** (PANN-2048, distributional). Distributional metrics are **descriptive** at our sample sizes; per
a variance analysis, only CLAP has usable per-prompt (interaction) power at our budget.

### 3.6 Recovery artifact characterization
We describe the recovery fine-tune **factually**: ~1M steps / ~40 epochs on AudioCaps, and a **large
parameter-space displacement** from the pruned initialization (weighted-mean cosine ≈ 0.376 to the pruned
init; ≈59% of tensors displaced beyond their own initial norm). We call this **large post-pruning
recovery drift** (parameter-space evidence only) — **not** retraining from scratch or functional
replacement.

## 4. Results (organized by FINDING, not chronology — DRAFT)
### 4.1 Finding 1 — Large domain-dependent recovered-vs-pruned interaction
Frozen primary CLAP (recovered−pruned): held-out **music R_music = −0.0941, CI95 [−0.1241, −0.0646]**;
in-domain **AudioCaps R_AC = −0.0024, CI95 [−0.0267, +0.0214]**; **interaction I = R_AC − R_music =
+0.0917, CI95 [+0.0535, +0.1311]**. Human-CLAP corroborates: **I_HC = +0.172, CI95 [+0.121, +0.224]**.
The recovered checkpoint is strongly degraded relative to pruned-only on the OOD music battery while
approximately matching pruned-only in-domain — a large, evaluator-corroborated domain-dependent
interaction. *(This is a positive measured result even though the pre-registered strict recovery-reversal
gate did not pass; we report the interaction, not the gate.)* **[Figure 2.]** We discuss forgetting /
domain specialization as **candidate** mechanisms only (§5).

### 4.2 Finding 2 — In-domain equivalence is metric-invariant at the controlled operating point
On the same in-domain outputs, no evaluated axis establishes a recovered-over-pruned advantage; the
pattern is **dense ≫ {pruned ≈ recovered}** throughout (recovered−pruned contrast; ↑ higher-better,
↓ lower-better):

| Metric | Dir | Dense | Pruned | Recovered | Δ(rec−pruned) | CI95 / status |
|---|---|---|---|---|---|---|
| CLAP | ↑ | 0.204 | 0.100 | 0.098 | −0.0024 | [−0.027, +0.021] |
| Human-CLAP | ↑ | 0.392 | 0.229 | 0.256 | +0.028 | [−0.012, +0.068] |
| PANN capture | ↑ | 0.446 | 0.339 | 0.362 | +0.023 | [−0.044, +0.088] |
| KL | ↓ | 2.852 | 3.424 | 3.358 | −0.067 | [−0.396, +0.251] |
| FAD | ↓ | 8.83 | 14.53 | 14.70 | +0.17 | descriptive (n=96) |
| FD | ↓ | 71.1 | 78.4 | 80.8 | +2.4 | descriptive (n=96) |

No metric — including the recovery paper's own FAD and KL — resolves a recovered advantage over
pruned-only at this operating point. This is stronger than a single-metric null: the in-domain
recovered≈pruned relationship is **metric-invariant**, not an artifact of one scorer. **[Table 1.]**

### 4.3 Finding 3 — Temporal-scale sensitivity (Arm D, prospectively specified) — INSERT AFTER OBSERVATION
Prospectively-specified paired follow-up (protocol frozen before generation; V1.1 outcome permanently
FALSE and untouched): change **only** temporal extent, 3.84 s → 10.24 s (DDIM50, guidance 2.5, single),
pruned & recovered, 80 ytids, r0, matched 80-item CLAP rescoring. Primary interaction **J_CLAP =
R_alt(10.24 s) − R_ctrl(3.84 s)**, paired bootstrap. Corrected sensitivity MDE ≈ 0.065 (80% power).
Insert **R_ctrl_80, R_alt, J_CLAP** with CIs, then keep exactly ONE branch:

**[BRANCH POSITIVE — if J_CLAP>0 and lo95(J)>0]** "Matching the temporal scale used during recovery
fine-tuning changes recovered-vs-pruned behavior (J_CLAP = __ [__ , __])." Only if additionally
`R_alt ≥ +0.025 and lo95(R_alt)>0`: "…and at the recovery model's training/evaluation duration the
recovered checkpoint shows a material advantage over pruned-only (R_alt = __ [__ , __]), supporting
temporal-scale conditionality for this checkpoint." If J>0 but R_alt not material: "…duration changes the
relationship but does not establish substantive recovery." (Do **not** call recovery restored.)

**[BRANCH NULL — if J_CLAP CI crosses 0]** "Matching the recovery model's training/evaluation temporal
scale does not resolve the recovered-vs-pruned tie (J_CLAP = __ [__ , __]). We do **not** infer that
guidance or best-of-3 therefore carries the published gain: a category-A difference — **DDIM 50 vs the
published 200 steps** — remains unresolved (see §6)." (No rescue experiment is run.)

Secondaries (Human-CLAP, KL, PANN capture, FAD/FD) reported descriptively as the same matched paired
interaction; none changes the primary conclusion.

### 4.4 Finding 4 — Parameter drift and adapter behavior (supporting, bounded)
Recovery induces a **large movement in weight space** (§3.6). A separate adapter probe (deterministic
mask-induced LoRA slicing; exact restricted-ΔW preservation; no adapter retraining/data) finds the
legacy adapter's **uplift collapses on the recovered checkpoint** (standalone CLAP loss E ≈ 0.174
[0.145, 0.204]) more than on pruned-only (E ≈ 0.080), with a positive differential-fragility statistic
(D ≈ 0.044 [0.019, 0.069]) — **but** this cannot be cleanly separated from the recovered checkpoint's
large generic degradation, so we report it as **bounded, descriptive** evidence, not a causal
adapter-specific claim. The **mask-induced LoRA slicing** is presented as reusable methodological
infrastructure (space permitting), not the main empirical claim.

## 5. Discussion (synthesis, not rescue — DRAFT skeleton)
Post-pruning "recovery" is **multi-dimensional and context-dependent**: a single benchmark number does
not establish restoration of the original model's behavior. Our evidence supports **domain/context
dependence** (Finding 1) and **metric-invariant in-domain non-advantage** (Finding 2); Arm D determines
whether **temporal scale** is a further demonstrated dependency (Finding 3). Implications: compressed
generative models should be evaluated under **multiple domains / declared operating regimes**; recovery
should be assessed **relative to pruned-only**, not only against absolute benchmark values; **paired**
evaluation reveals interactions hidden by aggregate tables; downstream parameter-efficient compatibility
should not be assumed after large recovery drift. Claims kept proportional to evidence; forgetting/
specialization named only as candidate mechanisms.

## 6. Limitations (write now — DRAFT)
1. One AudioLDM pruning severity and one published recovery checkpoint (case study, not a universal claim
   about post-pruning recovery). 2. No causal proof of catastrophic forgetting / domain specialization.
3. No subjective listening study. 4. CLAP and Human-CLAP are related model families. 5. FAD/FD are
   sample-size-limited on our controlled subsets. 6. The full published inference recipe is not
   reproducible from explicit author reporting. 7. The paper verifies 200 steps / ~10 s / 964 pairs, but
   guidance 3.5 and best-of-3-by-CLAP are upstream-framework defaults, not independently verified as the
   settings that produced the reported metrics. 8. **DDIM 200 remains untested after Arm D.** 9. Hence if
   Arm D is null we cannot conclude *why* the published numbers differ. 10. The study is a case study of
   one recovery procedure. *These limitations increase credibility; no additional GPU experiment is
   planned for this submission.*

## 7. Reproducibility note (careful wording — DRAFT)
The pruning paper explicitly reports 200 inference steps, ~10 s clips, and 964 AudioCaps test pairs; its
public repository defers evaluation to the upstream AudioLDM framework, whose defaults include guidance
3.5 and best-of-3-by-CLAP candidate selection. **Some inference parameters that materially affect
generation are inherited from the upstream framework but are not explicitly specified in the pruning
paper, complicating exact reproduction of the reported operating point.** We make no claim of misconduct
or error, and do **not** assert the published results "depend on" best-of-3 absent direct evidence.

## 8. Figures / tables (plan — generate from durable artifacts; no fabricated Arm-D value)
* **Fig. 1** — concept: dense → pruned → recovered, evaluated in-domain (AudioCaps) vs OOD (music) under
  paired operating points. (schematic)
* **Fig. 2** — cross-domain recovered−pruned contrast with CIs: music −0.094, AudioCaps −0.002,
  interaction +0.092; add the Arm-D duration contrast when observed. (from `reversal_v1_1_result.json` +
  frozen music baseline)
* **Table 1** — metric-concordance at the controlled OP (from `recovery_metric_audit_1_result.json`).
* **Table 2 / compact row** — Arm D: R_ctrl_80, R_alt, J_CLAP (from
  `op_duration_discriminator_1_result.json`, after observation).
Figure/table generation scripts to live under `scripts/research/paper_figs/` and read only durable
artifacts.

## References (to expand — primary sources only)
AudioLDM (Liu et al. 2023); Singh et al. [2607.13330]; Diff-Pruning [2305.10924]; BK-SDM [2305.15798];
2ndMatch [2506.05398]; PTQD [2305.10657]; Q-Sched [2509.01624]; MixDQ [2405.17873]; CLAP
(laion/clap-htsat-fused); Human-CLAP (sarulab human-clap-wsce-mae); PANNs (Cnn14); FAD/VGGish.
(Verify all ids/venues against primary sources before submission; several 26xx ids are unverified.)
