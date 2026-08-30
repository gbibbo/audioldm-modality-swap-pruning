# OPERATING-POINT-DISCRIMINATOR — DESIGN AUDIT (design only; no GPU, no audio, no prereg)

Purpose: choose the smallest scientifically informative recovered-vs-pruned experiment that tests
whether the ordering depends on inference conditions. **Wording discipline (binding):** we do NOT yet
have evidence that the published recovery is operating-point-conditional. We know only (1) Singh reports
recovery under their setup; (2) under our controlled 3.84 s/DDIM50/guidance2.5/single-gen setup
recovered ≈ pruned across six axes; (3) therefore operating-point dependence is a *plausible unresolved*
explanation. Everything below is design; nothing is launched or preregistered.

Remaining budget ≈ **1.46 cr**. Envelope for the planned run ≤ **1.20 cr** (retain ≥0.25–0.30 cr).

## 1. VERIFIED RECIPE — A (paper) / B (Singh repo) / C (upstream default)

Verified this session against the primary paper (via prior audit `2026-08-27_recovery_reversal_audit.md`,
quotes verbatim), the Singh public repo `_external/PruningAudioLDM`, and the upstream AudioLDM framework
config `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`.

| Factor | Value | Category | Evidence |
| --- | --- | --- | --- |
| AudioCaps test size | 964 pairs | **A** paper-stated | paper §5 |
| Clip duration | 10 s (config **10.24 s**) | **A** duration; **C** exact 10.24 | paper "10 s"; config `preprocessing.audio.duration: 10.24` (line 35) → latent_t 256 |
| DDIM steps | 200 | **A** paper-stated | paper §5; also config `ddim_sampling_steps: 200` (line 145) |
| Guidance scale | 3.5 | **B/C** framework default, **NOT paper** | paper leaves guidance unspecified; config `unconditional_guidance_scale: 3.5` (line 144) |
| Candidate count / best-of | 3, **best-of-3-by-CLAP** | **B/C** framework default, **NOT paper** | config `n_candidates_per_samples: 3` (line 146); selection `ddpm.py:1937-1945` `argmax(clap.cos_similarity)` |
| eta / sampler | DDIM (eta unspecified) | **C** | upstream |
| Precision / EMA | fp?, EMA on | **C** | upstream inference default |
| Caption policy | 1 caption/clip | **C** | upstream |

**Answer to the critical question:** best-of-3-by-CLAP and guidance 3.5 are **NOT VERIFIED AS PART OF THE
SINGH EVALUATION.** The Singh README (`_external/PruningAudioLDM/README.md:193`) says *"Please follow the
official AudioLDM repository for evaluation"* — there is **no eval config or eval script in the Singh
repo**; the numbers FAD 1.57 / KL 1.678 are produced by whatever the AudioLDM framework defaults to. The
paper independently states only **200 steps / 10 s / 964 pairs**. So guidance 3.5 and best-of-3 are
*framework-prescribed defaults*, not paper-confirmed evaluation settings. **We must NOT design around
best-of-3 or guidance 3.5 as if they were verified Singh choices.** (Note: best-of-3-by-CLAP is itself an
alignment-maximizing selection that would partially mask alignment collapse — relevant to interpretation,
but it stays out of the primary design because it is unverified and, at 3× compute, unaffordable.)

**Training temporal extent:** the recovery fine-tune uses "the same finetuning configuration as
AudioLDM-M-Full" (paper), i.e. `duration: 10.24 s` (config). So the recovered weights were adapted to a
**10.24 s temporal extent and evaluated at 10.24 s**; our V1.1 generated at **3.84 s** — a genuine
train/eval temporal-scale mismatch. This makes **duration** a mechanistically motivated factor, not
arbitrary tuning.

## 2. OPERATING-POINT FACTORS THAT TRULY DIFFER

| Factor | V1.1 control | Singh verified eval | Verified difference? | Compute multiplier | Scientific plausibility |
| --- | ---: | ---: | --- | ---: | --- |
| duration | 3.84 s (latent 96) | 10.24 s (latent 256) | **YES** (A) | ×2.67 (latent_t) | **High** — recovered was fine-tuned at 10.24 s; train/eval scale match |
| DDIM steps | 50 | 200 | **YES** (A) | ×4.0 | **Medium** — AudioLDM ablation: steps materially change eval, but saturate by ~100 |
| guidance | 2.5 | 3.5 | framework only (B/C) | ×1 | Low-verified — not a paper choice |
| best-of | 1 | 3 (by CLAP) | framework only (B/C) | ×3 | Low-verified + selection confound |
| eta/precision/EMA | eta0/fp32/EMA | unspecified | no | ×1 | Low |

**Primary-source step evidence (AudioLDM, Liu et al. 2023 ablation):** DDIM 50 → FD 35.71 / KL 2.01;
100 → 30.17 / 1.94; 200 → 29.48 / 1.97. Most gain is 50→100; little 100→200. So sampler discretization
is a genuine factor but **largely saturates by 100 steps** (relevant if a steps arm is ever run: DDIM100
captures most of the effect at half the DDIM200 cost). No causal inference drawn yet.

## 3. COST MODEL (from settled V1.1, measured)

Measured: **1.2623 cr / 576 WAV** at 3.84 s/DDIM50/single (T4) → gross **0.002191 cr/WAV**. Marginal
per-WAV ∝ `ddim × latent_t × best_of` (U-Net forwards dominate); a conservative fixed job overhead
**0.09 cr** (container + 2 model loads + CLAP/VAE init, from prior smokes) is added. **Pruned + recovered
only** (dense not needed to test whether the rec-vs-pru ordering moves; dense is descriptive and dropped
for budget). Full table in `configs/research/op_discriminator_design_audit.json`; key cells (cr):

| Arm (2 systems) | 96×1 | 80×1 | 64×1 | 48×1 | 32×1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **S** steps200 / 3.84 s / single | 1.77 | 1.49 | 1.21 | 0.93 | 0.65 |
| **D** dur10.24 s / DDIM50 / single | 1.21 | 1.02 | 0.84 | 0.65 | 0.46 |
| **F** full-Singh 200/10.24 s / **best-of-3** | 13.6 | — | 9.1 | 6.8 | 4.6 |
| **F′** full-Singh 200/10.24 s / single | 4.58 | — | 3.08 | 2.33 | 1.59 |

**Consequence:** the full verified-Singh recipe (F′, even dropping the unverified best-of-3) is
**unaffordable** at any powered N (≥2.3 cr for n≥48; only ~24×1 fits the envelope, and that is
underpowered — see §4). The affordable one-factor arms are **D (duration, cheapest)** and **S (steps)**.
No benchmark is needed to decide; the multipliers are anchored to a real settled cost.

## 4. STATISTICAL EFFICIENCY — clusters vs replicates, endpoint comparison

From the V1.1 96×2 grids (per-ytid recovered-advantage contrast, oriented so + = recovered better),
random-effects decomposition (`op_discriminator_design_audit.py`):

| Endpoint | mean | sd_between | sd_within | note |
| --- | ---: | ---: | ---: | --- |
| **CLAP** | −0.0024 | 0.074 | 0.138 | within-prompt (replicate) noise dominates |
| KL (pru−rec, nats) | +0.067 | 1.02 | 1.79 | enormous noise vs mean |
| PANN capture (events) | +0.057 | 0.48 | 0.97 | noisy vs mean |

**Clusters beat replicates.** For a fixed WAV budget `N = n·r`, SE²(mean contrast) =
(σ_b² + σ_w²/r)/n, so at equal cost 96×1 always beats 48×2 (adds independent clusters, which cut the
irreducible between-prompt term that replicates cannot). Confirmed numerically: CLAP interaction MDE
96×1 = 0.057 < 48×2 = 0.064. **Use 1 replicate, maximize independent ytids.** Ytid-level pairing across
OPs (reusing the same ytids) further cuts the between-prompt part by ~(1−ρ); with within-noise dominant
the gain is modest (~10% at ρ=0.7).

**Interaction MDE (80% power ≈ 2.8·SE_J), J = D_alt − D_ctrl:**

| Design | CLAP MDE (ρ0 / ρ0.7) | KL MDE (nats) | verdict |
| --- | --- | --- | --- |
| 96×1 | 0.057 / 0.051 | 0.75 | CLAP powered for a large restoration; KL/PANN not |
| 80×1 | 0.061 / 0.054 | ~0.80 | " |
| 64×1 | 0.067 / 0.059 | 0.88 | " |
| 48×1 | 0.075 / 0.066 | 1.00 | " |

**Endpoint choice is forced by the data:** CLAP is the ONLY endpoint with usable power at this budget
(MDE ≈ 0.06, i.e. it detects a *large* recovery restoration ≳0.06, roughly 2× SESOI). KL's MDE (~0.75
nats) dwarfs its control contrast (0.067) → unpowered; PANN capture similarly (MDE ≈ 0.4 events vs 0.057).
FAD is distributional with no per-prompt CI. This **overrides the intuition that KL should be primary** —
KL/PANN/FAD stay as *secondary, descriptive* Singh-metric links, but cannot carry the primary inference.

## 5. WHAT WE ARE DETECTING — the interaction J

Primary quantity = **operating-point × recovery interaction**, oriented so + always means "recovered
advantage grows at the alt OP":

```
R_ctrl = mean_ytid[ CLAP(recovered) − CLAP(pruned) ]  at 3.84s/DDIM50   (already observed, ≈ −0.002)
R_alt  = mean_ytid[ CLAP(recovered) − CLAP(pruned) ]  at the alt OP     (new arm)
J_CLAP = R_alt − R_ctrl        (ytid-paired: J_i = D_alt_i − D_ctrl_i, bootstrap ytids)
```

The interaction **cancels the OP main effect** (CLAP's absolute shift with duration cancels in the
within-OP difference), so it is robust to CLAP behaving differently at 10.24 s. V1.1 samples stay frozen;
the new arm reuses the same ytids (blind reuse — selection is independent of any V1.1 outcome). Any subset
< 96 is drawn by a NEW pre-data salt hash over ytid; never by R_AC/KL/PANN/CLAP/caption/difficulty/
waveform. Within the alt arm, pruned and recovered share x_T per ytid (common-random-number) to preserve
the paired-contrast precision.

## 6. PRIMARY ENDPOINT (recommended, data-driven)

**Primary = J_CLAP** (CLAP recovered-advantage interaction across OPs). Rationale, scored against the four
constraints: (1) relevance — CLAP is our V1.1 primary, so the arm directly extends the frozen negative;
(2) interpretability — paired, per-prompt, interaction cancels OP main effect; (3) power — the only
endpoint with MDE (~0.06) below a plausible large restoration at ≤1.2 cr; (4) robustness — paired,
distribution-free bootstrap, no encoder-set-size pathology. **Secondaries (descriptive, tie to Singh's
recovery metrics, individually underpowered): KL, PANN top-10 capture, FAD (distributional).** FAD is
never the sole primary at n = 24–96.

## 7. ABSOLUTE-FAD CAVEAT

Our n=96 controlled FAD (dense 8.83 / pruned 14.53 / recovered 14.70) is NOT comparable to Singh's 1.57,
because operating point, generated-set size, and eval details differ. The arm compares **recovered vs
pruned under a common new OP** (the interaction), and explicitly does NOT ask whether a small subset
reaches Singh's absolute 1.57. This is stated in the design so no reviewer reads a FAD-target into it.

## 8. STRATEGY COMPARISON

| | A: steps-only | B: duration-only | C: small screen (steps+dur, no interaction arm) | D: full verified-Singh subset |
| --- | --- | --- | --- | --- |
| Change | DDIM 50→200 | 3.84→10.24 s | both, separately | 200+10.24 s (+guidance if verified) |
| N (envelope) | 48×1 (0.93 cr) | 80×1 (1.02 cr) | ~32×1 each ≈ 0.65+0.46=1.11 cr | ~24×1 (1.21 cr, F′) |
| new WAV | 96 | 160 | 128 | 48 |
| CLAP MDE_J | ~0.075 | ~0.061 | ~0.09 per factor | ~0.10 (underpowered) |
| positive ⇒ | recovered advantage needs fine discretization | recovered advantage is temporal-scale-conditional | identifies which single factor moves the ordering | recipe-conditionality (but can't attribute to a factor) |
| negative ⇒ | doesn't rule out duration | doesn't rule out steps | doesn't rule out a factor *combination* | weak (small N) — doesn't rule out anything cleanly |
| residual confound | duration, guidance, best-of | steps, guidance, best-of | interaction of the two factors together | can't attribute; underpowered; possible best-of/guidance |

**Adversarial reviewer:** "What have you NOT ruled out?" For B: steps, guidance, best-of, and any
factor *combination*. For D: nothing cleanly, because N is too small AND multiple factors move at once
(a positive can't be attributed; a negative is underpowered). A clean one-factor result
("changing X flips the recovered-vs-pruned ordering, model/prompt/domain fixed") is more publishable than
a tiny multi-factor run whose simultaneous changes prevent interpretation.

## 9. LITERATURE (materially-relevant only; primary-source reconnaissance, ids flagged where unverified)

**The specific interaction "post-pruning recovery × inference step-count/solver, as an evaluation-time
ordering effect" appears UNOCCUPIED** in both audio and image diffusion — no source runs the
{pruned-only, pruned+recovered} × {step-counts/solvers} factorial and reports the interaction. The
ingredients exist only separately, and mostly for **quantization, not structural pruning**:

* **Compression error accumulates along the reverse trajectory (point 4) — quantization:** PTQD (He et
  al., NeurIPS 2023, arXiv 2305.10657); Timestep-Aware Correction (ECCV 2024); QNCD (2403.19140).
* **Compressed models disproportionately fragile at few steps (point 2) — quantization:** Q-Sched
  (2509.01624) — "few-step models … sensitive to further model compression," fix is the scheduler;
  MixDQ (2405.17873) / Q-DM (NeurIPS 2023) — low-bit few-step quantization markedly harder.
* **Pruning × step-reduction as a training-init problem (closest, still adjacent):** "Bridging Diffusion
  Pruning and Step Distillation" (arXiv 2607.06335, id unverified) — a pruned checkpoint is a poor
  *initialization* for step-distillation; this is training-time, not eval-time ordering.
* **Pruning+recovery baselines that fix the sampler and do NOT test the interaction (define the gap):**
  Diff-Pruning (Fang et al., NeurIPS 2023, 2305.10924; timestep-dependent *importance*, fixed eval);
  BK-SDM (ECCV 2024, 2305.15798); 2ndMatch (CVPR 2026, 2506.05398); MosaicDiff (2510.11962). Audio
  neighbor: Singh 2607.13330 — all eval at fixed 200 steps, lists faster sampling as future work.

**Bearing on the choice:** points 2 & 4 for *structural pruning* would be incremental-but-supported.
The **evaluation-time ordering interaction (point 1/3), especially in audio, is the defensible candidate
novelty.** This raises the value of a **step-count** manipulation — but see §10: a step interaction is
only scientifically meaningful in a regime where a recovered advantage exists to be modulated, which is
exactly what the cheaper duration arm establishes first. (Bounded caveat: absence of a hit is not proof
of absence; not every venue was searched.)

## 10. RECOMMENDATION — exactly one design

**Arm D (duration discriminator).** Change ONLY duration relative to the V1.1 control; hold DDIM50,
guidance 2.5, eta 0, single-gen, fp32, EMA — so J isolates the one factor with the strongest mechanistic
motivation (train/eval temporal-scale match) at the lowest cost.

**Why duration first, not the (more novel) steps interaction:** §9 makes the step-count × recovery
interaction the stronger novelty gap, so it is tempting to lead with steps. But a step interaction is
only meaningful in a regime where a recovered advantage *exists to be modulated*, and we do not yet know
that it exists anywhere under controlled single-generation. Duration at 10.24 s is the **home scale**
where recovery was trained and reported, so it is the cheapest, most direct test of "does a recovered
advantage exist at all under a controlled common OP." It **gates** the step experiment: if no advantage
appears even at 10.24 s single-gen, the step-interaction study is moot (the published gain is then likely
carried by best-of-3/guidance selection, not weights); if an advantage does appear, the step interaction
— ideally run at 10.24 s, DDIM 50→200 — becomes the clean, novelty-filling follow-up. A steps-only arm
at our 3.84 s scale risks a false null for the wrong reason (recovered was never adapted to 3.84 s).

* **Factor changed:** generation duration 3.84 s (latent_t 96) → **10.24 s (latent_t 256)**; reference =
  the real 10 s AudioCaps clips (already local). (latent_t 256 divisible by 8 ✓, per KIM-CLIP-LENGTH.)
* **Systems:** pruned + recovered only (dense dropped for budget).
* **N / replicates:** **80 ytids × 1 replicate** (deterministic 80/96 subset by a NEW pre-data salt over
  ytid, outcome-blind; pruned & recovered share x_T per ytid).
* **Primary endpoint:** **J_CLAP** = R_alt(10.24 s) − R_ctrl(3.84 s), ytid-paired percentile bootstrap
  (B=10000, new seed). MDE ≈ 0.061 (80% power) — powered for a large restoration.
* **Secondary (descriptive):** KL, PANN top-10 capture, FAD (all vs the same 80 refs at 10.24 s);
  Human-CLAP optional.
* **Estimated cost:** **≈ 1.02 cr** (160 new WAV). **Hard max spend: 1.20 cr.** Reserve ≥ 0.26 cr.
* **Positive (J_CLAP > 0, lower-CI > 0):** the recovered checkpoint's advantage over pruned is
  **temporal-scale-conditional** — realized near its 10.24 s training scale, absent at 3.84 s. Novel,
  publishable, bounds (does not contradict) Singh.
* **Negative (J_CLAP ≈ 0):** recovered ≈ pruned even near the training scale under single-gen/guidance2.5
  → the published advantage does not reproduce off best-of-3/guidance3.5, OR requires steps; the next
  single action would be a steps arm or the best-of-3 selection factor. Still a clean, reportable bound.
* **Fallback N:** 64×1 (0.84 cr, MDE 0.067, reserve 0.62) if more reserve is wanted; 96×1 (1.21 cr, MDE
  0.057, no subset needed) if using all ytids is preferred and the 0.25 reserve is acceptable.

## 11. STRONGEST CURRENT PAPER THESIS (before this experiment; no OP-dependence claimed)

> Under a single controlled operating point (3.84 s / DDIM50 / guidance 2.5 / single generation), the
> published post-pruning "recovered" AudioLDM-M is **statistically indistinguishable from the pruned-only
> model across text-alignment (CLAP, Human-CLAP), event-capture (PANN top-10), divergence (KL), and
> distributional (FAD, FD) metrics simultaneously**, while the reported recovery was established under a
> different, larger evaluation recipe (10.24 s / 200 steps, framework best-of-3-by-CLAP). We therefore
> **bound** the recovery claim: it is demonstrated only within the authors' evaluation regime and, under
> a controlled common operating point, does not manifest as a measurable advantage over pruned-only on
> any axis we tested. Whether the advantage is recipe-conditional is an open question the proposed
> duration experiment is designed to test.

Not claimed: that recovery *is* operating-point-conditional (unproven); that the paper is wrong (it is
in-domain-correct under its own recipe). The Singh 3.95→1.57 fact (their fine-tune beats their own
*unpruned* model) frames recovery as partly AudioCaps domain adaptation (literature note).

## 12. NEXT

STOP. Design only. No prereg, no manifest, no GPU, no audio until Gabriel reviews this and chooses an arm/N.
