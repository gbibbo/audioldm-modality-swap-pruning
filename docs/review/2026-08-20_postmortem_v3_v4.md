# Post-mortem: why plans v3 and v4 both ended in a dead end, and the rules for the next plan

Written 2026-08-20 21:55 at Gabriel's request, from the ledger and the plans — not from memory.
Scope: the *process* failures that are independent of which model we study next. The scientific
negatives themselves (RQ-swap, Gate A, Gate B, Tier-0 heterogeneity) were correctly obtained and
are not the problem; the problem is that each was foreseeable with evidence we already had.

---

## 1. The facts, with timestamps (ledger IDs in brackets)

**Plan v3 — "the modality swap organizes pruning-relevant structure".**

* 08-18 11:41 [M2-001]: conditioning-path validation measured `mean|ε_a − ε_t| = 1.148e-2`. The
  ledger records it as "**non-degeneration PASS**" (a binary check). Nobody read it as an effect
  size. Two days later, round-1 review [REVIEW-001/round1] restated the same number as "**the
  swap is 1.5 % of the ε signal — the instruments are insensitive to the hypothesis by
  construction**". The number did not change; the question asked of it did.
* 08-18 14:03 [AUDIT-M3-002]: Gate B as written in the plan was **mathematically unreachable**
  (prune-set overlap floor 0.75 by pigeonhole vs a 0.70 threshold). The threshold had been set
  without computing the chance level / geometry of the statistic. Fixed by amendment
  [DECISION-M3B-003], but the pattern — thresholds before null geometry — is the same one that
  hit v4's Gate E.
* 08-20 02:55 / 04:30 [M3B-SALIENCY-RUN, M3A-DIAG-RUN]: Gate B FAIL (audio/text saliency
  ρ = 0.98), Gate A FAIL (Δ_swap CI ⊃ 0). Both well audited. Both predictable from M2.

**Plan v4 — "pruning causes strongly event-dependent semantic forgetting; find the mechanism".**

* Premise (plan v4 §0, §1): Singh et al. Table 3 (Safety-critical −73.5 %, Mechanical −75 %,
  Speech −13 %), a **3-to-6-family table with no per-family counts or uncertainty analysed**.
  The contribution claim was written with the adjective "*strongly*" already in it.
* 08-20 12:30 → 14:13 [REVIEW-002 … DECISION-V4-00]: **five hostile-review rounds and
  adoption in 1 h 43 min**, the same morning the v3 negatives landed, with the 09-16 submission
  date kept as an invariant (§8 "retained"). The reviews converged on *internal consistency*
  (pre-registration, K_rand, disjoint sets, placebo) — none asked "what is the plausible
  magnitude, and is it worth the credits?".
* 08-20 19:20 [Q7]: the Gate-E power simulation — a Tier-1 *prerequisite* in the plan — ran
  **five hours after adoption**, with placeholder rates, and reported MDE δ = 0.35 ("underpowered
  for small effects"). It was filed, not acted on, because Tier 0 was "screening only".
* 08-20 19:55 [TIER0-SCREEN-RUN]: the screen (200 prompts over 103 events → 8 events with ≥ 5
  prompts) could not have decided anything: its power at the relevant effects is 0.16–0.26
  (`severity_sweep_power.json`, 35×6 row). The read-out nevertheless said "supports RQ1′ +
  funding Tier 1" on three bullets, two of which did not survive re-analysis
  [SWEEP-PROPOSAL]: P1-nat does not protect at the pooled level, and the latent between-event SD
  is ≈ 0.12 (35 events) / 0.00 (8 events) once sampling noise is removed.
* [M4-SCREEN-FOUND]: an overnight 1.40-cr screening job ran and was never recorded until found
  the next morning. A process failure, not a science failure — but the same root: speed over
  reading.

**Common denominator.** Spend so far: 6.915 cr, 20 jobs, ~3 days. Every gate that fired, fired
correctly. What was missing was the step *before* the gates: **an honest estimate of the effect
size the hypothesis predicts, and of the smallest effect that would matter — written down
before adoption and compared against data we already had.**

---

## 2. Root causes (ranked)

1. **Effect size was never a first-class object.** Both plans specified *thresholds* (overlap
   ≤ 0.80, p ≤ 0.05) but never *magnitudes* (how big should `|ε_a − ε_t|` be for the swap to
   organize channels? how big is the between-family SD in Table 3 once you subtract sampling?).
   Binary gates let a 1.5 % signal pass as "non-degenerate" and a 3-row table pass as "strong".
2. **Hypotheses were anomaly-driven and inherited the instrument.** v3 came from a quirk of
   Arshdeep's pipeline (train-with-CLAP-audio / infer-with-CLAP-text); v4 from a side table of
   his paper. In both, the model, dataset, pruning family and metric (AudioLDM-M + AudioCaps +
   L1 filter pruning + PANNs top-10) were fixed *before* the question, so the question had to
   fit an instrument that was never built to measure it (10 s clips, binary top-10 recall,
   captions that do not name every event, an extremely unequal event distribution).
3. **No positive control.** Every gate had a null (random masks) but no planted effect — we
   never showed the instrument *could* detect the thing we were looking for at a plausible
   magnitude. Sensitivity was assumed, then discovered to be absent after the spend.
4. **Rigor was equated with pre-registration.** The hostile reviews made v4 unassailable
   *as a protocol* and did nothing about its *expected value*. Five rounds in under two hours
   is not review; it is polishing.
5. **The calendar was an invariant while the science changed.** Keeping 09-16 forced a Tier 0
   that could only screen, and a screen design whose power nobody computed until afterwards.
   Deadline pressure plus sunk cost produced a same-day pivot into the same substrate.
6. **Power/feasibility analysis ran after adoption.** Q7 (power) and Q4 (materializer) were
   "CPU queue" items *inside* the adopted plan instead of *conditions for adopting it*.

---

## 3. Rules for the next plan (DECISION-RULES-001, Gabriel 2026-08-20 22:02)

Gabriel kept the **scientific** rules and removed the process/management ones (cooling period,
calendar, record-at-launch) from the protocol — those belong to `AGENTS.md`, not to the
science. Seven rules remain; each is an *adoption condition*, not a checklist applied afterwards.

* **S1 — Effect size before method.** For every primary quantity: the magnitude H1 predicts,
  the **smallest effect size of interest** (SESOI), and a cheap *functional* measurement that
  bounds the true magnitude — run before adoption. Parameter-space proxies (e.g.
  `‖W_P − W_B‖`) are explanatory covariates, never the effect estimate. *(Prevents root cause 1.)*
* **S2 — Positive control before any null result.** Two kinds: a *synthetic* planted effect of
  SESOI size that the instrument must recover, and an *ecological* one showing the phenomenon
  to be preserved exists before intervention (for the SA3 direction: the dense-base → dense-post
  LoRA transfer ceiling). *(Prevents 3.)*
* **S3 — Every inherited design choice requires scientific necessity.** Model, data, metric,
  pruning family: each is justified for the question or replaced. *(Prevents 2.)*
* **S4 — Analysis before intervention.** The first tier is forward-only / training-free, its
  decision table is written before the data, and method RQs stay as sketches until it has a
  result. *(Prevents 1, 5.)*
* **S5 — Strongest competing explanation / baseline first.** The "why not just do the obvious
  thing" baseline runs in the first experiment, not the last table. *(Prevents 4.)*
* **S6 — Novelty must survive the closest prior art before implementation.** Novelty ledger:
  nearest papers, the sentence separating us from each, and what we will *not* claim; re-checked
  at every gate. *(Prevents re-running someone else's result.)*
* **S7 — A statistically detectable effect is not automatically a scientifically valuable
  effect** (learned from AudioLDM). The SESOI is defined in *decision* terms: the effect must
  change which structures would be selected at the target budget, or move the
  deployment/compatibility Pareto front materially — not merely reach p < 0.05 or ρ = .92 vs .95
  on 100 000 samples. *(Prevents the next "supports funding Tier 1".)*

Power/feasibility before signing (old R6) is folded into S1/S4: the analysis tier *is* the
feasibility and magnitude measurement. Record-at-launch (old R10) remains an `AGENTS.md`
obligation.

## 4. Applying the rules to the Stable Audio 3 direction (what it still owes)

| Rule | Status for the SA3 proposal (22:02 version) | Owed before adoption |
|---|---|---|
| S1 | Kill thresholds sketched; no predicted magnitudes | Per-block functional quantities `D_B`, `D_P`, `I_PT` (normalized), `A` on the 20 blocks — forward-only; `‖ΔW_g‖` computed only as an explanatory covariate (`ρ(‖ΔW‖, I_PT) ≈ 0` would itself be a result) |
| S2 | Both controls now specified | Synthetic: a LoRA trained with `--include layers[a-b]`, the sensitivity instrument must mark `a–b`. Ecological: the dense-base → dense-post transfer ceiling for a real LoRA, measured *before* any pruning |
| S3 | Instrument chosen for the question (matched base/post pair, official LoRA transfer, depth pruning keeps block geometry so adapters map trivially) | List the remaining inheritances (AudioCaps captions, CLAP/FD metrics) with their necessity |
| S4 | RQ1/RQ2 forward-only; decision cases A–D written | Add case E: the post-trained 8-step model tolerates **no** block removal without repair (2607.06335) — pre-write that branch |
| S5 | Post-only pruning is now the *objective* of the constrained formulation, not a side baseline | Run it in the first GPU job; report the full Pareto front over `(ε_PT, ε_A)` rather than one `(λ, μ)` |
| S6 | Novelty ledger: TinyFusion, EcoDiff, 2607.06335, 2607.06631, TALL-Masks, NPS, tangent-space task arithmetic, **CAR-LoRA (ICLR 2026: adapter-side compression-aware training, LLMs, quantization/pruning — does not choose the backbone to preserve unseen adapters)**, Compress-then-Serve | Keep; headline = *backbone-side* compatibility preservation without seeing or retraining adapters |
| S7 | Stated | Define the SESOI as "the top-k removal sets chosen by `D_P` and by `A` differ at k ∈ {2, 4, 6}" (decision-relevant), not as a ρ threshold |

**Verified code facts that shape the probes (22:02):** SA3 `lora-xs` is `ΔW = U · M · Vᵀ` with
`U ∈ ℝ^{fan_out×r}`, `V ∈ ℝ^{fan_in×r}` = top-r singular vectors of the layer's `W0`
(sign-canonicalized), frozen, and only `M ∈ ℝ^{r×r}` trainable (`stable_audio_3/models/lora/model.py`).
`svd_bases.pt` ships **only with `small-sfx-base`**, not with `small-sfx`: a LoRA-XS trained on
base and applied to post is well-defined only if the *base's* bases are supplied; otherwise the
loader recomputes SVD from the post's weights and `M` lands in a different basis. Two
consequences: (i) the ecological probe family must use the base's `U, V`; (ii) for `lora-xs`
the official transfer contract is itself basis-dependent — a fact to report, and a reason the
standard `lora`/`dora-rows` adapters (which transfer basis-free) must be in the held-out set too.

**Exact enumeration beats greedy at the budgets that matter.** With 20 blocks, the feasible
masks are C(20,2) = 190, C(20,3) = 1 140, C(20,4) = 4 845, C(20,6) = 38 760. With ~108-token
sequences, forward-only proxies on a fixed latent panel make the Pareto front **exact** for
k ≤ 4 (and samplable for k = 6), removing the optimizer as a confound in RQ3. Greedy is then a
scalability note, not the method.

**One risk the rules surface immediately (S1/S4):** few-step adversarial generators may not
tolerate *any* block removal without repair (cf. 2607.06335). If removing even 1–2 of 20 blocks
destroys `small-sfx` at 8 steps, "pipeline-preserving *mask*" collapses into "mask + minimal
repair", which is the crowded territory. RQ1's first forward-only day must measure this before
anything else is designed; the branch for that outcome is case E above.
