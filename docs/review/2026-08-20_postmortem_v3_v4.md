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

## 3. Rules for the next plan (apply before adoption, not after)

Each rule names the failure it prevents. A plan that cannot answer a rule is not ready to adopt.

* **R1 — Effect-size statement.** For every primary quantity: (a) the magnitude H1 predicts,
  with the reasoning; (b) the **smallest effect size of interest** (SESOI) below which the
  result changes nobody's mind; (c) the cheapest measurement that bounds the true magnitude.
  Adoption requires (c) to have been *run* and to show the plausible range overlaps (b).
  *(Prevents 1.)*
* **R2 — Positive control before any null.** Before the first confirmatory GPU run, plant a
  known effect of SESOI size (a synthetic LoRA, a deliberately modified block, a known
  intervention) and show the instrument detects it with the pre-registered statistic. No
  detection ⇒ fix the instrument, not the hypothesis. *(Prevents 3.)*
* **R3 — Instrument chosen for the question, with an explicit inheritance list.** Write down
  every element inherited from a prior pipeline (model, data, metric, pruning family) and the
  reason each is *necessary* for the question. Anything inherited "because it was there" is
  replaced or justified. *(Prevents 2.)*
* **R4 — Analysis tier first; method tier only on evidence.** The plan is staged so that the
  first tier is a forward-only / training-free *analysis* whose every outcome is publishable
  (or at least informative), with kill thresholds written before the data. RQs that propose a
  *method* are not written in detail until the analysis tier has a result. *(Prevents 1, 5.)*
* **R5 — The strongest reviewer objection runs in the first experiment.** Name the
  "why not just do the obvious thing" baseline (for the new plan: *prune the deployed model on
  its own end-to-end damage*) and put it in the first GPU job, not the last table. *(Prevents 4.)*
* **R6 — Power and feasibility are adoption conditions.** Power simulation at the plausible
  magnitude (not a placeholder), materialization/loading of every system, and a measured
  per-unit cost exist *before* the plan is signed. *(Prevents 6.)*
* **R7 — Cooling period and an expected-value reviewer.** No plan is adopted within 24 h of
  the trigger that killed the previous one. One review round is explicitly assigned to
  "probability of a positive × value of a positive − cost", with numbers. *(Prevents 4, 5.)*
* **R8 — Novelty ledger.** The three nearest papers and the one sentence that separates us
  from each, re-checked at every gate, with "what we will NOT claim" written down. (The
  current draft already does this; keep it.) *(Prevents the next pivot from being a re-run of
  someone else's result.)*
* **R9 — Calendar follows evidence.** No submission date in the plan until the analysis tier
  has a result; venue chosen afterwards. *(Prevents 5.)*
* **R10 — Every run recorded before the next is launched.** Ledger entry (even a stub) at
  launch, completed at read-out; no job reads its own result without the entry existing.
  *(Prevents the M4-SCREEN-FOUND failure.)*

---

## 4. Applying the rules to the Stable Audio 3 direction (what it still owes)

| Rule | Status for the SA3 proposal | Owed before adoption |
|---|---|---|
| R1 | Kill thresholds sketched (ρ(D_B,D_P) > 0.9, `I_PT/D_P` flat) but no predicted magnitudes | Predict the size of `I_PT` under H1 (e.g. from the per-block weight deltas `‖W_P − W_B‖`, free to compute) and state the SESOI for RQ2/RQ3 (how much unseen-LoRA fidelity must a mask recover to matter) |
| R2 | Not yet | A planted test: a LoRA we *know* is carried by specific blocks (train it with `--include layers[a-b]`), then check the tangent-space probe and the mask score recover those blocks |
| R3 | Good: the instrument (matched base/post pair, official LoRA transfer) is chosen for the question; depth pruning justified | List what is still inherited (AudioCaps captions for eval, CLAP/FD metrics) and why |
| R4 | Good: RQ1 forward-only first | Write RQ1's decision table (cases 1–4) as the *only* adopted content; RQ2–RQ4 stay as sketches |
| R5 | Named (post-only end-to-end pruning) | Put it in the first GPU job |
| R6 | Costs sketched from repo numbers; T4/torch 2.7.1 untested | venv + load + one forward on T4; measured s/forward; materialize a block-skipped model |
| R7 | Being violated right now (three pivot proposals in one evening) | Sleep on it; assign the EV review |
| R8 | Done in the draft (TALL-Masks, NPS, TinyFusion, 2607.06335/06631, EcoDiff); add tangent-space fine-tuning / task arithmetic in tangent space (Ortiz-Jiménez et al. 2023) and LoRA cross-backbone transfer work | Keep |
| R9 | Calendar deliberately excluded from the discussion | Keep it excluded until RQ1 has data |
| R10 | Process in place | Keep |

**One risk the rules surface immediately (R1/R4):** few-step adversarial generators may not
tolerate *any* block removal without repair (cf. 2607.06335's finding that a pruned generator
produces unusable samples before teacher-aligned repair). If removing even 1–2 of 20 blocks
destroys `small-sfx` at 8 steps, "pipeline-preserving *mask*" collapses into "mask + minimal
repair", which is the crowded territory. RQ1's first forward-only day must measure this before
anything else is designed; the branch for that outcome must be written now, not after.
