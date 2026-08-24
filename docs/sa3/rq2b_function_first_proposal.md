# RQ2b — Function-First Positive-Control Qualification Proposal

**Status:** PROPOSAL for review — NOT authorized, NO compute spent. CPU/docs only.
**Author date:** 2026-08-24 (Montevideo 2026-08-23 23:38).
**Supersedes nothing.** This is a *new* proposal, separate from the completed RQ2 pre-registration
(`rq2_validation_protocol.md`, rc1.4). It does not alter RQ2's closed verdict or L6/L13.

---

## 0. What is closed, and what is not

RQ2 is **CLOSED** on commit `235f344` (Gabriel accepted, 2026-08-24). The pre-registered positive
controls **L_6 / L_13 FAILED** the task-level ΔT_AA gate and **remain failed positive controls**.
They will not be rescued by any change of metric, steps, rank, dataset, threshold, or CI rule.

**Preserved headline result (record):**

> Single-block LoRAs (r16/α16, 1000 steps, 32-clip domain task) can create a **measurable and
> algebraically localisable field perturbation** (~1 % of ‖F_P‖², precision-detectable; host removal
> collapses ΔF to 0 exactly) **without producing measurable dense task utility** on the held-out
> paired CLAP audio-audio metric T_AA. This is evidence for a **field / function gap** — but it is
> **not yet a standalone paper claim**, because a failed positive control is a control-qualification
> failure, not a demonstrated law.

**Interpretation adopted (Gabriel, 2026-08-24):** this is an **instrument / control qualification
failure**, *not* evidence that adapter-compatible pruning is false. L6/L13 were structurally real and
measurable and behaved exactly as designed under host removal, but neither established a positive
dense task-level uplift, so there was no demonstrated functional benefit for a pruning experiment to
preserve.

**The high-level research question remains provisionally alive.** The experimental direction becomes
**function-first**: *before any structural / adaptability analysis, a fresh control must first
demonstrate a robust held-out task uplift on the intact model.* Only an adapter that passes that
independent functional gate is eligible for localisation / A_eco analysis.

---

## 1. Zero-compute diagnosis of the L6/L13 null (from existing data)

Computed from the already-generated `control_taa_scores.json` `per_j` deltas (n=8). This
**characterises the T_AA instrument**; it does not re-decide L6/L13.

| pair | n | mean ΔT_AA | per-clip SD | SEM | **MDE @ ~80 % (n=8)** |
|---|---|---|---|---|---|
| L6_base | 8 | +0.005 | 0.080 | 0.028 | **0.079** |
| L13_base | 8 | −0.064 | 0.091 | 0.032 | **0.090** |
| L6_post | 8 | −0.017 | 0.190 | 0.067 | **0.188** |
| L13_post | 8 | −0.016 | 0.173 | 0.061 | **0.171** |
| L6_ext13 | 8 | +0.094 | 0.128 | 0.045 | 0.127 |
| L13_ext14 | 8 | +0.047 | **0.035** | 0.012 | **0.034** |

Three facts constrain any successor design:

1. **The measured single-block dense uplifts (+0.005 / −0.064) are an order of magnitude below the
   detectable floor** (MDE 0.079 base / 0.19 post). Even a *real* single-block effect could not have
   cleared the gate unless it were unusually large. → **measurement power (cause c) is genuinely
   implicated.**
2. **The instrument is not dead:** `L13_ext14` detected +0.047 cleanly (SD 0.035, tight CI). It
   resolves moderate, low-variance effects. → the failure is not "T_AA can never see anything".
3. **There is large recovery headroom:** pruning dropped T_AA from **0.435 → 0.204** (a **0.231** gap).
   An adapter that recovered even part of that gap would produce a large, detectable uplift — yet the
   single blocks recovered ≈0. → **single-block capacity (cause a) is also implicated.**

**Assessment of the three candidate causes** (Gabriel's three: single-block capacity / the 32-clip
control task / the suitability & power of the measurement):

- **Capacity (a):** strongly implicated — 0.231 of headroom existed and one block moved none of it.
- **Measurement power (c):** strongly implicated — n=8 gives an MDE (0.079–0.19) far above the
  plausible single-block effect; the *post* condition is especially underpowered (block-skip inflates
  variance to SD≈0.18).
- **32-clip task (b):** *not* the primary suspect — the recovery task has ample headroom (0.231); the
  task is not saturated. Data volume may limit adapter quality but is not the binding constraint here.

**Conclusion:** the L6/L13 null is most parsimoniously a **joint capacity × power** failure. A valid
successor control must (i) produce an effect large enough to sit **well above** a powered MDE, and
(ii) be measured with an instrument whose MDE is **verified below** the target effect *before* the
science is read. **"More steps" or "higher rank" is explicitly not assumed to be the fix** — neither
raises capacity across blocks nor lowers the MDE; both are ruled out as the lever.

---

## 2. The design tension

A valid function-first control must satisfy two goals that pull against each other:

- **Strong, robustly-detectable function** — a large, held-out, positive ΔT with a CI clear of the
  MDE. Favours *more* adapter capacity (more blocks).
- **Meaningful localisation ground truth** — a *known* spatial support, so that A_eco / A_tan
  localisation has something to be validated against. Favours *fewer* blocks (single-block is the
  crispest possible A_eco target).

The two designs below sit at opposite ends of this tension and are **diagnostic in combination**.

---

## 3. Design A — Known-support band control (capacity-first)

**Construction.** Train a LoRA restricted to a **known, small band of k contiguous blocks**
(k≈3–5). The band is chosen *a priori* (e.g. the mid-stack blocks with the largest measured recovery
leverage), and training only ever touches those k blocks. **The band is the localisation ground
truth** — the adaptation lives in exactly those k blocks and demonstrably not in the other ~11.

**Why it is a valid control.**
- *Localisation ground truth is unambiguous by construction* — we inserted adapters into exactly
  those k blocks. Coarser than single-block, but valid: A_eco must localise the effect *to the band*
  and not outside it.
- *Function has room to clear the gate* — the pruning-recovery task has a measured 0.231 T_AA gap; k
  blocks have the capacity to recover a large, detectable fraction of it (target ΔT ≥ ~0.10 ≫ MDE).
- *Function is established first* — the functional gate is decided on **held-out** eval before any
  per-block structural score (A_eco, A_tan, D_P) is inspected.

**Fresh calibration data.** Pruning-recovery on a domain with measured headroom (impact_percussion is
already characterised, or a fresh CC0 pull). New eval clips + prompts, disjoint from training and from
any later ecological set. No reuse of the L6/L13 eval units.

**How task functionality is established before inspecting structure.** Pre-register the eval n and the
MDE from the measured noise floor. Require **ΔT_AA(post+band vs post), lower-CI > MDE** on the
held-out eval. Only if this passes is the localisation phase (in-band removal collapses ΔT;
out-of-band removal does not; A_eco field test on the known support) allowed to begin.

**Held out for any later ecological test.** A separate domain, and its adapters, are never touched by
Design A — reserved for the eventual A_eco cross-adapter comparison.

**Cost estimate.** One band-LoRA training (still one T4 job, 1000 steps ≈ 0.15–0.20 cr) + a
**powered** generation eval (n≥24 eval units → ~4–6× the per-config gen of the n=8 run). One control
adapter end-to-end ≈ **0.5–0.8 cr**; a base/post/host/external config sweep as in rc1.4 pushes it
toward the upper end.

**Terminal STOP rule.** If the intact band adapter **fails** the held-out functional gate
(ΔT_AA lower-CI ≤ MDE), **STOP** — the band is not a valid task-level positive control; do **not**
inspect localisation. That branch closes RQ2b: if a multi-block, high-capacity adapter on a
high-headroom task cannot move T_AA, the task-level metric itself is unfit for this generative regime
and the whole ΔT_L instrument must be redesigned before any adaptability claim.

---

## 4. Design B — Single-block, high-leverage, powered-measurement control (localisation-first)

**Construction.** Keep the **sharpest** localisation ground truth — **a single block** — but attack
the two other suspected causes directly:
1. **Choose the highest-leverage block**, not an arbitrary 6/13: the block whose removal most degrades
   T_AA (largest single-block share of the 0.231 recovery gap), so a single-block adapter there has
   the maximum possible headroom to move the metric.
2. **Power the measurement**: raise n_eval until the pre-computed MDE drops **below** the target
   effect, and **qualify the metric first** on a synthetic known-uplift pair (a controlled generative
   difference the metric must detect) — a metric-sensitivity gate applied only to this fresh phase.

**Why it is a valid control.**
- *Single-block support is the exact A_eco ground truth* — the crispest localisation target possible,
  and the one RQ2's original question ("does A_tan predict A_eco better than D_P?") most directly
  needs.
- *Best single-block chance of clearing the gate* — high-leverage block selection maximises the
  achievable effect size given the one-block capacity ceiling.
- *A null becomes interpretable* — because the metric's MDE is verified below the target *before* the
  science, a null is evidence of real absence at the single-block scale, not of underpower.

**Fresh calibration data.** Fresh eval clips + prompts (disjoint as in A), **plus** a synthetic
known-uplift calibration set used only to qualify the metric's sensitivity (kept entirely separate
from the scientific comparison).

**How task functionality is established before inspecting structure.** Two ordered pre-gates:
(1) the metric must pass its sensitivity calibration (detect the synthetic known uplift, lower-CI>0);
(2) only then, require **ΔT_AA(intact+adapter vs intact), lower-CI > MDE** on held-out eval. Structural
scores are inspected only after both pass.

**Held out for any later ecological test.** As in A — a separate untouched domain and its adapters.

**Cost estimate.** Single-block LoRA training is cheap (≈0.15–0.20 cr), but the higher n_eval raises
generation cost; one control end-to-end ≈ **0.5–0.7 cr**. The metric-sensitivity calibration on
synthetic pairs is CPU-only (0 cr).

**Terminal STOP rules (two).**
- **B-stop-1 (measurement):** if the metric **fails its own sensitivity calibration**, STOP and do not
  spend generation compute — the task measurement is unfit and must be redesigned (this implicates
  cause c, the deepest problem, and would also condemn Design A).
- **B-stop-2 (capacity):** if the metric passes calibration but the high-leverage single-block adapter
  still fails the held-out functional gate, that is **decisive evidence that single-block capacity is
  insufficient** for a task-level positive control → close the single-block line; only Design A (bands)
  remains viable.

---

## 5. Diagnostic value of the pair, and recommendation

The two designs partition the remaining hypothesis space:

| outcome | conclusion |
|---|---|
| B's metric fails calibration (B-stop-1) | measurement is the problem — **redesign ΔT_L before any control**; A is also not worth compute |
| B metric OK, single-block fails (B-stop-2) | single-block **capacity** is the problem — **Design A is the required control** |
| Design A passes its functional gate | a **valid function-first control exists** → localisation / A_eco qualification may proceed |

### Recommendation

**Build Design A as the primary control, gated by Design B's (near-free) measurement calibration as a
mandatory precondition.** Rationale, grounded in §1:

- The dominant suspected cause is **joint capacity × power**. Design A's band directly resolves the
  **capacity** side (multi-block, high-headroom recovery task, target ΔT ≫ MDE) while keeping a valid
  (band-level) localisation ground truth. It is the design most likely to yield an *actually valid*
  positive control — the thing RQ2 never had.
- Design B's **metric-sensitivity calibration is CPU-only and decisive**: if the T_AA instrument
  cannot detect a *known* generative uplift at the chosen n, no amount of adapter capacity matters.
  So B's calibration should run **first, as a gate on A** — its key idea survives regardless of which
  design we build. (A's own functional gate uses the same calibrated, powered metric.)
- Pure single-block Design B as the primary control is **not** recommended first: §1 shows a single
  block moved none of 0.231 of headroom, so a single-block control repeats the most likely failure
  mode. Keep B in reserve for **B-stop-2** — i.e. run it only to *attribute* an A-adjacent null to
  capacity, or if Gabriel wants to preserve maximal localisation sharpness at the cost of a higher
  null risk.

**Sequence I recommend for review (no compute yet):**
1. (CPU, 0 cr) Metric-sensitivity calibration + full power/MDE table at the proposed n (Design B's
   pre-gate). Terminal branch **B-stop-1**.
2. (GPU, ~0.5–0.8 cr) Design A band control: train → **held-out functional gate first** → localisation
   only on PASS. Terminal branch: functional-gate fail closes RQ2b.

### Budget reality

Current SA3 headroom is **1.63 cr** under the cap-5 total (SA3_TOTAL 3.369). One Design-A control with
a powered eval fits (~0.5–0.8 cr). **A full RQ2b — powered controls plus any later ecological adapter —
will not fit the remaining 1.63 cr headroom.** RQ2b therefore needs either a **fresh budget
authorization** or a **strictly single-control scope** (qualify the instrument only; defer ecological
work to a funded phase). This is a decision for Gabriel, not an autonomous escalation.

---

## 6. Invariants for RQ2b (carried from RQ2, non-negotiable)

- **Function first, always:** no structural / A_eco / A_tan inspection until the intact-model held-out
  functional gate passes. This is the defining rule of RQ2b.
- **No retrospective rescue:** any metric / n / threshold change applies **only** to the new untouched
  control+data phase, never to L6/L13.
- **Freeze before generating:** primary scalar, aggregation, n, MDE, seeds, external panel, and all
  gate conditions frozen in a committed pre-registration before any adapter is trained or any audio is
  generated.
- **Negative results are terminal, not iterated:** each STOP rule above closes its branch; we do not
  invent a new metric after seeing a null.
- **No compute without explicit authorization.**
