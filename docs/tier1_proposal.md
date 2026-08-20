# Tier-1 proposal — sequential, decision-gated (2026-08-20)

**Status: PROPOSAL for Gabriel. NOTHING here is authorized; no GPU job runs until Gabriel
authorizes the first tranche explicitly.** Written after the Tier-0 campaign completed
(ledger `TIER0-D1D2-RUN`, `TIER0-GATEB-RUN`, `TIER0-SCREEN-RUN`). All costs use the
**measured** rate `0.00235 cr/clip` and `0.00234 GPU-h/clip` at DDIM S=50 (from `screen-1`:
600 clips → 1.412 cr). Design invariants (`K_rand = 20`, the frozen Q2 manifests, the
blinded holdout, the statistical criteria) are **unchanged** — the only optimizations below
preserve the science.

---

## 0. The headline finding (why this is not a monolithic ~45-cr ask)

The plan v4 §7 costed Tier 1 at **≈45 cr assuming a 20 × 15 sentinel** for Gate E. Re-running
the Gate-E power simulation (Q7 machinery, `scripts/research/gate_e_power_sim.py`) with the
**real Tier-0 rates** shows that sentinel is **badly underpowered**, and powering Gate E to a
defensible level is the single biggest cost in Tier 1 — larger than the plan's whole Tier-1
figure. Concretely:

* The Tier-0 screen's apparent event-level heterogeneity is **mostly sampling noise**. With
  the real per-event recall from `screen_eval.json`: `p_base = 0.586`, `mu_loss = 0.20`,
  observed between-event SD of loss = 0.29, but the **sampling SD alone is ≈0.27** (few
  prompts/event). The **noise-corrected true between-event SD (δ) ≈ 0.11** on the 35-event
  subset, and **≈0** on the 8-event ≥5-prompt subset. So the true effect is small and
  uncertain: `δ ∈ [0, ~0.11]`.
* At those real rates, the plan's **20 × 15 sentinel has only 23–57 % power** (δ = 0.11 → 0.20).
  It cannot decide Gate E.

Everything below follows from that: Gate E must be resized (design-preservingly — more
events/prompts, `K_rand = 20` untouched), which is expensive, and Tier 1 must be funded
**one gate at a time** so a Gate-E FAIL stops the ~35 cr of downstream work.

---

## 1. Recomputed Gate-E power (real Tier-0 rates)

`p_base = 0.586`, `mu_loss = 0.20`, `K_rand = 20`, exact rank test (p = (1+#{V_rand≥V_P0})/21 ≤ 0.05),
`n_sim = 4000`. Raw: `artifacts/gate_e_power/power_real_tier0.json`.

| sentinel (events × prompts) | sentinel prompts | power @ δ=0.11 | @ δ=0.15 | @ δ=0.20 |
|---|---|---|---|---|
| **20 × 15** (plan default) | 300 | 0.23 | 0.38 | 0.57 |
| 30 × 20 | 600 | 0.40 | 0.66 | 0.86 |
| **30 × 30** (recommended) | 900 | 0.61 | **0.87** | 0.98 |
| 40 × 30 | 1200 | 0.70 | 0.93 | 0.99 |
| 50 × 40 | 2000 | **0.91** | 1.00 | 1.00 |

**MDE recommendation — δ = 0.15, sentinel 30 × 30.** Rationale: δ = 0.15 (a 15-percentage-point
between-event SD of forgetting) is a scientifically meaningful heterogeneity; 30 × 30 reaches
**87 % power** there while the plan default reaches 38 %. This is the cost/power knee.

**The honest caveat you must weigh:** at the Tier-0 point estimate δ ≈ 0.11, 30 × 30 has only
**61 % power** — so a Gate-E FAIL would be ambiguous (no effect vs. underpowered for a small
effect). Only **50 × 40 (91 % power at δ=0.11, but ~2× the cost)** makes Gate E a *decisive*
test of the point-estimate effect. Cheaper (30 × 20, ~66 % @ δ=0.15) only detects strong
heterogeneity (δ ≥ 0.20). See the cost table in §2.

**Strategic note (my recommendation):** the Tier-0 evidence for event-level heterogeneity is
weak. Before committing ~50–110 cr to confirm a δ ≈ 0.10–0.15 effect, decide whether an
event-level forgetting heterogeneity of that magnitude is worth it. If yes, 30 × 30 is the
cost-effective bounded test; if you want Gate E to *settle* the point estimate, 50 × 40 is
required.

---

## 2. Tranche 1-E — Gate E only (the first funding decision)

Gate E generation = the sentinel panel × {base, P0-std, P1-nat, **RAND×20**}. base/P0-std give
`L(e)=recall_base−recall_P0std`; P1-nat gives the `Spearman(L_{P0-std},L_{P1-nat}) ≥ 0.5`
AND-condition; RAND×20 gives the between-mask null. (The plan's §7 line counted only the
RAND×20 6 000 clips; the base/P0-std/P1-nat on the sentinel are additional because the Q2
sentinel is disjoint from the mechanism set.)

| design (MDE) | total clips (×23) | GPU-h | **credits** | RAND×20-only (cr) | power @ δ=0.15 |
|---|---|---|---|---|---|
| 20 × 15 (underpowered — do not use) | 6 900 | 16.2 | 16.2 | 14.1 | 0.38 |
| 30 × 20 (δ=0.20 test) | 13 800 | 32.4 | 32.5 | 28.2 | 0.66 |
| **30 × 30 (recommended, δ=0.15)** | 20 700 | 48.5 | **48.7** | 42.4 | 0.87 |
| 50 × 40 (δ=0.11 decisive) | 46 000 | 107.8 | 108.3 | 94.1 | 1.00 |

* **Recommended Tranche 1-E (30 × 30):** ~48.7 cr generation + **smokes/failure margin**
  (~2 GPU smokes ≈ 0.22 cr + 10 % contingency ≈ 4.9 cr) → **≈ 54 cr**.
* **Jobs:** ~5–6 T4 jobs (split the 20 RAND masks into ~4 jobs of 4–5 masks each to keep each
  job ≤ ~10 h; one job for base/P0-std/P1-nat). Each preceded by a CPU dry-run + GPU smoke.
* **Decision gate at the end of Tranche 1-E:**
  * **Gate E FAIL** (V_P0-std not above the RAND×20 null at p ≤ 0.05, OR Spearman < 0.5) ⇒
    **STOP. RQ2′ and RQ3′ are not run; Tranches 2 and 3 (~35 cr) are not funded.** Report the
    negative (event-level forgetting not heterogeneous beyond a matched null at this budget).
  * **Gate E PASS** ⇒ proceed to the *second* funding decision (Tranche 2).

---

## 3. Tranche 2-M — Gate M (mechanism) + CFG grid (only if Gate E PASS)

| item | clips | credits |
|---|---|---|
| Mechanism set: 50 events × 20 prompts × {base, P0-std, P0-pub, P1-nat, mild P0-std, mild P1} | 6 000 | 14.1 |
| CFG grid (H-guidance): 3 CFG × {base, P0-std} × 300 prompts | 1 800 | 4.2 |
| Counterfactual ε forwards (H-guidance covariate) — forward-only, negligible | — | ~0.3 |
| smokes + 10 % contingency | — | ~2.3 |
| **Tranche 2-M total** | ~7 800 | **≈ 21 cr** |

* Gate M runs the three pre-registered covariate blocks (tail / guidance / acoustic — acoustic
  in the primary block because **FineLAP passed its Q3 smoke**) as nested block LRTs.
* **Decision gate:** if **no block wins** ⇒ RQ3′ is not run (Tranche 3 not funded). If a block
  wins ⇒ its intervention variant proceeds to Gate B′ (Tranche 3).

---

## 4. Tranche 3-I — RQ3′ intervention (only if a mechanism wins AND Gate B′ passes)

| item | clips | credits |
|---|---|---|
| P1-mech + P1-placebo variant per-slot saliency (GPU) + CPU Gate B′ null-split | — | ~1.0 |
| **Gate B′ gate:** overlap(P1-nat, P1-mech) < **0.9401** (the Tier-0 null 5th-pctile) | (CPU) | 0 |
| Intervention holdout: 500 prompts × {base, P1-nat, P1-mech, P1-placebo} | 2 000 | 4.7 |
| FAD/FD guardrail: 3 seeds × 300 clips × 3 systems (real-part Frechet, Q1 fix) | 2 700 | 6.4 |
| smokes + 10 % contingency | — | ~1.9 |
| **Tranche 3-I total** | ~4 700 | **≈ 14 cr** |

* **Gate B′** (CPU, free) runs *before* the holdout generation: if `overlap(P1-nat, P1-mech) ≥
  0.9401` the intervention does not change the mask beyond natural sampling ⇒ **RQ3′ dead, the
  2 000-clip holdout generation is not run** (~5 cr saved). Only if Gate B′ passes does the
  holdout + Gate I generation proceed. Same discipline that killed P2/P3.

---

## 5. Totals, top-up, and the sequential funding decisions

Current all-time spend **6.9150 cr** (SDK, 20 jobs). Under DECISION-CG-001 (~9.6 balance, 2.0
reserve), **spendable now ≈ 0.685 cr**; the 2.0-cr reserve stays untouched. So each tranche
needs a **top-up ≈ its cost** (the current headroom barely covers one smoke).

| tranche | when | credits | top-up to add to the org (keep 2.0 reserve) |
|---|---|---|---|
| **1-E Gate E** (30 × 30) | first decision | ~54 | **~53 cr** |
| 2-M Gate M + CFG | only if Gate E PASS | ~21 | ~21 cr |
| 3-I RQ3′ intervention | only if a mechanism wins AND Gate B′ passes | ~14 | ~14 cr |
| **Full Tier 1 (all pass)** | | **~89 cr** | ~88 cr cumulative |

**Decision structure (what I need from you, one at a time):**

1. **Decision 1 — fund Tranche 1-E (~53 cr top-up)?** Choose the MDE/design (30 × 30 recommended;
   30 × 20 cheaper/less sensitive; 50 × 40 decisive/expensive). If Gate E FAILs, we stop and
   save ~35 cr.
2. **Decision 2 — fund Tranche 2-M (~21 cr)?** Only asked if Gate E PASSes.
3. **Decision 3 — fund Tranche 3-I (~14 cr)?** Only asked if a mechanism wins *and* Gate B′
   passes.

Interruptible instances (verified `--interruptible`, exact resume proven) can cut the settled
cost of the long RAND jobs; that is a scheduling optimization, not a design change, and I would
trial it on one job first.

---

## 6. FINDING-P0-COLLAPSE — impact and confirmation for Tier 1

The M3B saliency artifact stored `P0_L1` and `P0_published` **identically** (sum-normalization
cancels the ±L1 sign). Impact carried into Tier 1:

* **All Tier-1 experiments use the corrected P0-standard** = the L1 pkl **reversed** per layer
  (keep-HIGHEST-L1), and **P0-published** = the L1 pkl direct (keep-LOWEST-L1). Never the
  collapsed saliency `P0_L1` entry. This is `system_rankings()` in `scripts/research/tier0_d1d2.py`
  and `tier0_screening.py` (0/2304 kept-set overlap between P0-std and P0-pub, verified). Tier-1
  runners (mechanism set, CFG grid, holdout) **must reuse `system_rankings()`**.
* **The mild-budget P0-std** (`channel_mult=(1,2,3,4)`, mechanism set) uses the **same reversed
  ranking at the mild budget** via the Q4-parameterized `materialize(..., channel_mult=[1,2,3,4])`
  — not the collapsed saliency.
* **Correction to the record:** the M4-SCREEN-FOUND P0_published-vs-P0_L1 KL gap was generation-seed
  noise, not a P0-convention effect (already recorded in the ledger). No Tier-1 number should cite it.

---

## 7. Design invariants NOT changed to cut cost

`K_rand = 20` (exact rank test, attainable p_min = 1/21); the frozen Q2 manifests (event set,
synonym maps, covariates, partitions, sentinel structure, seeds; sha256 in the ledger); the
blinded intervention holdout (unblinded only after the intervention criterion is frozen); and
the statistical criteria (α = 0.01 per Gate-M block, Gate I margins δ_target=+5pp / δ_harm=2pp,
FAD/FD +5 % / KL +0.05). The **only** cost levers used are the sentinel size (to reach power)
and job splitting / interruptible scheduling — both design-preserving.
