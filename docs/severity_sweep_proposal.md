# Severity-sweep proposal — cheap falsification of "generic capacity loss" before Gate E (2026-08-21)

**Status: PROPOSAL for Gabriel. Nothing here is authorized; no GPU job runs without an explicit
go. Adopting it is a plan-v4 amendment (a new pre-Gate-E step) and must be recorded as a
DECISION in `docs/experiment_ledger.md`.** Written in response to Gabriel's 2026-08-20 21:00
assessment (event-level heterogeneity mostly sampling noise; effect size shrinking with every
better measurement; do not fund Gate E at ~53 cr for a δ ∈ [0, 0.11] effect; ask first whether
heterogeneity grows with pruning severity). Companion to `docs/tier1_proposal.md`.

---

## 0. What the evidence says (verified this session, not quoted)

Recomputed directly from `artifacts/tier0_screening/screen_eval.json`:

| subset | events | mean L(P0-std) | observed SD | sampling SD | **latent SD** |
|---|---|---|---|---|---|
| ≥ 3 prompts | 35 | 0.234 | 0.294 | 0.268 | **0.122** |
| ≥ 5 prompts | 8 | 0.096 | 0.159 | 0.178 | **0.000** |

Two further facts that were not in `PROGRESS.md`:

* **Pooled over all 200 screen prompts, P1-nat does NOT protect better than P0-std**: recall
  base 0.533 → P0-std 0.336 → P1-nat 0.328 (loss 0.197 vs 0.204). The "P1 protects better"
  read-out (mean_L 0.048 vs 0.096) holds only on the 8-event subset. So the Tier-0 screen's
  three "supports RQ1′" bullets reduce to one weak one (Spearman 0.663, n = 8, p = 0.073).
* **Part of the latent SD is mechanical, not structural.** With heterogeneous base recalls
  (Sheep 0 … Speech 1) a *homogeneous* log-odds degradation already produces a between-event
  SD of loss of ≈ 0.07 at mean loss 0.17–0.20 (an event with recall 0.1 cannot lose 0.2). The
  Q7 power model (single `p_base`, constant loss under H0) does not contain this term, so its
  "δ ≈ 0.11" overstates the *structured* component (≈ 0.10 after removing it). Gate E's RAND
  null absorbs the mechanical term correctly; the point estimate just did not.

Conclusion shared with Gabriel's assessment: the structured event-level effect at −65 % is
small and uncertain (≈ 0–0.10), and Gate E 30×30 (87 % power only at δ = 0.15) is designed for
a larger effect than the evidence supports.

---

## 1. The question worth ~13 credits

> Is there **any** event-level structure in forgetting beyond a homogeneous capacity loss, at
> which pruning severity does it appear, and does it merely track mean damage?

This is prior to Gate E. Gate E asks "is P0-std's heterogeneity larger than that of random
masks at matched damage" (criterion-specificity) and pays 42 of its 48.7 cr for RAND×20 audio.
If there is no structure at all, Gate E cannot pass and the RAND audio is wasted; if structure
exists only at −65 % and scales with mean loss, it is capacity collapse, not a mechanism.

**Severity ladder (measured, nested, same P0-std ranking, Q4 materializer):**

| `channel_mult` | params | reduction | verified this session (CPU) |
|---|---|---|---|
| (1,2,3,4) | 317.308 M | −23.7 % | strict-load + forward ✓ |
| (1,2,3,3) | 239.047 M | −42.5 % | strict-load + forward ✓, kept-set ⊂ previous ✓ |
| (1,2,3,2) | 182.168 M | −56.2 % | strict-load + forward ✓, kept-set ⊂ previous ✓ |
| (1,2,3,1) | 145.674 M | −65.0 % | strict-load + forward ✓, kept-set ⊂ previous ✓ (= published budget) |

Nested kept-sets make this a clean 1-D severity axis (each level removes *more of the same
ranking*), so a change in structure along the axis cannot be attributed to a different mask.

---

## 2. Statistic and test (pre-registered here)

Per severity level *s*, on a panel of E events × n prompts with base and pruned recall
(PANNs top-10, the v4 primary metric, same scorer as `tier0_screen_eval.py`):

* `L̂_s(e) = recall_base(e) − recall_s(e)`; `V_s = var_e L̂_s(e)`; `μ̂_s = mean_e L̂_s(e)`.
* **Null H0 (generic capacity loss):** a common logit shift `q_e = expit(logit(p_e) − β_s)`.
  Differences in `L̂` under H0 come only from binomial sampling and the mechanical bound.
* **Test:** parametric bootstrap (B = 999) with plug-in smoothed `p̂_e = (y_base+½)/(n+1)` and
  the 1-parameter MLE `β̂_s`; `p_s = (1 + #{V* ≥ V_s}) / (B+1)`.
* **Read-out:** latent between-event SD `δ̂_s = sqrt(max(0, V_s − sampling var))` with a
  bootstrap 95 % CI, `μ̂_s`, `p_s`, and the curve `δ̂_s` vs `μ̂_s` across the four levels.
* Descriptive (no gate): Spearman of `L̂_s(e)` across adjacent levels (do the same events fall
  first?); if P1-nat is included, Spearman(L_P0-std, L_P1-nat) per level with n = 50 events
  instead of 8.
* α = 0.05 per level; the four levels are reported jointly, not cherry-picked.

Generation is seed-paired across systems (plan v4 §6), which only reduces Var(L̂) relative to
the simulation below (conservative power).

**Power (CPU, `scripts/research/severity_sweep_power_sim.py`, raw
`artifacts/gate_e_power/severity_sweep_power.json`; tests `tests/research/test_severity_sweep_power.py`
W1–W4 PASS; Type-I at τ = 0 is 0.042–0.071 across all cells):**

| panel (E × n) | prompts | cr / system | power @ δ≈0.09–0.11 | @ δ≈0.13 | @ δ≈0.17 |
|---|---|---|---|---|---|
| 35 × 6 (≈ the Tier-0 screen) | 200 | 0.49 | 0.15–0.16 | 0.25–0.26 | 0.46–0.50 |
| 25 × 20 (half the mechanism set) | 500 | 1.18 | 0.32–0.40 | 0.62–0.64 | 0.89–0.92 |
| 20 × 30 | 600 | 1.41 | 0.39–0.49 | 0.68–0.74 | 0.94–0.96 |
| **50 × 20 = frozen Q2 mechanism set** | **999** | **2.35** | **0.51–0.54** | **0.85–0.89** | **0.98–0.99** |
| 30 × 30 (Gate-E recommended sentinel) | 900 | 2.12 | 0.50–0.61 | 0.83–0.90 | 0.98–0.99 |
| 50 × 40 (would need a new manifest) | 2000 | 4.70 | 0.81–0.90 | 0.98–0.99 | 1.00 |

Reading: the frozen mechanism set (50 × 20) gives the overdispersion test **the same power as
Gate E 30×30 (0.87 @ δ=0.15, 0.61 @ δ=0.11) at ~1/10 of the clips per level**, because the
RAND×20 audio is replaced by a parametric null. The Tier-0 screen design could never have
decided this (power 0.16–0.26 at the relevant effects) — consistent with Gabriel's reading.
Against the point estimate δ ≈ 0.10 the test is still only ~50 % powered; only 50 × 40
(2000 prompts, 4.7 cr/system) is decisive there.

---

## 3. Costed options (measured 0.00235 cr/clip, 0.00234 GPU-h/clip, DDIM S=50, T4)

| option | panel | systems | clips | GPU-h | gen cr | + 2 smokes + 10 % | **ask** |
|---|---|---|---|---|---|---|---|
| **A (recommended)** | mechanism set 999 | base + P0-std × {−23.7, −42.5, −56.2, −65} | 4 995 | 11.7 | 11.7 | ≈ 1.4 | **≈ 13 cr** |
| A+ | mechanism set 999 | A + P1-nat × 4 levels | 8 991 | 21.0 | 21.1 | ≈ 2.3 | ≈ 23.5 cr |
| B (cheap, under-powered) | 25 events × 20 (deterministic half of the mechanism set) | as A | 2 500 | 5.9 | 5.9 | ≈ 0.8 | ≈ 6.7 cr |
| C (decisive at δ≈0.10) | new 50 × 40 manifest | as A | 10 000 | 23.4 | 23.5 | ≈ 2.6 | ≈ 26 cr |

* **Reuse:** Tranche 2-M of `docs/tier1_proposal.md` already requires the mechanism set ×
  {base, P0-std, mild P0-std (−23.7)}. Option A therefore pre-pays 3 of its 6 systems (≈ 7 cr);
  the *incremental* cost of the sweep over what Tier 1 would spend anyway is the two
  intermediate levels, ≈ 4.7 cr. If the line closes, nothing downstream is spent.
* **Jobs:** 2 on-demand T4 jobs of ~6 GPU-h (base + two levels each), each after a CPU dry-run
  + GPU smoke, from a clean pushed commit. Interruptible only if exact resume is trialled first.
* **Engineering (CPU, free, ~1 session):** `tier0_screening.py` gains `--prompts` (any Q2
  manifest) and per-system `channel_mult` (e.g. `P0-std@1234`); `tier0_screen_eval.py` gains
  the bootstrap overdispersion test (the simulator's `overdispersion_pvalue` on real counts);
  CPU dry-run of both; tests. No new pruning machinery — materialize at every level already
  strict-loads (verified above).

---

## 4. Decision branches — written before the data

Let `sig_s` ⇔ `p_s ≤ 0.05`.

1. **CLOSE** — no `sig_s` at any level, and upper 95 % CI of `δ̂_65` < 0.15. No event structure
   beyond homogeneous capacity loss is detectable at MDE ≈ 0.13. **Gate E is not funded;
   Tranches 1-E/2-M/3-I are not run.** The paper's evidence is the set of negatives (RQ-swap,
   paired saliency, event-level heterogeneity) plus the P1-vs-P0 data-aware comparison;
   the experimental line is closed rather than pivoted (no RQ5).
2. **COLLAPSE** — `sig_s` only at −56.2 % / −65 % and `δ̂_s/μ̂_s` flat or falling with severity.
   Heterogeneity is a by-product of severe capacity loss. Gate E (criterion-specificity) is
   worth funding only if `δ̂_65` ≥ 0.15 (where 30×30 has 87 % power); otherwise CLOSE.
3. **DISPARITY** — `sig_s` already at −23.7 % or −42.5 % with `μ̂_s` ≤ 0.10, and `δ̂_s/μ̂_s`
   rising as severity falls. This is compression-induced semantic disparity at budgets where
   global quality is nearly intact — the scientifically interesting outcome. Fund Gate E at the
   mildest significant level (smallest generic-damage confound) and Gate M on the mechanism set
   as already designed (Tranche 2-M, with the base / P0-std / mild audio already in hand).
4. Anything ambiguous (e.g. `sig` at one intermediate level only) is reported as such; it does
   not trigger Gate E funding by itself.

The branches are ordered by the simplest explanation first, as Gabriel asked: the sweep is
built to *falsify* capacity collapse, not to rescue the heterogeneity story.

---

## 5. What this does not do

* It does not test criterion-specificity (P0-std vs RAND×20 at matched `D_gen`) — that remains
  Gate E and is only worth its cost in branch 3 (or 2 with a large `δ̂_65`).
* It keeps AudioLDM-M + AudioCaps + PANNs top-10; the external-validity concern (one old model,
  one pruning family, one dataset) is unchanged and is a reason to keep the ask small.
* The power model uses a Beta fit of base recall from 35 screen events and independent draws;
  both are conservative-to-neutral assumptions, not guarantees.

**Decision requested:** (a) adopt the sweep as a pre-Gate-E step (DECISION-V4-08) and choose
option A / A+ / B / C, or (b) close the experimental line now on the Tier-0 evidence.
