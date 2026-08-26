# RQ2b — Function-First Positive-Control Qualification Protocol (rev2)

**Status:** PROPOSAL for review — GO conceptual / **NO-GO compute**. CPU/docs only, 0 cr spent.
**Revised:** 2026-08-25 (Montevideo 22:01) after Gabriel's review of rev1 (`07b7152`).
**Supersedes** the rev1 "Design A / Design B" framing in this file. It does **not** alter RQ2's closed
verdict (`235f344`) or L6/L13, which remain failed positive controls with no retrospective rescue.

## Revision history — what changed from rev1 and why

rev1 was accepted in *direction* (function-first) but rejected in *content*. Five corrections, applied
in full below:

1. **The 0.231 is NOT recovery headroom.** `T_AA=0.435` is `base_noL`, `0.204` is `post_noL` — a
   **base→post checkpoint difference between two different models**, not a pruning-induced gap a band
   adapter could "recover." Every claim built on "0.231 available recovery capacity" is deleted. A LoRA
   does not turn `post` into `base`; it learns a domain adaptation. That gap is used **only** as a rough
   scale reference for the metric's dynamic range, never as headroom.
2. **Known parameter support ≠ known functional localisation.** L6/L13 proved the opposite: adapter
   parameters lived entirely in one host block, yet the largest `A_eco` appeared in *other* blocks — the
   effect propagates through the network (this is exactly why we dropped host=top-1). A k-block band
   therefore gives known **parameter support**, and a valid *algebraic* ground truth, but **not** an
   expected in-band `A_eco` ranking. We do **not** require `A_eco` to rank the band above external blocks.
3. **Causal diagnosis downgraded.** Record now reads: **measurement power is demonstrably limiting;
   single-block support capacity and training/data adequacy remain unresolved alternative explanations.**
   Removed as scientific conclusions: "32 clips not the primary suspect" and "more steps / higher rank
   ruled out." Those are forbidden only as *retrospective rescues of L6/L13*; they are legitimate,
   un-falsified design variables for a fresh experiment.
4. **SESOI ≠ MDE.** MDE sizes the sample; it is not a PASS threshold. We define a scientifically
   justified SESOI first, then choose n so the powered MDE is comfortably ≤ SESOI. The functional PASS
   is `lower-CI > 0` **and** the effect compatible with / exceeding SESOI — never `lower-CI > MDE`.
   (Also corrected: the "order of magnitude below MDE" claim is false for `|L13_base|≈0.064` vs
   MDE≈0.090.)
5. **Primary sequence replaced** by a three-stage function-first qualification F0→F1→F2 (below): prove
   SA3 can learn *any* useful, base→post-transferring adaptation with the real LoRA mechanism *before*
   asking where to preserve it.

---

## 0. Corrected diagnosis of the L6/L13 null (from existing data)

From `control_taa_scores.json` `per_j` (n=8). This characterises the T_AA **instrument**; it does not
re-decide L6/L13.

| pair | mean ΔT_AA | per-clip SD | MDE @ n=8 |
|---|---|---|---|
| L6_base | +0.005 | 0.080 | 0.079 |
| L13_base | −0.064 | 0.091 | 0.090 |
| L6_post | −0.017 | 0.190 | 0.188 |
| L13_post | −0.016 | 0.173 | 0.171 |
| L13_ext14 | +0.047 | 0.035 | 0.034 |

- **Power is demonstrably limiting.** n=8 gives an MDE of 0.079 (base) to 0.19 (post); the post
  condition is especially noisy (one pathological eval clip drove SD≈0.18). The single-block dense
  uplifts sat below (base) or near (L13_base 0.064 vs 0.090) the detectable floor.
- **The instrument is not dead.** `L13_ext14` resolved +0.047 with a tight CI (SD 0.035).
- **`base_noL`=0.435 vs `post_noL`=0.204** is a checkpoint difference (scale reference only), **not**
  recoverable headroom.

**Open, unresolved alternatives** for why the single-block controls produced no detectable uplift:
(a) single-block **support capacity**; (b) **training/data adequacy** (32 clips, 1000 steps, r16);
(c) **measurement power/suitability** of T_AA for learned SA3 adaptations. Only (c) is partly
established (as limiting). F0→F1→F2 is designed to separate them without asserting any in advance.

---

## 1. SESOI and power framework (applies to every functional gate)

- **SESOI (proposed): ΔT_AA = 0.075.** Scientific anchor: it exceeds the largest *cleanly detected*
  real effect in the L6/L13 run (the ext14 removal, +0.047) by ~1.6×, i.e. the minimum paired
  audio-audio cosine uplift we would call a functionally relevant learned adaptation. **This value is
  the single most important reviewable parameter — Gabriel signs off before any data.**
- **Planning noise (to be re-estimated on fresh data):** σ_base≈0.08, σ_post≈0.19 (from L6/L13).
- **Sample-size rule:** choose n so MDE = 2.8·σ/√n ≤ SESOI at 80 % power, two-sided α=0.05.
  **n_eval = 48** ⇒ MDE_base 0.032, **MDE_post 0.077 ≈ SESOI** (the noisy post condition is the binding
  constraint; base is heavily overpowered). If the re-estimated σ_post is materially larger, n is raised
  before generation, not after.
- **Functional PASS (frozen):** paired bootstrap (seed 20260824, B=10000) over the n eval units,
  `lower-CI > 0` **and** point estimate ≥ SESOI. **FAIL / STOP** otherwise. No metric may be swapped in
  after seeing a result.

---

## 2. Stage F0 — scorer & power qualification (CPU only, 0 cr)

- Re-run `score_taa.py --selftest` and add unit tests pinning the paired aggregation and bootstrap
  determinism on the fresh manifest shape.
- Recompute the power/MDE table on the **fresh** eval σ once F1/F2 data exists (before generating).
- A synthetic audio-perturbation check (reference vs artificially degraded) is admissible **only** as an
  implementation/sensitivity unit test of the scorer. It is **explicitly not** evidence that T_AA is
  valid for *learned SA3 adaptations* — that validity is what F1 establishes. F0 cannot, by itself,
  qualify the metric.

**F0 gate:** scorer tests green + a frozen power table with n chosen for the proposed SESOI. No compute.

---

## 3. Stage F1 — functional sentinel: full-backbone LoRA (the control we never had)

**Question:** can SA3 learn *any* reproducible, held-out functional uplift with the real LoRA mechanism,
and does it transfer base→post? No structural inspection whatsoever.

- **Adapter:** one ordinary standard LoRA, **r16 / α16**, over the **entire `transformer.layers`
  stack** (full backbone). Same rank as L6/L13 — the only deliberate changes from the failed controls
  are (i) full support and (ii) ample data (§5), so F1 gives adaptation the best honest chance.
- **Data (fresh, §5):** a fresh CC0 domain, disjoint from impact_percussion (L6/L13) and from any future
  ecological domain. Train ≥ 64 clips, eval = 48 clips, frozen manifest + sha + provenance.
- **Recipe (frozen):** identical optimizer/LR/steps to `train_control_loras.py` (1000 steps, fp16 base,
  16-mixed), generation frozen (8-step, cfg 7.0, apg 1.0, 10 s, seed = 20260824 + idx), bootstrap
  seed 20260824 / B=10000. Recipe frozen and committed **before** training.
- **Configs (4):** `base_noL`, `base_Lfull`, `post_noL`, `post_Lfull` (× 48 eval = 192 gens).
- **Functional gate (before ANY structural analysis):**
  - **base uplift:** ΔT_AA(`base_Lfull` − `base_noL`) lower-CI > 0 **and** ≥ SESOI;
  - **base→post transfer:** ΔT_AA(`post_Lfull` − `post_noL`) lower-CI > 0 (robust positivity at
    powered n); report retention ΔT(post)/ΔT(base).

**Terminal STOP branches:**
- **base uplift fails** → the task / training / measurement chain is **not qualified**. STOP RQ2b.
  (SA3 could not learn a detectable adaptation even with full backbone + ample data + powered eval → the
  microscope, not a single block, is the problem.)
- **base passes, post fails** → the **base→post adapter-transfer contract that motivates the whole
  project does not hold for this task.** STOP RQ2b.
- **both pass** → the task **and** the functional metric are qualified on a learned adaptation. Only then
  is F2 eligible.

**F1 cost:** 192 gens ≈ 0.23 cr + training ≈ 0.20 cr ≈ **0.43 cr**.

---

## 4. Stage F2 — known-support band control (only if F1 fully passes)

**Question:** can a LoRA whose parameters are *restricted* to a known block band carry a functional
adaptation that the full-backbone sentinel demonstrably can — and does the algebraic support machinery
behave?

- **Adapter:** standard LoRA r16/α16 restricted to a **preregistered, adapter-blind band**. **Band =
  {11, 12, 13, 14}** (k=4): the four **lowest-D_P blocks** at N=32 (`rq1_reanalysis.json`), which
  **coincide exactly with the A_tan / I_PT k=4 tail** (jaccard 1.0, disagreement 0). Chosen *before*
  training, on structural grounds both criteria agree on — not on any observed adapter behaviour.
- **Data / recipe:** **identical** fresh domain, train set, eval set, seeds and recipe as F1. The **only**
  difference from F1 is adapter support (band vs full backbone) — so an F1-passes / F2-fails outcome
  isolates *restricted support* as the cause.
- **Ground truth is parameter support, NOT `A_eco` ranking.** Valid, checkable claims:
  1. the adapter has **zero parameters outside the band** (assert at build time);
  2. **removing the whole band removes the adapter algebraically** — `post^{−band}` + band-restricted
     adapter is bit-identical to `post^{−band}` with no adapter (ΔT_AA(`post^{−band}`) = 0 exactly);
  3. task uplift **remains observable under at least some frozen external removals** g ∉ band.
  We do **not** require `A_eco` maxima to fall inside the band.
- **Configs (reusing F1's `base_noL`/`post_noL`, same domain/seeds — 8 incremental):** `base_Lband`,
  `post_Lband`, `post^{−band}_noL`, `post^{−band}_Lband`, and external panel g ∈ **{9, 10}** (next-lowest
  D_P outside the band): `post^{−9}_noL`, `post^{−9}_Lband`, `post^{−10}_noL`, `post^{−10}_Lband`
  (× 48 = 384 gens).
- **F2 functional gate (before structural read-out):** same as F1 — ΔT_AA(`base_Lband`) lower-CI > 0 and
  ≥ SESOI, and ΔT_AA(`post_Lband`) lower-CI > 0.
- **Structural / algebraic read-out (only after the functional gate passes):** verify (1)–(3) above; i.e.
  band removal collapses ΔT to 0, and ≥1 external removal retains a positive uplift.

**Terminal STOP branches:**
- **F2 functional gate fails while F1 passed** → clean, interpretable evidence that **restricted support
  is insufficient** to carry a functional adaptation this task admits. STOP; report the support-capacity
  limitation.
- **F2 functional gate passes** → proceed to the algebraic support read-out; a passing F2 is the first
  genuinely valid task-level positive control, and only then does localisation-instrument work (A_eco /
  A_tan vs D_P) become scientifically meaningful.

**F2 cost:** 384 incremental gens ≈ 0.46 cr + training ≈ 0.20 cr ≈ **0.66 cr**.

---

## 5. Data separation (strict)

| set | domain | used for | never used for |
|---|---|---|---|
| impact_percussion (existing, sha aa76dc0f) | percussion/impact | L6/L13 (RQ2, closed) | RQ2b |
| **RQ2b qualification domain (fresh, to source)** | a new CC0 domain | **F1 + F2** (shared train+eval) | ecological validation |
| ecological domain(s) (future, untouched) | new CC0 domain(s) | later A_eco / ecological test | qualification |

- The RQ2b qualification domain is sourced CC0 from Freesound (CPU, 0 cr) under the frozen
  `freesound_selection_spec.md` rule, with its own frozen split_seed, manifest sha, and full sourcing
  provenance — **train ≥ 64 / eval = 48**, disjoint by clip id from impact_percussion and from any
  future ecological pull. (`water_liquid`, already sourced but only 32/8, is **too small/underpowered**;
  either re-source it larger under a new manifest or source a new domain.)
- **The F1/F2 sentinel adapters are calibration/control artefacts and can never become the later
  held-out ecological validation adapter** (Gabriel, explicit). Ecological domains stay untouched.

---

## 6. Cost envelope

| stage | compute | cr |
|---|---|---|
| F0 scorer/power qualification | CPU | 0 |
| data sourcing (fresh domain) | CPU | 0 |
| F1 full-backbone sentinel (train + 192 gens) | 1 T4 job | ~0.43 |
| F2 band control (train + 384 incremental gens) | 1 T4 job | ~0.66 |
| smoke / provisioning overhead | — | ~0.10 |
| **total qualification chain** | | **~1.19 cr** |

- **Fits the current 1.63-cr headroom** (SA3_TOTAL 3.369, cap 5) with ~0.4 cr margin. Watchdog on
  measured cost per job as in rc1.4; hard per-job ceilings set before launch.
- **Ecological work is NOT in this envelope** and must not be assumed to fit the remaining headroom —
  qualification comes first; ecological adapters need a separate budget decision.
- F2 is only ever launched if F1 fully passes, so a failed sentinel caps spend at ~0.43 cr.

---

## 7. Invariants (carried from RQ2, non-negotiable)

- **Function first, always:** no `A_tan` / `A_eco` / structural inspection until the intact-model
  held-out functional gate passes (F1 for existence, F2 for the band).
- **Known parameter support is not known functional localisation:** the band's ground truth is algebraic
  (support removal), never an `A_eco` ranking expectation.
- **SESOI is scientific, MDE is for sizing:** PASS = lower-CI > 0 and ≥ SESOI; never lower-CI > MDE.
- **No retrospective rescue:** any metric / n / SESOI / recipe choice applies only to this fresh phase,
  never to L6/L13.
- **Freeze before generating:** domain manifest, SESOI, n, seeds, band, external panel and all gates
  frozen in a committed preregistration before any adapter is trained or any audio generated.
- **Negatives are terminal, not iterated.** Each STOP branch closes RQ2b at that point.
- **No compute without explicit authorization.**

---

## 8. Open decisions for Gabriel (sign-off before any preregistration freeze)

1. **SESOI = 0.075** — accept, or set another value (drives n and cost).
2. **Fresh qualification domain** — re-source `water_liquid` larger, or a brand-new CC0 domain? (either
   is 0 cr CPU).
3. **Band = {11,12,13,14}, k=4** — accept the lowest-D_P / A_tan-tail band, or a different frozen rule.
4. **F1 recipe = L6/L13 recipe at full backbone (1000 steps, r16)** — accept, or grant F1 a more
   generous training budget to further de-risk the sentinel (a design choice, not a rescue).

No GPU is queued. On your answers I will freeze the preregistration (CPU) and only then request compute.
