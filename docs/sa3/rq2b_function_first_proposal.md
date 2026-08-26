# RQ2b — Function-First Positive-Control Qualification Protocol (rev3, FROZEN)

**Status:** **PREREGISTRATION FROZEN** (Gabriel signed off 2026-08-25). F0 + data sourcing are **GO**;
GPU F1 proceeds only after the CPU/data invariants and the final n=64 cost estimate are green. F2 only
if F1 fully passes. Ecological work is **not** authorized by this document.
**Revised:** 2026-08-25 (Montevideo 22:21 rev3 = four sign-off edits; 22:35 rev3.1 = two final gate
clarifications — n_eval frozen at 64 before data, symmetric base/post SESOI gate).
**Supersedes** the rev1 "Design A / Design B" framing. It does **not** alter RQ2's closed verdict
(`235f344`) or L6/L13, which remain failed positive controls with no retrospective rescue.

## Revision history — rev3 (frozen) changes vs rev2

Gabriel accepted the F0→F1→F2 structure and froze the protocol with four edits:

1. **SESOI 0.075 kept as a conservative ex-ante threshold, but its ext14 anchoring is removed.**
   ext14=+0.047 establishes *detectability*, not *practical relevance*; 0.075 stands simply as a
   conservative bar fixed before data to demand a non-trivial adaptation. **n_eval 48 → 64** so the
   powered MDE_post (≈0.067) is *comfortably* below SESOI, not merely equal to it (n=48 gave 0.077≈SESOI,
   which did not satisfy "comfortably ≤ SESOI").
2. **Qualification domain = `mechanical`** (frozen fallback in `freesound_selection_spec.md`, unused in
   RQ2 results → minimal new researcher choice), re-sourced fresh at **160 accepted / 96 train / 64
   eval**. `water_liquid` is not reused.
3. **Band = {8, 9, 10, 11}** — the four *central* transformer blocks, chosen purely by **architectural
   position**, independent of D_P / A_tan / I_PT / adapter / task results. (The rev2 band {11,12,13,14}
   is rejected: it was selected via D_P and coincides with A_tan — the very structural criteria whose
   relationship to adaptation we may later evaluate; the control's support must be exogenous to them.)
   External panel likewise architectural-only and symmetric: **{3, 16}** (not "next-best D_P").
4. **Recipe frozen, not made more generous:** standard LoRA r16/α16, 1000 steps, same optimizer/LR,
   fp16/16-mixed, for **both** F1 and F2. The fresh design already changes exactly the two variables we
   intend — full/band support and substantially more data. **If F1 fails, the interpretation stays
   modest: the task/training/measurement chain failed qualification under the frozen standard recipe — we
   do not infer which component is responsible and we do not iterate.**

## Revision history — rev2 changes vs rev1

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

- **SESOI (frozen): ΔT_AA = 0.075.** A **conservative ex-ante threshold** fixed before any data to
  demand a non-trivial learned adaptation. It is **not** anchored to ext14 (=+0.047, which only
  establishes detectability, not practical relevance). This is a scientific bar set by judgment, not
  derived from the noise floor.
- **Planning noise (conservative prior, from L6/L13):** σ_base≈0.08, σ_post≈0.19.
- **Sample size — FROZEN NOW at n_eval = 64** (not to be re-estimated after seeing F1 outputs). Using the
  conservative prior σ_post≈0.19: planned MDE = 2.8·0.19/√64 = **0.067 < SESOI 0.075 with margin**
  (MDE_base 0.028; the noisy post condition is binding, base is heavily overpowered). n=48 was rejected
  (MDE_post 0.077 ≈ SESOI, not "comfortably below").
  **Why n is frozen, not adaptive:** the relevant σ is the variance of the *paired adapter uplift*, which
  **cannot** be estimated from the 64 reference clips alone — it only exists once F1 generations exist,
  and resizing n on it would be resizing after seeing results. Descriptive properties of the fresh eval
  set may be **reported**, but they must **not** resize n or alter the gate after F1 generations exist.
- **Functional PASS (frozen):** paired bootstrap (seed 20260824, B=10000) over the 64 eval units,
  `lower-CI > 0` **and** point estimate ≥ SESOI (0.075). **FAIL / STOP** otherwise. No metric may be
  swapped in after seeing a result. MDE is a sizing quantity, never the PASS threshold.

---

## 2. Stage F0 — scorer & power qualification (CPU only, 0 cr)

- Re-run `score_taa.py --selftest` and add unit tests pinning the paired aggregation and bootstrap
  determinism on the fresh manifest shape.
- Emit the **frozen** n=64 power table (σ prior 0.19 → MDE 0.067). This table is descriptive/for the
  record; it does **not** resize n (n is frozen per §1).
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
- **Data (fresh, §5):** the `mechanical` CC0 domain, re-sourced at **160 accepted / 96 train / 64
  eval**, disjoint from impact_percussion (L6/L13) and from any future ecological domain. Frozen
  manifest + sha + provenance.
- **Recipe (frozen):** identical optimizer/LR/steps to `train_control_loras.py` (1000 steps, r16/α16,
  fp16 base, 16-mixed), generation frozen (8-step, cfg 7.0, apg 1.0, 10 s, seed = 20260824 + idx),
  bootstrap seed 20260824 / B=10000. Recipe frozen and committed **before** training; not made more
  generous than L6/L13's.
- **Configs (4):** `base_noL`, `base_Lfull`, `post_noL`, `post_Lfull` (× 64 eval = 256 gens).
- **Functional gate (before ANY structural analysis) — symmetric base/post:**
  - **base uplift:** ΔT_AA(`base_Lfull` − `base_noL`) lower-CI > 0 **and** point ≥ SESOI (0.075);
  - **base→post transfer:** ΔT_AA(`post_Lfull` − `post_noL`) lower-CI > 0 **and** point ≥ SESOI (0.075);
    report retention R_L = ΔT(post)/ΔT(base).
  - **Why post must also clear SESOI (not just positivity):** the pruning-compatibility question turns on
    the retention ratio R_L = ΔT_pruned/ΔT_dense; if ΔT_post is tiny, R_L is numerically unstable and
    there is little meaningful adapter function to preserve. A retention ratio is only interpretable once
    the dense adapter contribution clears a preregistered minimum functional effect. (This is exactly the
    footing on which we answer Arshdeep on prune+LoRA retention.)

**Terminal STOP branches:**
- **base fails** (lower-CI ≤ 0 or point < 0.075) → the task / training / measurement chain is **not
  qualified** under the frozen recipe. STOP RQ2b. (Modest conclusion only — no inference about which
  component, no iteration.)
- **base passes but post fails** (post lower-CI ≤ 0 **or** post point < 0.075) → the **meaningful
  base→post adapter-transfer contract is not qualified.** STOP RQ2b.
- **both pass** → the task **and** the functional metric are qualified on a learned adaptation. Only then
  is F2 eligible.

**F1 cost:** 256 gens ≈ 0.31 cr + training ≈ 0.20 cr ≈ **0.51 cr**.

---

## 4. Stage F2 — known-support band control (only if F1 fully passes)

**Question:** can a LoRA whose parameters are *restricted* to a known block band carry a functional
adaptation that the full-backbone sentinel demonstrably can — and does the algebraic support machinery
behave?

- **Adapter:** standard LoRA r16/α16 restricted to a **preregistered band chosen by architectural
  position only**. **Band = {8, 9, 10, 11}** (k=4): the four *central* transformer blocks, selected
  purely from stack position and **independently of D_P / A_tan / I_PT / adapter / task results** — so
  the control's support is exogenous to every criterion we may later evaluate. (The earlier {11,12,13,14}
  is rejected precisely because it was D_P-selected and coincides with A_tan.)
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
  `post_Lband`, `post^{−band}_noL`, `post^{−band}_Lband`, and an **architectural-only, symmetric**
  external panel g ∈ **{3, 16}** (outside the band, chosen by position — not by D_P): `post^{−3}_noL`,
  `post^{−3}_Lband`, `post^{−16}_noL`, `post^{−16}_Lband` (× 64 = 512 gens).
- **F2 functional gate (before structural read-out):** same symmetric criterion as F1 —
  ΔT_AA(`base_Lband`) lower-CI > 0 **and** point ≥ SESOI (0.075), **and** ΔT_AA(`post_Lband`)
  lower-CI > 0 **and** point ≥ SESOI (0.075).
- **Structural / algebraic read-out (only after the functional gate passes):** verify (1)–(3) above; i.e.
  band removal collapses ΔT to 0, and ≥1 external removal retains a positive uplift.

**Terminal STOP branches:**
- **F2 functional gate fails while F1 passed** → clean, interpretable evidence that **restricted support
  is insufficient** to carry a functional adaptation this task admits. STOP; report the support-capacity
  limitation.
- **F2 functional gate passes** → proceed to the algebraic support read-out; a passing F2 is the first
  genuinely valid task-level positive control, and only then does localisation-instrument work (A_eco /
  A_tan vs D_P) become scientifically meaningful.

**F2 cost:** 512 incremental gens ≈ 0.61 cr + training ≈ 0.20 cr ≈ **0.81 cr**.

---

## 5. Data separation (strict)

| set | domain | used for | never used for |
|---|---|---|---|
| impact_percussion (existing, sha aa76dc0f) | percussion/impact | L6/L13 (RQ2, closed) | RQ2b |
| **`mechanical` (fresh RQ2b manifest)** | `machine OR motor OR mechanical OR engine` | **F1 + F2** (shared train+eval) | ecological validation |
| ecological domain(s) (future, untouched) | new CC0 domain(s) | later A_eco / ecological test | qualification |

- The RQ2b qualification domain is **`mechanical`** (frozen fallback in `freesound_selection_spec.md`,
  unused in RQ2 results), re-sourced CC0 from Freesound (CPU, 0 cr) with its own frozen split_seed,
  manifest sha, and full sourcing provenance — **160 accepted / 96 train / 64 eval**, disjoint by clip id
  from impact_percussion and from any future ecological pull. `water_liquid` (32/8) is **not** reused.
- **The F1/F2 sentinel adapters are calibration/control artefacts and can never become the later
  held-out ecological validation adapter** (Gabriel, explicit). Ecological domains stay untouched.

---

## 6. Cost envelope

| stage | compute | cr |
|---|---|---|
| F0 scorer/power qualification | CPU | 0 |
| data sourcing (fresh domain) | CPU | 0 |
| F1 full-backbone sentinel (train + 256 gens) | 1 T4 job | ~0.51 |
| F2 band control (train + 512 incremental gens) | 1 T4 job | ~0.81 |
| smoke / provisioning overhead | — | ~0.10 |
| **total qualification chain (n=64)** | | **~1.42 cr** |

- **Fits the current 1.63-cr headroom** (SA3_TOTAL 3.369, cap 5) with ~0.21 cr margin. Watchdog on
  measured cost per job as in rc1.4; hard per-job ceilings set before launch, and **STOP after F1 if the
  real cost deviates materially from this estimate.**
- **Ecological work is NOT in this envelope** and must not be assumed to fit the remaining headroom —
  qualification comes first; ecological adapters need a separate budget decision.
- F2 is only ever launched if F1 fully passes, so a failed sentinel caps spend at ~0.51 cr.

---

## 7. Invariants (carried from RQ2, non-negotiable)

- **Function first, always:** no `A_tan` / `A_eco` / structural inspection until the intact-model
  held-out functional gate passes (F1 for existence, F2 for the band).
- **Known parameter support is not known functional localisation:** the band's ground truth is algebraic
  (support removal), never an `A_eco` ranking expectation.
- **SESOI is scientific, MDE is for sizing:** PASS = lower-CI > 0 and point ≥ SESOI, for **both** base
  and post; never lower-CI > MDE.
- **n is frozen at 64 before data:** the paired-uplift σ cannot be estimated from reference clips; fresh
  eval σ is reported descriptively only and never resizes n or moves the gate after F1 generations exist.
- **No retrospective rescue:** any metric / n / SESOI / recipe choice applies only to this fresh phase,
  never to L6/L13.
- **Freeze before generating:** domain manifest, SESOI, n, seeds, band, external panel and all gates
  frozen in a committed preregistration before any adapter is trained or any audio generated.
- **Negatives are terminal, not iterated.** Each STOP branch closes RQ2b at that point.
- **No compute without explicit authorization.**

---

## 8. Sign-offs (RESOLVED, Gabriel 2026-08-25 — preregistration frozen)

1. **SESOI = 0.075** ✓ — accepted as a conservative ex-ante threshold (ext14 anchoring removed).
2. **n_eval = 64, FROZEN NOW** ✓ — planned MDE_post ≈ 0.067 < SESOI on the conservative prior σ≈0.19;
   **not** re-estimated/resized after F1 outputs (the paired-uplift σ does not exist until F1 generates).
3. **Qualification domain = `mechanical`, 160/96/64** ✓ — fresh RQ2b manifest, `water_liquid` not reused.
4. **Band = {8,9,10,11}** ✓ (architectural centre, criterion-exogenous); **external panel {3,16}** ✓
   (architectural, symmetric).
5. **Recipe = standard LoRA r16/α16, 1000 steps, frozen** ✓ for both F1 and F2 — not made more generous.
6. **Symmetric base/post gate** ✓ — both base AND post require lower-CI>0 AND point ≥ 0.075 (post
   positivity alone is insufficient; retention R_L is only interpretable above a preregistered minimum).

## 9. Next steps (authorization state)

- **GO now (CPU, 0 cr):** F0 scorer/unit tests + emit the frozen n=64 power table; stream-source the
  `mechanical` domain until **160 clips pass all audio filters** and freeze the 96/64 manifest (sha,
  auto-prompts, provenance); update the cost estimate against the actual 256-generation F1 design.
  **n stays frozen at 64; fresh-eval σ is descriptive only, never a resize.**
- **GPU F1 is GO automatically** iff all manifest/F0 invariants pass **and** the revised projected spend
  keeps the whole qualification chain inside the 1.63-cr envelope. Watchdog mandatory.
- **Conditional:** F2 only if F1 fully passes; ecological work is **not** authorized here.

**Frozen F1/F2 terminal gates (symmetric):** base lower-CI ≤ 0 **or** base point < 0.075 → STOP RQ2b;
base passes but post lower-CI ≤ 0 **or** post point < 0.075 → meaningful base→post transfer contract not
qualified → STOP; both pass → F2 eligible (F2 applies the identical base+post criterion).
**If F1 fails, the conclusion is modest:** the task/training/measurement chain failed qualification under
the frozen standard recipe — no inference about which component, no iteration.
