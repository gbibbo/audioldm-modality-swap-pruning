# Master Plan v4 — Amendment: ICASSP-2027 publication pivot (rev4, ADOPTED = DECISION-V4-09)

**Status: rev4 ADOPTED by Gabriel 2026-08-26 03:02 as DECISION-V4-09 (binding; commit 63b11e3).**
Evidence base: `docs/publication_decision_memo.md` (verdict GO-AUDIOLDM, §5 rev4); ledger
ICASSP-PIVOT / ICASSP-PIVOT-REV4 / DECISION-V4-09. Authorization now in force is exactly §H
(CPU wiring + frozen held-out 64-prompt manifest + CPU dry-runs + ONE ≤0.1-cr GPU smoke); Gate 0
stays behind STOP-0; nothing downstream is authorized. Two implementation details frozen at
adoption are in §C (CI construction; exact sample size).

**rev4 = rev-PROPOSED + Gabriel's five binding reviewer-facing corrections (2026-08-26):**
(1) a standalone non-inferiority gate is now *required alongside* differential fragility;
(2) the statistical unit is the prompt (clustered/cluster-bootstrap), never the generated clip;
(3) the Gate-0 evaluation battery must be genuinely held out from the 193 training captions;
(4) the cheap falsifier is separated from a re-costed paper-completion tranche, with an
independent-replicate adapter B and the full severity curve prioritized over the Scenario-A
oracle; (5) a zero-cost request to Arshdeep for the recovered (post-finetuning) checkpoint, with
a time-box and an explicit pre-recovery limitation if it does not arrive. Plus: paper-facing
novelty wording softened to "we found no prior work…".

## A. Objective change

The project optimization target is **probability of a defensible ICASSP-2027 submission by
2026-09-16**, not continuity with prior lines. Prior spend confers no right to continue any line.

## B. Scope changes

1. **DEFERRED / NOT part of ICASSP:** the severity-sweep proposal (13–26 cr,
   `docs/severity_sweep_proposal.md`) — exceeds the ~6-cr balance and is off the critical path.
2. **Active line:** Scenario-B legacy-adapter compatibility on AudioLDM-M-Full, per memo §5.
3. v4 Tier-1/Tier-2, M5 recovery, and Gate E remain deferred as before.

## C. Primary endpoints and the analysis plan (frozen before any GPU)

All scoring is CLAP text-audio cosine with the LAION-CLAP **fused** checkpoint (Kim's scorer).
Define, per severity s (s=0 is dense):

* `C(s)` = standalone CLAP (backbone alone).
* `ΔCLAP(s)` = paired adapter uplift = mean over prompts of (adapter − backbone) at s.
* `E(s) = C(0) − C(s)` — standalone degradation.
* `D(s) = ΔCLAP(0) − ΔCLAP(s)` — adapter benefit lost.
* `F(s) = D(s) − E(s)` — **excess** adapter loss beyond standalone loss.

**Statistical unit = prompt (correction 2).** Evaluation is **exactly 64** held-out prompts × 3
seeds. Per prompt, compute its paired effect from its three matched seeds (seeds paired across
conditions, averaged within prompt); then **cluster bootstrap over the 64 prompts** (resample
whole prompts, keep all seeds/conditions together). The 192 generations are NEVER treated as
independent. Same principle for Gate 0 and every severity comparison.

**FROZEN AT ADOPTION (DECISION-V4-09, 2026-08-26) — do not change after seeing data:**

* **CI construction (ONE definition everywhere — Gate-0 ΔCLAP, E(s), F(s), every severity):**
  cluster **percentile** bootstrap, two-sided **95 %** interval (2.5/97.5 pctiles), **B = 10000**,
  **fixed RNG seed = 20260826**. Resampling unit = **prompt**; the three generation seeds stay
  clustered inside each resampled prompt. No other interval method (BCa/normal/t) and no switching
  after data. Implemented once in `research_pruning/eval/cluster_bootstrap.py`; every gate calls it.
* **Sample size (exact, not "approximately"):** the Gate-0 / falsifier primary battery is
  **exactly 64 held-out prompts × 3 paired generation seeds** = 192 generations per system. Any
  deviation is a protocol change requiring Gabriel, not an implementation detail.
* **Smoke measures execution/cost ONLY.** SESOI (+0.025), effective batch (2), epochs (200),
  rank (8), alpha (16), LR (1e-5), the prompt filter, and the primary scorer (LAION-CLAP fused)
  are fixed and are **not** tuned from smoke results.

**Claim-scope rule (correction 1, wording):** the preserved quantity is **standalone text-audio
semantic alignment (CLAP)**, not "generation quality" in general. FAD/KAD are corroborative only
and cannot rescue a failed CLAP gate; the broader word "quality" appears only if FAD/KAD agree.

## D. The gated chain (pre-registration — full spec in memo §5)

* **Smoke (≤0.1 cr):** measure real s/step (LoRA train, 4-s, batch 2) and s/clip (4-s, S=50) on
  M-Full; re-anchor every derived cost. STOP-0 if Gate 0 then projects > 1 cr — **report the
  measured numbers; never auto-shrink fidelity (e.g. batch 2→4) to fit the ceiling.**

* **Gate 0 (≤1 cr):** powered replication of the Kim recipe on **dense M-Full** — LoRA `to_q`/`to_v`
  all attention, **r8/α16**, lr 1e-5 AdamW, public 193-clip hip-hop set, 4-s crops, **200 epochs**.
  *200 epochs = first published checkpoint and the affordable pre-registered operating point, not
  a "best band" (correction 3); this is reproduction of a published recipe point on M-Full, not a
  claim of exact replication of the S-Full headline.* **Effective batch = 2, locked**; mixed
  precision only if numerically validated vs fp32; any scientific-optimization change needs review.
  Primary endpoint **ΔCLAP(0)** on the held-out battery, cluster-bootstrap CI. **SESOI +0.025;
  PASS = point ≥ SESOI AND lower-CI95 > 0.** FAIL ⇒ STOP-1 (thesis dead; no adapter iteration).

* **Held-out prompt battery (correction 3, frozen before the smoke):** ~64 hip-hop-relevant
  captions sampled deterministically from **MusicCaps outside the released 193-example training
  subset** (text only — no audio scraping needed), under a rule committed before the smoke
  (source CSV + seed + hip-hop/rap filter + exact IDs) with an exact/near-duplicate check against
  the 193 training captions. No hand-written or post-result curation. Gate 0 thus tests the
  recipe's **generalization**, not reconstruction of its training captions.

* **Phenomenon falsifier (generation only, no training):** severities {dense, (1,2,3,4) −23.7 %,
  (1,2,3,1) −65 %}; systems {backbone, backbone + mask-sliced legacy LoRA}. **A severity
  establishes the primary phenomenon iff BOTH (cluster-bootstrap CIs) (correction 1):**
  (i) **standalone non-inferiority:** `upper-CI95[E(s)] ≤ 0.025`; AND
  (ii) **differential fragility:** `point F(s) ≥ 0.025` AND `lower-CI95[F(s)] > 0`.
  If F(s) > 0 but (i) fails, the point is **descriptive only** (generic capacity loss, not a
  compatibility problem) and cannot carry the main claim. No severity satisfies both ⇒ STOP-2
  (pre-registered negative; no method work, no reframing).

* **Paper-completion tranche — ONLY if the falsifier passes; re-costed after the smoke; authorized
  separately, NOT by adopting this amendment (correction 4):**
  - **adapter A:** extend generation to the full nested ladder {0, 23.7, 42.5, 56.2, 65}% (a real
    severity curve; existing strict-load-verified checkpoints).
  - **adapter B:** train ONE independent replicate under the frozen Gate-0 recipe (different
    seed/data-shuffle), evaluate only at {dense + the qualifying severity} — evidence the effect
    is not a single-adapter accident.
  - **Priority under budget pressure: adapter B + full severity curve BEFORE the Scenario-A
    oracle.** A reviewer forgives a limited oracle, not a contribution resting on one realization
    of one adapter.

* **Scenario-A oracle (baseline (e), budget-permitting, after primary robustness):** LoRA
  retrained on the pruned backbone at the qualifying severity — quantifies "why not just retrain".

* **Recovered-backbone baseline (correction 5; UPGRADED 2026-08-26 — checkpoint is PUBLIC):**
  Arshdeep's recovered/finetuned **(1,2,3,1)** checkpoint is publicly downloadable at Zenodo
  **21977996** (`l1_p1_finetuned_global_step_999999.ckpt`, md5 `cfb7ca3f8c712850f5a4bfe2162f5d1c`,
  4.45 GB, CC-BY-4.0; recovered (1,2,1,1) also public; no recovered p2_dp2). **The Arshdeep email
  is CANCELLED.** This becomes an in-scope, reproducible system: apply the frozen sliced LoRA to
  the **published recovered** backbone; the strong result is *recovery restores standalone
  alignment but not legacy-adapter utility* — standard recovery metrics silently passing a broken
  downstream capability. **Caveats:** verify the file is weights-only vs optimizer/EMA state on
  download (it is larger than pruned-only p1); the sliced LoRA lands on the same pruned
  architecture/kept-set but recovered (non-selection) weights — reported honestly. **Time-box
  (scheduling only):** fetch at the phenomenon stage, never block Gate 0; if download/state-verify
  slips, fall back to the pre-recovery statement as a limitation.

* **Baselines:** (a) dense ± adapter; (b) pruned standalone per severity; (c) pruned + sliced
  legacy adapter; (d) independent-replicate adapter B; (e) Scenario-A oracle; (f) optional
  random-slicing control.

## E. Budget and STOPs

* **Falsifying chain (all this GO puts on the near-term table):** smoke 0.1 + Gate-0 train
  0.5–1.0 + Gate-0 eval 0.3 + 3-severity phenomenon gen 0.6–0.8 ≈ **1.5–2.2 cr ≤ 2.5 ✓**.
* **Completion tranche (only after a positive falsifier, re-costed post-smoke):** adapter B +
  ladder + oracle ≈ 1.5–2.2 cr.
* **Effective spend cap = 3.0 cr (BINDING; arithmetic correction DECISION-V4-10, 2026-08-26).**
  The measured balance is **5.0 cr** (`docs/compute_budget.md`), so the nominal 3.5-cr hard cap and
  the ≥2-cr reserve cannot both hold: **effective cap = min(3.5, 5.0 − 2.0) = 3.0 cr** while the
  balance is 5.0 and the reserve stays binding. The nominal 3.5 is kept only as a reference ceiling.
  Do not raise the overall spend now — after a positive phenomenon result, return with the measured
  balance and the cheapest confirmation plan. Watchdog `--max-cost` on every job; CPU dry-run first.
* **Clip length (DECISION-V4-10, 2026-08-26, PRE-DATA):** Gate 0 + all downstream ICASSP work run at
  **3.84 s / latent_t=96**, not 10.24 s. The earlier "256-fixed → must pad" premise was REFUTED by an
  executable audit (`scripts/research/kim_clip_length_audit.py`): the only constraint is latent_t
  divisible by 8; 3.84 s is the nearest shorter valid duration, strict-loads the same weights, has a
  valid loss, and leaves pruning + Scenario-B LoRA transfer unchanged. Deterministic **center** crop of
  each 4-s Kim clip (frozen; Kim released no boundary convention). Paper-facing: a **near-4-s M-Full
  transposition** of Kim's recipe, NOT an exact S-Full replication. Frozen in
  `configs/research/icassp_gate0_prereg.yaml` v2 (v1 preserved at `…_v1_superseded.yaml`).
* **STOP-0** smoke projects Gate 0 > 1 cr → re-cost to Gabriel (no fidelity auto-shrink).
  **STOP-1** Gate 0 FAIL → thesis dead; GO-ALTERNATIVE vs NO-GO-ICASSP decision to Gabriel.
  **STOP-2** phenomenon absent → pre-registered negative; venue fallback. **STOP-3** effective cap
  (3.0 cr) reached.

## F. Frozen technical facts this amendment relies on (verified, CPU, committed)

1. Mask-induced LoRA transfer is exact and deterministic at every ladder seam
   (`tests/research/test_lora_mask_transfer.py` 6/6; `test_scenario_b_seam_transfer.py` S1–S6:
   238/238 seams mapped, zero learned parameters; (1,2,3,3) drop rule).
2. Published pruned checkpoint = pure selection of dense, never finetuned, bit-exact (R5/P4) —
   backbone and sliced adapter share identical selections. **Pre-recovery** (correction 5 hook).
3. Kim et al.: recipe/data public, **weights not released**, published evaluation single-prompt
   (noise-compatible) — hence Gate 0's powered, held-out replication design.
4. No direct novelty collision (closest CA-LoRA, ASSP, LoRA-X, SVDQuant, X-Adapter — contrasted
   in memo §4); differential fragility not found measured in prior work (wording: "we found no
   prior work…", never "no study in any domain").
5. AudioLDM third-party adapter ecosystem is n=1 today → ecosystem motivation framed prospectively
   (MusicGen ~196 LoRAs; FLUX-lite/SSD-1B breakage as documented pain).

## G. Provenance rules (unchanged, restated)

Every number: commit, resolved config, checkpoint hash, manifest, seeds, raw output, GPU
metadata. Gate outcomes (PASS and FAIL) to `docs/experiment_ledger.md`; costs to
`docs/compute_budget.md`; claims only via `docs/claims_matrix.md`. The frozen held-out prompt
battery and bootstrap seeds are committed BEFORE the first GPU job.

## H. What adoption authorizes (and only this)

On Gabriel's adoption of rev4: (1) status flips to ADOPTED (DECISION-V4-09) with a ledger entry;
(2) CPU wiring proceeds — CLAP-fused scorer path, trainer data path, mask-transfer op,
cluster-bootstrap analysis, **and the frozen held-out prompt manifest** — each with a CPU
dry-run; (3) **exactly one ≤0.1-cr smoke** is authorized. Gate 0 stays behind the smoke cost gate
(STOP-0). No other GPU spend. The completion tranche and the oracle are authorized only later,
after a positive falsifier, with fresh measured costs.
