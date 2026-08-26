# Master Plan v4 — Amendment: ICASSP-2027 publication pivot (PROPOSED)

**Status: PROPOSED (2026-08-26), pending Gabriel's review. Becomes DECISION-V4-09 only on his
explicit adoption. NO GPU until then.** Evidence base: `docs/publication_decision_memo.md`
(verdict GO-AUDIOLDM); ledger ICASSP-PIVOT.

## A. Objective change

The project optimization target is **probability of a defensible ICASSP-2027 submission by
2026-09-16**, not continuity with prior lines. Prior spend confers no right to continue any line.

## B. Scope changes

1. **DEFERRED / NOT part of ICASSP:** the severity-sweep proposal (13–26 cr,
   `docs/severity_sweep_proposal.md`) — it exceeds the ~6-cr balance and is not on the critical
   path. It stays a documented, un-adopted proposal.
2. **Active line:** Scenario-B legacy-adapter compatibility on AudioLDM-M-Full, per the memo's
   §5 spec (gated chain: smoke → Gate 0 → phenomenon test → [oracle baseline] → figure/paper).
3. The v4 Tier-1/Tier-2 programme, M5 recovery, and Gate E remain deferred as before.

## C. The gated chain (pre-registration summary — full spec in memo §5)

* **Smoke (≤0.1 cr):** measure real s/step (LoRA train, 4-s, batch 2) and s/clip (4-s, S=50) on
  M-Full; re-anchor every derived cost below. STOP-0 if Gate 0 projects > 1 cr.
* **Gate 0 (≤1 cr):** powered replication of the Kim et al. recipe on dense M-Full
  (r8/α16, lr 1e-5, 193 public clips, 200 epochs). Primary endpoint ΔCLAP on a frozen 64-prompt
  × 3-seed battery, paired bootstrap CI. **SESOI +0.025; PASS = point ≥ SESOI AND lower CI95 >
  0.** FAIL ⇒ STOP-1 (thesis dead; no adapter iteration; fallback decision to Gabriel).
* **Phenomenon test (generation only):** severities {dense, (1,2,3,4), (1,2,3,1)}; systems
  {backbone, backbone + mask-sliced legacy LoRA}; **differential fragility at s iff
  D(s) − E(s) ≥ 0.025 with paired lower CI95 > 0**, where D = adapter uplift lost, E =
  standalone CLAP degradation (same scale). Absent ⇒ STOP-2 (pre-registered negative; no
  method work; venue-fallback decision).
* **Oracle baseline (budget-permitting):** LoRA retrained on the pruned backbone at (1,2,3,4)
  (Scenario-A oracle; quantifies "why not retrain").
* **Budget:** total through Sep 10 ≈ 2.4–3.3 cr, **hard cap 3.5 cr (STOP-3)**; ≥2 cr reserve.
  First falsifying chain ≤ 2.5 cr. Watchdog with `--max-cost` on every job; CPU dry-run first.
* **Schedule:** figure frozen Sep 10; paper skeleton from Sep 5; writing Sep 10–16; "do not
  submit underpowered" stands.

## D. Frozen technical facts this amendment relies on (all verified, CPU, committed)

1. Mask-induced LoRA transfer is exact and deterministic at every ladder seam
   (`tests/research/test_lora_mask_transfer.py` 6/6; `tests/research/test_scenario_b_seam_transfer.py`
   S1–S6: 238/238 seams mapped, zero learned parameters; (1,2,3,3) drop rule).
2. The published pruned checkpoint is pure selection of dense (never finetuned), bit-exact
   reproducible (R5/P4) — backbone and sliced adapter share identical selections.
3. Kim et al.: recipe/data public, **weights not released**, published evaluation is
   single-prompt (noise-compatible) — hence the powered-replication design of Gate 0.
4. No direct novelty collision (closest: CA-LoRA, ASSP, LoRA-X, SVDQuant, X-Adapter — all
   contrasted in memo §4); differential fragility unmeasured in any domain.
5. AudioLDM third-party adapter ecosystem is n=1 today → the paper frames ecosystem motivation
   prospectively (MusicGen ~196 LoRAs; FLUX-lite/SSD-1B breakage as documented pain).

## E. Provenance rules (unchanged, restated)

Every number: commit, resolved config, checkpoint hash, manifest, seeds, raw output, GPU
metadata. Gate outcomes (PASS and FAIL) to `docs/experiment_ledger.md`; costs to
`docs/compute_budget.md`; claims only via `docs/claims_matrix.md`. The frozen prompt battery
and bootstrap seeds are committed BEFORE the first GPU job.

## F. What adoption means

On Gabriel's adoption: (1) this file's status flips to ADOPTED (DECISION-V4-09) with a ledger
entry; (2) the CPU wiring (scorer path, trainer data path, transfer op, dry-runs) proceeds;
(3) the smoke job is the only GPU authorized until its numbers re-anchor the chain costs.
