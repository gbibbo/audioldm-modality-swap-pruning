# DECISION-V4-00..07 — decision sheet (RESOLVED 2026-08-20)

**Status: RESOLVED 2026-08-20 14:13.** Gabriel adopted **all eight recommended defaults
verbatim** (Tier 0 only; balance ~3–5 cr spendable). Recorded as ledger
`DECISION-V4-00..07`; `docs/master_plan_v4.md` is now the contract, v3 superseded. This
sheet is retained for traceability; the "Decision (Gabriel)" column below reads "adopted
default" for every row.

Each row carries the **unanimous recommendation of the two-reviewer audit** (rounds 1–5,
`docs/review/`). Gabriel can (a) write "adopt the recommended defaults for V4-00..07" —
the agent then records one ledger entry per decision with the default text verbatim and
Gabriel's authorship; or (b) override any row by editing the "Decision" column before the
agent records it. The agent must never fill a row on its own.

| ID | Question | Recommended default (both reviewers) | Decision (Gabriel) |
|---|---|---|---|
| V4-00 | Adopt `docs/master_plan_v4_draft.md` rc4 as the plan; v3 becomes historical (kept, not deleted) | **Adopt rc4.** On recording: move the draft to `docs/master_plan_v4.md`, mark it "contract", mark v3 "superseded 2026-08-20", update `AGENTS.md` step 2 and `docs/claims_matrix.md` (§9 of the draft). | PENDING |
| V4-01 | Primary L1 baseline | **P0-standard (keep-highest-L1) is the primary scientific baseline; P0-published (Singh et al. artefact, keeps lowest-L1) is the reproducibility control.** Amends DECISION-M3B-002/003; wording "vs the published L1 artefact" stays mandatory for P0-published. | PENDING |
| V4-02 | Credit tier authorized now, and the balance figure the plan may assume | **Tier 0 only (≈2.7 cr, screening) with the current balance; the 2.0-cr reserve of CG-001 stands. Tier 1 (≈45 cr) and Tier 2 (≈25–35 cr) require explicit new funding + a new decision.** Gabriel must state the real balance; the agent reads job spend from the SDK (4.205 cr on 2026-08-20) but cannot read the balance. | PENDING — balance: ____ |
| V4-03 | ELSA (arXiv 2606.17404; GPT-5.2 + SAM Audio + Human-CLAP) | **Defer. Optional secondary metric at most; never a gate or primary metric.** Primary event metric = PANNs top-10 recall per requested event (Singh et al. 2026 compatible); secondary local = CLAP score of the event phrase. | PENDING |
| V4-04 | Event-set and null parameters | **`N_min = 200` AudioCaps-train clips, `n_min = 10` (Tier 0) / `20` (Tier 1) eligible prompts; mild budget `channel_mult=(1,2,3,4)` (measured −23.7 %); RAND×5 for Tier-0 forward diagnostics; `K_rand = 20` minimum for Gate E with the exact rank test.** | PENDING |
| V4-05 | Keep the v3 modality-swap result in the paper | **Yes, as one paragraph of negative result; D1 (signed asymmetry + per-stratum) decides "not supported" vs "not detectable at this budget".** | PENDING |
| V4-06 | Temporal-occupancy estimator for H-acoustic | **FineLAP (`AndreasXi/FineLAP`, MIT) conditional on a CPU validity smoke under `torch 1.13.1`; if the smoke fails, H-acoustic leaves the primary Gate M and the single-label-clip subset is sensitivity only.** | PENDING |
| V4-07 | Partition sizes, sentinel panel, minimum detectable effect for the power simulation, Gate I margins | **Proposed: calibration pool 256 natural + 256 tail-enriched; mechanism set 50 events × 20 prompts; intervention holdout 500 prompts, disjoint at source-wav level; sentinel panel 20 events × 15 prompts stratified by exposure × family; MDE for Gate E fixed from Tier-0 rates by the power simulation (≥ 80 % power); Gate I `δ_target = +5 pp`, `δ_harm = 2 pp` (non-inferiority), FAD/FD +5 % relative, KL +0.05.** May stay open until before the holdout is unblinded; while open, the holdout stays blinded. | PENDING |

## What the agent does when a decision is recorded

1. One ledger entry per decision (`DECISION-V4-0x`, date, literal text, "Decided by Gabriel").
2. Propagate: `PROGRESS.md` state block + log; `docs/claims_matrix.md`; `docs/HANDOFF.md`
   head block; `AGENTS.md` step 2 if V4-00 is adopted.
3. Only then start the CPU queue in `docs/NEXT_SESSION.md`. No GPU job before V4-00 and
   V4-02 are recorded, and none without Gabriel's explicit go in the conversation.
