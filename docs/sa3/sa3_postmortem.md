# SA3 Postmortem — modality-swap-aware adapter-compatible pruning (stable-audio-3)

**Status: CLOSED as an active research line (Gabriel, 2026-08-26).** No F2, no ecological adapters, no
RQ3/CASE-C. Remaining 1.3092 cr is **not** authorized for SA3. This file is the durable postmortem;
raw numbers live in the ledger (`docs/experiment_ledger.md`) and the tracked JSON artifacts named below.

## One-paragraph summary

The intended contribution was to **preserve, under structured pruning, adapters the pruner does not yet
know about** — using a synthetic tangent proxy `A_tan` to predict which blocks are "adaptation-friendly"
better than the pruning-damage ranking `D_P`. After substantial methodological work we could never
establish the most elementary prerequisite: **a dense adapter with a functional utility large enough to
be worth preserving.** The evidence chain terminated at that prerequisite, so the line is closed.

## The three questions and where each terminated

- **SA3-RQ1 — distinct adaptation resource (`I_PT`).** FAILED/closed. `I_PT` is redundant with `D_P`
  (`artifacts/sa3/rq1_reanalysis.json`, `IPT_collapses_onto_DP=False` framing); the base→post
  post-training did **not** produce the strong structural differentiation the hypothesis needed.
- **SA3-RQ2 — localisable AND useful single-block adapters; `A_tan` beats `D_P`.** FAILED/closed. The
  positive controls L_6/L_13 passed the *field* conditions (precision guard, `A_eco(host)≈1`,
  host-removal algebraic collapse, an external-removal uplift) but **FAILED the task-level ΔT_AA gate**
  (`configs/sa3/adapters/control_verdict.json`). This is a **field/function gap**: a measurable,
  algebraically-localisable internal perturbation produced no measurable dense task utility. The
  pre-registered `A_tan`-vs-`D_P` prediction check was therefore never reached.
- **SA3-RQ2b — can SA3 learn *any* dense functionally-useful adapter?** FAILED/closed at the
  preregistered F1 STOP. A **function-first** sentinel (full-backbone LoRA r16/α16, all 20 blocks,
  `max|lora_B|=1.197`, 96 mechanical train / 1000 steps) scored with the frozen paired CLAP audio-audio
  `T_AA` on 64 held-out eval gave ΔT_AA(base)=+0.0118 CI[−0.020,+0.043] and ΔT_AA(post)=+0.0230
  CI[−0.011,+0.058]. Both point effects are ≈6.3× (base) / ≈3.3× (post) **smaller** than SESOI 0.075 and
  — crucially — **both upper CI95% bounds fall below SESOI**, so the data are *incompatible* with a
  SESOI-magnitude effect on either checkpoint (stronger than a mere failure to reject 0). Symmetric gate
  → `STOP_RQ2B_BASE_FAIL` (`configs/sa3/adapters/f1_verdict.json`).

## Exactly where the evidence chain terminated

At **RQ2b/F1**: the intended chain was `learn a useful dense adapter → localise it (A_eco) → show A_tan
predicts A_eco better than D_P → preserve such adapters under pruning`. It stopped at the **first link**
— we never demonstrated a useful dense adapter under the frozen recipe. Everything downstream
(localisation, A_tan-vs-D_P, ecological adapters, pruning-preservation) is consequently **NOT RUN**.

## The single most important allowed claim

> Under the frozen standard recipe (full-backbone LoRA r16/α16, 1000 steps, mechanical 96 train / 64
> held-out eval), the adapter failed to produce a functionally relevant paired `T_AA` uplift on either
> dense base or dense post.

**We may NOT claim** "LoRA does not work on SA3" or "adapter-compatible pruning is false." A different
task, more data, a different training recipe, or a different functional metric might yield useful
adapters — but exploring that would be **a new experimental project**, not a clean continuation of this
one.

## What survived (reusable methodological results)

1. **base→post post-training did not yield the strong structural differentiation we needed;** `I_PT`
   collapsed onto `D_P`.
2. **`A_tan` vs `D_P` showed a stable structural ranking difference** (δ=2 at k=6, robust N=16→N=32) —
   ESTABLISHED as structure only; **never linked to real functional adapters**, so not promotable.
3. **Field/function gap:** single-block LoRAs produced clearly measurable + algebraically localisable
   internal alterations with no functional utility, and the full-backbone F1 sentinel showed the problem
   is **not** solved by simply widening the support. This is the core cautionary result.

These remove the two ambiguities that clouded L6/L13: it is **no longer** attributable specifically to
single-block support, nor simply to the old n=8 power.

## Reusable machinery preserved (marked NOT RUN, not TODO, for future lines)

`train_control_loras.py` (`--backbone`/single-block, attach report+assert), `stage_trainL.py`,
`f1_task_gen.py`, `f1_verdict.py` + `aeco_predict.f1_functional_verdict`, `f1_run.py`, `f1_accept.py`,
`score_taa.py` (paired CLAP audio-audio), the decision core `research_sa3/aeco_predict.py`, and the
frozen Freesound sourcing/manifest tooling.

## Budget (final)

SA3_GPU 3.6382 + SA3_CPU 0.0526 = **SA3_TOTAL 3.6908 cr** of the cap-5 envelope; **headroom 1.3092 cr,
frozen and NOT authorized for SA3** (`docs/sa3/budget_reconciliation.md`, 14 jobs).

## Provenance index

- F1 run: job `sa3-f1-1` (T4, Completed, 0.3223 cr), code commit `ede6f99`; result committed at the
  closure commit. Adapter `F1_full.safetensors` sha256 `b6ad50c08f5ec9872050d48ddcab4e4f3a80a83c8e243e0703b6e85a4e55b4ad` (gitignored; provenance in `data/sa3/adapters/f1_run_summary.json`, gitignored).
- Frozen data: `configs/sa3/adapters/mechanical.manifest.json` (sha `c3ef1fc8…`, 96 train / 64 eval).
- Per-item deltas + seeds: `configs/sa3/adapters/f1_taa_scores.json` (bootstrap seed 20260824, B=10000;
  gen seeds 20260824+idx); verdict `configs/sa3/adapters/f1_verdict.json`.
- RQ2 controls: `configs/sa3/adapters/control_verdict.json`, `control_taa_scores.json`.
- Protocol: `docs/sa3/rq2b_function_first_proposal.md` (rev3.1 frozen); prior `rq2_validation_protocol.md`.

## Future-revival principle (the key lesson)

Do **not** restart by inventing a LoRA and then asking whether it is useful. Any revival of the
high-level "prune now, preserve unseen adapters later" idea must **begin from an independently-established
adapter or benchmark with a demonstrated dense functional uplift** — established *before* and
*independently of* the pruning hypothesis. Only then does it make sense to study how to compress the
backbone without breaking such adapters. Constructing a positive control inside SA3 to manufacture that
prerequisite is exactly the trap that ended this line.
