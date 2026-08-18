# AGENTS.md

Canonical instructions for coding agents in this repository.

## Start here

1. Read the bounded state block at the top of `PROGRESS.md` through `<!-- FIN-ESTADO -->`.
2. Read `docs/master_plan_v3.md`. If it has not yet been moved there, read `master_plan_v3.md` at the repository root and move it into `docs/` during bootstrap without changing its content.
3. Confirm that `docs/experiment_ledger.md`, `docs/compute_budget.md`, `docs/claims_matrix.md`, and `docs/pilot_protocol.md` exist. If any is missing, create the corresponding project template before substantial work.
4. Inspect the actual Git repository, code, tests, checkpoints, manifests, and outputs before asserting that prior work or a milestone exists.

The master plan is the scientific execution contract. Do not silently simplify its baselines, gates, budgets, metrics, or decision branches. If Gabriel explicitly changes the plan, record the change in `docs/experiment_ledger.md` and update the relevant durable state.

## Project boundary

This is the independent AudioLDM research project on modality-swap-aware structured pruning and fixed parameter-efficient recovery. It is not an Edge Audio Labs workspace.

Do not create or use Clockify logs, meeting transcript workflows, meeting schedules, Edge Audio Labs hooks, Jira conventions, team-chat publication gates, or EAL brief/claim checkers in this repository.

## Scientific invariants

* No expensive recovery work before the low-cost modality-swap hypothesis gates pass.
* P1 text-only Taylor is mandatory for any cross-modal pruning claim. P1, P2, and P3 must respect the matched gradient-evaluation budget defined in the master plan.
* `D_mod` and `R_mod` are diagnostics, not pruning losses.
* Do not run M3 scientific experiments until the first GPU benchmark has populated `docs/compute_budget.md`, `docs/pilot_protocol.md` is frozen and committed, and Compute Gate CG has been explicitly resolved.
* LoRA is not the claimed novelty. When biases and GroupNorm affine parameters are trainable, call the mechanism parameter-efficient recovery and report LoRA, bias, GroupNorm, and total trainable parameters separately.
* Preserve the frozen upstream/reference SHAs and keep upstream AudioLDM patches minimal and reviewable.
* Negative results are valid outcomes. Follow the predeclared M3 to M5 decision branches rather than inventing a new hypothesis after seeing results.
* Every reported experimental number must be traceable to Git commit, resolved config, checkpoint/hash, dataset manifest, calibration slots/timesteps when applicable, random and generation seeds, raw output, and runtime/GPU metadata.

## Evidence and verification

Verify before asserting. For debugging, review, surprising results, reproducibility questions, or any conclusion that could change a scientific decision, use `/auditar`.

Documentation and `PROGRESS.md` are context, not proof. Re-run the relevant command or inspect the underlying artifact when a claim matters.

Before declaring a coding unit complete, run the narrowest tests that exercise the changed path. Before declaring a milestone complete, run the acceptance checks specified by the master plan.

## Progress and provenance discipline

`PROGRESS.md` is compact living state, not a transcript.

When meaningful progress, a decision, a failure, or a new blocker occurs:

1. update CURRENT STATE and OPEN ITEMS;
2. add or correct a reproducible RUN RECIPE when needed;
3. add one concise dated LOG entry;
4. record every experiment or scientific gate, including failed and stopped runs, in `docs/experiment_ledger.md`;
5. update `docs/compute_budget.md` whenever measured throughput, VRAM, GPU-hours, pricing, or cost projections change;
6. update `docs/claims_matrix.md` only when evidence changes the status of a paper claim;
7. freeze and commit `docs/pilot_protocol.md` before inspecting M3 saliency results.

If a required provenance file is missing, create it immediately from the minimal schema. Its absence never makes provenance optional.

## Git discipline

Use Git as the durable checkpoint mechanism.

After a coherent, verified unit of work is complete, use `/cerrar-hito` or perform its equivalent workflow: validate, update durable state, commit, and push the current branch when a remote is configured. Do not accumulate hours of validated work only in the working tree.

Rules:

* Commit coherent verified units, not every tiny edit.
* Push normal commits to the current branch without asking each time when a valid upstream exists.
* Do not force-push, rewrite shared history, merge branches, delete remote branches, or open/merge PRs unless Gabriel asks.
* Never commit datasets, generated audio, large checkpoints, credentials, `.env` files, API keys, or other secrets.
* Do not add AI attribution, generated-with trailers, or session links to commits or PR text.
* If tests are failing or evidence is incomplete, record that state instead of disguising it as a completed checkpoint.

## Repository architecture

Follow the master plan unless the codebase requires a documented exception:

```text
audioldm_train/
audioldm_peft/
research_pruning/
configs/research/
scripts/research/
tests/research/
docs/
    master_plan_v3.md
    pilot_protocol.md
    compute_budget.md
    experiment_ledger.md
    claims_matrix.md
artifacts/        # gitignored
_external/        # gitignored reference clones
```

## Communication

Start every response with the timestamp injected by the turn hook, on its own first line. Keep status concise and distinguish completed, verified work from planned work.
