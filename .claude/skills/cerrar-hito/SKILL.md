---
name: cerrar-hito
description: Close a coherent verified unit of project work. Use after meaningful implementation, a completed audit, an experiment/gate decision, or before switching milestones. Validates, updates progress/provenance, commits, and pushes safely.
---

# Cerrar hito

Do not manufacture completion. If validation fails, record the failure and keep the unit open.

## 1. Inspect the unit

Run `git status --short --branch` and inspect the relevant diff. Confirm that generated artifacts, datasets, audio, checkpoints, secrets, and unrelated files are not staged.

## 2. Validate

Run the narrowest tests or reproduction commands that exercise the changed path. For a master-plan milestone or gate, also run its stated acceptance checks. Record exact commands and outcomes.

## 3. Update durable state

Update `PROGRESS.md` with current truth, the next concrete action, newly validated run recipes, and one concise dated log entry.

If an experiment or scientific gate ran, `docs/experiment_ledger.md` is mandatory. If missing, create it before closing the unit. Record commit, resolved config, data/checkpoint manifests or hashes, calibration slots/timesteps as applicable, random and generation seeds, command, runtime/GPU, raw-output path, result, and failure status.

Update `docs/compute_budget.md` when benchmark, throughput, VRAM, GPU-hour, provider-price, or cost numbers change. Update `docs/claims_matrix.md` only when a corresponding claim branch is resolved. Before M3, verify that `docs/pilot_protocol.md` is frozen and committed.

## 4. Commit

Stage only the coherent unit. Commit with a concise message describing the outcome. Do not add AI attribution or generated-with trailers.

Never commit large experiment artifacts or secrets. If `.gitignore` is insufficient, fix it before staging.

## 5. Push

If a normal remote/upstream exists, push the current branch. If no upstream exists but `origin` is clearly the intended project remote, set it with a normal `git push -u origin <current-branch>`.

Do not force-push, merge, rewrite history, delete branches, or open/merge a PR unless Gabriel explicitly asks.

## 6. Report

Return validation result, durable docs updated, and next open item. State anything still unverified explicitly. End the report with the binding checkpoint block (consistent with `AGENTS.md` Git discipline / Communication and the Stop hook):

```text
COMMIT: <full SHA>
SHORT: <short SHA>
PUSH: pushed/synced/no-upstream
TREE: clean
```

If the tree is not clean or a push did not happen, say so explicitly instead of reporting `clean`/`pushed`.
