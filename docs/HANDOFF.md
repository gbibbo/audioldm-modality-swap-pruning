# HANDOFF — read this first

Source of truth for resuming work on this project. A new session should be able
to continue from this file alone, without any prior chat history.

**Written:** 2026-08-18, at the end of the bootstrap/M0 session.

---

> ## STATUS UPDATE — 2026-08-19 (supersedes stale sections below)
>
> This banner reflects the current repository state; sections §5 (M1), §6 (M2)
> and §12 below were written on 2026-08-18 and are **partly stale**. Trust the
> bounded state block at the top of `PROGRESS.md` and `docs/experiment_ledger.md`
> for live status. Concretely, since this file was first written:
>
> * **M2 is COMPLETE (PASS)** — conditioning-path validation, CPU. See PROGRESS.
> * **M3A machinery is COMPLETE** (M3-000/001/002), CPU, no scientific result on
>   the real L1 checkpoint. M3B (Gate B) is "infeasible as written" (see ledger).
> * **M1 is RECOVERED, AUDITED, adoption PENDING — no longer blocked.** Gabriel
>   supplied the local scaffold; it is audited in `docs/m1_scaffold_audit.md`
>   (verdict: "adopt and fix, do not rebuild") with defect list F1..F8. The
>   pristine overlay is at `_external/m1_scaffold_recovered/` (gitignored).
>   `audioldm_peft/` in-repo is still only a skeleton `__init__.py`; adoption is
>   the current work.
>
> **Autonomous night run (2026-08-19, ~00:30→08:30 Montevideo).** Gabriel asked
> for autonomous completion across the 05:28 token-window renewal. A detached
> resume daemon (`artifacts/auto_resume/launcher.sh` + `resume_prompt.txt`,
> gitignored) relaunches headless `claude` at 05:30 Montevideo. **Priority queue:**
> * **(A) M1 adopt scaffold + fix F1..F8 to CPU acceptance — DONE** (M1-005/006/007,
>   commits 23ac4f2/200fbcc/317aea3). `audioldm_peft/` is a working, tested package;
>   full CPU suite 17/17 across 5 modules; all 8 audit defects addressed. Only M1
>   **GPU** acceptance remains (blocked on GPU + Compute Gate CG).
> * **(B) M0 remainder — IN PROGRESS/NEXT:** FAD/KL `audioldm_eval` end-to-end on a
>   tiny real folder pair; PANNs top-k semantics (section 5 of
>   `_external/PruningAudioLDM/README.md`).
> * **(C) prepare (do NOT run) the GPU benchmark; review `docs/pilot_protocol.md`.**
> Guardrails unchanged: no GPU numbers invented, no M3 scientific run, no saliency
> on the real L1 ckpt, `audioldm_train/` stays byte-identical to upstream-frozen.

---

## 1. Repository

```text
path      /teamspace/studios/this_studio/audioldm-modality-swap-pruning
branch    main
HEAD      6d321cbe91daacc1f0fbab18cda366fa657c79dc
status    clean, main == origin/main
```

Remotes:

| Name | URL |
|---|---|
| `origin` | https://github.com/gbibbo/audioldm-modality-swap-pruning (public) |
| `upstream` | https://github.com/haoheliu/AudioLDM-training-finetuning |
| `pruning-reference` | https://github.com/Arshdeep-Singh-Boparai/PruningAudioLDM |

Branches (identical locally and on `origin`):

| Branch | Commit |
|---|---|
| `main` | `6d321cbe91daacc1f0fbab18cda366fa657c79dc` |
| `upstream-frozen` | `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc` |
| `pruning-reference-frozen` | `6f65f628fabc4ad27770753698fc81944e820f9f` |

The scientific contract is `docs/master_plan_v3.md`. Agent rules are `AGENTS.md`.
Live state is `PROGRESS.md`. This file is the resume point.

## 2. Frozen references

| Reference | Commit | Date | Preserved as |
|---|---|---|---|
| `haoheliu/AudioLDM-training-finetuning` (MIT) | `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc` | 2024-12-13 | branch `upstream-frozen`; full 35-commit history merged into `main` |
| `Arshdeep-Singh-Boparai/PruningAudioLDM` (MIT) | `6f65f628fabc4ad27770753698fc81944e820f9f` | 2026-07-16 | branch `pruning-reference-frozen`; read-only clone in `_external/` (gitignored) |

```bash
git diff upstream-frozen -- audioldm_train/     # currently EMPTY — no upstream patch yet
```

Files on `main` that differ from `upstream-frozen`: only `.gitignore` (union of
the upstream block, kept verbatim, and the agent-kit block) and files we added
(`AGENTS.md`, `CLAUDE.md`, `PROGRESS.md`, `README.md`, `UPSTREAM_README.md`).
`pyproject.toml` and `poetry.lock` are byte-identical to `upstream-frozen`.

## 3. M0 — what is VERIFIED, and by what evidence

| Item | Status | Evidence |
|---|---|---|
| Frozen SHAs pinned and reachable | verified | `docs/m0_baseline_reproduction/frozen_references.md` |
| Public artifact inventory (3 Zenodo records) | verified | `docs/m0_baseline_reproduction/public_artifact_inventory.md` |
| All artifacts downloaded + md5-verified | verified | `artifacts/m0_baseline_reproduction/fetch.log` (`fetch rc=0`) |
| `(1,2,3,1)` architecture identified | verified from weights | `artifacts/m0_baseline_reproduction/architecture_check.log` |
| Published pruned ckpt is pre-recovery | proven | `artifacts/m0_baseline_reproduction/prerecovery_check.log` |
| L1 saliency manifest structure | verified | `artifacts/m0_baseline_reproduction/sorted_indexes_inspect.log` |
| Python 3.10 environment | built, 155 pkgs, 0 errors | `docs/environment_report.md`, `artifacts/m0_baseline_reproduction/poetry_install.log` |
| Import smoke tests 6/6 | pass | `docs/environment_report.md` |
| Both architectures rebuilt + `strict=True` load | pass | `artifacts/m0_baseline_reproduction/smoke_load_unet.log` |
| AudioCaps fetched, extracted, validated | pass | `artifacts/m0_baseline_reproduction/validate_dataset_checkpoint.log` |
| Dataset load smoke test (train + test) | pass | `artifacts/m0_baseline_reproduction/smoke_load_dataset.log` |

Key measured numbers (do not re-derive, do not contradict without re-running):

```text
base   channel_mult [1,2,3,5]  model_channels 192  U-Net 415.955 M params
pruned channel_mult [1,2,3,1]  model_channels 192  U-Net 145.674 M params  (-65.0%)
strict load: 690 tensors each, 0 missing, 0 unexpected
AudioCaps: 49 502 train + 964 test, 50 961 wav, 31 GB
```

## 4. M0 — what is still MISSING to close it

M0 is **NOT closed**. Remaining:

1. Run the FAD/KL pipeline end to end. `audioldm_eval 0.0.5` imports cleanly but
   has never been executed.
2. Reproduce the PANNs top-k semantic pipeline (section 5 of the
   PruningAudioLDM README, `_external/PruningAudioLDM/`).
3. Define a validation split disjoint from the test set (see finding 9.1).
4. Generation smoke test — **the only remaining M0 item that requires a GPU**.
5. Recovered full-FT `(1,2,3,1)` checkpoint — see section 9. Does not block M0.

## 5. M1 — status

**M1 does NOT exist in this repository.**

`audioldm_peft/`, `research_pruning/{diagnostics,taylor,paired_modality}/` and
`tests/research/` are **empty skeletons**. Each `__init__.py` says so explicitly.
There is no LoRA implementation, no injector, no merge/unmerge, no adapter
lifecycle, and no research test.

The master plan states that a LoRA/PEFT CPU scaffold with passing tests existed
as of 2026-08-17. That work is **local-only on Gabriel's Windows machine and was
never pushed**. It was searched for and not found: no matching repository exists
under any `gh`-accessible account, and none exists anywhere on this filesystem.

**Do not treat M1 as recovered, partially done, or reconstructible from memory.**
Gabriel is searching his Windows machine. Until he either delivers the scaffold
or explicitly authorises a rebuild, M1 is blocked. See section 10.B.

What to look for on the Windows machine, in priority order: the old repo's
`.git` directory (compare whole histories, do not copy loose files);
`audioldm_peft/`; `test_lora*.py`; older `PROGRESS.md` / `master_plan_v1|v2.md`.

## 6. M2 — status

**Not started.** No audio/text conditioning instrumentation exists. The
diagnostics `D_mod` and `R_mod` are defined in `docs/master_plan_v3.md` §3 and
are **diagnostics, never pruning losses**.

Relevant upstream entry points, unmodified and importable:
`audioldm_train.conditional_models`,
`audioldm_train.modules.latent_diffusion.ddpm`,
`audioldm_train.modules.diffusionmodules.openaimodel`.

## 7. Gates blocking M3

M3 must not run until **all** of these hold:

1. First real GPU benchmark executed and `docs/compute_budget.md` populated with
   measured values (it is currently 100% `TBD_MEASURED`).
2. **Compute Gate CG explicitly resolved** — currently unresolved.
3. `docs/pilot_protocol.md` completed, reviewed against the master plan, frozen
   and committed **before any saliency result is inspected** — currently UNFROZEN.
4. A validation split disjoint from test defined (finding 9.1).

## 8. Environment, data, and how to re-run

### Environment

```bash
cd /teamspace/studios/this_studio/audioldm-modality-swap-pruning
.venv/bin/python -V        # Python 3.10.20
```

There is no `activate` step to remember: call `.venv/bin/python` directly.
`.venv/` is gitignored. It is a `uv`-provisioned standalone CPython 3.10.20 with
the frozen `poetry.lock` installed — 155 packages, 0 errors, no pin relaxed:
`torch 1.13.1+cu117`, `torchvision 0.14.1`, `torchaudio 0.13.1`,
`transformers 4.30.2`, `pytorch-lightning 2.1.1`, `librosa 0.9.2`,
`numpy 1.23.5`, `taming-transformers-rom1504 0.0.6`, `audioldm_eval 0.0.5`,
`hear21passt 0.0.23`.

Rebuild instructions and the full package freeze: `docs/environment_report.md`.

**Platform traps, already solved — do not rediscover them:**
* Lightning **refuses `conda create` in every form**, including `-p` prefix envs.
* `/commands/python3.10` is a shim that actually executes Python 3.12.11.
* `torch.load(..., mmap=True)` does not exist in torch 1.13.1; our helper scripts
  carry a `torch_load()` fallback.
* Importing `...latent_diffusion.ddpm` downloads a tokenizer from HuggingFace at
  import time, so imports are not offline-safe.

### Data locations (all gitignored)

```text
data/checkpoints/audioldm-m-full.ckpt          base AudioLDM-M-Full
data/checkpoints/l1_audioldm-m-full_p1.ckpt    L1 (1,2,3,1) PRE-RECOVERY
data/checkpoints/Unet_model-m.ckpt             pretrained U-Net
data/checkpoints/{clap_*,audiomae_*,vae_mel_*,hifigan_*}   aux checkpoints
data/dataset/audioset/zip_audios/…             AudioCaps audio, 50 961 wav
data/dataset/metadata/…                        AudioCaps metadata
artifacts/m0_baseline_reproduction/            all raw logs + sorted_indexes_dict.pkl
_external/PruningAudioLDM/                     reference clone at the frozen SHA
```

Sizes and every md5: `docs/m0_baseline_reproduction/dataset_manifest.md`.

### Minimal re-run commands

```bash
cd /teamspace/studios/this_studio/audioldm-modality-swap-pruning

# upstream dataset/checkpoint validation
.venv/bin/python tests/validate_dataset_checkpoint.py

# rebuild both architectures and strict-load real weights
.venv/bin/python scripts/research/smoke_load_unet.py

# read samples from the dataset
.venv/bin/python scripts/research/smoke_load_dataset.py --split train --n 2

# derive the structural budget from checkpoint tensors
.venv/bin/python scripts/research/verify_pruned_architecture.py \
    data/checkpoints/audioldm-m-full.ckpt \
    data/checkpoints/l1_audioldm-m-full_p1.ckpt

# re-verify/complete public artifacts (idempotent, md5-checked, resumable)
bash scripts/research/fetch_public_artifacts.sh

# agent kit
python3 .claude/verify_agent_kit.py .
python3 .claude/hooks/check_progress.py PROGRESS.md
```

## 9. Important findings

**9.1 — `val` and `test` are the same 964 items.** `data/dataset/metadata/dataset_root.json`
maps both the `val` and `test` splits to
`audiocaps_test_nonrepeat_subset_0.json` (964 items). **Do not use `val` as an
independent validation set while this holds.** Any tuning, early stopping, or
model selection against it contaminates the evaluation set. A disjoint split must
be defined and recorded in `docs/pilot_protocol.md` before M3.

**9.2 — the published pruned checkpoint was never finetuned.** All 2061
same-shape tensors of `l1_audioldm-m-full_p1.ckpt` are bit-identical to
`audioldm-m-full.ckpt`. It is pure prune-and-merge output, i.e. the
**pre-recovery** checkpoint. This is proven, not inferred.

**9.3 — recovered full-FT `(1,2,3,1)` checkpoint is NOT publicly available.**
The public search is complete: no GitHub releases on either repository, and
Zenodo 10.5281/zenodo.21376822 contains only pre-recovery artifacts. Gabriel is
handling the request to Arshdeep directly. **This does not block M0.** Until it
arrives, RQ3 is downgraded to a published-reference comparison and **no exact
percentage-of-full-FT recovery may be claimed** from cross-pipeline numbers.

**9.4 — the L1 baseline ranks only 28 conv layers.** `sorted_indexes_dict.pkl`
covers `input_blocks.7..11` (9), `middle_block` (4), `output_blocks.0..6` (15) at
widths 384/576/960 — not the whole U-Net. For RQ2 to be structure-matched,
P1/P2/P3 must be computed over exactly this layer set with the same per-layer
channel counts.

**9.5 — provenance cross-check passed.** `audioldm-m-full.ckpt` has the same md5
in the official AudioLDM record (7884686) and in Arshdeep's record (21376822),
so the pruning work builds on the official weights.

**9.6 — checkpoint licensing.** Pretrained AudioLDM checkpoints are
**CC-BY-NC-4.0 (no commercial use)** per the upstream README. They are not
redistributed in this repository.

## 10. Provenance document status

| Document | Status |
|---|---|
| `docs/master_plan_v3.md` | the contract, unmodified |
| `docs/experiment_ledger.md` | entries BOOTSTRAP-000, M0-001..M0-004, plus this handoff |
| `docs/compute_budget.md` | **100% `TBD_MEASURED`** — no GPU has ever run; do not populate with estimates |
| `docs/claims_matrix.md` | all claims `unresolved`; RQ3 and EFF carry M0-derived wording constraints |
| `docs/pilot_protocol.md` | **UNFROZEN**; carries the 28-layer and val/test constraints; must be completed, frozen and committed before M3 |
| `docs/environment_report.md` | canonical environment record |
| `docs/m0_baseline_reproduction/` | frozen references, artifact inventory, dataset manifest |

## 11. Recorded deviations from the master plan

All already logged in `docs/experiment_ledger.md`; do not "fix" them silently.

1. Small text provenance documents live in `docs/m0_baseline_reproduction/` so
   they are version-controlled; `artifacts/m0_baseline_reproduction/` is
   gitignored and holds raw logs and binaries only. The plan names only the
   `artifacts/` path.
2. Upstream root `README.md` moved verbatim to `UPSTREAM_README.md`; a project
   `README.md` written in its place. Nothing under `audioldm_train/` was touched.
3. `PruningAudioLDM` preserved as branch `pruning-reference-frozen`, not only as
   a recorded SHA. Not merged into `main`.
4. Environment created with `uv` + `venv` instead of `conda create`, because the
   platform forbids the latter. Packaging deviation only: real unmodified
   CPython 3.10.20, every version from the frozen lock.
5. `POETRY_VIRTUALENVS_CREATE` passed as an env var rather than
   `poetry config --local`, so no `poetry.toml` enters the repository.
6. `torch_load()` compatibility helper added to our own scripts only. No
   upstream or scientific code modified.

## 12. Next work queue, prioritised

### A. Can be done now

1. **Close FAD/KL end to end.** Exercise `audioldm_eval` on a small real folder
   pair and record the exact invocation and outputs. Currently only proven to
   import.
2. **Reproduce PANNs top-k semantics.** Follow section 5 of
   `_external/PruningAudioLDM/README.md`; `hear21passt` is installed.
3. **Resolve a disjoint validation split.** Construct it from AudioCaps train or
   from the unused `audiocaps_test_nonrepeat_subset_{1..4}.json` files, record it
   in `docs/pilot_protocol.md`, and stop `val` from aliasing `test`.
4. **Prepare the reproducible GPU benchmark.** Write the single command/config
   that records every §7.2 variable. Prepare it now; **run it only when a GPU is
   attached**, then populate `docs/compute_budget.md` with measured values only.

Also available without a GPU: M2 conditioning instrumentation design and the
audio/text conditioning path inspection.

### B. Blocked — do not start

1. **M1 parameter-efficient recovery.** Blocked until Gabriel either delivers the
   recovered local scaffold or explicitly authorises a rebuild from scratch.
   Neither has happened.
2. **M3 pilot.** Blocked until the GPU benchmark is recorded, Compute Gate CG is
   explicitly resolved, `docs/pilot_protocol.md` is frozen and committed, and the
   validation split is disjoint.

Long GPU training and any M4/M5 work sit behind both of the above.

## 13. Do not assume

* **Do not assume M0 is closed.** It is not. See section 4.
* **Do not assume M1 exists.** It does not exist in this repository, in any form.
* **Do not reconstruct M1 without Gabriel's explicit authorisation.**
* **Do not use `val` as an independent validation set while it points at the same
  964-item file as `test`.**
* **Do not execute M3.**
* **Do not invent GPU numbers.** No GPU has ever run here. Every field in
  `docs/compute_budget.md` must stay `TBD_MEASURED` until it is measured.
* **Do not modify upstream code to hide an incompatibility.** Fix the
  environment, the paths, or our own scripts instead, and record the problem.
  `git diff upstream-frozen -- audioldm_train/` must stay empty until a patch is
  deliberately made and reviewed.
* **Do not use the account's MCP connectors** (Atlassian/Jira, Gmail, Google
  Calendar, Google Drive, Slack) unless Gabriel explicitly asks. They are
  account-level and reachable from any session, but are out of scope here.

Additionally, per `AGENTS.md`: verify before asserting, never present unvalidated
work as a completed checkpoint, and add no AI attribution to commits.
