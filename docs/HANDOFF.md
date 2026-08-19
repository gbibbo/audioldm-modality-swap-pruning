# HANDOFF — read this first

Source of truth for resuming work on this project. A new session should be able
to continue from this file alone, without any prior chat history.

**Written:** 2026-08-18, at the end of the bootstrap/M0 session.

---

> ## ⚠ READ FIRST — M3B-002 FINDING + DECISION (2026-08-19)
>
> The published PruningAudioLDM **L1 checkpoint keeps the LOWEST-magnitude conv filters**
> per pruned layer — inverted from standard L1 magnitude pruning. Verified 4 ways
> (Spearman -1 vs P0 on all 28 layers; reference code `np.argsort` ascending + the
> bit-exact materializer keeps `[:k]`; 15/15 pruned layers keep the lower-L1 set) and
> independently re-derived (ledger AUDIT-NIGHT2). It is a property of **Arshdeep's official
> artifact itself**.
> **DECIDED (Gabriel, DECISION-M3B-002):** since RQ2's L1 baseline IS Arshdeep's published
> pruning artifact (Zenodo 21376822; reference README "Official implementation"), the project
> **adopts the published inverted convention — P0 keeps LOWEST-L1**
> (`research_pruning.taylor.p0_importance(convention="published")`, default
> `P0_CONVENTION="published"`; `"standard"` kept only for a non-Arshdeep baseline). Verified to
> reproduce the published kept-set **exactly 12/12 ranking-driven layers**
> (`scripts/research/verify_p0_convention.py`; control test C8). RQ3's inverted-starting-point
> caveat stands. Finding write-up + reproduction:
> `docs/m0_baseline_reproduction/l1_pruning_direction_finding.md` /
> `scripts/research/verify_l1_direction.py`. Open for Arshdeep only: confirm intentionality
> for paper wording (non-gating).
>
> ## ⚠ GATE B AMENDED — 2026-08-19 (DECISION-M3B-003 / M3B-003)
>
> The master plan's Gate B was **infeasible as written** (audit finding G1): at the
> `(1,2,3,1)` budget each ranking-driven layer prunes 768 of 960 channels, so prune-set
> overlap is confined to `[0.75, 1.0]` with chance at `0.80` — its `<= 0.70` condition was
> mathematically impossible, and the draft protocol carried two contradictory overlap
> definitions (finding G2). **DECIDED (Gabriel, option (a)):** the single definition is the
> **KEPT set**, and the plan's numerals transfer verbatim — **Gate B PASS = weighted
> kept-set overlap `<= 0.80` AND `>= 2` ranking-driven layers `<= 0.70`** (chance `0.20`,
> full `[0,1]` range). Prune-set overlap is reported only, never the gate. Geometry
> re-derived from the real artifact: `scripts/research/verify_gate_b_geometry.py` (12
> ranking-driven layers, all N=960/k=192/p=768, floor 0.75 on every one).
> **Also decided:** `p0_importance('standard')` is reported as a **secondary reference**
> beside the primary published/inverted P0, because P1/P2/P3 keep the highest-saliency
> channels while P0 keeps the lowest-L1 ones; comparisons must be worded "vs the published
> L1 pruning artifact", never "vs standard L1".
> **Statistic implemented:** `research_pruning/paired_modality/overlap.py`
> (`evaluate_gate_b`), control-tested O1–O6 in `tests/research/test_overlap_gate_b.py`.
> Suite now **12/12 modules PASS**. Nothing evaluated on real saliencies.
> **Consequence: the pilot protocol's only remaining freeze prerequisite is the GPU
> benchmark (`T_sal`/`T_fwd`) plus Compute Gate CG.** Every decision blocker is cleared.
>
> ## STATUS UPDATE — 2026-08-19 autonomous night run (supersedes stale sections below)
>
> Sections §5 (M1), §6 (M2), §12 below were written 2026-08-18 and are **stale**.
> Trust the bounded state block at the top of `PROGRESS.md` and
> `docs/experiment_ledger.md`. Prior context: M2 COMPLETE; M3A machinery COMPLETE
> (M3-000/001/002); M3B Gate B "infeasible as written" (ledger AUDIT-M3-002, finding
> G1 — RESOLVED 2026-08-19 by DECISION-M3B-003, see the block above); M1 scaffold recovered+audited (`docs/m1_scaffold_audit.md`).
>
> **This night run (Gabriel authorised autonomous work ~00:30→08:30 Montevideo, across
> the 05:28 token-window renewal; a detached daemon `artifacts/auto_resume/` relaunches
> headless `claude` at 05:30). ALL FOUR CPU QUEUE ITEMS DONE + one extra, all pushed:**
>
> | Unit | What | Evidence |
> |---|---|---|
> | **M1-005/006/007** | M1 PEFT: adopt scaffold, fix all 8 audit defects F1–F8; real-U-Net tests; integration hooks. **M1 CPU acceptance COMPLETE.** | `audioldm_peft/`, tests L1-3/J1-4/S1-3/R6a-c/I1-4 (17/17) |
> | **M0-005** | GPU benchmark `scripts/research/gpu_benchmark.py` (write-only, refuses w/o CUDA) | compute_budget stays TBD_MEASURED |
> | **M0-006 (+follow-up)** | FAD/KL + PANNs top-k eval pipelines end-to-end; 6 library findings; IS=6.23, KL=-1(needs pairing), Frechet-2048(real-part fix)=15.36, FAD=NaN | `docs/m0_baseline_reproduction/eval_pipeline_closure.md` |
> | **M3B-000** | P0-P3 channel-gate Taylor saliency machinery + control tests | `research_pruning/taylor,paired_modality/`, C1-C7 |
> | **M3B-001** | 28 L1 prunable conv layers verified on real base U-Net; gates attach bit-identically | `test_prunable_layer_set.py` V1/V2 |
>
> **Full research suite: 10/10 modules PASS.** `git diff upstream-frozen -- audioldm_train/`
> stays empty. Run everything: `.venv/bin/python scripts/research/run_research_tests.py --all`.
>
> **WHAT REMAINS — all BLOCKED (do NOT do autonomously):**
> * **M1 GPU acceptance / GPU benchmark run / M3 pilot / M4 gen / M5 recovery** — need a GPU
>   (none attached). Run `gpu_benchmark.py` first, populate `compute_budget.md`, resolve CG.
> * **Real P0-P3 saliency on the base model** — this IS the M3B/M4 scientific run: blocked
>   until `pilot_protocol.md` is frozen (needs GPU T_sal/T_fwd; the Gate-B amendment is no
>   longer pending — DECISION-M3B-003). P1 is scientifically load-bearing → **must pass `/auditar`
>   before any real use.** Machinery + layer set are ready; only the slot construction,
>   real audio/text loss closures, and the run itself remain.
> * **pilot_protocol freeze** — now needs ONLY the GPU benchmark numbers (+ Compute Gate CG);
>   the Gate-B decision was made and implemented on 2026-08-19 (DECISION-M3B-003).
>
> **If you are the 05:30 resume:** the CPU-implementable queue is EXHAUSTED. Do NOT start any
> blocked item above. Do a genuinely useful, safe pass instead: re-run the full suite, verify
> the upstream invariant, and run an evidence-first `/auditar` of the night's NEW machinery
> (M1 PEFT correctness, esp. the load-bearing P0-P3/P1 Taylor saliency; the eval findings).
> Fix any real bug you find, commit it, then write the final status and create
> `artifacts/auto_resume/DONE` to stop the daemon. Never fabricate GPU numbers or run saliency
> on the real L1 ckpt.

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
