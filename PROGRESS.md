# AudioLDM Modality-Swap Pruning - Progress

> Living state document. Keep this compact. The SessionStart hook injects only the bounded state block below. Detailed experimental provenance belongs in `docs/experiment_ledger.md` and milestone artifacts.

## CURRENT STATE

* Repository: `/teamspace/studios/this_studio/audioldm-modality-swap-pruning`, branch `main`, remote `origin` = `gbibbo/audioldm-modality-swap-pruning` (public).
* No prior research repository was recoverable anywhere (`gh` or filesystem). This history was started fresh. The M0/M1 work the master plan describes as done is local-only on the author's Windows machine and unpushed.
* **Frozen references imported and verified in-repo.** `upstream-frozen` = `702a638d…` (full 35-commit upstream history merged into `main`); `pruning-reference-frozen` = `6f65f628…` (kept as reference branch, plus a working clone in `_external/`). `git diff upstream-frozen -- audioldm_train/` is **empty**.
* **`(1,2,3,1)` verified from artifacts, not from documents.** Base `channel_mult=[1,2,3,5]`, pruned `l1_audioldm-m-full_p1.ckpt` `channel_mult=[1,2,3,1]`, `model_channels=192`. U-Net 415.955 M -> 145.674 M params (-65.0%).
* **Full-FT checkpoint gate RESOLVED (deadline was today).** All 2061 same-shape tensors of the published pruned checkpoint are bit-identical to the base: it is pure prune-and-merge, **never finetuned**. The L1 `(1,2,3,1)` pre-recovery checkpoint is public and fetched; the **recovered full-FT `(1,2,3,1)` checkpoint is proven NOT public**. RQ3 is downgraded to a published-reference comparison until Arshdeep supplies it.
* Public artifacts fetched and md5-verified into gitignored `data/`: `audioldm-m-full.ckpt`, `Unet_model-m.ckpt`, `l1_audioldm-m-full_p1.ckpt`, `sorted_indexes_dict.pkl`. `checkpoints.tar` and `dataset.tar` (preprocessed AudioCaps) were still downloading at commit time; re-run the fetch script to confirm.
* **No environment exists.** The active `cloudspace` env is Python 3.12.11 and lacks every AudioLDM dependency except torch and pytorch-lightning; upstream requires Python 3.10 and pins `transformers==4.30.2`. Nothing has been installed. This decision blocks all model code.
* Structure created per master plan §13. `audioldm_peft/`, `research_pruning/{diagnostics,taylor,paired_modality}/`, `tests/research/` are **skeletons with no implementation**; no M1 code was written or reconstructed.
* No GPU is attached. `docs/compute_budget.md` is entirely unmeasured, Compute Gate CG is unresolved, and M3 stays blocked.

## OPEN ITEMS

1. **Request the recovered full-FT `(1,2,3,1)` checkpoint from Arshdeep today** — the public search is finished and the artifact is proven absent. Also ask him to confirm the pre-recovery reading of `l1_audioldm-m-full_p1.ckpt`.
2. Decide the environment strategy (dedicated Python 3.10 `audioldm_train` conda env per upstream, vs. relaxed pins on 3.12) and create it. Blocks everything below.
3. Recover the local-only M1 LoRA/PEFT CPU scaffold from the Windows machine, diff it against this repository, and re-run its tests here before touching it.
4. Finish the fetch of `checkpoints.tar` / `dataset.tar` and extract into `data/checkpoints/` and `data/dataset/`; then run upstream `tests/validate_dataset_checkpoint.py`.
5. Verify the FAD/KL pipeline (`audioldm_eval`) and reproduce the PANNs top-k semantic pipeline.
6. Implement M2 audio/text conditioning instrumentation; prepare the single reproducible GPU benchmark recording all §7.2 variables.
7. Resolve Compute Gate CG before M3.

## RUN RECIPES

* Fetch + md5-verify public artifacts: `bash scripts/research/fetch_public_artifacts.sh`
* Verify structural budget from checkpoints (CPU): `python3 scripts/research/verify_pruned_architecture.py data/checkpoints/audioldm-m-full.ckpt data/checkpoints/l1_audioldm-m-full_p1.ckpt`
* Review our patches to upstream: `git diff upstream-frozen -- audioldm_train/`
* Agent kit verification: `python3 .claude/verify_agent_kit.py .`
* Progress structure: `python3 .claude/hooks/check_progress.py PROGRESS.md`
* Git state: `git status --short --branch`
* Add build/train/benchmark commands only after they are verified in this repository.

<!-- FIN-ESTADO -->

## LOG

### 2026-08-18 | Bootstrap of the dedicated research repository

* Searched for the prior research repository before creating anything: `gh repo list` / `gh api /user/repos` (owner+collaborator+org) and a filesystem-wide search returned no AudioLDM, modality-swap, or pruning repository. Repository therefore initialized from scratch, not recovered.
* Installed `audioldm-agent-kit-minimal-v2` against this repository root.
* Verifications passed: `check_progress.py` OK, `verify_agent_kit.py` OK, `settings.json` valid JSON, `bash -n` on all three hooks OK, all three hooks execute cleanly, kit `tests/test_install.sh` self-test OK.
* Contamination audit: repository clean. External scopes clean except for account-level claude.ai MCP connectors (Atlassian/Jira, Gmail, Google Calendar, Google Drive, Slack) reachable from any session in this environment; these are outside the repository and cannot be removed by the kit.
* No scientific milestone is marked complete by this bootstrap.

### 2026-08-18 | M0: upstream import, frozen references, public-artifact inventory

* Imported the full upstream AudioLDM history and pinned both frozen reference SHAs as branches; `audioldm_train/` is byte-identical to `upstream-frozen`.
* Created the master-plan §13 directory structure as empty skeletons only.
* Fetched and md5-verified the public checkpoints; wrote `scripts/research/fetch_public_artifacts.sh` and `scripts/research/verify_pruned_architecture.py`.
* Derived `(1,2,3,1)` directly from checkpoint tensor shapes and proved the published pruned checkpoint was never finetuned, which resolved the full-FT strength gate against public availability.
* Recorded everything in `docs/experiment_ledger.md` (M0-001) and `docs/m0_baseline_reproduction/`; updated RQ3 and EFF wording in `docs/claims_matrix.md`.
* M0 remains open: no environment, no FAD/KL pipeline, no PANNs pipeline, no GPU smoke test. No M1 code written.
