# AudioLDM Modality-Swap Pruning - Progress

> Living state document. Keep this compact. The SessionStart hook injects only the bounded state block below. Detailed experimental provenance belongs in `docs/experiment_ledger.md` and milestone artifacts.

## CURRENT STATE

* **Resume point: `docs/HANDOFF.md`.** A new session should read that file first; it is self-contained and does not depend on any chat history.
* Repository: `/teamspace/studios/this_studio/audioldm-modality-swap-pruning`, branch `main`, remote `origin` = `gbibbo/audioldm-modality-swap-pruning` (public).
* No prior research repository was recoverable anywhere (`gh` or filesystem). This history was started fresh. The M0/M1 work the master plan describes as done is local-only on the author's Windows machine and unpushed.
* **Frozen references imported and verified in-repo.** `upstream-frozen` = `702a638d…` (full 35-commit upstream history merged into `main`); `pruning-reference-frozen` = `6f65f628…` (kept as reference branch, plus a working clone in `_external/`). `git diff upstream-frozen -- audioldm_train/` is **empty**.
* **`(1,2,3,1)` verified from artifacts, not from documents.** Base `channel_mult=[1,2,3,5]`, pruned `l1_audioldm-m-full_p1.ckpt` `channel_mult=[1,2,3,1]`, `model_channels=192`. U-Net 415.955 M -> 145.674 M params (-65.0%).
* **Full-FT checkpoint gate RESOLVED (deadline was today).** All 2061 same-shape tensors of the published pruned checkpoint are bit-identical to the base: it is pure prune-and-merge, **never finetuned**. The L1 `(1,2,3,1)` pre-recovery checkpoint is public and fetched; the **recovered full-FT `(1,2,3,1)` checkpoint is proven NOT public**. RQ3 is downgraded to a published-reference comparison until Arshdeep supplies it.
* **All public artifacts fetched, md5-verified and extracted** (`fetch rc=0`): base/pruned/U-Net checkpoints, `sorted_indexes_dict.pkl`, `checkpoints.tar` (7 aux checkpoints: CLAP, AudioMAE, VAE, HiFi-GAN 16k/48k) and `dataset.tar` (AudioCaps, 31 GB, 50 961 wav). Manifest: `docs/m0_baseline_reproduction/dataset_manifest.md`.
* **Upstream `tests/validate_dataset_checkpoint.py` PASSES**: structure complete and all 50 466 referenced audio files (49 502 train + 964 test) resolve on disk. Dataset load smoke test passes on both splits with the expected shapes (10.24 s @ 16 kHz, 64 mel bins).
* **Known trap:** `dataset_root.json` maps the **val split to the same file as test** (964 items). A disjoint validation split must be defined explicitly before any model-selection decision, or test contamination is structurally possible.
* **Reproducible environment BUILT and verified.** `.venv` on a `uv`-provisioned standalone CPython **3.10.20**, installed from the unmodified frozen `poetry.lock`: 155 packages, 0 errors, no pin relaxed (`torch 1.13.1+cu117`, `transformers 4.30.2`, `pytorch-lightning 2.1.1`, `librosa 0.9.2`, `numpy 1.23.5`, plus `audioldm_eval` and `hear21passt`). Lightning blocks `conda create` in every form and `/commands/python3.10` is a shim that runs 3.12; both are documented in `docs/environment_report.md`.
* **Model-loading smoke test PASSES.** `UNetModel` rebuilt from the frozen config and loaded `strict=True` for both budgets: `[1,2,3,5]` 415.955 M and `[1,2,3,1]` 145.674 M, 690 tensors each, 0 missing / 0 unexpected. Import smoke tests 6/6 including `audioldm_eval`. The M0 criterion "base and pruned architectures can be reconstructed deterministically" is **met**.
* **M2 conditioning-path validation COMPLETE (PASS), 100% CPU.** `research_pruning/diagnostics/conditioning.py` exposes the audio/text CLAP paths and FiLM epsilon prediction, faithful to `LatentDiffusion.apply_model` (proven by file:line in `docs/condition_swap_validation.md`). Tests `tests/research/test_conditioning_paths.py` **T1..T5 all PASS**: e_a/e_t are `[B,1,512]` into the same FiLM interface (`extra_film_condition_dim==512`, `film_emb 512→768`); determinism `max|Δ|==0.0`; pairing proven by tensor hash; non-degeneration `mean|eps_a−eps_t|=1.148e-2`. CLAP embeddings are **L2-normalized for both modalities** (‖e‖₂≈1.0, std ~4e-8, N=48); paired cosine 0.248 vs cross −0.079 (sanity PASS). **Two traps recorded:** (i) CLAP truncates the audio branch to 10.0 s (drops ~0.24 s of every 10.24 s clip, `data.py:452,460`); (ii) upstream `unconditional_prob=0.1` is **not** overridden by the config — the diagnostic forces 0.0, and M3 calibration must too. Report `docs/condition_swap_validation.md`; artifacts in `artifacts/m2_condition_swap/`. `git diff upstream-frozen -- audioldm_train/` still empty.
* **M3A machinery COMPLETE (M3-000), 100% CPU, NO scientific result.** Resolves audit findings A1/A2/A3. `build_paired_slots` now returns a noised REAL latent (`z_0 = scale_factor·VAE.encode(mel).mode()`, `z_t = √ā·z_0 + √(1−ā)·eps`); **`scale_factor = 0.9138255715370178`** read from the checkpoint. **The VAE embedded in `audioldm-m-full.ckpt` differs from `vae_mel_16k_64bins.ckpt`** (204/398 tensors, max|diff| 12.89) — `build_vae` uses the embedded weights (what LatentDiffusion uses). Diagnostics `D_gen/D_mod/R_mod` in `research_pruning/diagnostics/modality_diagnostics.py` (L2 flattened per-example norm; epsilon 1e-12; §6 strata aggregation); `tests/research/test_diagnostics.py` **D1..D5 all PASS** on control models. M2 tests re-run on the real latent: **T1..T5 still PASS** (T5 now reports mean|eps|=0.707, ratio 1.52%). **The real pruned checkpoint `l1_audioldm-m-full_p1.ckpt` was NEVER loaded** — pre-registration uncontaminated. Evidence: `artifacts/m3_pilot/`.
* **Disjoint validation split DEFINED (M0 9.1 resolved).** `configs/research/val_split_disjoint.json` = upstream AudioCaps val (495 items), proven disjoint by wav id from test (0) and train (0); sha256 `e540146d…`. `dataset_root.json` NOT modified.
* **`docs/pilot_protocol.md` DRAFTED (not frozen).** Reasoned proposals (B=256, K=5 timestep strata, bootstrap unit/seed, prune-tail + weighted-overlap definitions), both M2 traps carried in; Freeze fields left blank pending review + GPU benchmark.
* Structure created per master plan §13. `audioldm_peft/`, `research_pruning/{taylor,paired_modality}/`, remain **skeletons with no implementation**; no M1 code was written or reconstructed.
* No GPU is attached. `docs/compute_budget.md` is entirely unmeasured, Compute Gate CG is unresolved, and M3 stays blocked.

## OPEN ITEMS

1. **Request the recovered full-FT `(1,2,3,1)` checkpoint from Arshdeep today** — the public search is finished and the artifact is proven absent. Also ask him to confirm the pre-recovery reading of `l1_audioldm-m-full_p1.ckpt`.
2. Recover the local-only M1 LoRA/PEFT CPU scaffold from the Windows machine, diff it against this repository, and re-run its tests here before touching it.
4. Finish the fetch of `checkpoints.tar` / `dataset.tar` and extract into `data/checkpoints/` and `data/dataset/`; then run upstream `tests/validate_dataset_checkpoint.py`.
5. Verify the FAD/KL pipeline (`audioldm_eval`) and reproduce the PANNs top-k semantic pipeline.
6. ~~Implement M2 audio/text conditioning instrumentation~~ **DONE (M2-001, PASS).** Still pending: prepare the single reproducible GPU benchmark recording all §7.2 variables.
7. Resolve Compute Gate CG before M3.
8. ~~Carry the two M2 traps into `docs/pilot_protocol.md`~~ **DONE (draft).** Review the pilot-protocol draft against the master plan, then freeze it (fill Freeze commit/timestamp) — only after the GPU benchmark populates `T_sal`/`T_fwd`. Do NOT inspect any saliency/diagnostic result on the real L1 checkpoint before that freeze.
9. Run the M3A machinery on the real full/pruned models ONLY after CG is resolved and the protocol is frozen.

## RUN RECIPES

* Full resume context: `docs/HANDOFF.md`

* Environment: `.venv/bin/python` (CPython 3.10.20). Rebuild: see `docs/environment_report.md`.
* Model-loading smoke test: `.venv/bin/python scripts/research/smoke_load_unet.py`
* M2 conditioning-path tests (CPU): `.venv/bin/python tests/research/test_conditioning_paths.py`
* M2 evidence (norms, cosine, timing, figures): `.venv/bin/python scripts/research/m2_condition_swap.py --n 48`
* M3A diagnostic machinery tests (CPU, control models): `.venv/bin/python tests/research/test_diagnostics.py`
* A1 real-latent evidence (scale_factor, VAE divergence, schedule): `.venv/bin/python scripts/research/m3a_latent_check.py`
* Build/verify disjoint val split: `.venv/bin/python scripts/research/build_val_split.py`
* Dataset load smoke test: `.venv/bin/python scripts/research/smoke_load_dataset.py --split train --n 2`
* Upstream dataset/checkpoint validation: `.venv/bin/python tests/validate_dataset_checkpoint.py`
* Fetch + md5-verify public artifacts: `bash scripts/research/fetch_public_artifacts.sh`
* Verify structural budget from checkpoints (CPU): `.venv/bin/python scripts/research/verify_pruned_architecture.py data/checkpoints/audioldm-m-full.ckpt data/checkpoints/l1_audioldm-m-full_p1.ckpt`
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

### 2026-08-18 | M0: reproducible Python 3.10 environment and CPU load smoke tests

* Built the dedicated environment faithful to `upstream-frozen`: standalone CPython 3.10.20 via `uv` (Lightning blocks `conda create`; `/commands/python3.10` is a decoy that runs 3.12), then upstream's own `pip install poetry` + `poetry install` on the unmodified lock. 155 packages, 0 errors, no pin relaxed.
* Import smoke tests 6/6 pass, including the FAD/KL stack.
* Added `scripts/research/smoke_load_unet.py`: both budgets rebuilt and strict-loaded with 0 missing / 0 unexpected keys, so the M0 deterministic-reconstruction criterion is met.
* Fixed a `torch.load(mmap=True)` incompatibility in our own helper scripts (torch 1.13.1 predates `mmap`). No upstream or scientific code touched.
* Logged as M0-003. `docs/compute_budget.md` untouched: no GPU, nothing measured.

### 2026-08-18 | M0: AudioCaps fetched, upstream validation passes

* All six public artifacts downloaded and md5-verified (`fetch rc=0`); `dataset.tar` extracted cleanly to 31 GB / 50 961 wav.
* Upstream `tests/validate_dataset_checkpoint.py` passes on the first full run: structure complete, all 50 466 referenced audio files present. No fix was needed.
* Added `scripts/research/smoke_load_dataset.py`; train and test splits both load with the expected preprocessing shapes.
* Recorded the val == test trap in `docs/m0_baseline_reproduction/dataset_manifest.md` and `docs/pilot_protocol.md`.
* Logged as M0-004. No scientific code modified; `git diff upstream-frozen -- audioldm_train/` still empty.

### 2026-08-18 | Session close: handoff written

* Wrote `docs/HANDOFF.md` as the single self-contained resume point: repo/HEAD/remotes, frozen SHAs, verified M0 items with evidence paths, remaining M0 work, M1/M2 status, M3 gates, environment and data locations, re-run commands, findings, deviations, a prioritised available-vs-blocked work queue, and a "Do not assume" list.
* Logged as HANDOFF-001 in `docs/experiment_ledger.md`, explicitly as administrative and not an experiment.
* No experiment, download, installation or GPU work was performed during this close-out. No gate resolved: M0 open, M1 absent and blocked, M3 blocked.

