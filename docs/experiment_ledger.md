# Experiment Ledger

Append every scientific run or gate decision, including failed and stopped runs. Do not delete failed entries.

## Entry schema

### YYYY-MM-DD HH:MM | EXPERIMENT_ID | short name

* **Status:** planned | running | completed | failed | stopped
* **Milestone / gate:**
* **Git commit:**
* **Branch:**
* **Resolved config:**
* **Command:**
* **Dataset manifest / hash:**
* **Base checkpoint / SHA256:**
* **Pruning manifest:**
* **Calibration slots / timesteps:**
* **Random seed(s):**
* **Generation seed(s):**
* **GPU / runtime:**
* **Peak VRAM:**
* **Wall time / GPU-hours:**
* **Raw output path:**
* **Primary result:**
* **Acceptance / gate decision:**
* **Failure or uncertainty:**
* **Notes:**

---

## Entries

### 2026-08-18 00:55 | BOOTSTRAP-000 | Repository bootstrap and prior-state recovery attempt

* **Status:** completed
* **Milestone / gate:** M0 prerequisite. Not a scientific run; recorded because it changes M0 status.
* **Git commit:** this commit
* **Branch:** `main`
* **Resolved config:** n/a
* **Command:** `bash install.sh /teamspace/studios/this_studio/audioldm-modality-swap-pruning`
* **Dataset manifest / hash:** n/a
* **Base checkpoint / SHA256:** n/a
* **Pruning manifest:** n/a
* **Calibration slots / timesteps:** n/a
* **Random seed(s):** n/a
* **Generation seed(s):** n/a
* **GPU / runtime:** CPU only
* **Peak VRAM:** n/a
* **Wall time / GPU-hours:** 0
* **Raw output path:** n/a
* **Primary result:** Prior research repository could NOT be recovered. `gh repo list` and `gh api /user/repos?affiliation=owner,collaborator,organization_member` returned 21 repositories, none AudioLDM/pruning/modality-swap related; a filesystem-wide search of this Studio found no such repository. The repository was therefore initialized from scratch at `/teamspace/studios/this_studio/audioldm-modality-swap-pruning` rather than recovered.
* **Acceptance / gate decision:** M0 remains OPEN. The master plan's claim that repo bootstrap, frozen references, and the LoRA CPU scaffold/tests were completed (2026-08-13/17) refers to local-only work on the author's machine that was never pushed and is not present in this environment.
* **Failure or uncertainty:** The M1 LoRA/PEFT CPU scaffold and its reportedly passing tests are unverifiable here. No claim about M1 status may be made from this repository until that code is recovered and its tests re-run.
* **Notes:** Frozen reference SHAs verified to exist upstream via read-only API: `haoheliu/AudioLDM-training-finetuning` @ `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc` (2024-12-13), `Arshdeep-Singh-Boparai/PruningAudioLDM` @ `6f65f628fabc4ad27770753698fc81944e820f9f` (2026-07-16). Neither has been imported into this repository yet; `upstream-frozen` does not exist.

### 2026-08-18 01:05 | M0-001 | Upstream import, frozen references, public-artifact inventory

* **Status:** completed
* **Milestone / gate:** M0. Includes the **full-FT checkpoint strength gate**, whose master-plan deadline is 2026-08-18.
* **Git commit:** this commit (parent `2bbc0c6`, upstream import)
* **Branch:** `main`
* **Resolved config:** none; no model was constructed or run.
* **Commands:**
  * `git remote add upstream https://github.com/haoheliu/AudioLDM-training-finetuning.git`
  * `git fetch --no-tags upstream && git branch upstream-frozen 702a638d023b008a2d9a45cdf1e1f4fcdc590dfc`
  * `git merge upstream-frozen --allow-unrelated-histories`
  * `git remote add pruning-reference https://github.com/Arshdeep-Singh-Boparai/PruningAudioLDM.git`
  * `git fetch --no-tags pruning-reference && git branch pruning-reference-frozen 6f65f628fabc4ad27770753698fc81944e820f9f`
  * `bash scripts/research/fetch_public_artifacts.sh`
  * `python3 scripts/research/verify_pruned_architecture.py data/checkpoints/audioldm-m-full.ckpt data/checkpoints/l1_audioldm-m-full_p1.ckpt data/checkpoints/Unet_model-m.ckpt`
* **Dataset manifest / hash:** `dataset.tar` md5 `1c4e6642754c38f7041efdfeabe6e32d` (Zenodo 10.5281/zenodo.14342967), fetching; `checkpoints.tar` md5 `d9898f93372582119fa19c6464f59cdc`, fetching.
* **Base checkpoint / SHA256:** `audioldm-m-full.ckpt`, md5 `46bad9f176651404b3cf1484942749b9`, verified after download. Identical md5 in the official AudioLDM record 10.5281/zenodo.7884686 and in Arshdeep's record 10.5281/zenodo.21376822, which proves the pruning reference is built on the official weights.
* **Pruning manifest:** `sorted_indexes_dict.pkl` md5 `a4cd11ff83438ee0f9aa5fe0917f39e3`; L1 `(1,2,3,1)` checkpoint `l1_audioldm-m-full_p1.ckpt` md5 `2666e6fc108a9c4fc0d19bbf26832905`, verified.
* **Calibration slots / timesteps:** n/a
* **Random seed(s) / Generation seed(s):** n/a
* **GPU / runtime:** CPU only, no GPU attached to the Studio. `torch.cuda.is_available()` is False.
* **Peak VRAM:** n/a
* **Wall time / GPU-hours:** 0 GPU-hours.
* **Raw output path:** `artifacts/m0_baseline_reproduction/` (`fetch.log`, `architecture_check.log`, `prerecovery_check.log`, `sorted_indexes_dict.pkl`)
* **Primary result:**
  1. Structural budget recovered from weight shapes alone: base `channel_mult = [1,2,3,5]`, `l1_audioldm-m-full_p1.ckpt` `channel_mult = [1,2,3,1]`, both with `model_channels = 192`. The reference's `(dp, p)` parameterisation is therefore confirmed as `dp = b3`, `p = b4`, so `(1,2,3,1)` is `dp=3, p=1`.
  2. Measured parameter counts: U-Net 415.955 M -> 145.674 M (-65.0%); full checkpoint 1142.629 M -> 872.348 M (-23.7%, VAE/CLAP/vocoder untouched).
  3. All 2061 same-shape tensors of `l1_audioldm-m-full_p1.ckpt` are **bit-identical** to `audioldm-m-full.ckpt`. The published pruned checkpoint is pure prune-and-merge output with **no finetuning applied**.
* **Acceptance / gate decision:**
  * The L1 `(1,2,3,1)` **pre-recovery** checkpoint is public, fetched, hash-verified and architecture-verified. That master-plan M0 task is satisfied without reconstruction.
  * **Full-FT checkpoint strength gate: the recovered full-FT `(1,2,3,1)` checkpoint is NOT publicly available**, established by evidence rather than assumption. This triggers the master plan's fallback: request the final full-FT checkpoint from Arshdeep immediately, and until it is obtained treat RQ3 as downgraded to a published-reference comparison. `docs/claims_matrix.md` RQ3 wording updated accordingly.
  * M0 is **not** complete: environment creation, FAD/KL pipeline verification, PANNs pipeline, and GPU smoke tests remain open.
* **Failure or uncertainty:**
  * The Zenodo record description lists `l1_audioldm-m-full_p2_dp2.ckpt`, which does not exist in the record; the published `(1,2,2,2)` artifact is the U-Net-only `l1_unet_pruned_p2_dp2.pt`. Not load-bearing for `(1,2,3,1)`.
  * The active `cloudspace` env is Python 3.12.11 and lacks every AudioLDM dependency except torch and pytorch-lightning. Upstream specifies Python 3.10 and pins `transformers==4.30.2`. No environment has been created; this decision is recorded in `docs/environment_report.md` and must be made before any model code runs.
* **Deviations from the master plan (recorded as required by AGENTS.md):**
  * Small text provenance documents live in `docs/m0_baseline_reproduction/` so they are version-controlled; `artifacts/m0_baseline_reproduction/` is gitignored and holds only raw logs and binaries. The master plan names only the `artifacts/` path.
  * The upstream root `README.md` was moved verbatim to `UPSTREAM_README.md` and a project `README.md` written in its place. No file under `audioldm_train/` was touched.
  * `PruningAudioLDM` history is preserved as the branch `pruning-reference-frozen` rather than only as a recorded SHA, so the reference survives upstream deletion. It is not merged into `main`.
* **Notes:** `git diff upstream-frozen HEAD -- audioldm_train/` is empty at this commit. No M1 scaffold was written or reconstructed.

### 2026-08-18 01:30 | M0-002 | L1 saliency manifest inspection

* **Status:** completed
* **Milestone / gate:** M0, and a hard input constraint for the M3 pilot protocol.
* **Git commit:** this commit
* **Branch:** `main`
* **Command:** static `pickletools` scan, then `pickle.load`, on `artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl` (md5 `a4cd11ff83438ee0f9aa5fe0917f39e3`)
* **GPU / runtime:** CPU only. **Wall time / GPU-hours:** 0.
* **Raw output path:** `artifacts/m0_baseline_reproduction/sorted_indexes_inspect.log`
* **Primary result:** the file is `dict[str, list[int]]` over **28 conv layers** -- `input_blocks.7..11` (9), `middle_block` (4), `output_blocks.0..6` (15) -- at widths 384/576/960. Every value is a full permutation of `range(n_channels)`, so each entry is a complete L1 channel ranking, not a truncated keep-list. The pickle contains no `GLOBAL`/`REDUCE` opcodes and executes no third-party code on load; this was verified statically *before* loading it.
* **Acceptance / gate decision:** the published L1 baseline does **not** rank the whole U-Net. For RQ2 to be structure-matched, P1/P2/P3 must be computed over exactly this 28-layer set with the same per-layer channel counts. Recorded as a constraint in `docs/pilot_protocol.md` under "Targeted layers"; the protocol remains UNFROZEN.
* **Failure or uncertainty:** one covered layer (`output_blocks.6.0.in_layers.2.weight`, 384 channels) sits outside the nominal B3/B4 region. Confirm with Arshdeep whether that is intentional before freezing the pilot protocol.
* **Notes:** no saliency result was computed or inspected. This is manifest structure only.

### 2026-08-18 01:35 | M0-003 | Reproducible Python 3.10 environment + CPU load smoke tests

* **Status:** completed
* **Milestone / gate:** M0 environment task; prerequisite for every later milestone.
* **Git commit:** this commit
* **Branch:** `main`
* **Resolved config:** `pyproject.toml` + `poetry.lock` at `upstream-frozen`, unmodified.
* **Commands:**
  * `uv python install 3.10`
  * `<uv cpython-3.10.20>/bin/python3.10 -m venv .venv`
  * `.venv/bin/pip install poetry`
  * `POETRY_VIRTUALENVS_CREATE=false VIRTUAL_ENV="$PWD/.venv" .venv/bin/poetry install --no-interaction`
  * `.venv/bin/python scripts/research/smoke_load_unet.py`
* **GPU / runtime:** CPU only, no GPU attached. **Peak VRAM:** n/a. **Wall time / GPU-hours:** 0 GPU-hours.
* **Raw output path:** `artifacts/m0_baseline_reproduction/poetry_install.log`, `artifacts/m0_baseline_reproduction/smoke_load_unet.log`
* **Primary result:**
  1. Python 3.10.20 environment installed from the frozen lock: **155 packages, 0 errors**. Key pins honoured exactly — `torch 1.13.1+cu117`, `torchvision 0.14.1`, `torchaudio 0.13.1`, `transformers 4.30.2`, `pytorch-lightning 2.1.1`, `librosa 0.9.2`, `numpy 1.23.5`, `taming-transformers-rom1504 0.0.6`, plus the git dependencies `audioldm_eval 0.0.5` and `hear21passt 0.0.23`.
  2. Import smoke tests 6/6 PASS, including `audioldm_train.modules.latent_diffusion.ddpm`, `audioldm_train.conditional_models` and `audioldm_eval` (the FAD/KL pipeline).
  3. **Model-loading smoke test PASS.** `UNetModel` rebuilt from the frozen config for both budgets and loaded with `strict=True`: base `[1,2,3,5]` 415.955 M params, 690 tensors, 0 missing / 0 unexpected; pruned `[1,2,3,1]` 145.674 M params, 690 tensors, 0 missing / 0 unexpected. Built parameter counts match the counts derived independently from the raw checkpoints in M0-001.
* **Acceptance / gate decision:** the M0 acceptance criterion "base and pruned architectures can be reconstructed deterministically" is **met**, now by actual model construction and strict weight loading rather than by shape inspection alone.
* **Failure or uncertainty / deviations:**
  * Lightning **blocks `conda create` in every form**, including `-p` prefix envs; the upstream `conda create -n audioldm_train python=3.10` recipe cannot be followed. Resolved with a `uv`-provisioned standalone CPython 3.10.20 plus `venv`. Packaging deviation only: real unmodified CPython 3.10, all versions from the frozen lock, no pin relaxed.
  * `/commands/python3.10` on this Studio is a shim that actually executes Python 3.12.11. Do not use it.
  * `POETRY_VIRTUALENVS_CREATE=false` passed as an env var rather than `poetry config --local`, so no `poetry.toml` is added and `pyproject.toml`/`poetry.lock` stay byte-identical to `upstream-frozen`.
  * `torch.load(..., mmap=True)` does not exist in torch 1.13.1. Our own helper scripts, written against the Studio's torch 2.8, failed with `TypeError`. Fixed with a `torch_load()` fallback in both helper scripts. **No upstream or scientific code was modified**; `git diff upstream-frozen -- audioldm_train/` is still empty.
  * Importing `...latent_diffusion.ddpm` downloads a tokenizer from HuggingFace at import time, so imports are not offline-safe.
* **Notes:** `docs/compute_budget.md` deliberately left entirely `TBD_MEASURED` — no GPU was attached, so there is nothing measured to record. No LoRA/PEFT scaffold was written or reconstructed.

### 2026-08-18 02:05 | M0-004 | AudioCaps fetch, upstream validation, dataset load smoke test

* **Status:** completed
* **Milestone / gate:** M0 dataset task.
* **Git commit:** this commit
* **Branch:** `main`
* **Resolved config:** `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml` at `upstream-frozen`, unmodified.
* **Commands:**
  * `bash scripts/research/fetch_public_artifacts.sh`  (completed, `fetch rc=0`)
  * `tar xf data/dataset.tar -C data/`
  * `.venv/bin/python tests/validate_dataset_checkpoint.py`
  * `.venv/bin/python scripts/research/smoke_load_dataset.py --split train --n 2`
* **Dataset manifest / hash:** `dataset.tar` md5 `1c4e6642754c38f7041efdfeabe6e32d`, verified; `checkpoints.tar` md5 `d9898f93372582119fa19c6464f59cdc`, verified. Full manifest in `docs/m0_baseline_reproduction/dataset_manifest.md`.
* **GPU / runtime:** CPU only. **Peak VRAM:** n/a. **Wall time / GPU-hours:** 0 GPU-hours.
* **Raw output path:** `artifacts/m0_baseline_reproduction/{fetch.log,dataset_extract.log,validate_dataset_checkpoint.log,smoke_load_dataset.log}`
* **Primary result:**
  1. All six public artifacts downloaded and md5-verified; `dataset.tar` extracted with no `tar` errors to 31 GB / 50 961 wav files.
  2. **Upstream `tests/validate_dataset_checkpoint.py` PASSES**: structure complete, and all 50 466 audio files referenced by the AudioCaps metadata (49 502 train + 964 test) resolve on disk. Output: "All audio files are present. You are good to go!"
  3. **Dataset load smoke test PASSES** on train and test. Sample shapes confirm the frozen preprocessing: `waveform (1, 163840)` = 10.24 s @ 16 kHz, `log_mel_spec (1024, 64)`, `stft (1024, 512)`, caption text present.
* **Acceptance / gate decision:** the M0 dataset/checkpoint preparation task is **complete and verified by the upstream validator**, not merely by file presence. No fix was needed: the validator passed on the first full run after extraction.
* **Failure or uncertainty:**
  * `dataset_root.json` maps the **val split to the same file as test** (`audiocaps_test_nonrepeat_subset_0.json`, 964 items). Any protocol needing a validation split disjoint from the test set must define one explicitly. This must be settled before M3 and before any model-selection decision, or test contamination is structurally possible.
  * librosa 0.9.2 emits `FutureWarning`s from upstream `stft.py` and `dataset.py` about positional arguments. Upstream code, deliberately not modified.
* **Notes:** no scientific code was modified to make any test pass. `git diff upstream-frozen -- audioldm_train/` is still empty.
