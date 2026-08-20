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

### 2026-08-18 02:20 | HANDOFF-001 | Session close (administrative, not an experiment)

* **Status:** completed
* **Milestone / gate:** none. This is a session-continuity record, **not a scientific run**, and resolves no gate.
* **Git commit:** this commit
* **Branch:** `main`
* **Resolved config / dataset / checkpoints / seeds:** n/a — nothing was executed.
* **Command:** none. No experiment, download, installation or GPU work was performed during this close-out.
* **GPU / runtime:** none. **Peak VRAM:** n/a. **Wall time / GPU-hours:** 0.
* **Raw output path:** n/a
* **Primary result:** `docs/HANDOFF.md` written as the single resume point for a new session, covering repository/branch/HEAD, remotes, both frozen SHAs, verified M0 items with their evidence paths, the remaining M0 work, M1/M2 status, the gates blocking M3, environment and data locations, minimal re-run commands, the recorded findings, provenance-document status, the six recorded plan deviations, a prioritised work queue split into available vs blocked, and a "Do not assume" prohibition list.
* **Acceptance / gate decision:** no gate was resolved. M0 remains **open**, M1 remains **absent and blocked**, M3 remains **blocked**.
* **Failure or uncertainty:** none introduced. The handoff asserts only state already verified and logged in BOOTSTRAP-000 and M0-001..M0-004.
* **Notes:** M1 must not be reconstructed without Gabriel's explicit authorisation; he is searching his Windows machine for the local scaffold. `docs/compute_budget.md` deliberately left entirely `TBD_MEASURED`.

### 2026-08-18 11:41 | M2-001 | Conditioning-path validation (audio vs text CLAP → FiLM)

* **Status:** completed, PASS. M2 fail condition NOT triggered.
* **Milestone / gate:** M2 (conditioning-path validation). Hard prerequisite of M3. 100% CPU.
* **Git commit:** this commit
* **Branch:** `main`
* **Resolved config:** `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml` at `upstream-frozen`, unmodified.
* **Base checkpoint:** `data/checkpoints/audioldm-m-full.ckpt` (base U-Net `[1,2,3,5]`, strict load); CLAP `data/checkpoints/clap_music_speech_audioset_epoch_15_esc_89.98.pt` (HTSAT-base).
* **Dataset manifest:** AudioCaps `test` split (964 items); N=48 real items (indices 0..47) for norm/cosine stats; slots for the paired eps tests use indices [0,1,2,3], seed 1234.
* **Seeds:** paired-slot seed 1234 (z_t noise realisation + timesteps). CLAP `unconditional_prob` forced to 0.0.
* **Commands:**
  * `.venv/bin/python tests/research/test_conditioning_paths.py`
  * `.venv/bin/python scripts/research/m2_condition_swap.py --n 48`
* **GPU / runtime:** CPU only. **Peak VRAM:** n/a. U-Net forward: 1.52 s/fwd @ batch 1, 4.94 s/fwd @ batch 4 (1.24 s/item). **Wall time / GPU-hours:** 0 GPU-hours.
* **Raw output path:** `artifacts/m2_condition_swap/{test_conditioning_paths.log,condition_swap_metrics.json,condition_swap_metrics.log,embedding_norm_hist.png,paired_cosine_hist.png}`. Report: `docs/condition_swap_validation.md`.
* **Primary result:**
  1. **T1..T5 all PASS.** e_a, e_t are `[B,1,512]`; both enter the same FiLM interface (`film_emb` `512→768`, `openaimodel.py:871-872`); `extra_film_condition_dim==512`. Determinism `max|Δ|==0.0` for eps_a and eps_t; pairing proven by `hash(z_t)==hash(noise)`; non-degeneration `mean|eps_a−eps_t|=1.148e-02`.
  2. **CLAP embeddings are L2-normalized in this checkpoint** for BOTH modalities (`F.normalize` at `model.py:749,777`): `||e||_2` mean 1.0000, std ~4e-8 over N=48. No cross-modality normalization mismatch.
  3. **Paired cosine** (same item) mean 0.248 ± 0.102 vs **cross-item** mean −0.079; sanity same>cross PASS. Modest absolute value reflects the CLAP audio–text modality gap; clearly above the cross baseline.
  4. FiLM wiring proven identical to `LatentDiffusion.apply_model` by file:line (see report table).
* **Acceptance / gate decision:** M2 **PASS**. No unresolved normalization or code-path difference. M3 is not blocked by M2 (remaining M3 gates — GPU benchmark, Compute Gate CG, frozen `pilot_protocol.md`, disjoint val split — are independent and still open).
* **Failure or uncertainty:**
  * **Audio-branch 0.24 s truncation:** CLAP resamples 16k→48k then truncates to 480000 samples = 10.0 s (`training/data.py:452,460`); every 10.24 s AudioCaps clip loses its last ~0.24 s before audio embedding. Deterministic, identical for full/pruned models, does not confound the swap; recorded for reproducibility.
  * **Upstream unconditional dropout is 0.1 and the config does NOT override it** (`conditional_models.py:1151`); the raw conditioner would apply 10% stochastic unconditional-token replacement. The diagnostic forces `unconditional_prob=0.0`; downstream M3 calibration must set this explicitly too.
* **Notes:** no scientific code modified. `git diff upstream-frozen -- audioldm_train/` still empty. New code only in `research_pruning/diagnostics/conditioning.py`, `tests/research/test_conditioning_paths.py`, `scripts/research/m2_condition_swap.py`. The `~0.24 s` audio truncation and the `unconditional_prob=0.0` requirement should be carried into `pilot_protocol.md` before M3.

### 2026-08-18 12:05 | AUDIT-M2-001 | Independent audit of M2-001

* **Status:** completed. **M2-001 verdict: PASS, confirmed independently.** Three corrections required before the M2 code is reused by M3.
* **Milestone / gate:** audit of M2. No gate resolved, no new experiment run.
* **Git commit:** this commit. Audited commit: `9ea0272`.
* **Role:** auditor, not implementer. Every claim below was re-derived from source or re-executed, not read from `docs/condition_swap_validation.md`.

**Re-verified independently (all confirmed):**

1. `.venv/bin/python tests/research/test_conditioning_paths.py` re-run: **T1..T5 PASS**, reproducing the reported values exactly — `mean|eps_a−eps_t| = 1.148377e-02`, `max = 3.519993e-01`, `z_t` hash `7c4ad16b81ddc2fb`.
2. All six `file:line` citations in the report's wiring table check out: `ddpm.py:1996-2000` (`y = cond_dict[key].squeeze(1)`), `ddpm.py:2039-2041` (`self.diffusion_model(xc, t, context_list=..., y=y, ...)`), `openaimodel.py:871-872` (`emb = th.cat([emb, self.film_emb(y)], dim=-1)`), `openaimodel.py:555`, `conditional_models.py:1321`.
3. L2 normalisation confirmed in source for both heads: `F.normalize(text_embeds, dim=-1)` at `clap/open_clip/model.py:749` and `F.normalize(audio_embeds, dim=-1)` at `:777`.
4. The 0.24 s audio truncation is real **and stronger than reported**: this vendored `get_audio_features` (`clap/training/data.py:438-466`) **ignores `data_truncating` and `data_filling` entirely** and always takes `audio_data[..., :max_len]` with `longer=True` hardcoded. It therefore diverges from upstream `laion_clap`, where `data_truncating="fusion"` selects chunks stochastically. The practical consequence is favourable — the audio branch is **deterministic** — but anyone reading `laion_clap` would wrongly assume random chunking.
5. `git diff upstream-frozen -- audioldm_train/` is empty. All five declared artifacts exist under `artifacts/m2_condition_swap/`.

**Findings requiring correction before M3 reuses this code:**

* **A1 — blocking for M3, not for M2. `z_t` is pure Gaussian noise, not a noised real latent.** `build_paired_slots` sets `z_t = noise.clone()` (`research_pruning/diagnostics/conditioning.py`); there is no VAE encode and no `q_sample`. The docstring states this honestly and it is defensible for M2, whose claim is only about determinism, pairing and wiring. But master plan §3 defines the diagnostics "for the same example, noisy latent `z_t`, diffusion timestep `t`, and noise realization", and `build_paired_slots` is the exact function M3A will reuse. Under pure noise the latent carries no audio content, so at low `t` the model is evaluated in a region it never sees. Must be replaced by `z_t = sqrt(alphacumprod_t) * z_0 + sqrt(1 - alphacumprod_t) * eps` with `z_0` the scaled VAE encoding of the real item.
* **A2 — reporting. T5's magnitude has no scale.** `mean|eps_a − eps_t| = 1.148e-02` is uninterpretable without `mean|eps|`. The ratio must be reported, because M3A's `D_mod`/`D_gen` statistics live at exactly this scale.
* **A3 — precision.** The report cites the call as `data_truncating="fusion"` without noting that the argument is dead in this vendored copy (see item 4). Record the divergence explicitly, since the determinism it buys is load-bearing for paired diagnostics.

* **Additional fact recorded for M3:** `scale_factor = 0.9138` is stored **inside** `data/checkpoints/audioldm-m-full.ckpt` (scalar tensor, key `scale_factor`), because the config sets `scale_by_std: true`. M3 must read it from the checkpoint and must not recompute it from a data batch.
* **Failure or uncertainty:** none in M2's own claims. A1 is a scope limitation of M2 that becomes a defect only when the code is reused.
* **Notes:** `pytest` is absent from `.venv`; M2's tests are standalone executable scripts, which sidesteps the problem without any environment deviation. Keep that convention.

### 2026-08-18 12:10 | M3-000 | M3A machinery (diagnostics, real latent, val split) — NO scientific result

* **Status:** completed. Machinery + tests only. **No D_gen/D_mod/R_mod was computed on the real pruned checkpoint** (`l1_audioldm-m-full_p1.ckpt` was never loaded). Pre-registration is uncontaminated.
* **Milestone / gate:** M3A machinery. Resolves audit findings A1, A2, A3. Does NOT resolve Gate A/B, Compute Gate CG, or freeze the pilot protocol.
* **Git commit:** this commit. Builds on M2-001 (`9ea0272`) and audit AUDIT-M2-001.
* **Resolved config:** `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`, unmodified.
* **Checkpoints (real):** `data/checkpoints/audioldm-m-full.ckpt` (base U-Net + embedded first_stage VAE + `scale_factor`), CLAP `clap_music_speech_audioset_epoch_15_esc_89.98.pt`. **Never** `l1_audioldm-m-full_p1.ckpt`.
* **Seeds:** paired-slot seed 1234; control-model seeds in `test_diagnostics.py`; proposed master seed for the pilot 20260818.
* **Commands:**
  * `.venv/bin/python tests/research/test_conditioning_paths.py`  (M2 T1..T5, re-run on real latent)
  * `.venv/bin/python tests/research/test_diagnostics.py`  (D1..D5)
  * `.venv/bin/python scripts/research/m3a_latent_check.py`  (A1 evidence)
  * `.venv/bin/python scripts/research/build_val_split.py`  (disjoint val split)
* **GPU / runtime:** CPU only. **Peak VRAM:** n/a. **Wall time / GPU-hours:** 0 GPU-hours.
* **Raw output path:** `artifacts/m3_pilot/{test_diagnostics.log,a1_latent_check.json,a1_latent_check.log,val_split_check.json}`; `artifacts/m2_condition_swap/test_conditioning_paths.log`. Report updated: `docs/condition_swap_validation.md`. Draft: `docs/pilot_protocol.md`.
* **Primary result:**
  1. **A1 fixed.** `build_paired_slots` now returns a noised REAL latent: `z_0 = scale_factor * VAE.encode(mel).mode()`, `z_t = sqrt(a_t) z_0 + sqrt(1-a_t) eps`. `scale_factor = 0.9138255715370178` read from the checkpoint (not recomputed). Encoder uses the posterior `.mode()` (deterministic; justified over `.sample()`). Schedule reuses upstream `make_beta_schedule` + `extract_into_tensor`; `q_sample` byte-identical to `DDPM.q_sample`. z_0/z_t bit-identical across builds; `z_t == q_sample(z_0,t,noise)`; `z_t != pure noise`. **M2 T1..T5 still PASS** on the real latent.
  2. **VAE source divergence found.** The `first_stage_model.*` embedded in `audioldm-m-full.ckpt` differs from standalone `vae_mel_16k_64bins.ckpt`: 204/398 common tensors differ (max|diff| 12.89; encoder/decoder/quant_conv all differ). LatentDiffusion uses the embedded weights, so `build_vae` loads those.
  3. **A2 fixed.** T5 now reports `mean|eps| = 0.707` and the ratio `mean|eps_a-eps_t|/mean|eps| = 0.0152` (1.52% of the epsilon scale).
  4. **A3 fixed.** `docs/condition_swap_validation.md` records that the vendored `get_audio_features` (`clap/training/data.py:438-467`) ignores `data_truncating`/`data_filling` and hardcodes `longer=True`, diverging from `laion_clap`; the resulting determinism is load-bearing for pairing.
  5. **Diagnostics implemented + tested.** `modality_diagnostics` (D_gen, D_mod, R_mod; L2 flattened per-example norm; epsilon 1e-12) + `aggregate_over_strata` (§6). `test_diagnostics.py` D1 IDENTITY / D2 BOUNDS / D3 MONOTONE / D4 SYMMETRY / D5 ISOLATION all PASS on control models.
  6. **Disjoint val split defined.** `configs/research/val_split_disjoint.json` = upstream AudioCaps val (495 items), proven disjoint by wav id from test (0) and train (0). sha256 `e540146d62d01ca70ed92e8b1adc1991da8c967e3e5229241c13f78edc8ff45e`. `dataset_root.json` NOT modified.
  7. **Pilot protocol drafted** (`docs/pilot_protocol.md`) as reasoned proposals (B=256, K=5 timestep strata, bootstrap unit/seed, prune-tail and weighted-overlap definitions), carrying both M2 traps; marked DRAFT, Freeze fields left blank.
* **Acceptance / gate decision:** machinery ready; **M3 remains blocked.** Not frozen, CG unresolved, no GPU benchmark. No scientific number produced.
* **Failure or uncertainty:** schedule buffers were verified by construction (same upstream `make_beta_schedule` + identical `register_schedule` arithmetic and `extract_into_tensor`) rather than against a live DDPM buffer, because instantiating DDPM builds a CLAP+UNet just to read a buffer. Budget/strata numbers are provisional pending the GPU benchmark (`T_sal`, `T_fwd`).
* **Notes:** no scientific code modified. `git diff upstream-frozen -- audioldm_train/` empty. New code only in `research_pruning/diagnostics/`, `tests/research/`, `scripts/research/`, `configs/research/`.

### 2026-08-18 12:34 | AUDIT-M3-000 | Independent audit of M3-000

* **Status:** completed. **M3-000 verdict: machinery ACCEPTED.** One blocking inconsistency and four corrections in the *draft protocol*; the code itself is sound. No gate resolved. `pilot_protocol.md` stays UNFROZEN.
* **Git commit:** this commit. Audited commit: `27e2b12`.
* **Role:** auditor. Every claim re-derived or re-executed, not read from the report.

**Re-verified independently (all confirmed):**

1. `tests/research/test_diagnostics.py` re-run: **D1..D5 PASS**, values reproduce (`D3` 0→253.526, `D5` `D_mod=0` exact with `D_gen=30.811`, `epsilon=1e-12`).
2. **Validation split — verified stronger than claimed.** `audiocaps_val_label.json` holds 2 475 entries over **495 unique wav ids**. Intersection by wav id is **0 in all eight comparisons**: against `audiocaps_test_nonrepeat_subset_0` (the operative test set), `audiocaps_test_label`, `audiocaps_test_nonrepeat`, `audiocaps_train_label`, and subsets 1, 2, 3, 4. The implementer checked only test and train. sha256 of `configs/research/val_split_disjoint.json` matches the reported `e540146d…`. His reason for discarding subsets 1–4 is confirmed: `sub0 ∩ subK = 964` for every K and their union is 964 — they are reorderings of the same 964 items.
3. **VAE divergence confirmed exactly.** `first_stage_model.*` inside `audioldm-m-full.ckpt`: 398 tensors, all present in the standalone file; **194 identical, 204 different, max|diff| = 12.8866** at `encoder.mid.block_2.norm2.weight`. The standalone `vae_mel_16k_64bins.ckpt` carries 56 extra keys. His choice of the embedded VAE is **correct**, and the reason is verifiable: `audioldm_train/train/latent_diffusion.py:195` calls `latent_diffusion.load_state_dict(ckpt, strict=False)` with the full checkpoint *after* `AutoencoderKL` has already loaded `reload_from_ckpt`, so the embedded weights overwrite the standalone ones.
4. **Schedule gap CLOSED — the verification he could not perform.** The trained checkpoint itself stores `betas`, `alphas_cumprod`, `alphas_cumprod_prev`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, … Reconstructing `alphas_cumprod` with upstream `make_beta_schedule` from the frozen config (`linear_start 0.0015`, `linear_end 0.0195`, `timesteps 1000`) matches the checkpoint buffer to **max|diff| = 2.979e-08**, i.e. float32 storage precision (`ckpt[0]=0.9984999895` vs `recon[0]=0.9985000000`; `ckpt[-1]=1.42303979e-04` vs `recon[-1]=1.42303975e-04`). `q_sample` is now verified against ground truth, not only by construction.
5. **A2 and A3 correctly applied:** `mean|eps_a − eps_t| = 1.078e-02` against `mean|eps| = 7.070e-01`, ratio **1.52 %**; the `data_truncating` divergence is now stated.

**Findings — all in the DRAFT protocol, none in the code:**

* **B1 — blocking. The calibration budget is internally inconsistent by a factor of 5.** `pilot_protocol.md` declares `B = 256` **base slots**, then "Draw `B` base **examples**", then `K = 5` strata with "1 timestep per stratum per base example". That yields 256 × 5 = **1 280** `(example, noise, timestep)` slots, so "all criteria consume `2B = 512` gradient evaluations" is wrong by 5×. Master plan §5 defines one gradient-evaluation unit as one evaluation at one slot. The matched-budget contract is the entire fairness guarantee of RQ2 and cannot be ambiguous. State explicitly whether `B` counts slots or examples and write the arithmetic out.
* **B2 — P1 fairness under stratification is undeclared.** P2/P3 draw 1 timestep per stratum per example. P1 draws "two pre-registered (noise, timestep) draws per base example". Unless P1's `2B` evaluations are declared to cover the **same strata with the same weights**, P1 can be handicapped by an unfavourable timestep distribution — and P1 is the mandatory baseline the entire cross-modal claim must beat.
* **B3 — caption-selection rule undeclared, and a pseudo-replication risk.** The val manifest has **5 captions per wav**; `scripts/research/build_val_split.py:62` keeps the *first caption in source order* via `setdefault`. Deterministic and fine, but unwritten, and text conditioning depends on it. Additionally the bootstrap unit must be the **wav**, never the caption-wav entry: expanding the 5 captions and resampling them as independent would narrow the CI artificially and let Gate A pass by construction.
* **B4 — D3/D4/D5 are formula tests, not model-level tests.** They operate on constructed error tensors, which is legitimate and is documented, but the module docstring says "the algebraic properties (D2..D5) are checked on error tensors" while D2 in fact perturbs real U-Nets end-to-end. Fix the docstring and label D3–D5 explicitly as formula tests, so `D5` is never later mistaken for empirical evidence of modality isolation.
* **B5 — minor.** Record in `docs/m0_baseline_reproduction/dataset_manifest.md` that `vae_mel_16k_64bins.ckpt` is **not** the VAE of AudioLDM-M-Full, so nobody reintroduces it by following the upstream config's `reload_from_ckpt`.

* **Notes:** the pre-registration discipline held — no `D_gen`/`D_mod`/`R_mod` was computed on `l1_audioldm-m-full_p1.ckpt`, and `Freeze commit`/`Freeze timestamp` are correctly blank. `git diff upstream-frozen -- audioldm_train/` still empty.

### 2026-08-18 12:46 | M3-001 | M3A random-null generator + matched-null statistic; protocol B1-B5 — NO scientific result

* **Status:** completed. Machinery + tests only. **`l1_audioldm-m-full_p1.ckpt` never loaded; no D_gen/D_mod/R_mod computed on the real pruned model.** Pre-registration intact.
* **Milestone / gate:** M3A machinery (random null + Gate A statistic). Resolves audit findings B1..B5. Does NOT resolve Gate A/B, Compute Gate CG, or freeze the protocol.
* **Git commit:** this commit. Builds on M3-000 and AUDIT-M3-000.
* **Resolved config:** `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`, unmodified.
* **Checkpoints (real):** base weights + rankings from `data/checkpoints/audioldm-m-full.ckpt` and `artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl`. **Never** `l1_audioldm-m-full_p1.ckpt`.
* **Seeds:** random-null master 20260818, seeds 20260818..20260837; matched-null bootstrap seed 20260818.
* **Commands:**
  * `.venv/bin/python tests/research/test_random_masks.py`  (R1..R4)
  * `.venv/bin/python tests/research/test_matched_null.py`  (N1..N4)
  * `.venv/bin/python tests/research/test_diagnostics.py`  (D1..D5, docstring B4 fix)
  * `.venv/bin/python scripts/research/build_random_null.py`  (persist seeds + sha256)
* **GPU / runtime:** CPU only. **Peak VRAM:** n/a. **Wall time / GPU-hours:** 0 GPU-hours.
* **Raw output path:** `artifacts/m3_pilot/{test_random_masks.log,test_matched_null.log,random_null_masks.json}`. Docs: `docs/pilot_protocol.md`, `docs/m0_baseline_reproduction/dataset_manifest.md`, `docs/condition_swap_validation.md` (unchanged this unit).
* **Primary result:**
  1. **Random-null generator** (`random_masks.py`): mechanic ported verbatim from `_external/PruningAudioLDM/scripts/pruned_unet_dict_creation.py` (`prune_with_indices` + 46-entry `LAYER_MAP`). Per-layer `k` derived from the pruned `[1,2,3,1]` target SHAPES, NOT from the pkl (which holds permutations) and NOT from the L1 ckpt. k-histogram: 15 layers 960→192, 12 keep 576, 1 keeps 384; 10 176 kept channels/mask. L1 and random masks materialise to **145.674 M params**, strict-load, and forward at `[B,8,256,16]`. Random-mask-set sha256 `90a05395…`; L1 reference-mask sha256 `5fef7d7f…`.
  2. **R1..R4 PASS**: per-layer k respected; 20 masks pairwise distinct and distinct from L1 (at the 15 pruned layers); bit-identical from seed; materialised random model loads strict=True from the BASE ckpt and forwards.
  3. **Matched-null statistic** (`matched_null.py`): linear OLS fit of `R_mod` vs `D_gen` across per-mask controls (justified; R²/resid_sd stored); `Delta_swap = R_mod^L1 - E[R_mod^random|D_gen^L1]`; standardized residual in control-SD units; bootstrap over wavs AND masks, **unit = wav** (raises on repeated wav ids).
  4. **N1..N4 PASS** on synthetic data: on-curve → delta≈0, CI contains 0; +2 SD shift → CI excludes 0, standardized residual ≈1.97; bootstrap 95% CI coverage ≈0.96; wav-unit guard raises. Demonstrates Gate A cannot pass by construction.
  5. **Protocol B1 resolved:** `B` counts SLOTS; `B = E·K`; proposal `E=256, K=5 ⇒ B=1280`; P1==P2==P3==2560 gradient evals; arithmetic table added. **B2:** P1 uses 2 draws per (example, stratum) to match P2/P3 per-stratum weights. **B3:** caption rule = first-in-file (setdefault) documented; bootstrap unit = wav documented + enforced. **B4:** `test_diagnostics.py` docstring corrected (D2 model-level; D3-D5 formula). **B5:** `dataset_manifest.md` warns `vae_mel_16k_64bins.ckpt` is NOT the AudioLDM-M-Full VAE (use the embedded first_stage; latent_diffusion.py:195 load order). M3B "per-layer counts from the pkl" phrasing corrected to "from target shapes".
* **Acceptance / gate decision:** machinery ready; **M3 remains blocked.** Protocol still UNFROZEN (Freeze fields blank), CG unresolved, no GPU benchmark. No scientific number produced.
* **Failure or uncertainty:** the random-null structural validity is guaranteed by construction (correct shapes → strict load), verified by materialising L1+random to the known 145.674 M architecture and forwarding; I did not re-derive the reference's exact published-weight reproduction. Matched-null tests use synthetic data with known response, not real diagnostics. Budget numbers (E, K, N_eval) provisional pending the GPU benchmark.
* **Notes:** no scientific code modified. `git diff upstream-frozen -- audioldm_train/` empty. New code only in `research_pruning/diagnostics/`, `tests/research/`, `scripts/research/`.

### 2026-08-18 13:30 | AUDIT-M3-001 | Independent audit of M3-001 — materialization does NOT reproduce the published L1 checkpoint

* **Status:** completed. **M3-001 verdict: tests and statistic ACCEPTED; the mask MATERIALIZER is REJECTED until it reproduces the published artifact bit-exactly.** One major finding about the published baseline itself.
* **Git commit:** this commit. Audited commit: `e72c1fe`.
* **Golden-rule note:** this audit opened `l1_audioldm-m-full_p1.ckpt` for **structural tensor-equality comparison only** — the same class of check as M0's `prerecovery_check` (which already compared the two checkpoints tensor by tensor). No `D_gen`/`D_mod`/`R_mod`, saliency, or any diagnostic was computed on it. The pre-registration is intact.

**Re-verified independently (confirmed):**

1. R1..R4 and N1..N4 re-run: all PASS, values reproduce (mask-set sha256 `90a05395…`, N2 residual 1.97, N3 coverage 0.960).
2. Per-layer `k` histogram re-derived from the `(1,2,3,1)` state-dict shapes without using his code: `{192: 15, 576: 12, 384: 1}`, 10 176 kept channels, exactly 15 selecting tensors — matches his report.
3. Budget table (B1), stratum fairness (B2), caption/bootstrap rules (B3), docstring and manifest fixes (B4/B5): all present and arithmetically consistent (E=256, K=5, B=E·K=1280 slots, P1=P2=P3=2B=2560 evaluations).

**Major finding — closing the gap he declared.** He wrote "no re-derivé que la referencia reproduzca los pesos publicados exactos". The audit did. Result: **materialize(base ckpt, L1 ranking) ≠ published `l1_audioldm-m-full_p1.ckpt`. 686/690 tensors bit-identical; 4 differ:**

| Tensor | Published convention (proven bit-exact) | Our port (verbatim reference) |
|---|---|---|
| `input_blocks.10.0.in_layers.2.weight` | out = `perm[:192]`, **in = identity** (all 576) | in permuted by `input_blocks.9.0.op` ranking (per reference `layer_map`) |
| `output_blocks.0.0.in_layers.2.weight` | **out = positional `[:192]`**, in = positional `[:384]` | out = `perm[:192]` (per reference `layer_map`) |
| `output_blocks.1.0.in_layers.2.weight` | **out = positional `[:192]`**, in = positional `[:384]` | out = `perm[:192]` (per reference `layer_map`) |
| `output_blocks.2.0.in_layers.2.bias` | **`base[perm[:192]]` (ranked)** | positional `[:192]` (reference comments this entry out) |

Trivial-identity explanations were ruled out: `perm[:192] != range(192)` for both output_blocks rankings.

**Consequences, in decreasing order of importance:**

* **C1 — the public reference script does not reproduce the published artifact.** The divergence is upstream of us: sometimes the reference ranks where the artifact is positional/identity, sometimes it truncates where the artifact ranks. The implementer's port was faithful; the port target was not.
* **C2 — the published L1 artifact is internally inconsistent at its seams.** Proven examples: (a) `output_blocks.0/1.0.out_layers.3.weight` select their **input** columns by the `in_layers` **ranking** (ours==pub) while the producing `in_layers.2` outputs **positional** first-192 channels — the consumer's column selection does not correspond to the producer's actual outputs; (b) in `output_blocks.2.0.in_layers.2`, the **weight** rows are positional while the **bias** entries are ranked — the bias values are attached to channels other than the ones whose weights were kept. These are properties of the *published baseline*, pre-recovery; they are part of what the paper compares against and belong in the M0 record and possibly in the questions for Arshdeep.
* **C3 — the matched null as built is NOT matched to the artifact that will be diagnosed.** M3A will measure `R_mod^L1` on the published checkpoint. The random masks must therefore be produced by a materializer that, given the L1 ranking, reproduces the published checkpoint **690/690 bit-exactly**; random masks then reuse that exact materializer with random rankings, so they differ from L1 *only* where L1's construction was actually ranking-driven, and inherit the same seam conventions everywhere else.

**Required fix (blocking for the null):** rewrite the materializer's per-layer conventions to match the published artifact; acceptance = `materialize(L1 ranking) == published ckpt` on all 690 tensors; regenerate the 20 masks + sha256 through the fixed materializer; document each deviation from the public reference script alongside the proof.

**Not affected:** `matched_null.py` (statistic + bootstrap), the protocol arithmetic, split, captions, docstrings. R4's strict-load/forward criterion stays necessary but was proven insufficient — only artifact-level bit-equality catches seam-convention errors.

### 2026-08-18 13:42 | M3-002 | Materializer made bit-exact to the published L1 artifact — NO diagnostic

* **Status:** completed. **Acceptance met: `materialize(base, L1 ranking) == l1_audioldm-m-full_p1.ckpt`, 690/690 bit-exact.** Resolves AUDIT-M3-001's rejection of the materializer.
* **Golden-rule note:** the L1 checkpoint is opened for **tensor EQUALITY ONLY** (test R5 / `verify_l1_bitexact.py`), the same class of check as M0's `prerecovery_check`. **No `D_gen`/`D_mod`/`R_mod`, saliency, or any diagnostic** is computed on it. Pre-registration intact.
* **Git commit:** this commit. Builds on M3-001 (`e72c1fe`) and AUDIT-M3-001.
* **Resolved config:** `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`, unmodified.
* **Checkpoints (real):** base weights from `audioldm-m-full.ckpt`; ranking from `sorted_indexes_dict.pkl`; `l1_audioldm-m-full_p1.ckpt` opened only for equality (R5).
* **Commands:**
  * `.venv/bin/python scripts/research/verify_l1_bitexact.py`  (R5 evidence)
  * `.venv/bin/python tests/research/test_random_masks.py`  (R1..R5)
  * `.venv/bin/python tests/research/test_matched_null.py`  (N1..N4, unchanged, re-run PASS)
  * `.venv/bin/python scripts/research/build_random_null.py`  (regenerate seeds+sha)
* **GPU / runtime:** CPU only. **Peak VRAM:** n/a. **Wall time / GPU-hours:** 0 GPU-hours.
* **Raw output path:** `artifacts/m3_pilot/{l1_bitexact_check.json,test_random_masks.log,random_null_masks.json}`. Docs: `docs/pilot_protocol.md` (+ "Seam conventions" section), `docs/m0_baseline_reproduction/dataset_manifest.md`.
* **Primary result:**
  1. **The public reference script does NOT reproduce the published artifact** (686/690). The 4 deviating tensors and their published conventions, each proven bit-exact:
     * `input_blocks.10.0.in_layers.2.weight`: out=`perm[:192]`, **in=identity** (ref reorders input by `input_blocks.9.0.op` ranking). Fix: `idx2=None`.
     * `output_blocks.0.0.in_layers.2.weight`, `output_blocks.1.0.in_layers.2.weight`: **fully positional** `[:192,:384]` (ref ranked output). Fix: removed from `LAYER_MAP` → positional fallback.
     * `output_blocks.2.0.in_layers.2.bias`: **ranked** `base[perm[:192]]` (ref comments it out → positional). Fix: added ranked override. Its **weight** stays positional (artifact inconsistency, reproduced).
  2. **`LAYER_MAP` corrected** = reference map minus the 2 positional-out seams, with `input_blocks.10.in` input set to identity and the ranked-bias override added. `materialize(L1)` → **690/690 bit-exact** (`l1_bitexact_check.json`). The corrected model still has 145.674 M params and strict-loads.
  3. **Ranking-driven layers enumerated: 12 of the 15 selectors** are ranking-driven in output; 3 (`output_blocks.0/1/2.0.in_layers.2.weight`) are positional. Random masks (same materializer, random rankings) coincide with L1 at every positional/identity seam and differ **only** in those 12 layers — the null is matched exactly in the non-chosen channels. M3B prune-tail competes only on these 12.
  4. **Tests:** R5 BIT-EXACT PASS (690/690); R1..R4 updated (R2 distinctness now measured over the 12 ranking-driven layers) PASS; N1..N4 re-run PASS. Regenerated fingerprints (full-ranking hashes): random-mask-set sha256 `3e6666bcdf0bab77568650732aaf9aab37241527903c6023031d01aac84e8f7e`; L1 reference-mask sha256 `9a2593c20555d510d0edef76deb2075121f5e865f5f931d0c28584fa83524360`.
  5. **Artifact inconsistency recorded** in `docs/m0_baseline_reproduction/dataset_manifest.md`; question for Arshdeep logged in PROGRESS.
* **Acceptance / gate decision:** materializer ACCEPTED (bit-exact). **M3 remains blocked** — protocol UNFROZEN, CG unresolved, no GPU benchmark. No scientific number produced.
* **Failure or uncertainty:** the seam conventions are reproduced from the artifact, not explained; whether they are intentional is a question for Arshdeep. R4's strict-load/forward remains necessary but insufficient — only R5 bit-equality catches seam-convention errors.
* **Notes:** no scientific code modified. `git diff upstream-frozen -- audioldm_train/` empty. New/changed code only in `research_pruning/diagnostics/`, `tests/research/`, `scripts/research/`.

### 2026-08-18 14:03 | AUDIT-M3-002 | Independent audit of M3-002 — fix accepted; the master plan's Gate B is mathematically broken at this budget

* **Status:** completed. **M3-002 verdict: ACCEPTED.** The audit then uncovered a defect in the *master plan itself* that requires Gabriel's decision before `pilot_protocol.md` can freeze.
* **Git commit:** this commit. Audited commit: `926bf72`.

**Re-verified independently (confirmed):**

1. **R5 with the auditor's own script from AUDIT-M3-001, unchanged: 690/690 bit-identical, 0 missing, 0 extra.** The materializer now reproduces the published artifact exactly.
2. R1..R5 and N1..N4 re-run: all PASS. R2 correctly restricted to the 12 ranking-driven layers.
3. Seam conventions pre-registered in `pilot_protocol.md` (12 ranking-driven / 3 positional of the 15 selecting tensors) and propagated into the M3B overlap definition. Freeze fields still blank. New null sha256 `3e6666bc…`, L1 reference `9a2593c2…` recorded.

**Finding G1 — blocking for the freeze, requires Gabriel: master plan Gate B is infeasible as written.** At the `(1,2,3,1)` budget the ranking-driven layers prune `p = 768` of `N = 960` channels (`k = 192` kept). For **prune-set** overlap — the plan's explicit object ("Jaccard of prune sets", M3B) — set algebra gives:

* range `[0.75, 1.0]` — the pigeonhole floor is `(2·768−960)/768 = 0.75`;
* chance level under independent selections = `768/960 = 0.80`.

Therefore the plan's Gate B conditions read, at this budget: **condition 1** (`weighted overlap ≤ 0.80`) = "audio and text agree *no more than pure chance*" — an extreme demand, since any two magnitude-correlated saliencies will exceed chance; **condition 2** (`≥ 2 layers with overlap ≤ 0.70`) is **mathematically impossible** — 0.70 is below the 0.75 floor. Gate B as pre-registered can never PASS. The plan was evidently written without noticing that an 80 % prune fraction pushes prune-set overlap into a compressed `[0.75, 1]` range.

**Finding G2 — internal contradiction in the draft protocol.** The M3B section currently defines overlap **both** ways: one bullet says "`overlap@k` uses the **kept sets** of size `k_l = 192`", the next defines `S_a^l, S_t^l` as "the **bottom-`p_l`** channels" with `overlap_l = |∩|/p_l` (prune sets, `p_l = 768`). These are different quantities (exact correspondence: `prune_overlap = (576 + kept_intersection)/768`; plan's 0.80 prune ⇔ 0.20 kept = chance; plan's 0.70 prune ⇔ negative kept = impossible). The draft must keep exactly one definition — after Gabriel amends the gate.

**Options prepared for Gabriel (decision recorded here once made):**

* **(a) Kept-set overlap with the plan's numerals transferred** (weighted ≤ 0.80, ≥2 layers ≤ 0.70): well-defined, lenient — passes unless modalities agree on ≥80 % of kept channels (chance is 20 %).
* **(b) Kept-set overlap at midpoint calibration** (weighted ≤ 0.60, ≥2 layers ≤ 0.50): stricter, roughly halfway between chance (0.2) and identity (1.0).
* **(c) Chance-adjusted overlap** `(obs − 0.2)/0.8` with thresholds ≤ 0.75 / ≤ 0.625 (equivalent to (a)/(b) but reported on a 0-at-chance scale).
In every case report raw kept-set, prune-set, and chance-adjusted overlap; the gate decision uses the amended primary. Amendment must be recorded in this ledger *before* the protocol freezes, per the plan's own rule.

* **Notes:** the implementer's honesty chain worked exactly as designed — M3-001 declared the unverified gap, AUDIT-M3-001 closed it and found the seam divergence, M3-002 fixed it to bit-exactness, and this audit confirmed the fix with independent code. `git diff upstream-frozen -- audioldm_train/` still empty. Pre-registration intact: no diagnostic has touched the L1 checkpoint.

### 2026-08-19 00:45 | M1-005 | Adopt recovered PEFT scaffold into the repo; fix audit defects F2–F5, F7; CPU dummy-model tests

* **Status:** completed (CPU dummy-model portion of M1; F6 real-U-Net and F8 upstream integration are separate follow-up units).
* **Milestone / gate:** M1 parameter-efficient recovery. Moves `audioldm_peft/` from skeleton to a working, tested package. GPU acceptance still pending (no GPU attached).
* **Git commit:** this commit.
* **Branch:** main.
* **Resolved config:** `configs/research/peft_r8_full_unet.yaml` (rank 8, alpha 16, dropout 0, full-U-Net scope, train_bias/train_groupnorm_affine true, train_layernorm_affine **false**).
* **Command:** `.venv/bin/python scripts/research/run_research_tests.py` (M1 suite); `.venv/bin/python scripts/research/cpu_smoke_peft.py`.
* **Random seed(s):** per-test `torch.manual_seed` (0/1); no data seeds (dummy tensors).
* **GPU / runtime:** CPU only, CPython 3.10.20, torch 1.13.1+cu117 (CPU path).
* **Raw output path:** test stdout (self-checking, exit 0). Pristine overlay preserved at `_external/m1_scaffold_recovered/` (gitignored).
* **Primary result:** adopted `audioldm_peft/{config,layers,inject,report,state,optimizer,ema,__init__}.py` from the audited overlay and fixed five of the eight audit defects:
  * **F2 (scientific):** LayerNorm affine no longer half-trained. New explicit `train_layernorm_affine` flag (default false); GroupNorm **and** LayerNorm are excluded from the generic bias sweep; LayerNorm is trained weight+bias together only under the flag and reported as its own `layernorm_affine` category in `report.py` / `build_parameter_groups`.
  * **F3 (correctness):** `configure_auxiliary_trainables` counts unconditionally → identical nonzero counts on repeated calls, independent of call order (was zeros on the second call).
  * **F4 (correctness):** `freeze_for_peft` preserves LoRA adapter params, so a late/mis-ordered call can no longer silently disable training; `assert_peft_ready` guard added and regression-tested (adversarial freeze-after-inject).
  * **F5 (efficiency):** `LoRAConv2d.forward` factorised into `conv2d(x,A)->conv2d(.,B_1x1)`; numerically identical to the materialised-delta path (max|Δ| 1.19e-7) with `delta_weight()`/merge unchanged.
  * **F7 (gap):** `training_state_dict` / `load_training_state_dict` bundle adapter+optimizer+scheduler+EMA+global_step and round-trip through `torch.save`; verified restoring into a fresh model/optimizer/EMA (adapter, AdamW moments, EMA shadows and step all match).
  * Also adopted: `configs/research/peft_r8_full_unet.yaml`, `scripts/research/cpu_smoke_peft.py`, `docs/integration_notes.md`, `docs/M1_CHECKLIST.md`; added `scripts/research/run_research_tests.py` (stdlib runner).
* **Acceptance / gate decision:** M1 **CPU dummy-model** checks PASS — L1/L2/L3 (merge/unmerge/factorisation), J1–J4 (inject/freeze/F2/F3/F4), S1–S3 (adapter roundtrip / trainable-only EMA / full resume). M1 CPU acceptance is **not yet complete**: F6 (tests on the real pruned U-Net) and F8 (upstream integration hooks) remain. M1 GPU acceptance stays blocked (no GPU).
* **Failure or uncertainty:** F1 environment deviation — `pytest` is absent from the frozen lock and no pinned version was relaxed; tests run via the stdlib runner and each module's `__main__`. Numbers here are from dummy `nn.Sequential` models, not AudioLDM; the real-U-Net evidence is F6 (next unit). Two audit defects (F6, F8) remain open by design.
* **Notes:** no scientific/upstream code modified; `git diff upstream-frozen -- audioldm_train/` still empty. New code only in `audioldm_peft/`, `tests/research/`, `scripts/research/`, `configs/research/`, `docs/`. The overlay's stale `docs/` copies were NOT adopted (would regress project state), per the audit's adoption constraint.

### 2026-08-19 01:10 | M1-006 | PEFT tests on the REAL pruned U-Net (audit finding F6)

* **Status:** completed (CPU).
* **Milestone / gate:** M1. Closes audit defect F6 (dummy-only tests). Injection, parameter accounting and merge algebra now verified on the actual `(1,2,3,1)` diffusion U-Net.
* **Git commit:** this commit.
* **Branch:** main.
* **Command:** `.venv/bin/python tests/research/test_peft_real_unet.py`.
* **Resolved config:** UNet built from `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml` with `channel_mult=[1,2,3,1]`; PEFT `configs/research/peft_r8_full_unet.yaml` (rank 8, alpha 16). No checkpoint loaded — counts and merge algebra are weight-independent; random init.
* **Random seed(s):** `torch.manual_seed(0)`.
* **GPU / runtime:** CPU only, ~30 s (3 forwards on the full U-Net).
* **Primary result:** on the real pruned U-Net, injection wraps **284 modules (185 Linear + 99 Conv2d)** — every eligible module — with decomposition **LoRA 3,718,784 · bias 108,680 · GroupNorm affine 48,768 · LayerNorm affine 0 (default) · other_trainable 0 · trainable 3,876,232 of 149,392,648 total** (base 145,673,864 + LoRA 3,718,784). This reproduces the audit's structural numbers except that the **F2 fix** removes the 17,856 LayerNorm biases from the bias bucket (audit reported bias 126,536 / trainable 3,894,088; now 108,680 / 3,876,232). Merge/unmerge on a real forward is numerically invariant: max|unmerged−merged| **1.04e-7**, unmerge restores to **6.52e-8**. `train_layernorm_affine=True` adds exactly **35,712** params (48 LayerNorms × weight+bias) under `layernorm_affine`.
* **Acceptance / gate decision:** F6 tests **R6a/R6b/R6c PASS**. M1 CPU acceptance now covers dummy + real U-Net; only **F8** (upstream integration hooks) remains for the CPU portion. GPU acceptance still blocked.
* **Failure or uncertainty:** none in the tested path. Numbers are from random weights; real-weight behaviour is identical for counts/merge (weight-independent) and only matters for the GPU training run. First test iteration miscounted `base_total` from post-injection `unet.parameters()` (a test bug, not a package bug); fixed to capture it pre-injection.
* **Notes:** `git diff upstream-frozen -- audioldm_train/` still empty. Test is checkpoint-independent (config + import only), so it runs anywhere the environment builds.

### 2026-08-19 01:35 | M1-007 | PEFT upstream-integration hooks + CPU tests (audit finding F8) — closes M1 CPU acceptance

* **Status:** completed (CPU). **M1 CPU acceptance is now COMPLETE.** GPU acceptance remains blocked (no GPU attached).
* **Milestone / gate:** M1. Closes audit defect F8. Provides the CPU-testable core of the minimal upstream patch **without modifying `audioldm_train/`** (diff still empty).
* **Git commit:** this commit.
* **Branch:** main.
* **Command:** `.venv/bin/python scripts/research/run_research_tests.py` (full M1 suite, 5 modules).
* **GPU / runtime:** CPU only.
* **Primary result:** added `audioldm_peft/integrate.py` — `setup_peft` (freeze→inject→aux in the one correct order + readiness assertion), `build_peft_optimizer` (AdamW over only the LoRA/bias/GroupNorm/LayerNorm groups; raises on any stray trainable), `build_trainable_only_ema` (post-setup EMA), `peft_config_from_yaml` (parses `configs/research/peft_r8_full_unet.yaml`). Tests `tests/research/test_peft_integration.py` **I1–I4 PASS**: I1 setup leaves only PEFT params trainable; I2 optimizer routes lora/bias/groupnorm with per-group LRs (1e-4 / 5e-5) and its optimized tensor-id set equals the model's trainable set; I3 config parse matches; I4 proves the post-load ordering constraint (loading the original checkpoint after wrapping fails strict key matching — keys renamed to `...base.weight` — while the base weights are preserved exactly through wrapping). `docs/integration_notes.md` documents the exact upstream patch points these functions plug into.
* **Acceptance / gate decision:** **M1 CPU acceptance criteria (master plan §M1) all met:** Linear & Conv2d merge/unmerge (L1–L3), injector freezes base (J1/J4), auxiliaries reported separately (J1/J2), adapter-only state save/reload (S1) — plus, beyond the minimum, real-U-Net evidence (R6a–c), full-resume state (S3), trainable-only EMA (S2), and the integration hooks (I1–I4). Full suite: **17/17 checks PASS across 5 modules.** All eight audit defects addressed: F1 (stdlib runner), F2–F5 & F7 (M1-005), F6 (M1-006), F8 (M1-007). **M1 GPU acceptance** (several hundred real steps, VRAM, sec/step, resume) is the only remaining M1 item and is blocked on GPU + Compute Gate CG.
* **Failure or uncertainty:** the upstream Lightning wiring itself is written as documented patch points, not yet applied to `audioldm_train/` — deliberately, to keep the frozen diff empty until a GPU session applies and validates it. The hooks are unit-tested in isolation but their end-to-end behaviour inside `LatentDiffusion.configure_optimizers`/EMA/resume is only verifiable with a training run.
* **Notes:** `git diff upstream-frozen -- audioldm_train/` verified empty (0 lines). New code only in `audioldm_peft/`, `tests/research/`.

### 2026-08-19 02:20 | M0-005 | GPU benchmark script prepared (write-only; §7.2) — NOT run, no GPU

* **Status:** completed (script written + guarded); **no GPU numbers produced.**
* **Milestone / gate:** Compute Gate CG prerequisite. Prepares the reproducible benchmark so the first GPU session can populate `docs/compute_budget.md` with measured values.
* **Git commit:** this commit.
* **Command (GPU only, NOT run tonight):** `.venv/bin/python scripts/research/gpu_benchmark.py --steps 30 --batch 8 --out docs/compute_budget_measured.json`.
* **Primary result:** `scripts/research/gpu_benchmark.py` measures every §7.2 variable on the real pruned `(1,2,3,1)` U-Net — TRAIN_SEC_PER_STEP + PEAK_TRAIN_VRAM (PEFT fwd+bwd+opt via the tested M1 `setup_peft`/`build_peft_optimizer`), SALIENCY_SEC + PEAK (Taylor fwd+bwd weight grads), FORWARD_SEC + PEAK (diagnostic fwd), and GPU_MODEL/VRAM. Generation timing is a marked stub (needs the LatentDiffusion sampler wired). Uses warmup + `cuda.synchronize` + `max_memory_allocated`. `docs/compute_budget.md` gains a "how to populate" pointer; every field stays `TBD_MEASURED`.
* **Acceptance / gate decision:** **the script REFUSES to run without CUDA** (verified on CPU: exits non-zero with an explicit message) so no value is ever fabricated. Syntax verified. CG remains UNRESOLVED; compute_budget untouched (still 100% TBD_MEASURED).
* **Failure or uncertainty:** the timing loops cannot be executed here (no GPU), so they are unverified beyond syntax + the shared, already-tested build path (M1 F6). Generation timing is not implemented.
* **Notes:** invariant intact — no GPU numbers invented; `git diff upstream-frozen -- audioldm_train/` empty.

### 2026-08-19 02:35 | M0-006 | FAD/KL and PANNs top-k evaluation pipelines exercised end-to-end (CPU)

* **Status:** completed for PANNs top-k; FAD/KL pipeline proven to run end-to-end (FAD library bug worked around), numeric KL/IS/FID values captured in a follow-up when the CPU run finishes. Pipeline SMOKE only — arbitrary disjoint AudioCaps subsets, values are not scientific.
* **Milestone / gate:** M0 remaining items 1 (FAD/KL) and 2 (PANNs top-k). Both were previously only proven to import.
* **Git commit:** this commit (+ a follow-up for the KL/IS/FID numbers).
* **Command:** `scripts/research/fad_kl_smoke.py --gen ... --gt ... --sr 16000 --fresh`; `scripts/research/panns_topk.py --dir ... --k 10 --limit 20`.
* **Dataset manifest / hash:** two disjoint 256-clip symlink folders (gt = AudioCaps items 1..256 sorted, gen = 257..512) under `artifacts/m0_baseline_reproduction/fad_kl_smoke/` (gitignored).
* **Base checkpoint / SHA256:** PANNs Cnn14 16 kHz `ckpt/Cnn14_16k_mAP=0.438.pth` (Zenodo 3987831), CPU-sanitised; 32 kHz `Cnn14_mAP=0.431.pth` (Zenodo 3576403) auto-fetched by the library; both gitignored. VGGish via torch.hub.
* **GPU / runtime:** CPU only.
* **Primary result:** documented in `docs/m0_baseline_reproduction/eval_pipeline_closure.md`. PANNs top-10: 20/20 real clips classified with semantically coherent events (Train/Rail, Waterfall/Stream, Neigh/Horse, Bee/Insect, Applause). FAD/KL: full dependency chain resolved; pipeline runs through VGGish FAD (returns NaN via workaround), Cnn14 classifier feature extraction, and KL/IS/FID.
* **Acceptance / gate decision:** both M0 eval pipelines now execute end-to-end with recorded invocations — M0 items 1 and 2 satisfied at the pipeline level.
* **Failure or uncertainty:** six library findings recorded (F-eval-1 CPU-deserialize without map_location → checkpoint CPU-sanitised; F-eval-2 path-keyed feature cache → `--fresh`; F-eval-3 FAD `sqrtm` imaginary component exceeds tolerance and `eval.py` mishandles the sentinel → NaN workaround, real eval must use standard-FAD real-part behaviour; F-eval-4 Cnn14 IS pretrained; **F-eval-5** the Cnn14-2048 FID crashes the same way via AssertionError and, running after KL/IS in `calculate_metrics`, discards them → computed KL/IS directly from cached features via `scripts/research/fad_kl_from_cache.py`, and demonstrated the standard `covmean.real` Frechet fix is finite; **F-eval-6** `calculate_kl` returns the sentinel -1 unless files are same-name paired). **Captured smoke values (non-scientific, disjoint 256-clip subsets):** IS = 6.230 ± 1.531; KL = -1 (needs pairing); Frechet-2048 (real-part fix) = 15.361; FAD = NaN. FAD/FID unusable as-is in audioldm_eval 0.0.5.
* **Notes:** no scientific/upstream code modified; `git diff upstream-frozen -- audioldm_train/` empty. New tracked code in `scripts/research/{fad_kl_smoke,panns_topk}.py` and `docs/m0_baseline_reproduction/eval_pipeline_closure.md`; checkpoints/label CSV/eval folders gitignored.

### 2026-08-19 03:15 | M3B-000 | P0-P3 pruning-criteria machinery (channel-gate Taylor) — NO scientific result

* **Status:** completed (machinery + control-model tests). **No saliency computed on the real pruned/L1 checkpoint.** Serves M3B (Gate B saliency disagreement) and M4 (matched pruning experiment).
* **Milestone / gate:** master plan §4-5. Moves `research_pruning/taylor/` and `research_pruning/paired_modality/` from skeletons to a working, tested criteria package.
* **Git commit:** this commit.
* **Branch:** main.
* **Command:** `.venv/bin/python tests/research/test_taylor_saliency.py`.
* **GPU / runtime:** CPU only, control models.
* **Primary result:** implemented the channel-gate first-order Taylor saliency machinery:
  * `research_pruning/taylor/gates.py` — `ChannelGate` (per-output-channel gate g_c=1 on a Conv2d; `out = conv(x)·g`), `attach_gates`/`remove_gates` (in-place, fail loudly on non-Conv2d), `conv_modules`.
  * `research_pruning/taylor/saliency.py` — `accumulate_taylor` (S_c = mean_slots |g_c·∂L/∂g_c|), `normalize_within_layer` (sum/max/l2), `p0_l1_magnitude` (data-free), `combine_mean` (P2), `combine_max` (P3), `prune_order`/`keep_topk`, `assert_matched_budget` (§5: P1 2B == P2/P3 B+B, audio==text).
  * `research_pruning/paired_modality/criteria.py` — `compute_criteria` orchestration producing P1/P2/P3 at the matched 2B budget, sharing S_a/S_t between P2 and P3 (no duplicate compute), with real-model wiring notes (reuse `conditioning.build_paired_slots`/`paired_eps`, gate the 28 L1 conv layers).
* **Acceptance / gate decision:** control tests **C1–C7 PASS** — gate gradients (dead channel scores ~0 and is pruned first; C1), audio≠text swap is real with max≥mean (C2), exact mean/max combine (C3), sum/max/l2 normalization (C4), matched-budget enforcement incl. rejection of mismatches (C5), P0 L1 equals the manual per-channel weight norm (C6), and the full P1/P2/P3 orchestration at 2B with a budget-mismatch rejection (C7).
* **Failure or uncertainty:** tested on control `nn.Conv2d` nets, NOT the real U-Net. The real prunable-layer→module-path mapping (the 28 L1 conv layers) and the exact slot/timestep construction are NOT wired/frozen here — they belong to the M3B/M4 run behind the unfrozen pilot protocol. Within-layer normalization mode is a protocol parameter (default "sum"); the choice must be frozen. **P1 is scientifically load-bearing (any cross-modal claim depends on a correct P1) — this path must pass `/auditar` before any real use.**
* **Notes:** follows the M3-000 "build machinery, run no scientific experiment" precedent. `git diff upstream-frozen -- audioldm_train/` empty. New code only in `research_pruning/taylor/`, `research_pruning/paired_modality/`, `tests/research/`.

### 2026-08-19 03:40 | M3B-001 | Prunable layer set verified + gates attach on the real base U-Net — NO saliency

* **Status:** completed (structural/plumbing). **NO saliency or P0-P3 ranking computed on the real model.**
* **Milestone / gate:** master plan §4, finding 9.4. Wires the P0-P3 machinery to the structure-matched L1 layer set.
* **Git commit:** this commit.
* **Command:** `.venv/bin/python tests/research/test_prunable_layer_set.py`.
* **Primary result:** `research_pruning/taylor/layer_set.py` — `l1_prunable_layer_names`/`verify_prunable_layers`/`load_and_verify` derive the 28 prunable Conv2d module paths from the public `sorted_indexes_dict.pkl` and check them against the real base `(1,2,3,5)` U-Net. Test **V1/V2 PASS**: all 28 L1 keys resolve to Conv2d with out_channels == the ranking full length (widths 384/576/960); attaching channel gates (init 1.0) to those 28 layers leaves the U-Net output **bit-identical** (max|Δ| = 0.0) and `remove_gates` restores the bare convs.
* **Acceptance / gate decision:** the P0-P3 criteria now have a verified, structure-matched real-model layer set. Computing actual saliency over it remains the M3B/M4 scientific run — blocked until the pilot protocol is frozen and CG resolved.
* **Failure or uncertainty:** none in the tested structural path; no scientific value produced by design. Saliency is computed on the base (unpruned) model, which this test uses.
* **Notes:** reads only architecture + the public ranking. `git diff upstream-frozen -- audioldm_train/` empty.

### 2026-08-19 04:05 | M3B-002 / FINDING | Published L1 baseline keeps the LOWEST-magnitude filters (inverted L1)

* **Status:** completed — high-severity finding, rigorously verified (4-way); interpretation open for Gabriel/`/auditar`. **No gate changed.** Surfaced while validating the P0 machinery on the real base weights.
* **Milestone / gate:** M0/M3B baseline integrity. Affects RQ2 (P0/L1 baseline) and RQ3 (recovery starting point).
* **Git commit:** this commit.
* **Command:** `.venv/bin/python scripts/research/verify_l1_direction.py` (exit 0 = CONFIRMED).
* **Base checkpoint:** `audioldm-m-full.ckpt` (base weights loaded strict); public ranking `sorted_indexes_dict.pkl`; reference `_external/PruningAudioLDM/scripts/layerwise_sorted_index_generation.py`.
* **Primary result:** the published pruned checkpoint keeps, per pruned layer, the `k` conv filters of **lowest** output-channel L1 magnitude and removes the highest — inverted from standard L1 magnitude pruning. Evidence: (1) P0 (my per-filter L1, descending) vs the published ranking has **Spearman = -1.000000 on all 28 layers** (exact reversal); (2) published ranking lists low-L1 filters first, high-L1 last; (3) the reference's own `l1_imp_index` = per-filter sum(|w|) (== my P0) and `sorted_idx = np.argsort(scores)` is **ascending**, and the bit-exact materializer (M3-002) keeps `[:k]` → keeps lowest; (4) on **15/15** actually-pruned layers the kept set has lower mean L1 than the pruned set. Because the materializer is bit-exact to the artifact, this is a property of the **checkpoint itself**. Full write-up: `docs/m0_baseline_reproduction/l1_pruning_direction_finding.md`.
* **Acceptance / gate decision:** **no gate changed.** This reframes what the published L1/P0 baseline is. The project's own `p0_l1_magnitude` implements STANDARD L1 (highest-kept) and is correct (C6 + this validation) — so it will DIFFER from the published checkpoint. Whether the project's P0 should be standard L1 or reproduce the published inverted convention is a Gabriel decision to record before M3B/M4.
* **Failure or uncertainty:** the *interpretation* — intentional design vs. a direction bug in the reference — is NOT decided here; it is a question for Arshdeep (via Gabriel) and for `/auditar`. Do not act on this beyond surfacing it.
* **Notes:** data-free P0 on base weights + public-ranking comparison only; NO diagnostics, NO calibration slots, NO saliency on the L1 checkpoint. `git diff upstream-frozen -- audioldm_train/` empty.

### 2026-08-19 12:35 | AUDIT-NIGHT2 | Evidence-first audit pass (the 05:30 resume, run by hand after the VM slept)

* **Status:** completed. This is the safe audit pass the resume prompt scheduled for 05:30 Montevideo. The detached daemon (`artifacts/auto_resume/`, pid 5102) never fired: the Lightning Studio VM slept during the idle window and killed the process before the token window renewed (`resume.log` has only the "daemon armed" line; no "window renewed"; the VM later cold-booted 12:09 UTC). Gabriel re-ran it manually while present. **No code changed — no real bug found.**
* **Milestone / gate:** integrity audit only. No milestone advanced; every blocked item stays blocked (no GPU attached).
* **Git commit:** this commit.
* **Branch:** main.
* **Commands:** `.venv/bin/python scripts/research/run_research_tests.py --all`; `git diff upstream-frozen -- audioldm_train/`; `.venv/bin/python scripts/research/verify_l1_direction.py`.
* **GPU / runtime:** CPU only. No saliency computed on the real L1/pruned checkpoint; no GPU numbers produced.
* **Primary result:**
  * **Full research suite 11/11 modules PASS** (exit 0), re-confirmed. `git diff upstream-frozen -- audioldm_train/` = **0 lines**.
  * **(a) M3B-002 independently re-derived and CONFIRMED; refutation attempts failed.** Re-read both sides of the reference's own code: producer `layerwise_sorted_index_generation.py` (`l1_imp_index` = per-filter sum(|w|), `np.argsort(scores)` ascending → index 0 = lowest L1) and consumer `pruned_unet_dict_creation.py:118` (`out_idx = out_idx_full[:out_k]` → keeps the first = lowest-L1 filters). `verify_l1_direction.py` reproduces Spearman −1.000000 on all 28 layers and kept-set-lower-L1 on 15/15 pruned layers. The three refutation angles (argsort direction, keep-vs-prune semantics of the `[:k]` slice, normalization-by-max preserving order) all fail. Finding holds; **not acted on** (Gabriel/Arshdeep decision).
  * **(b) P0-P3 Taylor machinery re-derived correct.** `gates.py` gate `S_c=|g_c·∂L/∂g_c|` at g=1 is the standard first-order channel-Taylor criterion; `accumulate_taylor` zeroes grads per slot and means correctly; `assert_matched_budget` enforces §5 (P1 2B == P2/P3 B+B, audio==text); `keep_topk`/`p0_l1_magnitude` implement STANDARD L1 (highest-kept), i.e. deliberately the opposite convention to the published checkpoint — this is exactly the open M3B-002 decision, already flagged, not a bug. `layer_set.py` maps the 28 ranking keys to Conv2d paths and fails loudly on any out_channels mismatch.
  * **(c) M1 PEFT re-derived correct.** LoRA F5 factorised conv is exact: `conv2d(x, A_kernel, stride,pad,dil)` then a 1×1 `conv2d(·, B)` equals `conv2d(x, B@A, stride,pad,dil)` because the stride-1 1×1 preserves the strided conv's spatial grid (algebraic re-derivation). `merge`/`unmerge` use the identical `delta_weight()`, so the weight-space round trip is exact; `LoRALinear` forward matches its merged weight. B initialised to zero (adapter starts as identity). Eval findings F-eval-1..6 are sound and correctly caveated as non-scientific (real bugs in third-party `audioldm_eval`, worked around without patching upstream).
* **Acceptance / gate decision:** the night-run machinery (M1 PEFT, P0-P3/P1 Taylor, eval pipelines) passes an evidence-first read, not just its tests. No fix required. M3B-002 stays open for Gabriel's convention decision.
* **Failure or uncertainty:** none found in code. The only open items are decisions/inputs, not bugs: the M3B-002 P0-convention choice, and everything GPU-gated.
* **Notes:** the auto-resume daemon design has a demonstrated gap — `setsid` survives the Claude process dying but not the VM sleeping/cold-booting. Not re-armed: Gabriel is present and the CPU queue is exhausted. `git diff upstream-frozen -- audioldm_train/` empty.

### 2026-08-19 12:45 | DECISION-M3B-002 | P0 baseline adopts Arshdeep's published inverted-L1 convention

* **Status:** completed — resolves the open M3B-002 convention question with Gabriel's decision, implemented and verified. Supersedes the earlier "no gate changed, Gabriel's call" hold.
* **Milestone / gate:** RQ2 (L1/P0 baseline definition). Unblocks the P0 baseline direction for M3B/M4.
* **Git commit:** this commit.
* **Branch:** main.
* **Decision (Gabriel, 2026-08-19):** "use the published inverted convention iff the published pruning work is Arshdeep's; otherwise standard L1." Provenance check: `_external/PruningAudioLDM/README.md` = "Official implementation of our pruning framework" (`Arshdeep-Singh-Boparai/PruningAudioLDM`, arXiv 2607.13330); `l1_audioldm-m-full_p1.ckpt` + `sorted_indexes_dict.pkl` from Zenodo 21376822 (Arshdeep Singh, 2026-07-15), md5-verified. It IS Arshdeep's → **adopt the published (inverted) convention: P0 keeps the LOWEST-L1 filters.**
* **Change:** `research_pruning/taylor/saliency.py` adds `P0_CONVENTION = "published"` (default) and `p0_importance(convs, convention)` — `"published"` returns `-L1` so `keep_topk` keeps the low-L1 channels (reproducing the artifact); `"standard"` returns `+L1` (Li et al. 2017, retained for non-Arshdeep baselines only). `p0_l1_magnitude` (raw magnitudes) unchanged. Exported from `research_pruning/taylor/__init__.py`.
* **Command / acceptance:** `.venv/bin/python scripts/research/verify_p0_convention.py` → on the real base `(1,2,3,5)` U-Net, `keep_topk(p0_importance("published"), k)` reproduces the published kept-set **EXACTLY on 12/12 ranking-driven pruned layers**; `"standard"` is disjoint from it (12/12 where k≤full/2). Control-model unit test `tests/research/test_taylor_saliency.py::C8` PASS (published==lowest-L1, standard==highest-L1, opposites). Full Taylor suite now **C1–C8 PASS**.
* **Failure or uncertainty:** none in code. Still open for Arshdeep (informational, non-gating): confirm whether the inverted direction was intentional, for accurate paper wording. No saliency computed on the real L1/pruned checkpoint; no GPU work; `git diff upstream-frozen -- audioldm_train/` empty.
* **Notes:** this is a plan-affecting decision recorded per AGENTS.md — the L1 baseline the project compares P1/P2/P3 against is now defined as the published inverted-L1 kept-set. RQ3's inverted-starting-point caveat (finding write-up) stands.

### 2026-08-19 17:55 | DECISION-M3B-003 | Gate B amended onto the KEPT-set definition (resolves G1/G2) + P0-standard as secondary reference

* **Status:** completed — two Gabriel decisions recorded, implemented in the protocol draft, and verified against the real artifact geometry. Resolves the last non-GPU blocker for freezing `docs/pilot_protocol.md`.
* **Milestone / gate:** M3B Gate B definition; RQ2 baseline reporting. Master-plan amendment, recorded per AGENTS.md before any protocol freeze and before any saliency is inspected.
* **Git commit:** this commit.
* **Branch:** main.
* **Decision 1 (Gabriel, 2026-08-19) — Gate B, option (a).** The master plan states Gate B against the **prune-set** overlap (weighted `<= 0.80`, `>= 2` layers `<= 0.70`). At the `(1,2,3,1)` budget that object is confined to `[0.75, 1.0]` with chance at `0.80`, so condition 1 demanded "no more agreement than pure chance" and condition 2 was **mathematically impossible** (audit finding G1); the draft also carried both a kept-set and a prune-set definition in adjacent bullets (finding G2). **Amendment: the single overlap definition is the KEPT set, and the plan's two numerals transfer verbatim onto it** — Gate B PASS requires weighted kept-set overlap `<= 0.80` AND `>= 2` ranking-driven layers with kept-set overlap `<= 0.70`. Kept-set overlap spans the full `[0, 1]` with chance at `k/N = 0.20`, so the gate now fails only if audio and text agree on `>= 80 %` of kept channels. The prune-set number is reported for transparency via the exact identity `(N - 2k + |K_a ∩ K_t|)/p` but is **never** the gate. Options (b) midpoint recalibration and (c) chance-adjusted scale were offered and not taken; the chance-adjusted value `(obs - k/N)/(1 - k/N)` is still reported alongside.
* **Decision 2 (Gabriel, 2026-08-19) — P0-standard as a SECONDARY reference.** DECISION-M3B-002 makes P0 keep the LOWEST-L1 filters (reproducing Arshdeep's published artifact), while P1/P2/P3 keep the HIGHEST-saliency channels. A P1-vs-P0 margin therefore confounds criterion *direction* with criterion *quality*. **The run additionally computes and reports `p0_importance(convention="standard")` (keep-highest-L1) as a secondary reference point**, so RQ2 can separate "beats the published artifact" from "beats a competently-directed L1 criterion". The pre-registered primary baseline is unchanged (published/inverted); the gate decision uses the primary; both numbers are reported side by side. Cost: zero extra compute (same per-channel L1, opposite sign). A mandatory wording constraint is also pre-registered: comparisons are worded "vs the published L1 pruning artifact", never "vs standard L1 magnitude pruning".
* **Change:** `docs/pilot_protocol.md` M3B section rewritten — P0 convention pre-registered (was missing entirely: DECISION-M3B-002 had reached `claims_matrix.md` and the finding write-up but not the protocol), direction-asymmetry reporting constraint added, P0-standard secondary added, single kept-set overlap definition with formulas, amended Gate B with its derivation. `docs/claims_matrix.md` RQ2a updated.
* **Command / acceptance:** `.venv/bin/python scripts/research/verify_gate_b_geometry.py` → exit 0. Re-derived from the published ranking + pruned target shapes (structure only, no saliency): **12 ranking-driven layers of 28 ranked, every one N=960 / k=192 / p=768**, kept-set chance `0.2000` on all 12, prune-set pigeonhole floor `0.7500` on all 12 → the plan's `0.70` prune-set threshold is confirmed unreachable and the amended thresholds sit strictly between chance and identity.
* **GPU / runtime:** CPU only, seconds. No saliency computed, no checkpoint ranked, `l1_audioldm-m-full_p1.ckpt` not opened.
* **Failure or uncertainty:** none. Gate B is now well-posed but still **UNFROZEN and unevaluated** — freezing `pilot_protocol.md` additionally needs the GPU benchmark (`T_sal`/`T_fwd`) and Compute Gate CG. No overlap has been computed on real saliencies.
* **Notes:** `git diff upstream-frozen -- audioldm_train/` empty.

### 2026-08-19 18:05 | M3B-003 | Gate B kept-set overlap statistic implemented + control-tested — NO scientific result

* **Status:** completed (machinery only). Gate A had a statistic (`matched_null.py`, M3-001); Gate B had none — the amended definition is now executable.
* **Milestone / gate:** M3B. Implements DECISION-M3B-003. No gate evaluated.
* **Git commit:** this commit.
* **Branch:** main.
* **Change:** new `research_pruning/paired_modality/overlap.py` — `kept_set_overlap` (per-layer records: intersection, overlap, chance, chance-adjusted, prune-set overlap for reporting), `weighted_overlap`, `weighted_adjusted`, `evaluate_gate_b` (returns `GateBResult` with both pre-registered conditions and a `summary_lines()` report), and the pre-registered constants `GATE_B_WEIGHTED_MAX = 0.80`, `GATE_B_LAYER_MAX = 0.70`, `GATE_B_MIN_LAYERS = 2`. Exported from `research_pruning/paired_modality/__init__.py`. `weighted_overlap` is evaluated as `sum_l |K_a ∩ K_t|_l / sum_l k_l` — algebraically identical to `sum_l k_l·overlap_l / sum_l k_l` but exact (integer sums, one correctly-rounded division), because the gate compares with `<=` and the plan's numerals are exactly reachable (32/40 == 0.80); accumulating rounded products can land a boundary case just above the threshold.
* **Command / acceptance:** `.venv/bin/python tests/research/test_overlap_gate_b.py` → **O1–O6 PASS**. O1 IDENTITY (audio==text → overlap/adjusted 1.0 everywhere, Gate B FAILS — a criterion that cannot disagree must not pass a disagreement gate). O2 CHANCE (independent random saliencies at the real geometry N=960/k=192/12 layers → weighted 0.2170 vs chance 0.2000, adjusted +0.0213, PASS). O3 PRUNE-FLOOR (reported prune-set overlap matches brute-force set intersection exactly at 4 intersection levels; floor over all intersections == 0.75 == pigeonhole, so 0.70 is unreachable — finding G1 made executable). O4 GATE LOGIC (both conditions necessary: weighted 0.775 with only 1 layer `<= 0.70` FAILS; weighted 0.85 with 2 such layers FAILS; aggregation is k-weighted, 0.40 vs an unweighted 0.60). O5 BOUNDARY (weighted exactly 0.80 and per-layer exactly 0.70 both PASS — inclusive and FP-exact — and one notch worse flips to FAIL). O6 GUARDS (9 malformed inputs raise; and leaving the 3 positional seams in dilutes the gate — 3 ranked layers at 0.70 PASS alone but FAIL at 0.85 once seams that agree by construction are included, which is why the comparison is restricted to `ranking_driven_layers`).
* **GPU / runtime:** CPU only, seconds. Synthetic saliencies with hand-derived expected values throughout; no checkpoint loaded, no real saliency computed, the L1 checkpoint never opened.
* **Primary result:** none — this is machinery. **Full research suite 12/12 modules PASS.**
* **Failure or uncertainty:** the statistic is control-tested but has never been fed a real saliency; doing so is the M3B scientific run, still blocked on the protocol freeze (GPU benchmark + CG). P1 remains scientifically load-bearing and must pass `/auditar` before any real use.
* **Notes:** `git diff upstream-frozen -- audioldm_train/` empty.

### 2026-08-19 19:20 | M1-008 / FINDING | First real GPU session: two M1 defects found, benchmark still NOT measured

* **Status:** partially completed. GPU attached and verified; **F9 fixed**; **F10 open, needs a Gabriel decision**. `docs/compute_budget.md` is still 100% `TBD_MEASURED` — the benchmark has never completed, and no number was invented.
* **Milestone / gate:** M1 GPU acceptance / M0-005. Compute Gate CG stays unresolved.
* **Git commit:** this commit.
* **GPU / runtime:** Studio switched CPU -> **Tesla T4, 15360 MiB, driver 580.173.02** (`nvidia-smi`), after Gabriel topped the org balance to 10.00 credits. The earlier attempt had been refused for insufficient balance (see `docs/environment_report.md`).
* **Command:** `.venv/bin/python scripts/research/gpu_benchmark.py --steps 30 --batch 8 --out artifacts/m3_pilot/compute_budget_measured.json` — **failed twice, no JSON produced.**
* **F9 (FIXED) — LoRA adapters were created on CPU regardless of the base model's device.** `LoRALinear.__init__` / `LoRAConv2d.__init__` built `lora_A`/`lora_B` with bare `torch.empty`/`torch.zeros`, i.e. on the default device. Since PEFT is injected AFTER the checkpoint load (`docs/integration_notes.md` I4), the model is already on the GPU by then, so the first forward raised `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`. Fixed by constructing both adapters with `device=base.weight.device, dtype=base.weight.dtype`, so injection order no longer matters. `audioldm_peft/ema.py` was checked and is already device-safe (`p.detach().clone()` inherits the device); `state.py` already casts on load.
* **F10 (OPEN, needs decision) — gradient checkpointing is incompatible with PEFT's frozen base weights.** After F9, the forward succeeds and `loss.backward()` raises `RuntimeError: One of the differentiated Tensors does not require grad` at `audioldm_train/utilities/diffusion_util.py:161`. `CheckpointFunction.backward` calls `torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, ...)`; `ctx.input_params` are the block's own parameters, which PEFT sets to `requires_grad=False`. `allow_unused=True` covers *unused* tensors, not *non-requiring-grad* ones. **This is NOT GPU-specific — it reproduces on CPU.** Options: (a) run PEFT training with `use_checkpoint=False` (no upstream patch; higher VRAM, faster step — but then the budget must be measured in that same configuration); (b) a minimal, reviewable upstream patch filtering `input_params` by `requires_grad` before the `autograd.grad` call — this is the standard fix for this known upstream pattern, and `AGENTS.md` allows deliberate reviewed patches; (c) keep base params requiring grad and discard their gradients (wasteful). **Gabriel decides;** until then `git diff upstream-frozen -- audioldm_train/` stays empty.
* **Root cause of the miss — test-coverage gap:** `tests/research/test_peft_real_unet.py` (R6a-c) never performs a `backward`; it only wraps modules, counts parameters, inspects `requires_grad` and checks merge/unmerge. So no test ever ran an optimization step through the real U-Net's checkpointed blocks. A regression test that does exactly that is required before M1 GPU acceptance can be claimed.
* **Failure or uncertainty:** the benchmark has produced NO measured value. `docs/compute_budget.md` remains untouched and `TBD_MEASURED`; Compute Gate CG unresolved; M3 blocked. Do not quote any GPU number for this project — none exists yet.
* **Notes:** cost discipline (Gabriel, 2026-08-19): developing inside a billed GPU Studio is wrong when the GPU work is bursty. Future GPU work should be submitted with `lightning job run --studio ... --machine T4` while the Studio stays on free CPU. Both defects above are CPU-reproducible, so the fixes belong on CPU.

### 2026-08-19 19:35 | DECISION-F10 | PEFT + gradient checkpointing: adopt a minimal reviewable upstream patch

* **Status:** decision recorded; **NOT yet implemented** (deliberately — see below). No code changed by this entry.
* **Milestone / gate:** M1 GPU acceptance. F10 is what currently blocks a PEFT optimization step on the real U-Net (ledger M1-008).
* **Git commit:** this commit.
* **Decision (Gabriel, 2026-08-19):** take the **minimal, reviewable upstream patch**. In `CheckpointFunction.backward` (`audioldm_train/utilities/diffusion_util.py`), filter `ctx.input_params` by `requires_grad` before the `torch.autograd.grad` call, so frozen base weights are not passed as differentiation targets. Rejected alternatives: `use_checkpoint=False` (keeps upstream pristine but raises VRAM, risks OOM on a 16 GB T4, and would force the whole compute budget to be measured in that configuration) and keeping base params trainable to discard their grads (wasteful).
* **Consequence — the first deliberate upstream patch.** `git diff upstream-frozen -- audioldm_train/` has been empty for the whole project and will become non-empty. `AGENTS.md` permits this ("keep upstream AudioLDM patches minimal and reviewable"), and it must be recorded as a deviation, kept to the smallest possible diff, and reviewed. This entry is that record.
* **Deliberately NOT implemented in the GPU session.** The regression test that would validate it does not exist yet (`tests/research/test_peft_real_unet.py` performs no `backward` — the coverage gap that let F9/F10 through). Applying an unvalidated upstream patch and immediately benchmarking with it would produce numbers from unreviewed code that would have to be re-measured. Implementation + test belong on **CPU, at zero cost**.
* **Required order of work:** (1) implement the patch, (2) add a regression test that performs a real PEFT optimization step (forward + backward + optimizer) through the real checkpointed U-Net — the missing R6 case, (3) re-run the full research suite, (4) re-run `verify_l1_bitexact.py` and confirm the patch changes no numerical result, (5) **only then** submit `gpu_benchmark.py` as a GPU job, (6) populate `docs/compute_budget.md` with measured values, (7) resolve Compute Gate CG, (8) freeze `docs/pilot_protocol.md`.
* **Failure or uncertainty:** none decided beyond the above. `docs/compute_budget.md` remains 100% `TBD_MEASURED` — no GPU number exists for this project.
* **Notes:** cost discipline (Gabriel): the Studio returns to free CPU immediately after this entry; future GPU work goes through `lightning job run --studio ... --machine T4`, not by moving the Studio.

### 2026-08-19 20:05 | M1-009 | F10 implemented (first upstream patch) + the missing backward regression test; F9/F10 closed

* **Status:** completed and verified on CPU. Implements DECISION-F10 and closes the test-coverage gap that let F9 and F10 reach the first GPU session. **M1 GPU acceptance itself is still NOT claimed** — no optimization step has yet run on a GPU, and no GPU number exists.
* **Milestone / gate:** M1. Unblocks the benchmark; Compute Gate CG still unresolved.
* **Git commit:** this commit.
* **Branch:** main.
* **F10 reproduced on CPU first, as claimed.** `BasicTransformerBlock` count 16 on the real pruned U-Net, all with `checkpoint=True`; forward succeeds (2.3 s), `loss.backward()` raises `RuntimeError: One of the differentiated Tensors does not require grad`. **Root cause is broader than recorded in M1-008:** `attention.py:379` defaults `BasicTransformerBlock.checkpoint=True` and calls `checkpoint(...)` at lines 400/402 **independently of the U-Net's `use_checkpoint` flag** — which is absent from the config and therefore `False`. So the rejected option (a) (`use_checkpoint=False`) **would not have fixed F10 at all**; the transformer blocks checkpoint regardless. The chosen patch was the only workable option of the three.
* **THE UPSTREAM PATCH (project's first; recorded deviation).** `audioldm_train/utilities/diffusion_util.py`, `CheckpointFunction.backward`: differentiate w.r.t. `ctx.input_tensors + [p for p in ctx.input_params if p.requires_grad]`, then re-expand the result with `None` in the frozen slots so the returned tuple keeps the arity of `forward`'s argument list. `git diff upstream-frozen -- audioldm_train/` is now **1 file, 16 insertions, 2 deletions**, of which 8 lines are the explanatory comment — i.e. 8 functional lines. Nothing else under `audioldm_train/` is touched.
* **Command / acceptance:** `.venv/bin/python tests/research/test_peft_backward_real_unet.py` → **R7a–R7d PASS**.
  * **R7a STEP** — real pruned U-Net with REAL pretrained weights + PEFT, one full fwd+bwd+optimizer step: **284/284 `lora_B` adapters receive a NON-ZERO gradient** (so gradient really reaches through all 16 checkpointed transformer blocks), 0/284 `lora_A` non-zero (correct: `B` is zero-initialised so `dL/dA = 0` at step 0), **0 frozen base params received a gradient**, and after `opt.step()` **568/568 LoRA tensors changed while 0 frozen base tensors changed**.
  * **R7b CKPT-EQUIV** — gradients through the checkpointed path vs the non-checkpointed path over 568 tensors: **max|Δ| = 0.000e+00** (exactly equal), max|grad| = 2.077e-02. This is what proves the patch computes the *correct* gradients rather than merely not raising.
  * **R7c ZEROINIT** — documents and asserts the trap that would make the module vacuous (see finding below).
  * **R7d FACTORY** — F9 regression: a float64 base layer yields float64 adapters on the base's device, checked non-vacuously on CPU; the CUDA branch additionally asserts device equality when a GPU is present.
* **FINDING (zero-init gradient trap).** On a **freshly initialised** U-Net the final output conv `out.2` is `zero_module`-ed (`sum|W| = 0.0`), so **no gradient propagates backward past it** and only **1 of 284** adapters (`out.2.lora_B`) is exercised. With the real published weights that same tensor has `sum|W| = 174.9828` and gradient reaches all 284. Any backward-based test or diagnostic built on a randomly-initialised model is therefore **almost entirely vacuous**. R7c asserts both halves so the test cannot be silently weakened back to random init. This also explains why the earlier ad-hoc check showed "only 1 non-zero gradient" — an artifact of the fresh model, not a defect.
* **Regression coverage added:** `tests/research/test_peft_real_unet.py` (R6a-c) performs no `backward`; the new module does, on the real weights, through the real checkpointed blocks.
* **No numerical result changed by the patch:** `.venv/bin/python scripts/research/verify_l1_bitexact.py` → **R5 690/690 bit-identical, `bit_exact: true`**, L1 reference-mask sha256 `9a2593c2…` unchanged. **Full research suite 13/13 modules PASS.**
* **GPU / runtime:** CPU only. No GPU was attached for this unit; `docs/compute_budget.md` remains 100% `TBD_MEASURED` and no GPU number is quoted.
* **Failure or uncertainty:** the patch is validated for *correct gradients*, not yet for *GPU behaviour under memory pressure* — the benchmark and M1 GPU acceptance still have to run on real hardware. Per Gabriel's cost instruction that must be a `lightning job run --studio ... --machine T4` submission, with the Studio left on free CPU; whether such a job mounts `data/checkpoints/` and `data/dataset/` is still unverified and should be tested with a cheap **CPU** job first.

### 2026-08-19 20:35 | M0-007 | gpu_benchmark.py corrected: real weights + fail-fast preflight + staged batch escalation

* **Status:** completed on CPU (payload validated); **no measurement produced** — `docs/compute_budget.md` stays 100% `TBD_MEASURED`.
* **Milestone / gate:** M0-005 / Compute Gate CG prerequisite.
* **Git commit:** this commit.
* **Defect corrected (found by Gabriel's audit).** The script benchmarked a **fresh-init** U-Net and never loaded `l1_audioldm-m-full_p1.ckpt`. Combined with finding R7c (ledger M1-009) that invalidates TRAIN and SALIENCY in particular: on a fresh-init model `out.2` is `zero_module`-ed (`sum|W| = 0.0`), so gradient reaches only 1 of 284 adapters and the backward graph timed is almost empty. Any number it had produced would have been wrong.
* **Changes:** `build_pruned_unet(real_weights=True)` strict-loads the published pruned weights by the same path as `test_peft_backward_real_unet.py`; `assert_real_weights()` refuses to measure unless `sum|out.2.W| > 0` and is called before every timed section (train/saliency/forward). **Fail-fast preflight** on: expected commit (`--expect-commit`), clean working tree (measurements must be traceable; `--allow-dirty` to override), checkpoint present, CUDA device present and matching `--expect-gpu`, and the **R7a PEFT-backward gate** on CPU with real weights. **Staged execution** replaces the old blind `--batch 8`: `smoke` (batch=1, 2 warmups, 5 steps; asserts 284/284 adapters get gradients and 0 frozen params do, on CUDA) -> `escalate` (ladder 1,2,4,8; OOM is caught, not fatal, and the full sec/step + peak-VRAM + headroom curve is recorded together with `MAX_STABLE_BATCH`) -> `measure` (long run, with an OOM step-down so a paid job is not lost if the long run peaks higher than the short probe). Exact git commit, branch, dirty flag and the upstream-patch diffstat are recorded in the JSON, which is both written to `--out` and printed to stdout so it survives in the job log.
* **`--dry-run-cpu` added:** exercises the entire staged flow on the free CPU Studio, stubbing the CUDA-only calls. It forces `DRY_RUN: true` and `GPU_MODEL: "DRY-RUN-CPU (no measurement)"` into the JSON and **refuses `--out`**, so a flow check can never be mistaken for a benchmark or reach `compute_budget.md`.
* **Command / acceptance (all CPU, free):** guards verified individually — dirty tree refused; `--expect-commit` mismatch refused; missing checkpoint refused; **no CUDA refused with the message pointing at Lightning jobs**; real weights load with `sum|out.2.W| = 174.9828` and the guard passes; a fresh-init model is **refused** by `assert_real_weights`. Full staged flow dry-run: PREFLIGHT -> SMOKE (census `lora_B_nonzero_grad: 284/284`, `frozen_with_grad: 0`) -> ESCALATE (1, 2) -> MEASURE, producing every §7.2 key. Two self-inflicted bugs were caught by this CPU validation before any job was submitted: an infinite recursion in `_reset_peak` and a stray raw `torch.cuda.empty_cache()`, both from an over-broad text replacement.
* **GPU / runtime:** none. No GPU was attached; the Studio stayed on free CPU throughout.
* **Failure or uncertainty:** the script has still never run on CUDA. No §7.2 value exists. This benchmark, when it runs, does **not** constitute M1 GPU acceptance — that additionally requires a several-hundred-step run plus a resume test, also as a job.
* **Notes:** operating policy now frozen (Gabriel, 2026-08-19): all development, debugging, tests and preparation happen on the CPU Studio; the GPU is used **only** through Lightning Jobs; the interactive Studio is never switched to T4 again; and any GPU failure reproducible on CPU returns immediately to the CPU Studio — the T4 is not a debugger. The job-filesystem question is closed: Lightning Jobs snapshot the full Studio environment.

### 2026-08-19 20:50 | OPS-001 | Lightning Job execution model verified; first GPU job failed on a missing `cd`

* **Status:** operational unit. `gpu-benchmark-1` **FAILED** (cost 0.107 credits); `gpu-benchmark-2` relaunched with the fix and pending at the time of writing. **No measurement exists yet; `docs/compute_budget.md` stays 100% `TBD_MEASURED`.**
* **Milestone / gate:** M0-005 execution. No scientific gate touched.
* **Git commit:** this commit.
* **Policy being implemented (Gabriel, frozen 2026-08-19):** development, debugging, tests and preparation on the free CPU Studio; GPU **only** via Lightning Jobs; the interactive Studio is never switched to T4 again; any GPU failure reproducible on CPU returns immediately to CPU.
* **Failure and its cause.** `gpu-benchmark-1` died in 6 minutes with `.venv/bin/python: No such file or directory`. A job starts in `$HOME`, not in the repository. The `cd` had been present in the first draft of the command and was dropped while fixing the teamspace flags — an avoidable 0.107-credit loss, caused by changing a command and not re-checking the part that was already right.
* **Job execution model, verified by direct inspection rather than assumption.** A Job snapshots the entire Studio, and from the Studio both halves are readable for free: `/teamspace/jobs/<name>/snapshot/` is what the job saw and `/teamspace/jobs/<name>/artifacts/` is what it wrote (writes sync back, mirroring the home layout). The snapshot of `gpu-benchmark-1` contained `.venv/bin/python`, `data/checkpoints/l1_audioldm-m-full_p1.ckpt` (3 490 506 986 B), `data/dataset` (30 GB) and a `.git` whose `HEAD` was exactly `ce8815aa…`, the commit pinned by `--expect-commit`. **So checkpoints, dataset and git provenance are all present inside jobs** — the previously open question is closed, at zero cost, by reading the snapshot instead of running a probe job. Results do **not** appear in the Studio's own `artifacts/`, which is why `gpu_benchmark.py` prints its JSON to stdout as well as writing `--out`.
* **CLI facts worth not rediscovering:** `job run` takes `--teamspace general --org independentaudioresearch` (separate flags) while `job list` and `studio switch` take the combined `independentaudioresearch/general`, and `job list` has no `--org`. Job names must be unique per teamspace. **There is no `lightning job logs`** — read logs via the SDK `Job(...).logs`, which additionally refuses while the job is Pending or Running ("not supported yet"), so logs are only retrievable after a terminal state. `Job` also exposes `.status`, `.total_cost`, `.artifact_path`, `.snapshot_path`.
* **Relaunched as** `gpu-benchmark-2` with `cd audioldm-modality-swap-pruning && …`, same staged arguments and the same `--expect-commit ce8815aa…`.
* **Failure or uncertainty:** the benchmark has still never completed on CUDA. No §7.2 value exists; no GPU number may be quoted. Whatever it returns will not constitute M1 GPU acceptance.

### 2026-08-19 21:15 | M1-010 / FINDING | M1 GPU acceptance script written; it found F11 (snapshot aliased live training state)

* **Status:** completed on CPU. **F11 found and FIXED**; the acceptance script is validated by dry run but has **not yet run on a GPU**, so **M1 GPU acceptance is NOT claimed**.
* **Milestone / gate:** M1. No scientific gate touched.
* **Git commit:** this commit.
* **New script `scripts/research/m1_gpu_acceptance.py`.** Several hundred real PEFT steps on the real pruned `(1,2,3,1)` U-Net with the real published weights, plus an **exact resume test**: run A trains `0..N` snapshotting at `K`; run B rebuilds the model from scratch, loads the snapshot and replays `K..N` over a byte-identical batch sequence (per-step CPU generators seeded `seed+step`); the two final trainable-parameter sets must agree. Records steady-state sec/step, first/last-decile sec/step (drift), peak VRAM, the loss trace with a NaN guard, and `RESUME_MAX_DELTA`. Reuses the benchmark's verified preflight and `assert_real_weights` guard by explicit path import (`scripts/research` is not a package). Has the same `--dry-run-cpu` mode, which refuses `--out`.
* **FINDING F11 (real defect, FIXED) — `training_state_dict` returned a snapshot that aliased live training state.** `adaptation_state_dict` cloned its tensors, but the `optimizer` / `scheduler` / `ema` payloads were the raw `state_dict()` results, whose tensors ARE the live objects the next `step()` mutates in place. Holding the dict in memory while training continued therefore silently rewrote the "snapshot", and a resume from it replayed with the wrong Adam moments. **Measured on the real U-Net: resume diverged by max|Δ| = 1.570e-04** across 878 trainable tensors after replaying only 2 steps — the order of a full update, i.e. as if the optimizer state had never been restored. Confirmed in isolation on a toy `nn.Linear`: after 5 further steps the snapshot's own `exp_avg` had drifted by 0.461. **Fix:** `_deep_clone` detaches, moves to CPU and clones every tensor in the optimizer/scheduler/EMA payloads. **After the fix the same test gives max|Δ| = 0.000e+00 — exact resume.**
* **Why the existing S3 test could not catch it:** `test_state_ema_optimizer.py::S3` round-trips the state through `torch.save`/`torch.load`, and serialising silently breaks the aliasing. The bug only bites when the dict is held in memory — which is exactly what a resume test does, and what a training loop doing periodic in-memory checkpointing would do. **New regression `S4 SNAPSHOT-IMMUTABLE`** takes a snapshot, keeps training, and asserts the snapshot is unchanged (adapter, optimizer moments, `global_step`). S1–S4 PASS.
* **Command / acceptance:** `.venv/bin/python scripts/research/m1_gpu_acceptance.py --dry-run-cpu --steps 4 --snapshot-at 2` → RUN A 878 trainable tensors snapshotted, RUN B restored `global_step = 2`, **878 tensors compared, 0 key mismatches, RESUME max|Δ| = 0.000e+00 (EXACT)**, loss finite throughout. **Full research suite 13/13 modules PASS.**
* **Second bug caught by the same CPU dry run:** the script initially called `ema.update(unet)` while the EMA had been built on the holder, so every shadow lookup missed with `KeyError`. Fixed by passing the holder through `run_steps`. On a GPU this would have wasted a job.
* **GPU / runtime:** CPU only. No GPU number produced; `docs/compute_budget.md` stays 100% `TBD_MEASURED`.
* **Failure or uncertainty:** the acceptance script has never run on CUDA. Its `--resume-tol` default is 1e-6 and cuDNN determinism is forced, but GPU nondeterminism may make the resume delta non-zero there; the measured value is reported either way. **M1 GPU acceptance remains unclaimed.**

### 2026-08-19 21:20 | OPS-002 | Second GPU job refused by its own dirty-tree guard (working as designed)

* **Status:** `gpu-benchmark-2` **FAILED at preflight**, cost 0.091 credits (mostly machine startup). Still no measurement.
* **Cause:** the job snapshot carried `HEAD = ce8815aa…` — matching `--expect-commit` exactly — but a **dirty working tree**, because uncommitted documentation edits existed at launch time. The preflight guard refused rather than produce measurements that could not be traced to a commit. **The guard behaved correctly; the sequencing error was mine.**
* **Rule added to the recipe:** launch a job only from a **clean, committed, pushed** tree. A Lightning Job snapshots the Studio filesystem as-is, so any uncommitted edit travels with it.
* **Running total of avoidable job cost:** 0.107 (missing `cd`) + 0.091 (dirty tree) ≈ 0.198 credits. Both failures were preflight-fast by design, and both are now encoded as rules rather than as memories.

### 2026-08-19 21:55 | M0-005-RUN | FIRST REAL GPU MEASUREMENT — compute_budget.md populated; CG analysed, still unresolved

* **Status:** completed. **The project's first measured GPU numbers exist.** `docs/compute_budget.md` is no longer `TBD_MEASURED`. Compute Gate CG is **still UNRESOLVED** and now needs a Gabriel decision, not more measurement of the same kind.
* **Milestone / gate:** M0-005 executed; Compute Gate CG analysed against master plan §7.4.
* **Git commit:** this commit. **Measured at code commit `e6f50f48ce498652bf5e29652aeec3f17113047c`** (clean tree, verified by the job's own preflight).
* **Command:** Lightning **Job** `gpu-benchmark-3`, `--machine T4`, Studio left on free CPU. Status **Completed**, **cost 0.1372 credits**. Full invocation in `docs/compute_budget.md`.
* **Provenance:** raw JSON `artifacts/m3_pilot/compute_budget_measured.json` (md5 `12f8fef8577bfbff8b053e9ae90dd81e`), retrieved from `/teamspace/jobs/gpu-benchmark-3/artifacts/…` and also printed verbatim into the job log. Model: the real pruned `(1,2,3,1)` U-Net with the **real published weights**; the R7c guard was active.
* **Primary result (MEASURED, Tesla T4 14.562 GB, batch 8):** `TRAIN_SEC_PER_STEP` **1.672427**, `PEAK_TRAIN_VRAM_GB` **4.177**; `SALIENCY_SEC_PER_GRAD_EVAL_OR_BATCH` **1.596534**, `PEAK_SALIENCY_VRAM_GB` **4.152**; `FORWARD_SEC_PER_DIAGNOSTIC_BATCH` **0.465546**, `PEAK_FORWARD_VRAM_GB` **1.540**. `GEN_*` **NOT MEASURED** — the generation stack is not wired.
* **F9/F10 confirmed on CUDA, not just CPU:** the job's R7a preflight gate passed on the GPU — **284/284 LoRA adapters received non-zero gradients, 0 frozen base parameters received any**, and the smoke stage independently reproduced the same census at batch 1.
* **Batch escalation (MEASURED, no OOM at any rung):** batch 1/2/4/8 → 0.3897/0.5597/0.9586/1.6410 s per step and 1.048/1.497/2.388/4.177 GB peak. **`MAX_STABLE_BATCH = 8` is the largest TESTED, not the largest that fits** — the ladder ended by configuration with **10.385 GB still free**, and per-sample cost was still falling (0.3897 → 0.2051 s/sample, 1.90× better at batch 8). **The prior worry that a 16 GB T4 would be tight was wrong:** PEFT training peaks at 4.18 GB, ~29 % of the card. Extend the ladder to 16/32 before costing any long run.
* **Milestone projections (DERIVED via §7.3 from the measured values):** M1 smoke **0.232** GPU-h; M3A random-null **0.136**; M3B saliency (B=256, 4B batches) **0.454**; M2 **<0.05**. **Subtotal excluding M4/M5: 0.82 GPU-hours** — the entire RQ1/RQ2 diagnostic and saliency programme costs under one GPU-hour. **M5 recovery is 46.46 GPU-h PER MODEL** (`100000*Ttrain/3600`), i.e. **185.8 GPU-h for four criteria** — roughly 50× everything else combined. M4 **cannot be projected at all** without `Tgen`.
* **Cost rate (DERIVED, weak — do not rely on it):** 0.1372 credits for ~9 minutes wall implies **~0.91 credits/GPU-hour**, but provisioning dominates a 9-minute job so this is an **upper bound**. Even so, one M5 model would be ~42 credits against ~9.6 available.
* **Compute Gate CG assessed against §7.4:** condition 1 **SATISFIED** (Lightning + Jobs verified end to end); condition 2 **SATISFIED** (this measurement); condition 3 **CANNOT BE SATISFIED AS SPECIFIED** — M5 alone exceeds the balance by a wide margin and `Tgen` is missing; condition 4 **UNDECIDABLE** until 3 resolves. **CG therefore stays UNRESOLVED and is escalated to Gabriel as a schedule decision** — per §7.4 a failed compute gate is explicitly a schedule decision, not a scientific failure.
* **What the numbers actually mean scientifically:** the cost is **not** in the modality-swap or paired-saliency work. RQ1 and RQ2 (M1/M2/M3A/M3B) fit in under one GPU-hour. The entire budget problem is concentrated in **M5 recovery (RQ3)** and the unmeasured **M4 generation**. The core contribution is affordable now; the recovery arm and generation-based evaluation are what the gate is really about.
* **Failure or uncertainty:** `Tgen`, `GEN_BATCH_SIZE` and `PEAK_GENERATION_VRAM_GB` remain unmeasured. `MAX_STABLE_BATCH` is a tested maximum, not a true one. The credits/GPU-hour figure is a weak single-observation upper bound. **M1 GPU acceptance is still NOT claimed** — `m1_gpu_acceptance.py` has never run on CUDA.
* **Notes:** `docs/pilot_protocol.md` can now be frozen on the compute side (`T_sal` = 1.596534, `T_fwd` = 0.465546 are measured), but the freeze also depends on the CG decision above.

### 2026-08-20 00:10 | REVIEW-001 | External review accepted on two points; one point already implemented; full credit estimate

* **Status:** completed (documentation + estimate). No code changed. No scientific result.
* **Git commit:** this commit.
* **ACCEPTED — correction 1 (a real error in `docs/compute_budget.md`).** The file stated that 0.82 GPU-hours covered "the entire RQ1/RQ2 programme". It does not. `docs/claims_matrix.md` requires **"M3 Gate B + M4"** for RQ2a, so closing RQ2 needs pruning with P0-P3, **generating audio and evaluating FAD/KL/PANNs** — M4, whose cost was unknown because `Tgen` is unmeasured. The 0.82 figure covers the RQ1 diagnostics and the RQ2 saliency computation only. Corrected in place with an explicit CORRECTION block so the error stays visible rather than being quietly overwritten. This mattered: the wrong framing would have fed a Compute Gate CG decision.
* **ACCEPTED IN PRINCIPLE — correction 2 (P0 split), needs Gabriel's ruling.** The reviewer argues P0-published (Arshdeep's inverted artifact) and P0-L1 (conventional keep-highest) should be **two co-equal named baselines**, not primary + "secondary reference" as DECISION-M3B-003 currently has it. The reasoning is sound and is the sharper form of the confound already flagged: *if P1/P2/P3 beat the published artifact but not conventional L1, there is no pruning improvement.* **Cost note the reviewer did not raise:** P0 itself is data-free, but promoting P0-L1 to a full baseline adds one more model to every M4 generation and evaluation pass. Recorded as a proposed amendment to DECISION-M3B-002/003; **not applied unilaterally.**
* **NOT A CORRECTION — already implemented.** The reviewer asks that the RQ1 random null "control for `D_gen`, not simply compare against 20 random masks". `research_pruning/diagnostics/matched_null.py` already does exactly this: it fits `R_mod ~ f(D_gen)` (linear OLS) across the controls, evaluates the expected random `R_mod` at L1's observed `D_gen`, and defines `Delta_swap` as the residual, with a standardized residual in control-SD units. Implemented in M3-001, tested N1-N4. No change needed.
* **CREDIT RATE CORRECTED — ~0.89 cr/GPU-h, measured, not ~0.19 as quoted.** Final settled job costs: `gpu-benchmark-1` 0.1168, `gpu-benchmark-2` 0.1179, `gpu-benchmark-3` 0.1674 — **0.402 credits total**. Three jobs of different durations converge on **0.88-0.91 cr/GPU-h**, so this is the real on-demand T4 rate, not a provisioning artefact of one short job. **The 0.19 cr/h figure quoted to the project is 4.7× lower than observed** and should be checked against the published price; if it refers to interruptible instances that is a major lever. Also: settled costs came in 10-30 % **above** the values read while the jobs were still running.
* **FULL-EXPERIMENT ESTIMATE (added to `docs/compute_budget.md`).** Backbone measured (`Ttrain`, `Tfwd`, rate); **`Tgen` DERIVED** as `DDIM_steps × (Tfwd/batch) × 1.15` = 13.38 s/clip at S=200, from the measured 0.0582 s per-sample forward. At `Neval=200` and the six-model generation set: **A — RQ1+RQ2 complete (incl. generation and eval, no recovery): 11.8 GPU-h ≈ 10 credits**; **B — A + RQ3 with 2 recovered models: 125 GPU-h ≈ 111 credits**; **C — A + RQ3 with 5 models: 295 GPU-h ≈ 263 credits** (all +20 % contingency).
* **Levers, ranked:** (1) recovery step count — 100k is the plan's number, not an optimised one; at 20k, M5 falls from 46.46 to 9.3 GPU-h/model and B→~32, C→~64 credits; (2) interruptible instances (~half price), now safe to use because **exact resume is proven** (F11 + S4 + the acceptance resume test) so a preemption costs a restart rather than a run — combined with (1), scenario C lands near **~32 credits**; (3) batch 16/32; (4) S=50 for screening generation.
* **Failure or uncertainty:** `Tgen` is derived, not measured — every M4/M5-evaluation number above inherits that. Measuring it is the cheapest next job. The reviewer's own draft email contained two overstatements that were corrected before recording: it claimed the modality-swap diagnostics and saliency code "are running on GPU" (only the PEFT path has run on CUDA), and quoted the ~1.6 s figure as a saliency measurement when `time_saliency` is a generic all-parameter forward+backward **proxy** that does not exercise the channel-gate Taylor path.

### 2026-08-20 00:30 | OPS-003 | Public repository prepared for external collaborators

* **Status:** completed (documentation only). No code, no scientific result.
* **Git commit:** this commit.
* **Context:** the repository (`gbibbo/audioldm-modality-swap-pruning`, PUBLIC, 1.69 MiB tracked) is to be shared with Arshdeep Singh, whose published pruning artifact this project reproduces and questions.
* **RISK FOUND AND MITIGATED — an internal working document about a named collaborator was public.** `docs/collaboration/arshdeep_update_draft.md` was committed in `56b4aac` and contained a section headed *"Notes for Gabriel (not part of the message)"* discussing how to frame the pruning-direction finding to him, plus a claims table. Nothing in it is disparaging, but a collaborator invited to browse the repository could find a file named after himself containing notes on how to handle him — the opposite of the intended effect. **Moved to `artifacts/collaboration/` (gitignored) and removed from tracking.** **Caveat: it remains in the public git history at `56b4aac` and `6f31af2`.** Purging it needs a history rewrite and force-push, which `AGENTS.md` forbids without Gabriel's explicit request; flagged to him as his decision.
* **README rewritten** to tell the project's story plainly for an external reader: the modality-swap premise (trained on audio CLAP, used with text CLAP, both entering the same FiLM interface), the three research questions, an explicit built-vs-not-claimed status table, reproduction commands for every claim, and the frozen references. It states prominently that **no result exists for RQ1/RQ2/RQ3** and why (the protocol is a pre-registration and must be frozen first). The previous README was stale — it described `audioldm_peft/` and `research_pruning/` as empty skeletons and the upstream patch set as empty.
* **Framing decision on the pruning-direction finding.** The README reports the positive result first (690/690 bit-exact reproduction; pre-recovery confirmation; clean md5 provenance) and presents the direction as an **open question for the original authors**, explicitly stating "We are not claiming this is an error." Rationale: we genuinely do not know whether it is intentional, and a public repository should not headline a defect claim about a colleague's released work before they have had the chance to answer.
* **Language:** all tracked files are now English. The two remaining Spanish strings (`BORRADOR — PENDIENTE DE REVISIÓN` in `pilot_protocol.md` and a `BORRADOR` reference in this ledger) were translated. The local skill names `/auditar` and `/cerrar-hito` are unchanged — they are identifiers of Gabriel's own tooling, not prose.
* **Verified:** all 11 relative links in the README resolve to existing files; no checkpoints, datasets, audio or credentials are tracked.

### 2026-08-20 00:45 | AUDIT-COMM-001 | Evidence-first audit of the four claims Gabriel intends to send to Arshdeep

* **Status:** completed. **Three claims CONFIRMED by re-execution; one statement in the draft asserts a decision the ledger records as NOT taken.** One reproducibility gap found and closed.
* **Git commit:** this commit. Audited at `f730b1c67c08eb53c70f62aa9488ef607f108263`, working tree clean. Artifact md5s re-verified in-session: base `46bad9f1…`, published pruned `2666e6fc…`, ranking pkl `a4cd11ff…`.
* **Method:** every number re-derived by running code in this session. No claim accepted from prose in `docs/` — that was the explicit ask.

**C1 — "the released L1 checkpoint retains the lowest-L1 filters" — OBSERVED EVIDENCE, CONFIRMED.** `scripts/research/verify_l1_direction.py` re-run: on the **15 layers that are actually pruned**, the kept set has lower mean L1 than the removed set on **15/15**.

**C2 — "exactly reversed across all 28 ranked layers, Spearman = −1.0" — OBSERVED EVIDENCE, CONFIRMED.** Same run: 28 layers, **mean Spearman −1.000000 and min −1.000000**, so it is every layer, not an average over a mixed set. *Precision note for the message:* 28 is the count of layers in the published ranking file; only 15 of them actually change channel count. Both numbers are correct but describe different objects, and stating both pre-empts an obvious question.

**C3 — "four tensors where the public reconstruction logic differs, yet we reproduce the artifact exactly" — OBSERVED EVIDENCE, CONFIRMED, and a reproducibility gap CLOSED.** The 686/690 figure existed **only as prose** in `AUDIT-M3-001`; no script re-derived it. Added `scripts/research/verify_reference_divergence.py`, which re-derives it from the artifacts and **names** the divergent tensors: `input_blocks.10.0.in_layers.2.weight`, `output_blocks.0.0.in_layers.2.weight`, `output_blocks.1.0.in_layers.2.weight`, `output_blocks.2.0.in_layers.2.bias` — **686/690 identical, 4 divergent**; our corrected map gives **690/690, 0 divergent** (independently re-confirmed by re-running `verify_l1_bitexact.py`). **Scope caveat now stated in the script:** `_REFERENCE_LAYER_MAP` is *our reading* of the reference channel mapping, not the upstream script executed verbatim, so the defensible claim is "the public reconstruction logic, as we implement it, diverges on these four tensors". This matters because the message offers to send details — the offer is now backed by something runnable.

**C4a — "I would keep the released L1 artifact and conventional L1 as separate baselines" — NOT A RECORDED DECISION. This is the audit's main finding.** The ledger currently records the opposite: `DECISION-M3B-002` states `"standard"` L1 "must not be used for RQ2"; `DECISION-M3B-003` decision 2 makes P0-standard a **secondary reference** with "the pre-registered primary baseline unchanged"; and `REVIEW-001` records the co-equal-baselines proposal as **"ACCEPTED IN PRINCIPLE — needs Gabriel's ruling … not applied unilaterally."** Sending the sentence as written would commit the project externally to a design its own pre-registration does not yet carry. **The science is not in question — the split is the better design and removes a real confound — the problem is provenance order.** Resolution required before sending: either record the amendment (a new DECISION entry superseding M3B-002/003 on this point, plus the `pilot_protocol.md` M3B section) and then send, or soften the sentence to an intention.

**C4b — "do you still have the fully finetuned (1,2,3,1) checkpoint?" — WELL FOUNDED.** Re-verified in-session: `gh release list` returns **no releases on either repository**. The premise that such a checkpoint exists is supported by the reference README, which documents step **"4. Finetuning — Finetune the pruned AudioLDM-M-Full model using the AudioCaps training dataset"** and evaluates semantics "before and after pruning and finetuning". So asking whether he still has it is reasonable rather than speculative.

* **Hostile-reviewer pass.** Attempted refutations that failed: (i) that Spearman −1 might be a mean over a mixed set — refuted, the minimum is also −1.000000; (ii) that "four tensors" might be an artefact of counting our corrections rather than measured divergence — refuted, the divergence is now measured directly against the released tensors and the count matches the correction count; (iii) that the kept-set-lower-L1 result might hold only on ranking-driven layers — it holds on 15/15 actually-pruned layers, which is the superset. Surviving caveat: C3 depends on our reconstruction of the reference mapping (stated above).
* **Failure or uncertainty:** none in C1–C3. C4a is a provenance-order problem requiring Gabriel's ruling, not a measurement problem.

### 2026-08-20 00:50 | DECISION-CG-001 | Compute Gate CG RESOLVED by Gabriel: descoped core + S=50 screening; M5 probe; 2-credit reserve; interruptible trial

* **Status:** completed (gate decision).
* **Milestone / gate:** Compute Gate CG (master plan §7.4) — **RESOLVED**.
* **Git commit:** this commit.
* **Branch:** main.
* **How the decision was taken:** Gabriel, interactively, 2026-08-20 00:41–00:45 Montevideo, via four explicit structured questions answered before leaving the session in autonomous mode. Verbatim scope of each answer recorded below.
* **Decision 1 — scope: "Núcleo + screening S=50".** Authorized for immediate execution: measure `Tgen` for real (`--with-generation`), M1 GPU acceptance, M2 paired diagnostics, M3A random-null, M3B saliency (P0–P3), pruning materialization, and **screening generation at DDIM S=50** with FAD/KL/PANNs evaluation. The S=200 confirmatory generation is **deferred until a top-up**, not cancelled. Projected ~4–5 credits.
* **Decision 2 — M5 convergence probe authorized at ~2k steps** (~0.9 GPU-h ≈ 0.8 cr): produce the loss-vs-steps curve that informs the recovery step-count lever (100k vs ~20k). **Full M5 recovery is NOT authorized** by this decision.
* **Decision 3 — hard credit reserve: 2.0 credits untouched.** Spendable ≈ 7.6 of the ~9.6 balance. Settled-vs-live cost bias (10–30 % above live readings) is the stated reason.
* **Decision 4 — interruptible instances: trial first.** CLI supports `--interruptible` (verified in `lightning job run --help` this session). One cheap job first to observe real rate and preemption semantics; if sound, long jobs go interruptible. Exact resume is proven (F11 fix + S4 + acceptance resume test), so preemption costs a restart, not a run.
* **§7.4 condition-by-condition:** (1) SATISFIED — Lightning Jobs end-to-end verified. (2) SATISFIED — measured `Ttrain`/`Tsal`/`Tfwd` in `docs/compute_budget.md`. (3) **RESOLVED BY DESCOPING** — the core RQ1+RQ2 programme with S=50 screening fits the current balance; the RQ3 recovery arm and the S=200 confirmatory are explicitly deferred pending top-up and the probe-informed step count. (4) SATISFIED under the descoped plan — core completion projected well before the 2026-09-05 writing start.
* **Consequences:** M3 blocking gates now read: (1) GPU benchmark DONE, (2) CG RESOLVED (this entry), (3) pilot-protocol freeze **UNBLOCKED — must land in a commit before any saliency result is inspected**, (4) disjoint validation split DONE (`configs/research/val_split_disjoint.json`).
* **P0 handling at screening (superset, no new ruling):** both `P0-published` (inverted, per DECISION-M3B-002) and `P0-L1` (`standard` convention) will be materialized and generated at S=50. This collects the superset the REVIEW-001 co-equal-baselines proposal needs while leaving the primary/secondary labeling decision open — DECISION-M3B-003's wording constraint ("vs the published L1 pruning artifact", never "vs standard L1") unchanged. Marginal cost ≈ 0.19 GPU-h.
* **Failure or uncertainty:** `Tgen` is still derived, not measured; the ~4–5-credit projection inherits that until the `Tgen` job lands. Budget checkpoints against the 2.0-credit reserve are mandatory before each job launch.

### 2026-08-20 00:53 | M3B-FREEZE | Pilot protocol FROZEN; calibration slot manifest materialized and hashed

* **Status:** completed (pre-registration). No scientific result; no model or checkpoint opened.
* **Milestone / gate:** M3 freeze gate (AGENTS.md; master plan §5/§6) — **CLOSED**.
* **Git commit:** this commit. Freeze commit references itself; code commit at freeze `42f015d`, suite 13/13 PASS.
* **Both freeze prerequisites satisfied:** (a) measured `T_sal`/`T_fwd` (`M0-005-RUN`), (b) Compute Gate CG resolved (`DECISION-CG-001`).
* **What was frozen.** `docs/pilot_protocol.md` status DRAFT → **FROZEN (2026-08-20 00:53 America/Montevideo, 03:53 UTC)**. All `(PROPOSAL)` numerals are now pre-registered decisions; the two DECISION-M3B-002/003 conventions and the amended Gate B were already in the draft and are unchanged.
* **Calibration manifest materialized.** New `scripts/research/build_calibration_manifest.py` (pure CPU / deterministic; reads only the train JSON + seeded RNGs, never opens a model, the L1 ckpt, or a GPU) writes `configs/research/calibration_manifest.json` (tracked, 146 058 bytes) — **256 wav ids in seeded-permutation draw order, first-caption per wav, and every per-slot timestep** (`t_paired[K]`, `t_p1[K][2]`) drawn stratum-wise from the K=5 equal-width strata. **sha256 `8d7de0659554385389d3d71d349037d39c39e5842a7488e85037c060532b2d80`**, verified byte-identical across two independent runs. Per-stratum coverage exactly 256 (paired) / 512 (P1 units); all 256 wavs unique and drawn from train (disjoint from test/val by construction). 25 spot-checked wavs resolve on disk under `data/dataset/audioset/`.
* **Recipe executed verbatim from the frozen protocol:** E=256, K=5 strata `[0,200)…[800,1000)`, first-caption rule (`dict.setdefault`), master seed 20260818 (example permutation), timestep generator sub-seeded `MASTER_SEED+1`, consumed example-major then stratum. P1 gets 2 timesteps/(example,stratum), P2/P3 get 1 shared audio+text timestep — matched budget `2*E*K = 2560` grad-evals per criterion; P0 data-free.
* **Affordability re-checked with the measured `T_sal`.** Est. per-slot grad-eval on the base `(1,2,3,5)` U-Net ≈ 0.57 s (measured pruned proxy 0.1996 s/slot × 415.955/145.674), total 5120 evals ≈ **0.81 GPU-h ≈ 0.72 credits** — within the descoped budget, so pre-registered E=256 stands unmodified (not shrunk post hoc to save credits).
* **Acceptance / gate decision:** the freeze gate that blocked inspecting any saliency/diagnostic on the real base/L1 checkpoint is now CLOSED. **P1 still requires an evidence-first `/auditar` of the Taylor implementation + real loss closures before any saliency number is treated as a result** (stated in the freeze banner) — freezing does not waive it.
* **Failure or uncertainty:** the per-slot cost is estimated from a pruned-model proxy, not measured on the base model with real conditioning; the real saliency job will measure it. No number here depends on a GPU.
