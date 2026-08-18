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
  7. **Pilot protocol drafted** (`docs/pilot_protocol.md`) as reasoned proposals (B=256, K=5 timestep strata, bootstrap unit/seed, prune-tail and weighted-overlap definitions), carrying both M2 traps; marked BORRADOR, Freeze fields left blank.
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
