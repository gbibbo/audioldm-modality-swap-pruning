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
  * The active `cloudspace` env is Python 3.12.11 and lacks every AudioLDM dependency except torch and pytorch-lightning. Upstream specifies Python 3.10 and pins `transformers==4.30.2`. No environment has been created; this decision is recorded in `docs/m0_baseline_reproduction/environment_report.md` and must be made before any model code runs.
* **Deviations from the master plan (recorded as required by AGENTS.md):**
  * Small text provenance documents live in `docs/m0_baseline_reproduction/` so they are version-controlled; `artifacts/m0_baseline_reproduction/` is gitignored and holds only raw logs and binaries. The master plan names only the `artifacts/` path.
  * The upstream root `README.md` was moved verbatim to `UPSTREAM_README.md` and a project `README.md` written in its place. No file under `audioldm_train/` was touched.
  * `PruningAudioLDM` history is preserved as the branch `pruning-reference-frozen` rather than only as a recorded SHA, so the reference survives upstream deletion. It is not merged into `main`.
* **Notes:** `git diff upstream-frozen HEAD -- audioldm_train/` is empty at this commit. No M1 scaffold was written or reconstructed.
