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
