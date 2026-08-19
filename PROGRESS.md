# AudioLDM Modality-Swap Pruning - Progress

> Living state document. Keep this compact. The SessionStart hook injects only the bounded state block below. Detailed experimental provenance belongs in `docs/experiment_ledger.md` and milestone artifacts.

## CURRENT STATE

* **Resume point: `docs/HANDOFF.md`.** A new session should read that file first; it is self-contained and does not depend on any chat history.
* **⚠ HIGH-SEVERITY FINDING (2026-08-19, M3B-002) — read `docs/m0_baseline_reproduction/l1_pruning_direction_finding.md`.** The published PruningAudioLDM L1 checkpoint keeps the **lowest**-magnitude conv filters per pruned layer (inverted from standard L1), verified 4 ways incl. the reference's own code (`np.argsort` ascending + bit-exact materializer keeps `[:k]`) and 15/15 pruned layers; Spearman -1 vs our P0 on all 28 layers. **A property of the artifact itself.** Affects the RQ2 L1/P0 baseline and RQ3 recovery starting point. **No gate changed** — reproduce with `scripts/research/verify_l1_direction.py`; needs `/auditar` + a Gabriel/Arshdeep decision on whether it is intentional and which P0 convention the project uses.
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
* **M3A random-null + Gate-A statistic COMPLETE (M3-001), materializer BIT-EXACT (M3-002), CPU, NO diagnostic.** `research_pruning/diagnostics/random_masks.py` + `matched_null.py`. Per-layer k from the pruned `[1,2,3,1]` target shapes (the pkl holds permutations, not counts). **M3-002: the public reference script reproduces only 686/690 tensors of the published L1 ckpt; `LAYER_MAP` was corrected (2 positional-out seams removed, `input_blocks.10.in` identity-input, ranked-bias override for `output_blocks.2.0.in_layers.2.bias`) so `materialize(base, L1 ranking) == l1_audioldm-m-full_p1.ckpt` 690/690 bit-exact** (R5). The published artifact is internally inconsistent at its seams (documented; question for Arshdeep). Random masks use the same materializer → differ from L1 only in the **12 ranking-driven layers** (of 15 selectors; 3 are positional). 20 pre-registered masks (seeds 20260818..37); random-mask-set sha256 `3e6666bc…`, L1 ref sha256 `9a2593c2…` (full-ranking fingerprints). `matched_null.py`: linear OLS fit R_mod~D_gen, `Delta_swap`, standardized residual, bootstrap over wavs+masks (**unit=wav, raises on repeats**). Tests **R1..R5** and **N1..N4 PASS**; Gate A can't pass by construction. **L1 ckpt opened ONLY for R5 tensor-equality; no D_gen/D_mod/R_mod on it.** Evidence: `artifacts/m3_pilot/`.
* **Disjoint validation split DEFINED (M0 9.1 resolved).** `configs/research/val_split_disjoint.json` = upstream AudioCaps val (495 items), proven disjoint by wav id from test (0) and train (0); sha256 `e540146d…`. `dataset_root.json` NOT modified. **Note:** `vae_mel_16k_64bins.ckpt` is NOT the AudioLDM-M-Full VAE (204/398 tensors differ) — use the embedded first_stage; recorded in `dataset_manifest.md`.
* **`docs/pilot_protocol.md` DRAFTED (not frozen).** Reasoned proposals (B=256, K=5 timestep strata, bootstrap unit/seed, prune-tail + weighted-overlap definitions), both M2 traps carried in; Freeze fields left blank pending review + GPU benchmark.
* **M1 PEFT scaffold ADOPTED + audited-defects fixed (M1-005), CPU dummy-model tests PASS.** `audioldm_peft/` is now a working package (was a skeleton): LoRA Linear/Conv2d (F5 factorised conv, merge/unmerge bit-exact), injector + order-safe `freeze_for_peft` + `assert_peft_ready` guard (F4), unconditional aux-trainable counting (F3), explicit `train_layernorm_affine` with separate `layernorm_affine` reporting (F2), full-resume `training_state_dict`/`load_training_state_dict` (F7), trainable-only EMA, optimizer groups. Tests `tests/research/{test_lora_layers,test_injector,test_state_ema_optimizer}.py` **L1–L3, J1–J4, S1–S3 all PASS** via stdlib runner `scripts/research/run_research_tests.py` (F1: pytest absent, no pin relaxed). **F6 DONE (M1-006):** `test_peft_real_unet.py` R6a/R6b/R6c PASS on the actual `(1,2,3,1)` U-Net — 284 modules wrapped (185L+99C), LoRA 3,718,784 / bias 108,680 / GroupNorm 48,768 / LayerNorm 0, trainable 3,876,232 of 149,392,648; merge/unmerge max|Δ| 1.0e-7. **F8 DONE (M1-007):** `audioldm_peft/integrate.py` (`setup_peft`, `build_peft_optimizer`, `build_trainable_only_ema`, `peft_config_from_yaml`) + `test_peft_integration.py` I1–I4 PASS (incl. post-load-order proof) — CPU hooks only, `audioldm_train/` NOT patched (diff empty). **M1 CPU ACCEPTANCE COMPLETE: full suite 17/17 checks across 5 modules; all 8 audit defects F1–F8 addressed.** Only M1 **GPU acceptance** (several hundred real steps, VRAM, sec/step, resume) remains — blocked on GPU + Compute Gate CG. `research_pruning/{taylor,paired_modality}/` remain skeletons. `git diff upstream-frozen -- audioldm_train/` still empty.
* **M0 eval pipelines EXERCISED end-to-end (M0-006), CPU, pipeline-smoke only.** FAD/KL (`audioldm_eval` 0.0.5) and PANNs top-10 both run on real AudioCaps clips; invocations + findings in `docs/m0_baseline_reproduction/eval_pipeline_closure.md`. PANNs top-10 fully works (20/20 clips, coherent labels). FAD/KL: resolved deps (PANNs Cnn14 16k `ckpt/Cnn14_16k_mAP=0.438.pth` from Zenodo, CPU-sanitised; VGGish via torch.hub) and 4 library findings — **F-eval-3: audioldm_eval's VGGish FAD is unusable as-is** (sqrtm imaginary > tol, and `eval.py` crashes on the error sentinel); worked around to NaN so KL/IS/FID run; a real eval must use standard real-part FAD. Values are non-scientific (arbitrary disjoint subsets). Scripts `scripts/research/{fad_kl_smoke,panns_topk}.py`; checkpoints/CSV/folders gitignored.
* **GPU benchmark PREPARED (M0-005), write-only, NOT run.** `scripts/research/gpu_benchmark.py` measures every §7.2 variable on the real pruned U-Net (reusing the tested M1 setup) and **refuses without CUDA**; `docs/compute_budget.md` has a populate-pointer. No GPU numbers invented.
* **P0–P3 pruning-criteria MACHINERY COMPLETE (M3B-000), CPU, control-tested, NO scientific result.** `research_pruning/taylor/` and `research_pruning/paired_modality/` moved from skeletons to a working criteria package: channel-gate first-order Taylor saliency (`gates.py` `ChannelGate` g_c=1, `attach_gates`), `accumulate_taylor` (S_c=mean|g_c·∂L/∂g_c|), within-layer normalization (sum/max/l2), P0 L1 magnitude, P2 mean / P3 max combine, `assert_matched_budget` (§5: P1 2B == P2/P3 B+B), and `compute_criteria` orchestration sharing S_a/S_t. Tests `tests/research/test_taylor_saliency.py` **C1–C7 PASS** on control models. **Prunable layer set now VERIFIED on the real U-Net (M3B-001):** `research_pruning/taylor/layer_set.py` maps the 28 public L1 ranking keys to Conv2d modules of the base `(1,2,3,5)` U-Net (widths 384/576/960), all out_channels matching; `test_prunable_layer_set.py` V1/V2 PASS — gates attach with **bit-identical** output (max|Δ|=0.0). **Still NOT done (needs GPU/frozen protocol + `/auditar`):** the slot/timestep construction, real audio/text loss closures, and any actual saliency computation on the base model — no P0-P3 ranking has been computed on the real model. **P1 is scientifically load-bearing.** Full research suite **10/10 modules PASS**. `git diff upstream-frozen -- audioldm_train/` still empty.
* No GPU is attached. `docs/compute_budget.md` is entirely unmeasured, Compute Gate CG is unresolved, and M3 stays blocked.

## OPEN ITEMS

1. **Request the recovered full-FT `(1,2,3,1)` checkpoint from Arshdeep today** — the public search is finished and the artifact is proven absent. Also ask him to confirm the pre-recovery reading of `l1_audioldm-m-full_p1.ckpt`, and (new, AUDIT-M3-001) whether the seam conventions found in the published pruned checkpoint are intentional: 4 tensors deviate from his public script, and the artifact is internally inconsistent at `output_blocks.0/1` (consumer selects by ranking, producer outputs positional channels) and at `output_blocks.2.0.in_layers.2` (weight positional, bias ranked).
2. **M1 GPU acceptance (only remaining M1 item, blocked on GPU + CG):** apply the minimal upstream patch (`setup_peft`/`build_peft_optimizer`/EMA/resume per `docs/integration_notes.md`), then run several hundred real optimization steps and record VRAM, sec/step and a resume test in `docs/compute_budget.md`. M1 CPU acceptance is COMPLETE (M1-005/006/007; F1–F8 all addressed).
4. Finish the fetch of `checkpoints.tar` / `dataset.tar` and extract into `data/checkpoints/` and `data/dataset/`; then run upstream `tests/validate_dataset_checkpoint.py`.
5. ~~Verify the FAD/KL pipeline (`audioldm_eval`) and reproduce the PANNs top-k semantic pipeline.~~ **DONE (M0-006, pipeline-smoke).** Both run end-to-end on CPU. **Carry into the eval protocol:** replace audioldm_eval's broken VGGish FAD with a standard real-part FAD (F-eval-3), keep `--fresh` cache discipline (F-eval-2), and use the CPU-sanitised Cnn14 (F-eval-1). PANNs top-k ready for M4/M5 semantic analysis.
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
* M3A random-mask tests (CPU): `.venv/bin/python tests/research/test_random_masks.py`
* M3A matched-null tests (synthetic): `.venv/bin/python tests/research/test_matched_null.py`
* Persist random-null record (seeds+sha256): `.venv/bin/python scripts/research/build_random_null.py`
* R5 bit-exact check vs published L1 ckpt (equality only): `.venv/bin/python scripts/research/verify_l1_bitexact.py`
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

### 2026-08-19 | Autonomous night run: M1 scaffold adopted, audit defects F2–F5/F7 fixed

* Gabriel authorised an autonomous ~8h run (00:30→08:30 Montevideo) across the 05:28 token-window renewal; a detached resume daemon (`artifacts/auto_resume/`, gitignored) relaunches headless `claude` at 05:30 to continue.
* Adopted the recovered, audited M1 PEFT scaffold into `audioldm_peft/` and fixed audit defects **F2** (LayerNorm no longer half-trained; explicit flag + separate reporting), **F3** (unconditional counting), **F4** (order-safe freeze + guard), **F5** (factorised LoRA conv), **F7** (full resume state). Added stdlib test runner (**F1**).
* CPU dummy-model tests **L1–L3, J1–J4, S1–S3 all PASS** (**M1-005**). Then **F6** real pruned-U-Net tests R6a–c PASS (**M1-006**) and **F8** integration hooks I1–I4 PASS (**M1-007**). **M1 CPU acceptance COMPLETE** — full suite 17/17 across 5 modules; all 8 audit defects F1–F8 addressed. Only M1 GPU acceptance remains (blocked). No GPU work; `git diff upstream-frozen -- audioldm_train/` still empty.

### 2026-08-19 | Autonomous night run: M0 eval pipelines + GPU benchmark prep

* Prepared the §7.2 GPU benchmark (`scripts/research/gpu_benchmark.py`, **M0-005**) — write-only, refuses without CUDA; `compute_budget.md` unchanged (no numbers invented).
* Exercised both M0 evaluation pipelines end-to-end on CPU (**M0-006**): PANNs top-10 fully reproduced (20/20 clips, coherent labels); FAD/KL runs after resolving the PANNs Cnn14 16k checkpoint dependency + CPU sanitize + a path-keyed cache trap, and working around a genuine `audioldm_eval` FAD bug (documented, F-eval-3). Findings + invocations in `docs/m0_baseline_reproduction/eval_pipeline_closure.md`.
* FAD/KL numeric KL/IS/FID values captured in a follow-up commit once the CPU run finishes. `git diff upstream-frozen -- audioldm_train/` still empty.


### 2026-08-19 | Evidence-first audit pass (05:30 resume run by hand; VM had slept)

* The detached auto-resume daemon (pid 5102) never fired: the Lightning Studio VM slept during the idle window and killed it before the 05:28 token-window renewed (`resume.log` shows only "daemon armed"; the VM cold-booted 12:09 UTC). `setsid` survives the Claude process dying but NOT the VM sleeping — a real gap in the resume design. Gabriel re-ran the pass manually while present.
* Ran the safe pass the resume prompt scheduled: **full research suite 11/11 modules PASS** (exit 0); `git diff upstream-frozen -- audioldm_train/` = 0 lines; evidence-first `/auditar` of the night's machinery. **No bug found; no code changed.**
* **M3B-002 independently re-derived and CONFIRMED** (both sides of the reference code + `verify_l1_direction.py`: Spearman −1 on 28 layers, 15/15 kept-set-lower-L1). Refutation attempts failed. **Not acted on** — the P0-convention choice is Gabriel's. P0-P3 Taylor machinery, LoRA F5 factorised conv (exact), and eval findings F-eval-1..6 all re-derived correct. See ledger AUDIT-NIGHT2.
* Everything still open is a decision or GPU-gated, not a bug. Daemon NOT re-armed (Gabriel present, CPU queue exhausted).
