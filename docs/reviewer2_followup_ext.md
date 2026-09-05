# REVIEWER2-FOLLOWUP-EXT — frozen protocol: the 2×2 fine-tuning control (items 1 + 2 of the round-2 review)

**Status: FROZEN BEFORE ANY OUTPUT.** Nothing in this file may be edited after the first training step or the first
WAV of any job below exists. The sha256 sidecar `docs/reviewer2_followup_ext.md.sha256` is committed together with
this file and before launch.

## 0. Authorization and scope

* **Trigger.** The round-2 review of Draft 13 (`docs/review/2026-09-05_review_round2_methodological_response.md`,
  Accept 4/5) asked for (item 1) the symmetric control of E3 — a 20 000-step fine-tune of the pruned checkpoint at
  **10.24 s** — and (item 2) the reduced-scale **2×2** (dense and pruned, each trained 20 000 steps at 3.84 and at
  10.24 s), the paired design the paper has been lamenting it lacks.
* **Gabriel, 2026-09-05 18:06 MVD:** "tenemos 20 créditos exactamente. Autorizo los experimentos completos. Corre
  todo, si es posible en paralelo." → authorization to launch the full 2×2 with a **total ceiling of 20 cr** and a
  preference for parallel execution. Recorded in `docs/experiment_ledger.md` (REVIEWER2-FOLLOWUP-EXT) and
  `docs/compute_budget.md`.
* **Plan change (AGENTS.md / DENSE-FT-CLOSURE).** DENSE-FT-CLOSURE (2026-08-31) closed dense training and
  "approximate dense-FT reconstruction". This package **reopens dense training at reduced scale** by Gabriel's
  explicit instruction, and only as a *reduced-scale analogue* (20 000 steps, not Singh's 10⁶): it is **NOT** a
  reconstruction of Singh's deleted dense-FT checkpoint and **NOT** the matched control. Singh's checkpoint stays
  unrecoverable; that limitation sentence stays in the manuscript.
* **What is NOT claimed by any arm.** No "restored to dense"; no "matched dense control"; no causal attribution to
  pruning. Every arm is 2 % of the released recovery budget, so absolute R/G values are **not** comparable with the
  released checkpoint; only the **sign and the ΔJ contrasts across training duration** are read.

## 1. Shared conventions (identical to the frozen runs and to E3 unless stated)

* Recipe (Singh's = upstream `audioldm_original_medium.yaml`, with the training DURATION as the deliberate factor):
  start from a backbone; train ALL U-Net parameters (VAE, CLAP conditioner, vocoder frozen); AdamW lr 1e-4 constant,
  torch defaults (betas 0.9/0.999, weight decay 0.01, eps 1e-8), no scheduler, no grad clip; **effective batch 2**;
  CFG dropout 0.1; FP32; data = the preprocessed AudioCaps TRAIN split (49 502 clips), random `duration`-second
  crops (the dataset's own `random_segment_wav`); **N = 20 000 optimizer updates**; seed 20260905.
* **Backbones.** `pruned2_A` = A′ L1 selection on the dense EMA, [1,2,1,1] (E3's start); `dense` = AudioLDM-M-Full
  EMA (`data/checkpoints/audioldm-m-full.ckpt`, materialized EMA). Trainer `scripts/research/e3_shortft_trainer.py`.
* **Gradient accumulation.** The dense 10.24-s arm runs `batch 1 × accum 2` to fit a 16 GB T4; effective batch stays
  2. The U-Net uses GroupNorm (not BatchNorm), so accumulation is numerically equivalent to batch 2. This is a
  declared, minimal recipe deviation for that one arm; the other three arms run batch 2 directly.
* **Self-gate.** Every job times its first 200 optimizer updates and STOPS before the full run if the projected
  training cost (`sec/step × 20 000 / 3600 × 0.89 cr/h`) exceeds the job's `--cap-cr`. A bench-only stop produces no
  checkpoint and no eval, is reported as such, and is not rescued.
* **Evaluation (in the same job, CRN).** The saved raw U-Net is generated on the frozen 192 AudioCaps prompts
  (`configs/research/xsev_audiocaps_manifest.json`) at both durations with the frozen `GEN_SALT`
  (`RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30`), so each fine-tuned checkpoint is **noise-paired** with the
  frozen P (`pruned2_A`) and dense clips of the same prompt. Systems: `longft` (pruned arch), `denseft` (dense arch);
  contexts `ac_short` (3.84 s, latent 96) and `ac_native` (10.24 s, latent 256). DDIM 50 / guidance 2.5 / eta 0 /
  fp32 / single. Weights: **raw** (no EMA at this horizon), as E3.
* **Scorer.** Fused CLAP `laion/clap-htsat-fused` rev `365dea6e`, one seed-once fixed-order call per group, shuffled
  caption floor from the same embeddings; unit = prompt; percentile bootstrap `B = 10000`, seed namespace
  `REVIEWER2-FOLLOWUP|BOOTSTRAP|2026-09-05` (the EXT2×2 verdict draws under `…|EXT2x2`). SESOI 0.025.
* **Baselines already on disk (0 cr).** Frozen P at both durations (`xsev_sev2_groups_out.json`) and the frozen dense
  at both durations on the same 192 prompts (`xsev_dense192_groups_out.json`, XSEV-DENSE-192-CONTROL) — both are the
  CRN partners of the new checkpoints. E3's `shortft` is the pruned 3.84-s arm (already run, `r2_E3_result.json`).
* **Structural validation before any score:** sha256 + prompt-index + sample count per WAV (3.84 s → 61 472; 10.24 s
  → 163 872).

## 2. The four cells (E3 supplies one; three are new)

| Backbone | Train @3.84 s | Train @10.24 s |
|---|---|---|
| Pruned (P = pruned2_A) | **E3 `shortft`** (done) | **item 1 `longft`** (new, `r2-longft`) |
| Dense (AudioLDM-M-Full) | **item 2 `denseft_short`** (new, `r2-denseft-s`) | **item 2 `denseft_native`** (new, `r2-denseft-n`) |

Each cell is evaluated at both 3.84 and 10.24 s on the 192 prompts.

## 3. Estimands and readings (verdict `EXT2x2` → `configs/research/r2_EXT2x2_result.json`)

* Pruned arm: `R_sf(d) = CLAP(shortft) − CLAP(P)`, `R_lf(d) = CLAP(longft) − CLAP(P)`, `J_sf`, `J_lf`.
* Dense arm: `G_ds(d) = CLAP(denseft_short) − CLAP(dense)`, `G_dn(d) = CLAP(denseft_native) − CLAP(dense)`,
  `J_ds`, `J_dn`.
* **PRIMARY contrast, per arm: `ΔJ = J(trained@10.24) − J(trained@3.84)`, paired per prompt** (the P / dense baseline
  terms cancel, so `ΔJ_pruned = (longft_{10.24} − longft_{3.84}) − (shortft_{10.24} − shortft_{3.84})`, likewise for
  dense). This is exactly the comparison the specialisation hypothesis predicts.
* Cross-arm secondary: `(pruned duration response) − (dense duration response)` at matched training duration.

**Pre-specified readings** (per arm, on the 95 % CI of ΔJ):

* **`lo95(ΔJ) > 0`** → training at 10.24 s buys **more** interaction than training at 3.84 s → the training duration
  DOES modulate the favourable evaluation duration → **specialisation contributes**. The manuscript's abstract must
  then read **"does not support"** the specialisation prediction, not "contradicting"; the operating-point claim is
  qualified accordingly.
* **CI of ΔJ within ±0.025 (SESOI)** → the interaction is **independent of the training duration** at this budget →
  the operating-point reading is established, and the abstract keeps "contradicting"/"does not support" as decided by
  the pruned arm. Item 7's out-of-distribution hypothesis gains its discriminating datum.
* **`hi95(ΔJ) < 0`** → reversed; reported as is.
* Otherwise UNRESOLVED. If both `R_lf`/`G_dn` and their short counterparts are unresolved (2×10⁴ steps changed too
  little), the arm is UNINFORMATIVE — declared possible in advance (especially for dense, which M-Full already
  fine-tuned 0.25 M AudioCaps steps), and reported as such, not rescued.

The abstract-wording decision is driven by the **pruned** arm's ΔJ (the object of study). The dense arm is a
reduced-scale analogue that informs, but does not by itself set, the causal wording.

## 4. Budget (Gabriel's 20-cr ceiling; T4 0.89 cr/h; §A10 per-WAV model 0.001329 + 9.0e-6·L; job overhead 0.145 cr)

Training throughput measured on E3 (`r2-shortft`): 0.327 s/step, pruned 70.5 M U-Net, latent 96, batch 2, peak VRAM
5.83 GB. Scaled per arm (latent 256 ≈ 2.67× activations; dense ≈ the gate0-smoke 0.307 s/step full-backward proxy):

| Job | Cell | latent / batch×accum | s/step (est.) | Train cr (est.) | Eval cr | Self-gate `--cap-cr` | Watchdog cap / min |
|---|---|---|---:|---:|---:|---:|---:|
| `r2-longft` | pruned @10.24 | 256 / 2×1 | 0.8–0.9 | 4.0–4.5 | 1.1 | **4.8** | **6.2 / 360** |
| `r2-denseft-s` | dense @3.84 | 96 / 2×1 | 0.32–0.36 | 1.6–1.8 | 1.1 | **3.2** | **4.2 / 240** |
| `r2-denseft-n` | dense @10.24 | 256 / 1×2 | 0.9–1.05 | 4.7–5.3 | 1.1 | **5.6** | **7.2 / 420** |
| **total** | | | | **10.3–11.6** | **3.3** | | **17.6 caps** |

Expected settled ≈ 14–16 cr of generation/training + ≈ 1.5–2 cr Studio (watchdogs run on the Studio through the
jobs). Watchdog caps sum to 17.6; with ≈ 2 cr Studio the hard ceiling is ≈ 19.6 < 20. If any 200-step bench projects
over its `--cap-cr`, that arm bench-stops at ≈ 0.15 cr and the decision returns to Gabriel rather than overrunning.
The funded balance is not exposed by the SDK; the 20-cr figure is Gabriel's. Insurance: `--mid-step 7500` saves an
extra raw checkpoint of each 10.24-s arm at 7 500 updates (≈ matched audio-seconds to E3's 20 000 short steps) for
an optional later eval under a separate GO.

## 5. Compute-discipline record (AGENTS.md, required before every GPU launch)

1. **Why CPU is unsuitable.** 60 000 U-Net optimizer updates + 1 152 diffusion clips: infeasible in Studio wall time
   (E3's 20 000 updates took ≈ 1.8 h on a T4; on CPU that is days per arm).
2. **GPU-only work.** The three fine-tunes and their evaluation sampling. Everything else — validation, scoring,
   floors, bootstrap, the EXT2×2 verdict, manuscript — is CPU (0 cr), run after the jobs settle.
3. **Smallest compatible class.** T4 on-demand (device rule; every frozen clip came from a T4). The dense 10.24-s arm
   is kept on the T4 by batch 1 × accum 2 rather than moving to a pricier 24 GB class.
4. **Ceilings.** Per-job self-gate (`--cap-cr`) + external watchdog caps above, enforced by
   `scripts/sa3/job_watchdog.py` (Running-time clock) from the Studio.

## 6. Order of execution and reporting

The three jobs launch together (parallel, Gabriel's preference) from a clean, committed, pushed tree via
`scripts/ops/launch_job_with_watchdog.sh`, after this file is committed with its sidecar and every trainer/generator
CPU dry-run passes. Each job self-contains training + evaluation. Scoring and the `EXT2x2` verdict run on CPU after
the jobs settle (`scripts/research/r2_verdict.py --emit/--score --job {longft,denseft_s,denseft_n}` then
`--verdict EXT2x2`). Every result — including UNRESOLVED / UNINFORMATIVE and any bench-only stop — is recorded in
`docs/experiment_ledger.md`; the manuscript changes only after the verdict exists. Stop the Studio between stages.
