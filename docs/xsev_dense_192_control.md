# XSEV-DENSE-192-CONTROL — paired dense control on the severity-2 prompt set (protocol, frozen before generation)

**Status:** PROPOSED / FROZEN DESIGN, **NOT LAUNCHED** (2026-09-02, America/Montevideo). No output exists.
**Class:** prospective design completion (post-result of RECOVERY-CROSS-SEVERITY-REP-1); **no gate**; every
outcome is reportable. It cannot change any frozen verdict (V1.1, Arm-D, xsev CASE C, music-native branch (a)).

## 1. Why (the gap it closes)

The manuscript's sharpest sentence at the primary severity — "at 83 % pruning the pruned checkpoint has
lost most of the dense model's response to clip duration, and fine-tuning restores a response of dense
magnitude" — and the recovery ratio "fraction of the pruned checkpoint's gap to dense closed by
fine-tuning" are **cross-set** at severity 2: the dense model was generated only on the severity-1 Arm-D
80 prompts (`DRAFT4-DENSE-DURATION-CONTROL`), never on the severity-2 192 prompts. This arm generates the
dense model on the same 192 prompts, at both durations, with the same common random numbers as the frozen
P (pruned2_A) and P+FT (recovered2) clips, so that s(dense), the recovery ratio and the dense−P+FT gap at
severity 2 become paired, inferential quantities.

## 2. Design (frozen)

* **System:** dense AudioLDM-M-Full, EMA convention (the same `build_backbone("dense")` used for the
  frozen Arm-D dense control), DDIM 50 / guidance 2.5 / eta 0 / fp32 / single generation.
* **Prompts:** `configs/research/xsev_audiocaps_manifest.json` (192, `prompts_sha256` as recorded there),
  replicate 0 only.
* **Operating points:** `ac_short` (3.84 s, latent 96) and `ac_native` (10.24 s, latent 256).
* **Common random numbers:** `x_T = f(GEN_SALT, ytid, 0)` via `derive_paired_seed` — identical to the
  frozen pruned2_A / recovered2 clips of the same (context, ytid). Generator: `reversal_xsev_gen.py`
  with `--system dense --context ac_short|ac_native` (one-line relaxation of the "dense only for
  dense_native" guard; no other change).
* **Device rule:** all 384 dense WAVs come from ONE device (either one T4 job or one CPU run). If the run
  is split or restarted on another device, the partial output is discarded. (The paired systems were
  generated on T4; a device-consistency check regenerates 4 pruned2_A native clips on the chosen device
  and requires |ΔCLAP| < 0.01 per clip — descriptive, reported.)
* **Scoring:** frozen fused-CLAP convention (rev 365dea6e; ONE seed-once 192-item call per cell in
  `prompt_index` order), plus the Draft-5 shuffled-caption floor for both dense cells.

## 3. Estimands (unit = prompt, n = 192, percentile bootstrap B = 10 000, seed namespace
`XSEV-DENSE-192-CONTROL|BOOTSTRAP|<launch date>`)

| Symbol | Definition | Role |
|---|---|---|
| s(dense) | mean_i [CLAP(dense,10.24)_i − CLAP(dense,3.84)_i] | primary descriptive |
| s(P)−s(dense), s(P+FT)−s(dense) | paired per prompt | primary (no gate) |
| ρ_short, ρ_native | R_op / (dense_op − P_op), ratio of means | primary (no gate) |
| G_native(P+FT) = dense − P+FT at 10.24 s; G_short | paired | secondary; "restored to dense" only if a TOST at ±0.025 passes |
| floor-corrected versions of all of the above | Draft-5 floor method | sensitivity |

No gate; the paper reports whatever comes out. Predeclared reading: if s(dense) on the 192 set is
≈ +0.15 (as on the 80 set) and s(P)=+0.040 ≪ s(dense) ≈ s(P+FT), the cross-set sentence becomes paired;
if s(dense) is much smaller on this set, the "dense magnitude" wording is withdrawn.

## 4. Cost and launch (needs Gabriel's explicit authorization for GPU; CPU is 0 cr)

* **T4 (recommended):** 192 native ≈ 0.0036 cr/WAV + 192 short ≈ 0.0022 cr/WAV ≈ **1.1–1.3 cr** incl.
  provisioning; **hard cap 1.5 cr** via the job watchdog. ≈ 60–70 min wall.
* **CPU fallback (0 cr):** measured on this 4-core Studio: dense at latent 96 ≈ 1.7 s/DDIM step
  (≈ 95 s/clip → 192 clips ≈ 5 h); latent 256 ≈ 4× → ≈ 4 min/clip → ≈ 13 h. Total ≈ 18 h; resumable via
  `--indices`.
* Launch script: `scripts/research/run_xsev_dense_192_gen.sh` (`DEV=cuda` in a job, or `DEV=cpu`).
* After generation (CPU, 0 cr): emit the two 192-item groups, score with `gate0_clap_scorer.py
  --score-groups`, run the verdict script (to be written on launch; mirrors `draft4_dense_duration_control.py`).

## 5. What it does NOT do

It does not add a dense **fine-tuned** control (the mechanistic confound stays); it does not change the
CASE C verdict; it does not touch severity 1.
