# REVIEW3-DENSERAW — frozen protocol addendum: the raw dense baseline (third review, item 1b)

**Status: FROZEN BEFORE ANY OUTPUT.** Nothing in this file may be edited after the first WAV of the job below exists. The
sha256 sidecar `docs/review3_denseraw.md.sha256` is committed together with this file and before launch.

## 0. Authorization and scope

* **Trigger.** The third review of Draft 14 (`docs/review/2026-09-06_review_round3_methodological_response.md`, Accept 4/5),
  item 1: the dense 2×2 is "broken" (both 20 000-step dense fine-tunes lose ≈ 0.2 CLAP against the EMA dense baseline) and
  the paper attributes it to the raw-vs-EMA weight convention. The EMA weights of the 20 000-step runs were never kept
  (trainer `use_ema=False`; §1a of the response), so the only test left is the reviewer's second option: **evaluate the
  raw dense baseline**.
* **Gabriel, 2026-09-06 17:52 MVD:** "tenemos 4 créditos cargados"; **18:00 MVD: "GO"** for this item alone (item 3's
  second draw does not fit the 4-cr pool and is not launched). Ceiling: job hard cap **2.6 cr**; expected total incl.
  Studio 2.3–3.0 cr; worst case ≈ 3.4 cr < 4.
* **What is NOT claimed.** Nothing about Singh's deleted million-step dense control (still unavailable). This addendum
  only decides whether the reduced-scale dense 2×2 measured a convention artefact or a genuine degradation, i.e. what the
  two dense paragraphs of Sec. 4.2 / Sec. 5 may say.

## 1. System, contexts, pairing (identical to the frozen runs unless stated)

* **`dense_raw`** = the trajectory weights `model.diffusion_model.*` of `data/checkpoints/audioldm-m-full.ckpt` (file sha256
  prefix `936914a388905e1f`; 690 tensors; mean-rel 2.4 % from the EMA shadow per the DECISION-V4-12 audit), strict-loaded
  into a bare dense-architecture U-Net (`channel_mult [1,2,3,5]`), no EMA materialization. Generator
  `scripts/research/reversal_xsev_gen.py --system dense_raw` (added 2026-09-06; CPU dry-run PASS on `ac_short`, prompt 0,
  seed `2823286605618390972` = the frozen seed of that prompt). Every frozen "dense" clip in the paper used the EMA.
* **Contexts.** `ac_short` (3.84 s, latent 96) and `ac_native` (10.24 s, latent 256) on the frozen 192 AudioCaps prompts
  (`configs/research/xsev_audiocaps_manifest.json`, manifest sha256 `4da90661…`), frozen `GEN_SALT`
  (`RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30`), replicate 0. Recipe DDIM 50 / guidance 2.5 / η 0 / fp32 /
  single sample. Each `dense_raw` clip is therefore **noise-paired** (same x_T) with the frozen dense-EMA clip
  (`xsev_dense192_groups_out.json`), the `denseft_short` and `denseft_native` clips (`r2_denseft_s`, `r2_denseft_n`) and
  the P / P+FT clips (`xsev_sev2_groups_out.json`) of the same prompt.
* **Device check (descriptive).** `dense` (EMA) `ac_native` prompt indices 0–3 regenerated on the job machine and scored
  against the frozen `xsev_dense192` clips (max |ΔCLAP| reported; the frozen bit-identical rule has been FAIL at ≈ 4e-6 on
  every previous job and is kept as written).
* **Structural validation before any score:** sha256 + prompt-index + sample count per WAV (3.84 s → 61 472; 10.24 s → 163 872).

## 2. Scorer and bootstrap

Fused CLAP `laion/clap-htsat-fused` rev `365dea6e`, one seed-once fixed-order call per group (`r2_verdict.py` conventions),
shuffled-caption floor from the same embeddings; unit = prompt; percentile bootstrap `B = 10 000`; seed namespace
`REVIEW3|BOOTSTRAP|2026-09-06`; SESOI 0.025. Scorer `scripts/research/r3_denseraw_score.py` (to be written after the job
settles, following `r2_ext2x2_score.py`; prompt-index aligned so the n = 170 `denseft_short @10.24` cell is handled) →
`configs/research/r3_denseraw_result.json`.

## 3. Estimands (paired per prompt, d ∈ {3.84, 10.24})

* `O(d) = CLAP(dense_raw) − CLAP(dense_EMA)` — the raw-vs-EMA offset at step 0 (n = 192).
* `G′_s(d) = CLAP(denseft_short) − CLAP(dense_raw)`, `G′_n(d) = CLAP(denseft_native) − CLAP(dense_raw)` — the fine-tuning
  effect net of the weight convention (n = 192, except `denseft_short @10.24` n = 170 → common set).
* `J′_s`, `J′_n` and `ΔJ′_dense = J′_n − J′_s` — algebraically identical to the reported `J_ds`, `J_dn`, `ΔJ_dense`
  (the baseline cancels); recomputed only as a consistency guard.
* `Δ_O = O(10.24) − O(3.84)` — duration dependence of the convention offset: the sign and scale of the bias the raw-vs-EMA
  convention can put on the individual pruned `J_3.84` / `J_10.24` (reported as a caveat magnitude; **not** applied as a
  correction to any pruned number).
* Descriptive: `CLAP(dense_raw)` levels and floors at both durations; `ρ`-type ratios are not computed.

## 4. Pre-specified readings (on the 95 % CIs, both dense arms)

* **(i) CONVENTION.** `G′_s(d)` and `G′_n(d)` both inside ±SESOI at both durations → the −0.2 "degradation" is the weight
  convention; the dense 2×2 becomes a genuine short-budget control ("20 000-step dense fine-tuning neither helps nor hurts").
  Manuscript: the two dense paragraphs (Sec. 4.2, Sec. 5) collapse to one control sentence plus a table row; the
  raw-vs-EMA explanation is confirmed and stated as measured (`O(d)`).
* **(ii) DEGRADATION.** `hi95(G′(d)) < 0` for both arms at both durations **and** `|O(d)| < |G(d)| / 2` → the recipe
  (constant lr 1e-4, batch 2, no EMA, 20 000 steps) genuinely degraded an already-converged dense model; the raw-vs-EMA
  explanation is **deleted** from Sec. 4.2 and Sec. 5 and replaced by the recipe statement; the dense 2×2 stays a negative.
* **(iii) MIXED.** Anything else → both components reported with the split (`O(d)` vs `G′(d)`), wording follows the numbers.
* `O(d)` on its own: `hi95(O(d)) < 0` → raw weights underperform the EMA at step 0 by the stated amount (the size of the
  convention offset); `Δ_O` resolved ≠ 0 → the offset is duration-dependent (caveat magnitude for the pruned J values).
* If the job stops early, contrasts are computed on the common prompt set actually generated (prompt-index aligned), and
  the partial n is reported; no rescue run without a new GO.

## 5. Budget (Gabriel's 4-cr pool; T4 0.89 cr/h; dense per-WAV rate measured on the r2 dense jobs 0.0041–0.0052 cr/WAV)

| Component | Basis | Point (cr) |
|---|---|---:|
| 384 dense WAVs (192 × {3.84, 10.24} s) + 4 device-check WAVs | 388 × 0.0041–0.0052 | 1.6–2.0 |
| Provisioning / lifecycle | one job | 0.15 |
| **Job `r3-denseraw` point / hard cap** | watchdog `--max-cost 2.6 --max-minutes 180`; in-job `timeout` 3 300 s (short) + 6 600 s (native) + 600 s (check) | **1.7–2.15 / 2.6** |
| Studio hours (launch, watchdog, CPU scoring) | 2–3 h × 0.27 | 0.6–0.8 |
| **Total** | | **≈ 2.3–3.0, worst ≈ 3.4 < 4** |

**Compute-discipline record (AGENTS.md).** (1) CPU unsuitable: the dense U-Net takes ≈ 5–8 min per clip on this CPU →
384 clips ≈ 30–50 Studio-hours ≈ 8–14 cr, and the device rule requires T4 clips. (2) GPU-only work: the 388 diffusion clips.
Scoring, floors, bootstrap, verdict and manuscript are CPU. (3) Smallest class: T4 on-demand. (4) Bounds: in-job `timeout`
(primary, survives a stopped Studio), external watchdog cap 2.6 cr / 180 running-minutes (insurance), self-bounded WAV count.

## 6. Order of execution and reporting

Commit this file + sidecar + `scripts/research/run_r3_denseraw.sh` → push → launch from the clean tree via
`scripts/ops/launch_job_with_watchdog.sh r3-denseraw 2.6 180 "DEV=cuda bash scripts/research/run_r3_denseraw.sh"` →
ledger REVIEW3-DENSERAW-LAUNCH → stop the Studio while the job runs (the in-job timeout is the bound) → on return:
structural validation, CPU scoring, `r3_denseraw_result.json`, ledger REVIEW3-DENSERAW-RESULT, then (and only then) the
manuscript edits under the pre-specified reading.
