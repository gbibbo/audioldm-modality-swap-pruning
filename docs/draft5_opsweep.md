# DRAFT5-OPSWEEP-1 and DRAFT5-PUBRECIPE-1 — frozen protocol

**Status: FROZEN BEFORE ANY OUTPUT.** Nothing in this file may be edited after the first WAV of either
job exists. The sha256 sidecar `docs/draft5_opsweep.md.sha256` is committed together with this file and
before launch.

## 0. Authorization and plan change (READ FIRST)

* **Trigger.** The external-reviewer simulation (`docs/review/2026-09-03_manuscript_draft5_icassp_reviewer_simulation.md`)
  asked two questions that only new generation can answer: (1) is the recovery gain monotone in
  duration or peaked at the fine-tuning duration? (2) does the duration interaction survive the
  published sampler recipe? Both were costed in `docs/compute_budget.md` §A10 and NOT launched.
* **Gabriel, 2026-09-03 12:28 MVD.** Asked what the `total_spent` reading of 85.855 refers to and
  pre-approved a job with a **maximum estimated budget of 7 cr** conditional on that number not being
  credits. **It IS credits** (lifetime cumulative spend; proven by the settled deltas: V1.1 cost
  1.262 cr and moved `total_spent` 66.770→68.032; the smoke test cost 0.1835 cr and moved it
  55.62→55.80). The condition was therefore reported as FALSE and the choice put back to him with the
  corrected fact. **He then selected "barrido + receta publicada" (E1 + E2b) with the 7-cr ceiling.**
* **PLAN CHANGE, explicitly authorized (AGENTS.md: record it).** `DDIM200` is listed as a REJECTED /
  closed experiment in four frozen records: PROGRESS OPEN ITEM 0, FINAL-STORY-FREEZE, the XSEV CASE C
  entry ("NO manuscript, NO DDIM200, NO third severity") and RECOVERY-REVERSAL-V1.1 ("NO rescue (no
  alt op-point/DDIM200/...)"). Gabriel's selection reopens **DDIM200 for E2b only**, as a
  **pre-specified sensitivity check of an already-established interaction** — NOT as a rescue of any
  failed gate, and NOT as a reopening of the V1.1 negative or of the third severity. Every other item
  on the rejected list stays closed.
* **What this protocol may NOT do.** It cannot change any frozen verdict (V1.1 PASS=FALSE, XSEV CASE C,
  FineLAP Branch A2). It adds new operating points to an evaluation; it does not re-litigate a gate.

## 1. Systems, prompts, seeds (identical conventions to the frozen runs)

* **Systems.** `pruned2_A` (P: L1 channel selection A′ applied to the dense EMA weights, `[1,2,1,1]`),
  `recovered2` (P+FT: the public dp1 recovered checkpoint, EMA), `dense` (AudioLDM-M-Full dense EMA).
* **Prompts.** The frozen `configs/research/xsev_audiocaps_manifest.json` — 192 AudioCaps prompts,
  disjoint from every severity-1 prompt. Replicate 0 only.
* **CRN seeds.** `derive_paired_seed(GEN_SALT, ytid, 0)` with the frozen
  `GEN_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30"`, i.e. the **same integer seed per
  ytid** as `ac_short` / `ac_native`; the `x_T` tensor shape follows the latent length. This is exactly
  the convention already used for `music_native`. Consequence, stated in advance: systems are
  **noise-paired within a duration**; duration contrasts are **prompt-paired but not noise-paired**,
  as the manuscript already declares.
* **Weights.** EMA for every system. Checkpoints and their conventions unchanged.

## 2. E1 — DRAFT5-OPSWEEP-1 (duration sweep)

**Question.** Is the recovery gain monotone in clip duration, or does it peak at the fine-tuning
duration (10.24 s)?

* **New generation:** `dense`, `pruned2_A`, `recovered2` × 192 prompts × {**5.12 s** (latent 128),
  **7.68 s** (latent 192)} = **1152 WAVs**, DDIM 50 / guidance 2.5 / eta 0 / fp32 / single generation.
* **Reused, not regenerated:** the frozen 3.84 s and 10.24 s cells of the same three systems.
* **Estimand.** `R(d)` = mean per-prompt `CLAP(P+FT) − CLAP(P)` at duration `d`, prompt-level
  percentile bootstrap, `B = 10000`, unit = prompt, seed namespace
  `DRAFT5-OPSWEEP-1|BOOTSTRAP|2026-09-03`.
* **Pre-specified shape rule** (declared here, before any score). Let
  `D1 = R(5.12) − R(3.84)`, `D2 = R(7.68) − R(5.12)`, `D3 = R(10.24) − R(7.68)`, each paired per prompt:
  * **MONOTONE-INCREASING** — all three point estimates `> 0` and no `hi95 < 0`.
  * **PEAKED BEFORE NATIVE** — `hi95(D3) < 0` (a resolved fall at the last step).
  * **SATURATING** — `lo95(D1) > 0` and `lo95(D2) > 0`, while `D3`'s CI contains 0 **and**
    `|point(D3)| < 0.025` (the project SESOI).
  * **UNRESOLVED** — anything else. This is a legitimate outcome and will be reported as such.
* **Secondary (CPU, 0 cr):** chance floor of every new cell (shuffled captions, same embeddings);
  real-audio ceiling at 5.12 s and 7.68 s (the same source clips truncated to 81 952 and 122 912
  samples, matching the generated lengths); `rho_real(d)` and `rho_dense(d)` at all four durations.
* **Expected cost:** 3.35 cr (model in `docs/compute_budget.md` §A10). **HARD CAP 4.00 cr**,
  max 300 min, enforced by `scripts/sa3/job_watchdog.py`.

## 3. E2b — DRAFT5-PUBRECIPE-1 (published-recipe spot check)

**Question.** Does the duration interaction hold at Singh et al.'s published sampler recipe rather than
at this project's frozen one?

* **New generation:** `pruned2_A`, `recovered2` × **the first 64 prompts of the frozen manifest in
  `prompt_index` order** (an outcome-blind, deterministic subset fixed by a manifest frozen long before
  any of these scores) × {3.84 s, 10.24 s} = **256 WAVs**, **DDIM 200 / guidance 3.5** / eta 0 / fp32,
  **single generation**.
* **Declared limitation, not fixed here:** the published recipe also uses **best-of-3**; this check
  varies the sampler budget and the guidance scale only. Reproducing best-of-3 would triple the cost
  and was not authorized. Any conclusion is therefore about DDIM steps and guidance, not about
  candidate selection.
* **Estimand.** `J_pub = R_pub(10.24) − R_pub(3.84)`, paired per prompt, same bootstrap settings, seed
  namespace `DRAFT5-PUBRECIPE-1|BOOTSTRAP|2026-09-03`.
* **Pre-specified gate.** `lo95(J_pub) > 0` → **the interaction holds at the published recipe**. If the
  gate fails, that is reported as a limitation of the manuscript's operating point, and the frozen
  DDIM-50 result stands as measured — the published-recipe run cannot retroactively validate or
  invalidate it.
* **Pre-specified secondary.** `J_frozen|64`, the frozen DDIM-50 interaction recomputed on the *same*
  64 prompts, so the recipe comparison is like-for-like. Reported as a difference
  `J_pub − J_frozen|64` with a paired CI; **descriptive** (the two recipes are not noise-paired).
* **Expected cost:** 2.47 cr. **HARD CAP 3.00 cr**, max 240 min.

## 4. Device rule and integrity checks

* Every WAV of a job comes from **one Lightning T4 job**, the same hardware class as every frozen clip.
  A resume via `--indices` in another **T4** job is allowed; CPU/GPU mixing is not.
* **Device-consistency check, per job:** 4 `pruned2_A` / `ac_native` clips (`prompt_index` 0–3) are
  regenerated under the frozen recipe into a separate `device_check/` directory and compared with the
  frozen clips. Expectation, from the XSEV-DENSE-192-CONTROL job: **bit-identical, ΔCLAP 0.0**. A
  non-zero difference invalidates cross-job comparability and must be reported before any result.
* Structural validation before scoring: every WAV matches its manifest sha256, ytid and sample count
  (5.12 s → 81 952; 7.68 s → 122 912; 3.84 s → 61 472; 10.24 s → 163 872 at 16 kHz).

## 5. Budget ceiling (Gabriel's approval)

| Job | WAVs | Expected | Hard cap | Watchdog max-minutes |
|---|---|---|---|---|
| `draft5-opsweep-1` (E1) | 1152 + 4 | 3.35 cr | **4.00** | 300 |
| `draft5-pubrecipe-1` (E2b) | 256 + 4 | 2.47 cr | **3.00** | 240 |
| **total** | **1416** | **5.82 cr** | **7.00** | — |

The two caps sum to exactly the 7-cr ceiling Gabriel approved. If either watchdog fires, the job is
STOPPED and the partial state is reported as partial — no silent overrun, no rescue run.

## 6. Compute-discipline record (AGENTS.md, required before every GPU launch)

1. **Why CPU is unsuitable.** Measured CPU generation on this Studio is 1.7 s per DDIM step at latent
   96. The two jobs total 1416 clips with a mean of ~103 DDIM-step-equivalents at larger latents;
   a CPU run is on the order of **several days** and would block the Studio.
2. **What genuinely requires GPU.** Only the diffusion sampling. Scoring, floors, ceilings, bootstrap,
   verdicts, structural validation and manuscript integration are CPU and cost 0 cr.
3. **Smallest compatible class.** T4 on-demand — the class every settled generation job in this
   project used, and the class the frozen clips came from (required by the device rule).
4. **Cost ceiling.** Per-job hard caps above, enforced externally by the cost watchdog, not by trust.

## 7. What will NOT be claimed

* No causal or mechanistic claim. These are additional evaluation operating points on released
  checkpoints; nothing is trained.
* No "restored to dense" claim at any new duration.
* No re-opening of the V1.1 negative, the third severity, text-FT generation, new prompts, new scorers
  or the human listening study.
* If E2b's gate fails, the manuscript reports it as a limitation of the frozen operating point. It does
  not become a reason to re-run anything.
