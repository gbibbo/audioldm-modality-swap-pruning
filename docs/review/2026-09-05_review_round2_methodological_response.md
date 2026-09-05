# Round-2 review of Draft 13 (Accept, 4/5) — methodological items re-verified against the record and costed

**Date:** 2026-09-05 (MVD). **Type:** read-only re-verification + CPU-only post-hoc pooling (0 cr). **Nothing launched.**
**Manuscript under review:** Draft 13 (`icassp/icassp_operating_point.tex` → `icassp/sections/draft13_*.tex`, marker
`%% draft13-reviewer-followup`, bundle `icassp/icassp_operating_point_draft13_reviewer_followup.zip`, evidence baseline
`12546d5`). **Instruction (Gabriel):** keep Draft 13 as the current version; for every *methodological* weakness, re-check
whether the constraint that produced the design decision still holds and, where it does not, work out how the
correction would be run. Presentation items (3, 4-wording, 6-minor, URL) are deferred by instruction and only noted.

## 0. Budget fact that conditions every "correctable" verdict

* Lightning `total_spent` = **114.95 cr** (SDK `billing_service_get_user_balance()`, 2026-09-05 ~20:45 UTC; `balance`
  field still the static 5.0, uninformative). At the moment Gabriel set the **15-cr ceiling** (01:22 MVD, reading
  99.18 cr at ~04:00 UTC) the counter has since moved **+15.77 cr** = 8.94 cr of settled R2 jobs (2.399 + 2.278 +
  0.697 + 0.638 + 2.926) + ≈ 6.8 cr of Studio uptime (the Studio ran through the jobs and this session).
* **The 15-cr ceiling is therefore spent.** Every GPU item below needs a *new* authorization (and, since the funded
  balance is not exposed by any endpoint, Gabriel's confirmation that the credit exists). CPU-only items are free.

## 1. Item 1 — the 3.84-s intervention lacks its symmetric control (20 000 steps at 10.24 s) — **CORRECTABLE, GPU**

**What the reviewer asks.** E3 (P fine-tuned 20 000 steps at 3.84 s) gains +0.009 at its own duration and +0.075 at
10.24 s. Reading A (ours): the duration effect is not specialisation. Reading B (reviewer): 20 000 steps cannot adapt
anything to short generation, and what little is learned only shows at 10.24 s because that is the only duration at
which the U-Net produces coherent audio — compatible with "short generation is broken for everyone" and *not* a
refutation of specialisation at an adequate budget. The symmetric control is the same 20 000 steps at 10.24 s:
J(long-FT) vs J(short-FT) is the comparison the specialisation hypothesis actually predicts. Until it exists,
"contradicting" must become "does not support".

**What the record says about why it was not run.** `docs/reviewer2_followup.md` §8, "Declared limits: single arm
(no matched 10.24-s arm at the same step count: **budget**)". The two-arm design (E3′) was costed at 8.2 cr point /
9.8 cap in `docs/review/2026-09-05_reviewer2_methodological_response.md` and dropped to fit the whole package under
15 cr. **The constraint was budget only** — no checkpoint, data, code or ethics obstacle.

**Re-verification today — everything except credit is in place.**

| Requirement | Status |
|---|---|
| Start checkpoint (`pruned2_A`, A′ L1 selection on the dense EMA, [1,2,1,1]) | built from local checkpoints by `reversal_xsev_gen.build_backbone`; identical to E3's start |
| Trainer | `scripts/research/e3_shortft_trainer.py` — E3's exact code; `DURATION`/`LATENT_T` are two module constants (3.84 / 96) → add `--duration 10.24 --latent-t 256`; 200-step self-gate, resume every 5 000 steps, raw-weight export already there |
| Data | preprocessed AudioCaps TRAIN split on disk (49 502 clips). At 10.24 s the dataset's `random_segment_wav` pads the 10-s clip instead of cropping it, i.e. the long arm is **Singh's recipe exactly** (upstream `duration: 10.24`), while E3 used random 3.84-s crops. Declare: "matched steps, not matched audio-seconds" (the long arm sees 2.67× more mel frames per step) |
| Evaluation path | `--system shortft` loads any raw U-Net state dict of the pruned2_A architecture from `SHORTFT_UNET`; CRN-paired with P / P+FT on the frozen 192 prompts at both durations; add an alias `longft` so manifests/group names stay unambiguous |
| Scoring/verdict | `r2_verdict.py` conventions (fused CLAP rev 365dea6e, seed-once calls, prompt bootstrap B = 10⁴); a new `E3L` verdict block |
| VRAM | E3 peak **5.83 GB** at latent 96 (`bench.json` in the job artifacts); activations scale ≈ 2.67× → ≈ 10–11 GB, inside a T4's ≈ 15 GB usable |
| Speed | E3 measured **0.327 s/step** at latent 96 (0.351 in the 200-step bench). Latent 256: convolutions ×2.67, attention up to ×7 on a minority of the FLOPs → **≈ 0.75–0.90 s/step** (the bench decides) |

**Cost (T4 on-demand 0.89 cr/h; per-WAV model §A10 = 0.001329 + 9.0e-6·L; job overhead 0.145 cr).**

| Component | Basis | Point (cr) |
|---|---|---:|
| 200-step benchmark + 20 000 steps at latent 256 | 0.75–0.90 s/step × 20 200 / 3600 × 0.89 | 3.7–4.5 |
| Evaluation, 192 prompts × {3.84 s, 10.24 s} = 384 WAVs | 192 × (0.00219 + 0.00363) | 1.12 |
| Provisioning / lifecycle | one job | 0.15 |
| **Job `r2-longft` point / hard cap** | cap = point × 1.2, self-gate stops training if the bench projects > 5.0 cr | **5.0–5.8 / 7.0** |
| Studio hours (scoring, verdict, bookkeeping) | ≈ 2–3 h × 0.27 | 0.6–0.8 |
| **Total ask** | | **≈ 6–7 cr, hard cap 8 cr** |

**Pre-specified design (to be frozen with a sha256 sidecar before launch, as `docs/reviewer2_followup.md` §12).**

* Systems on the frozen 192 AudioCaps prompts: P, `shortft` (E3, exists), `longft` (new); CRN `x_T` shared per
  (context, prompt) with the frozen P / P+FT clips.
* Estimands: `R_lf(d) = CLAP(longft) − CLAP(P)`, `J_lf = R_lf(10.24) − R_lf(3.84)`; **primary contrast
  `ΔJ = J_lf − J_sf`, paired per prompt** (both checkpoints are evaluated on the same prompts and noise), prompt
  bootstrap CI. Secondaries: `R_lf(3.84)` vs 0; `R_lf(10.24) − R_sf(10.24)` (does training at the evaluation
  duration buy more gain *there*?); `R_lf` vs the released `R` (descriptive, 2 % of the budget).
* Readings: **lo95(ΔJ) > 0** → the training duration modulates the interaction: specialisation contributes and the
  abstract keeps "does not support" (not "contradicts"); **CI of ΔJ inside ±0.025 (SESOI)** → the interaction is
  independent of the training duration: the operating-point reading is established at this budget and item 7's
  hypothesis gains its discriminating datum; **hi95(ΔJ) < 0** → reversed, reported as is; otherwise UNRESOLVED.
  `R_lf(3.84)` and `R_lf(10.24)` both unresolved → UNINFORMATIVE, declared in advance.
* Free secondary: save the step-7 500 checkpoint of the long arm (≈ matched audio-seconds to E3's 20 000 short
  steps; the trainer already checkpoints every 5 000 steps — add 7 500). Evaluate it only under a separate
  +1.1-cr GO.
* Compute-discipline record (AGENTS.md): CPU infeasible for 20 000 U-Net updates and 384 diffusion clips; T4 by the
  device rule (every frozen clip came from a T4); cap enforced by `job_watchdog.py` (Running-time clock).

**Wording, 0 cr, independent of the run:** "contradicting the prediction of training-duration specialization"
(abstract) and "a direct intervention contradicts" (conclusion) → "does not support"; Sec. 4.2's "It produces the
opposite ordering" stays (it is a description of E3). This is the reviewer's camera-ready condition and costs nothing.

**Verdict: correctable; the single most valuable remaining experiment; needs ≈ 6–7 cr of new credit.**

## 2. Item 2 — a 2×2 (dense and P, each 20 000 steps at 3.84 and 10.24 s) — **FEASIBLE, ≈ 2× the cost, lower value per credit**

**What the record says.** DENSE-FT-CLOSURE (2026-08-31): Singh's dense fine-tuned checkpoint was deleted (author
confirmation); the supervisor closed "approximate dense-FT reconstruction" because the source recipe is
under-specified (step count and trainable modules unknown). Those two facts are unchanged today: **the matched
control is still not recoverable.** What the reviewer now proposes is different — not a reconstruction of Singh's
run but a *reduced-scale paired analogue* at 20 000 steps, made possible by the fact that the paper now trains.

**Re-verification.**

| Requirement | Status |
|---|---|
| Dense start checkpoint | `audioldm-m-full.ckpt` on disk (md5 = official release); `build_backbone("dense")` materialises its EMA |
| Trainer | the same script with `build_backbone("dense")` and the `unet_is_71M` assertion relaxed; the frozen-VAE/CLAP path is model-agnostic |
| VRAM, 3.84-s arm | AdamW FP32 on 415.96 M parameters = 1.66 (weights) + 1.66 (grads) + 3.33 (moments) = **6.7 GB** + 2.1 GB frozen VAE/CLAP/vocoder + activations (dense channel_mult [1,2,3,5]; `gate0-smoke-1` dense+LoRA peaked at 5.4 GB at latent 96, so activations ≈ 1.5–2 GB) → **≈ 11 GB, fits a T4** |
| VRAM, 10.24-s arm | activations ×2.67 → ≈ 4–5.5 GB → **≈ 14–15 GB, at the T4 limit**: batch 1 × 2 gradient-accumulation steps (a documented recipe deviation, ≈ +20 % time) or a 24-GB class (L4/A10G, ≈ 1.5–2× the hourly rate) |
| Speed | `gate0-smoke-1` measured 0.307 s/step for a full backward through the dense U-Net at latent 96; + the 416 M-parameter AdamW update → ≈ 0.36–0.40 s/step at 3.84 s; ≈ 0.9–1.1 s/step at 10.24 s (bench decides) |
| Evaluation | dense-architecture generation on 192 × 2 durations per arm; dense per-WAV cost ≈ the pruned one (Arm-D native 0.0036 cr/WAV) |

**Cost of the two dense arms** (on top of item 1): training 20 000 steps at 3.84 s ≈ 2.0–2.2 h ≈ 1.8–2.0 cr; at
10.24 s ≈ 5–6 h ≈ 4.5–5.5 cr on a T4 with accumulation (or ≈ 5.5–8 cr on a 24-GB class); evaluation 2 × 384 WAVs
≈ 2.3 cr; two jobs' overhead 0.3 → **≈ 9–11 cr point, cap ≈ 13 cr**. Full 2×2 including item 1: **≈ 15–17 cr point,
caps ≈ 20 cr, plus ≈ 1.5 cr of Studio time.**

**Scientific caveats that lower the value per credit (to be stated if it runs).**

1. AudioLDM-M-Full was **already fine-tuned on AudioCaps for 0.25 M steps by its authors** (ledger, M0). Twenty
   thousand further steps start from a model that has little AudioCaps headroom; the public dense text-FT reference
   (a much longer, different-corpus fine-tune) moved dense CLAP by only −0.022 (3.84 s) / +0.091 (10.24 s). The
   dense arms' `J_dense` may well be unresolved at n = 192 (paired MDE ≈ 0.03–0.04): a real risk of paying ≈ 10 cr
   for an UNINFORMATIVE cell, which the pruned arms do not carry (P has 0.2–0.3 CLAP of pruning damage to repair).
2. It is still not "the matched dense control": Singh's dense step count is unknown, so the 2×2 is a reduced-scale
   analogue and the paper would keep the limitation sentence (softened, not removed).
3. One training run per cell, no training-seed replication → a descriptive 2×2, as the 2026-08-31 audit already
   noted for the (then hypothetical) dense-FT comparison.

**Verdict: technically feasible (checkpoint, code, data, T4 memory with one documented deviation), ≈ 10 cr on top of
item 1, with a material risk of an uninformative dense cell. Recommendation: run item 1 first; add the dense pair only
under a separate ≥ 12-cr authorization, 3.84-s arm before 10.24-s arm (the cheap arm answers "does dense gain
anything at 3.84 s at this budget?", which is the half of the 2×2 that item 7 needs).**

## 3. Item 4 — "plateaus at 15.36 s" overstates D4 = +0.021 [−0.023, +0.067] — **WORDING; more power NOT justified**

* The pre-specified reading rule (`docs/reviewer2_followup.md` §6) labels a CI containing 0 with |point| < 0.025 as
  "PLATEAU", but the reviewer is right that the interval admits a +0.067 step (40 % of R). The statement the data
  support is "does not clearly increase beyond 10.24 s" (equivalently: "peaked at the fine-tuning duration" is
  excluded, "still increasing" is not).
* Could more prompts settle it? Half-width at n = 96 is 0.045; the other 96 prompts of the frozen manifest at 15.36 s
  (P, P+FT: 192 WAVs at latent 384, 0.00479 cr/WAV → **≈ 1.1 cr, cap 1.3**) would give n = 192 and ≈ ±0.032 — still
  unable to bound the step inside ±SESOI. Bounding it within ±0.025 needs n ≈ 96 × (0.045/0.025)² ≈ **310 prompts**,
  i.e. a new manifest and ≈ 860 WAVs (≈ 3.7 cr) for a question whose answer does not change any conclusion.
* **Verdict: correct the wording (0 cr); do not buy power for it.** Also change "The gain therefore plateaus" (Sec. 4.1)
  and "The plateau at 15.36 s removes…" (Sec. 5) to the weaker form.

## 4. Item 5 — severity-1 heterogeneity (+0.044 on the original 80 vs the new 96) — **CORRECTED at 0 cr (CPU)**

`scripts/research/r2_posthoc_pooled_anchors.py` → `configs/research/r2_posthoc_pooled_anchors.json` (post-hoc,
descriptive; the frozen Arm-D n = 80 value remains the pre-specified primary):

| Set | J | 95 % CI | R(3.84 s) | R(10.24 s) |
|---|---:|---:|---:|---:|
| frozen Arm-D 80 (paper) | +0.044 | [−0.000, +0.088] | +0.008 | +0.052 |
| new 96 (E8) | +0.169 | [+0.115, +0.222] | −0.032 | +0.137 |
| **difference new − frozen** (unpaired) | **+0.124** | **[+0.058, +0.194]** | | |
| pooled n = 176 (E8, pre-specified) | +0.112 | [+0.076, +0.149] | | |

The between-set difference is resolved and larger than the original effect, as the reviewer inferred. **What changed in
the selection** (recorded in the artifact): the 80 are a seeded-hash subset of the V1.1 96 (Convention-B hash draw over
the eligible AudioCaps test set), the 96 are the first 96 `prompt_index` entries of the severity-2 192 manifest
(different salt, pool disjoint by construction, and the 192 manifest additionally required 5-caption rows). Both draws
are outcome-blind; checkpoints, sampler, scorer convention and hardware class are identical; x_T seeds and generation
jobs differ. The per-set means show both components moving (P+FT at 10.24 s is 0.369 on the new set vs 0.304 on the
old; P at 3.84 s 0.136 vs 0.104), so nothing points at a single mechanism beyond prompt-set sampling. **Manuscript
action (0 cr):** report both subsets and the difference in Sec. 4.1, keep the pooled value as the higher-power
estimate with that caveat, and state the selection difference in one sentence.

## 5. Item 6 (methodological part) — dense hip-hop anchors cover 64 of 127 prompts — **CORRECTED at 0 cr (CPU)**

The E7 job did generate dense clips for the 63 extension prompts (`docs/reviewer2_followup.md` §4), so the anchors
can be pooled; the same script computes (post-hoc, descriptive):

| Cell (n = 127) | R | A_dense (dense − floor) | ρ_dense pooled | ρ_dense frozen 64 | ρ_dense ext 63 |
|---|---:|---:|---:|---:|---:|
| 3.84 s | +0.026 [+0.007, +0.044] | +0.110 [+0.091, +0.128] | **0.106 [0.031, 0.177]** | 0.038 | 0.172 |
| 10.24 s | +0.027 [+0.003, +0.050] | +0.108 [+0.087, +0.130] | **0.119 [0.015, 0.215]** | 0.020 | 0.241 |

The battery discriminates for the dense model on all 127 prompts (lower bounds > SESOI), so the floor objection stays
answered at full coverage. But the fraction of the dense gap that recovery closes on hip-hop is **≈ 11–12 % pooled**,
not the "about 2 %" of the pre-specified 64-prompt subset — the extension prompts behave differently (ρ ≈ 0.17–0.24).
**Manuscript action (0 cr):** Table 1's ρ_dense column and footnote, and Sec. 4.3's "closes only about 2 % of the dense
gap", should report the pooled n = 127 value with the between-subset spread; the hip-hop conclusion ("much smaller
than AudioCaps' 0.44–0.84") is unchanged.

## 6. Item 7 — "3.84 s is an out-of-distribution operating point for the U-Net regardless of fine-tuning" — **FORMULATE; existing data test part of it; item 1 supplies the discriminating datum**

**Hypothesis H_OP (reviewer).** Short generation is out of distribution for AudioLDM-M's U-Net irrespective of any
fine-tune; fine-tuning gains only express where the base model works. **Hypothesis H_SPEC.** Recovery gain is
largest at the training operating point.

**Predictions that separate them.**

| Prediction | H_OP | H_SPEC | Data |
|---|---|---|---|
| (a) the dense model's own alignment at 3.84 s is degraded beyond the scorer's crop effect | yes | silent | **Not supported as stated.** Paired sev-2 192 (`xsev_dense_192_control_result.json`, `draft5_floor_ceiling_result.json`): floor-corrected duration response of dense +0.142 [+0.111, +0.172] vs real-audio crop +0.150 [+0.133, +0.166] — dense loses with duration exactly what real audio loses under CLAP; dense keeps 74 % of real audio's above-chance alignment at 3.84 s vs 81 % at 10.24 s (mild). P is near chance at both durations (+0.020 / +0.035 above floor) |
| (b) every fine-tune, whatever its duration or recipe, gains ≈ 0 at 3.84 s | yes | no (a short-trained model should gain at 3.84 s) | **Supported so far:** E3 +0.009 [−0.006, +0.024]; dense text-FT −0.022 [−0.061, +0.017]; released recovery +0.085 (small relative to +0.244) |
| (c) a checkpoint trained at 10.24 s gains no more at 10.24 s, relative to 3.84 s, than one trained at 3.84 s (ΔJ ≈ 0) | yes | no (ΔJ > 0) | **Missing — item 1** |
| (d) KL / PANNs event metrics of dense vs real at 3.84 s (a non-CLAP view of (a)) | degraded | silent | dense WAVs at both durations exist on the 192 prompts; scoring is CPU (0 cr, ≈ 30 min) — optional |

Existing data therefore support a **weaker** form than the reviewer's: under CLAP the base model is *not* broken at
3.84 s (its alignment tracks real audio's), yet no fine-tune tested so far expresses a gain there. Sec. 5 can be
built around that formulation ("fine-tuning gains express at the longer operating point; the base model's own
alignment does not explain the difference"), with (c) named as the open test — which is exactly item 1. The
author-listening remark ("noise in both systems at 3.84 s") concerned P and P+FT at severity 2, not the dense model,
and cannot be upgraded to human evidence (see §8).

**Verdict: 0 cr to formulate and to add predictions (a), (b); item 1 delivers (c); (d) is an optional CPU analysis.**

## 7. Items deferred by instruction (presentation) — noted only

Item 3 (revision narrative in the manuscript), item 4's wording, item 6's PANNs-capture definition, uncited [6] [7]
[17], Discussion length, repository URL. All 0 cr; none changes evidence. One provenance slip found while verifying
Draft 13's numbers (`scripts/research/paper_figs/verify_draft13_numbers.py`, 38/40 OK): the primary interaction is
printed as **J = +0.159 [+0.131, +0.188]** in Sec. 4.1 and Table 1, but the frozen primary artifact
(`xsev_result.json`, `PRIMARY_A.J`) gives **[+0.131, +0.187]** (Draft 12 and the companion print 0.187); the 0.188
comes from the re-bootstrap `released_J` in `r2_E3_result.json` (a different seed namespace). Camera-ready: use the
frozen 0.187 in both places. The Fig. 1 coordinates and all other 38 printed numbers reproduce from the artifacts.

## 8. Constraints re-tested and still binding

* **Singh's dense fine-tuned checkpoint:** deleted (author confirmation 2026-08-31). Nothing new. Not recoverable.
* **Human listening:** cancelled pre-launch because the co-author did not approve it without ethics clearance
  (`docs/listening_study_closure.md`); a compute budget cannot correct it. The author listening stays descriptive.
* **Credit:** the 15-cr ceiling is exhausted (§0). Every GPU item above is conditional on a new, explicit
  authorization; nothing has been launched.

## 9. Summary

| Item | Nature | Correctable? | Cost | Action |
|---|---|---|---:|---|
| 1 — 10.24-s FT control (20 000 steps) | methodological | **Yes** (budget was the only obstacle) | ≈ 6–7 cr, cap 8 | freeze a §12 protocol addendum; launch on Gabriel's GO |
| 1 — "contradicting" → "does not support" | wording | Yes | 0 | camera-ready |
| 2 — dense 2×2 at 20 000 steps | methodological | Feasible, lower value/credit; still not the matched control | + ≈ 10 cr (≈ 17 total) | only under a separate ≥ 12-cr GO, after item 1; 3.84-s arm first |
| 4 — "plateaus" | wording (+ power question) | Wording yes; power no | 0 (1.1 cr would not resolve it) | camera-ready |
| 5 — severity-1 heterogeneity | analysis | **Done** | 0 | report both subsets + difference +0.124 [+0.058, +0.194] |
| 6 — anchors on 64/127 | analysis | **Done** | 0 | Table 1 / Sec. 4.3 to the pooled n = 127 (ρ_dense ≈ 0.11–0.12) |
| 7 — H_OP hypothesis | framing + test | Formulate now; (c) = item 1 | 0 (+ item 1) | Sec. 5 rewrite around the weaker, data-supported form |
| listening / matched dense control | constraints | No | — | unchanged |
