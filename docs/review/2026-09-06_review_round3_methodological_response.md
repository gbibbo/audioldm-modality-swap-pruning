# Third review of Draft 14 (Accept, 4/5) — methodological items re-verified against the record and costed

**Date:** 2026-09-06 (MVD). **Type:** read-only re-verification + two CPU-only post-hoc re-analyses (0 cr) + one additive
generator path with a CPU dry-run (0 cr). **Nothing launched; no WAV generated for science.**
**Manuscript under review:** Draft 14 (`icassp/icassp_operating_point.tex` → `icassp/sections/draft14_*.tex`, marker
`%% draft14-second-review`, bundle `icassp/icassp_operating_point_draft14_second_review.zip`, PDF sha256 `c895339e…`),
installed today as the current version; Draft 13 moved to `icassp/archive/` (sections, main file, Overleaf notes, delivery
note, PDF). **Instruction (Gabriel):** keep Draft 14 as the current version; for every *methodological* weakness, re-check
whether the constraint that produced the design decision still holds and, where it does not, work out in depth how the
correction would be run. Presentation items are deferred by instruction and only noted.

**Classification of the reviewer's six points.** Methodological: **1** (dense 2×2 / raw-vs-EMA), **3** (prompt-set
sensitivity at severity 2), **4** (hip-hop has no duration interaction) and the evidential half of **5** (what supports
"short generation is not broken"). Presentation, deferred: **2** (text-FT reference in Sec. 5 but not in Sec. 4.2),
**6** (ρ interval, URL, page budget) and the wording half of **5**.

## 0. Budget fact that conditions every "correctable" verdict

* Lightning `total_spent` = **135.5623 cr** (SDK `billing_service_get_user_balance()`, 2026-09-06 ≈ 22:45 UTC; `balance`
  still the static 5.0). Unchanged since the OUT_OF_FUNDS reading of 13:5x MVD (`docs/compute_budget.md`): Gabriel's 20-cr
  pool for the 2×2 campaign is spent (114.95 → 135.56 = 20.6 cr) and Lightning refused the last two tail jobs with
  `USER_STOP_WORKLOAD_REASON_OUT_OF_FUNDS`.
* **Every GPU item below therefore needs a top-up and a new explicit authorization.** CPU items cost only Studio time
  (≈ 0.27 cr/h). The two CPU re-analyses in this document are done; nothing GPU has been launched.

## 1. Item 1 — the dense 2×2 is "broken" (raw vs EMA); evaluate the EMA weights, or the raw dense baseline — **HALF NOT CORRECTABLE, HALF CORRECTABLE (cheap, GPU, blocked only by funds)**

**What the reviewer asks.** (a) If the EMA weights of the 20 000-step dense runs were saved, evaluate them. (b) Otherwise
evaluate the *raw* dense baseline, so that the −0.2 drop is either explained by the weight convention or shown to be
real degradation. (c) Close the consequence for the pruned arm: are the 20 000-step pruned checkpoints compared against
an EMA or a raw P, and can the mismatch bias J_3.84 and J_10.24 even though it cancels in ΔJ?

**(a) EMA weights of the 20 000-step runs — NOT correctable: they were never kept.**

| Fact | Evidence |
|---|---|
| The trainer never maintained an EMA | `scripts/research/e3_shortft_trainer.py`: `model.use_ema = False` (l.127); `save_unet` writes `unet.state_dict()` only (l.81-84); exported meta says `"weights": "raw (no EMA)"` (l.213) |
| The protocol declared it | `docs/reviewer2_followup_ext.md` §1: "Weights: **raw** (no EMA at this horizon), as E3" (frozen, sha256 sidecar) |
| What is on disk per job (`/teamspace/jobs/r2-{longft,denseft-s,denseft-n}/artifacts/…/r2_*/`) | final raw U-Net at 20 000 (`*_unet.pt`), raw mid-step 7 500 for the two 10.24-s arms (`*_unet_step7500.pt`), `resume/latest.pt` = raw weights + AdamW state at 20 000, `train_log.jsonl`, `trainer_report.json`, 384/362/384 eval WAVs. **No EMA shadow anywhere.** |

An EMA cannot be reconstructed after the fact: it is a running average over the whole optimizer trajectory, and only two
snapshots (7 500 and 20 000) exist. Averaging them would be a new, unregistered estimator, not the EMA the reviewer means.
**Verdict (a): not correctable.**

**(b) Evaluate the dense RAW baseline — CORRECTABLE; all pieces exist; ≈ 2.3 cr; blocked only by credit.**

* **The raw weights exist and are known to differ from the EMA.** `data/checkpoints/audioldm-m-full.ckpt` stores both the
  trajectory weights `model.diffusion_model.*` (690 tensors) and the EMA shadow `model_ema.*`; the EMA-convention audit
  (ledger 2026-08-2x, DECISION-V4-12) measured dense raw-vs-EMA **mean-rel 2.4 %, max-rel 20 %**, U-Net output max |Δ| ≈ 5.9.
  Every frozen "dense" clip in the paper used the EMA (materialized into the U-Net). The raw dense model has never been
  scored with CLAP (the only raw-weight run, M4-SCREEN 2026-08-20, used KL/IS at guidance 3.5 on other prompts).
* **Code — done today, CPU dry-run PASS (0 cr).** `scripts/research/reversal_xsev_gen.py` gains `--system dense_raw`: a
  bare dense-architecture U-Net (`channel_mult [1,2,3,5]`, built without a second full pipeline so it fits the 15 GB Studio)
  strict-loaded with the raw `model.diffusion_model.*` tensors (strict load proves those tensors ARE the dense
  architecture); every other code path is untouched. The dry-run (`--dry-run-cpu`, 1 prompt, 6 DDIM steps) generated a
  61 472-sample clip with the frozen seed of prompt 0 (`2823286605618390972`, identical to every other system's x_T for that
  prompt) and manifest `checkpoint_convention = dense_raw:936914a388905e1f`; output kept in the session scratchpad only.
  Contexts allowed: `ac_short`, `ac_native` (the two cells of the 2×2).
* **Design (to be frozen as a protocol addendum with a sha256 sidecar before launch).** 192 frozen AudioCaps prompts ×
  {3.84, 10.24} s = 384 WAVs, frozen recipe (DDIM 50 / 2.5 / η 0 / fp32), CRN-paired by prompt with the existing
  dense-EMA (`xsev_dense192`), `denseft_short`, `denseft_native`, P and P+FT clips. Estimands (paired per prompt):
  * `O(d) = CLAP(dense_raw) − CLAP(dense_EMA)` — the raw-vs-EMA offset at step 0;
  * `G′(d) = CLAP(denseft) − CLAP(dense_raw)` — the fine-tuning effect net of the weight convention, per dense arm;
  * `J′` per arm and `ΔJ′_dense` (algebraically identical to the reported ΔJ_dense: the baseline cancels);
  * `O(10.24) − O(3.84)` — the duration dependence of the convention offset (feeds (c)).
* **Pre-specified readings.** (i) `|G′(d)| < SESOI (0.025)` at both durations → the dense "degradation" IS the weight
  convention; the dense 2×2 becomes a genuine short-budget control ("dense fine-tuning neither helps nor hurts at 20 000
  steps") and the two paragraphs of Sec. 4.2/5 collapse to one control sentence plus a table row. (ii) `hi95(G′(d)) < 0`
  with `|O(d)| ≪ |G(d)|` → the recipe (constant lr 1e-4, batch 2, no EMA, 20 000 steps) genuinely degraded an
  already-converged dense model; the raw-vs-EMA explanation in Sec. 4.2 and Sec. 5 must be **deleted**, not softened.
  (iii) Anything between → both effects, reported with the split. In every branch the text shrinks and the claim gets
  sharper, which is why the reviewer calls it "cheap and decisive".
* **Cost (T4 0.89 cr/h; DENSE per-WAV rate measured on `r2-denseft-s`: ≈ 0.0041 cr/WAV at mixed 96/256, the lesson of
  `docs/compute_budget.md` 2026-09-06 14:4x; job overhead 0.145 cr).**

  | Component | Basis | Point (cr) |
  |---|---|---:|
  | 384 dense WAVs (192 × 2 durations) | 384 × 0.0041 | 1.57 |
  | Provisioning / lifecycle | one job | 0.15 |
  | **Job `r3-denseraw` point / hard cap (×1.2)** | | **1.7 / 2.1** |
  | Studio hours (launch, watchdog, CPU scoring ≈ 2 h) | 2 × 0.27 | 0.55 |
  | **Total ask** | | **≈ 2.3 cr, hard cap ≈ 2.7** |

  A first-64 pilot (128 WAVs, ≈ 0.7 cr) is possible but leaves the pairing with the 192-prompt dense cells incomplete;
  the full 192 is recommended. **CPU alternative rejected:** the dense U-Net on this CPU runs ≈ 5–8 min per clip → 384 clips
  ≈ 30–50 Studio-hours ≈ 8–14 cr, more expensive than the T4 and in breach of the device rule (every frozen clip is T4).
* **Compute-discipline record (AGENTS.md).** (1) CPU unsuitable: 384 diffusion clips at 5–8 min each. (2) GPU-only
  work: the generation; scoring, floors, bootstrap, verdict and manuscript are CPU. (3) Smallest class: T4 (device rule).
  (4) Cap: `--cap` 2.1 cr on the job via `scripts/ops/launch_job_with_watchdog.sh`; self-bounded (384 WAVs, exits after
  the last write).

**Verdict (b): correctable, the cheapest decisive experiment left, ≈ 2.3 cr; needs a top-up and Gabriel's GO.**

**(c) The consequence the text does not close — 0 cr now, quantified by (b).** Facts from the frozen protocol and the
audit: **P (`pruned2_A`) is EMA-derived** (A′ L1 selection applied to the dense **EMA**), and **both 20 000-step pruned
checkpoints are raw exports**; so `R_sf(d)`, `R_lf(d)`, `J_3.84` and `J_10.24` are raw-fine-tune-minus-EMA-derived-P
differences and each may carry a raw-vs-EMA offset, while `ΔJ_pruned` is raw-vs-raw with the P terms cancelled
(`docs/reviewer2_followup_ext_results.md` §Audit). Whether that offset is duration-dependent is *unknown*: the dense arm
shows the drop is larger at 10.24 s (`J_ds = −0.051 [−0.090, −0.013]`, `J_dn = −0.057 [−0.091, −0.023]`) but cannot
separate convention from degradation. `O(10.24) − O(3.84)` from (b) is exactly the sign and scale of the bias the
convention can put on `J_3.84` and `J_10.24`. **Manuscript (0 cr, independent of the run):** one sentence in Sec. 3.2 or
4.2 stating that P is EMA-derived, the two fine-tunes are raw exports, the mismatch cancels in ΔJ and may bias the two
individual J values, whose sign is to be settled by the raw dense baseline (or, if (b) is not run, "is not quantified here").

## 2. Item 2 — the public dense text-FT reference appears in Sec. 5 but not in Sec. 4.2 or Table 1 — **PRESENTATION (deferred)**

Numbers are in `configs/research/r2_B_result.json` (pre-specified follow-up B, n = 96): `G_tf(3.84) = −0.022 [−0.061,
+0.017]`, `G_tf(10.24) = +0.091 [+0.042, +0.141]`, `J_tf = +0.113 [+0.051, +0.173]`. Either option (restore them to
Sec. 4.2 or drop the Sec. 5 clause) costs 0 cr; the round-2 figure specification already excludes it from panel (b).

## 3. Item 3 — two outcome-blind severity-1 samples differ by +0.124; is the 192-prompt severity-2 result equally sensitive? — **CORRECTED at 0 cr (CPU); GPU replication optional**

**What the record says about the reviewer's proposed sentence.** It is partly wrong on facts and must not be written as
proposed: the published-recipe check (E2b) ran on the **first 64 prompts of the same 192** (`--first-n 64`,
`draft5_pubrecipe_result.json`, `J_frozen|64` like-for-like), and the Human-CLAP / KL / PANNs sweeps **re-score the same
WAVs**. Neither is an independent prompt sample. The independent samples that do reproduce the interaction are **Clotho**
(96 different prompts, different dataset, same checkpoints: `J_clo = +0.112 [+0.079, +0.146]`, `r2_E5_result.json`) and
**severity 1 on two disjoint draws** (+0.044 / +0.169, different checkpoints). At severity 2 every AudioCaps number in the
paper (sweep, Holm family, secondary scorers, published recipe, 15.36-s extension) comes from the single 192 draw.

**NEW (post-hoc, CPU, 0 cr) — `scripts/research/r2_posthoc_review3.py` → `configs/research/r2_posthoc_review3.json`
(bootstrap conventions of `r2_verdict.py`, B = 10⁴, unit = prompt, seed namespace `…|POSTHOC3`).**

1. **Split-half of the 192.** `prompt_index` follows the ascending selection hash `sha256(salt|YTID|ytid)`
   (`xsev_select_audiocaps.py`; verified `selection_key` is sorted), so prompts 0–95 and 96–191 are exchangeable
   outcome-blind sub-samples of one draw — same checkpoints, jobs, x_T seeds and scorer calls as the primary result.

   | Half (n = 96) | R(3.84 s) | R(10.24 s) | J |
   |---|---:|---:|---:|
   | prompt_index 0–95 | +0.084 [+0.058, +0.110] | +0.242 [+0.197, +0.285] | **+0.159 [+0.116, +0.200]** |
   | prompt_index 96–191 | +0.086 [+0.058, +0.115] | +0.246 [+0.207, +0.285] | **+0.160 [+0.122, +0.198]** |
   | difference (unpaired, last − first) | | | **+0.001 [−0.056, +0.057]** |

   The severity-2 interaction is not sensitive to prompt sub-sampling at n = 96: the two halves differ by 0.001 where the
   two severity-1 sets differed by 0.124.
2. **Same prompts, both severities.** The severity-1 "new 96" (E8) ARE `prompt_index` 0–95 of this manifest, so the two
   severities can be compared on identical prompts (paired by prompt; x_T seeds differ between the two generation
   campaigns): `J_sev1 = +0.169 [+0.116, +0.221]`, `J_sev2 = +0.159 [+0.116, +0.200]`, paired difference **+0.010
   [−0.049, +0.067]** (per-prompt Pearson r = 0.28). On shared prompts the two severities agree; the severity-1
   heterogeneity is therefore between the V1.1-derived 80-prompt set and this manifest, not a property of the 192 set.

**GPU option — the reviewer's "second disjoint draw at severity 2": feasible, optional.**

| Requirement | Status |
|---|---|
| Prompt pool | frozen eligibility (`apply_exclusions` + V1.1-96 exclusion) leaves 868 ytids; minus the 192 drawn = **676 candidates** for a disjoint 192 (AudioCaps test rows carry 5 captions; the 5-row rule is asserted on the selection) |
| Selection | `xsev_select_audiocaps.py` with a new salt (`…|AUDIOCAPS-DRAW2|<date>`) and the 192 excluded; frozen manifest + sha256 before generation |
| Generation | `reversal_xsev_gen.py --system {pruned2_A, recovered2} --context {ac_short, ac_native}` on the new manifest (a new `CTX` entry with its own generation salt), T4 |
| Scoring / verdict | `r2_verdict.py` conventions; readings: `lo95(J_draw2) > 0` (reproduces); `two_sample(J_draw2 − J_draw1)` descriptive |

Cost (§A10 pruned per-WAV model 0.001329 + 9.0e-6·L): 192 × 2 × (0.00219 + 0.00363) = 2.24 cr + 0.15 overhead ≈ **2.4 cr,
cap 2.9**; Studio ≈ 3–4 h ≈ 1 cr → **≈ 3.4 cr, hard cap ≈ 4**. What it can and cannot buy: n = 192 per draw gives a
half-width ≈ 0.04 on the between-draw difference, so it can *replicate* the interaction but cannot bound the difference
within ±SESOI (that needs ≈ 600 prompts per draw). **Recommendation:** the split-half, the same-prompt cross-severity
comparison and Clotho already answer the reviewer's question at 0 cr; the second draw is lower priority than item 1(b).

**Manuscript sentence (0 cr), corrected version of the reviewer's proposal:** "Clotho (96 held-out prompts) and the
severity-1 replication on a disjoint draw reproduce the interaction, and the two halves of the 192-prompt severity-2
sample give J = +0.159 [+0.116, +0.200] and +0.160 [+0.122, +0.198]; the published sampler and the secondary scorers
re-use the same prompts and are robustness checks, not new samples."

## 4. Item 4 — hip-hop shows no duration interaction (R 0.026 → 0.027) — **ONE SENTENCE, 0 cr; the numbers exist**

* **Pre-specified (E7, `r2_E7_result.json`):** `J_music,127 = +0.001 [−0.026, +0.028]` (n = 127) — an interval essentially
  inside ±SESOI. `ρ_dense` pooled 0.106 [0.031, 0.177] / 0.119 [0.015, 0.215].
* **NEW (post-hoc, CPU, same artifact as §3):** AudioCaps − hip-hop difference of J (unpaired) = **+0.158 [+0.120,
  +0.197]**; AudioCaps − Clotho = **+0.047 [+0.003, +0.090]** (Clotho keeps most of the interaction, +0.112).
* **Reading for the sentence.** The duration interaction is largest where recovery is largest (AudioCaps, R ≈ 0.24 at
  10.24 s), intermediate on Clotho (R ≈ 0.21, J ≈ 0.11) and absent on hip-hop where recovery itself is ≈ 0.03 at both
  durations. The interaction scales with the recovered gain rather than existing independently of it, which is what the
  "expression of adaptation gains" hypothesis of Sec. 5 predicts and a "long generation is easier for everyone"
  account does not (the dense model's own alignment is above chance on hip-hop at both durations). Caveat for the same
  sentence: on hip-hop the gain is small relative to its interval, so the absence of an interaction is bounded at
  ±0.03, not shown to be exactly zero.

## 5. Item 5 — "short generation is not broken" rests on one CLAP comparison; soften — **WORDING; the record supports the reviewer; not correctable by compute**

* The supporting comparison is scorer-internal (dense floor-corrected response +0.142 [+0.111, +0.172] vs real-audio
  crop response +0.150 [+0.133, +0.166], both CLAP). No perceptual evidence exists: the blinded author listening
  (`docs/review/2026-09-04_author_listening_1_responses.md`, descriptive, one listener) recorded "ruido estrambótico
  insoportable" in **both** systems at 3.84 s (B2) and constant high-pitched noise (B10, B13) — about P and P+FT at
  severity 2, not the dense model — and the human listening study was **cancelled pre-launch** for lack of ethics
  clearance (`docs/listening_study_closure.md`; co-author decision). Nothing has changed: **no human evidence can be
  added, and a compute budget cannot correct it.** Adopt the reviewer's wording ("under CLAP, base-model degradation does
  not account for the interaction").
* **Optional, 0 cr, not required by the reviewer:** the round-2 prediction (d) is still unrun — KL and PANNs top-10 capture
  of the dense model vs the real clip at 3.84 vs 10.24 s on the existing 192 dense WAVs (CPU ≈ 30 min). It would let the
  sentence read "under CLAP and event-level metrics", a modest strengthening. Run only if Gabriel wants the extra clause.

## 6. Item 6 — minor (presentation, deferred; noted)

* **ρ_dense interval for hip-hop [0.015, 0.215] is wide because ρ is a ratio** whose denominator (dense − P ≈ +0.22 on
  hip-hop) has its own noise and whose numerator (R ≈ +0.027) is small relative to its interval; the reviewer's clause is
  correct and costs nothing. The rho routine is `Boot.ratio` (ratio of means, bootstrapped jointly).
* **URL under double-blind:** ICASSP 2027's Paper Kit states the regular review is *not* double-blind
  (`docs/reviewer2_response_manuscript.md` §7); the citation may stay. Gabriel decides.
* **Page budget:** `pagecheck_times.py` on the installed Draft 14 (Times metrics; the Overleaf-faithful check) → 5 pages,
  body ends at the references heading on page 5 (2 characters of body on page 5), verdict **fits 4 content pages**, 0
  overfull boxes reported. The two placeholder boxes occupy the intended footprint (0.49 textwidth × 4.80 cm each), so the
  final two-panel figure must keep that footprint or the text will spill.

## 7. Provenance found while installing Draft 14 (0 cr)

* **`scripts/research/paper_figs/verify_draft14_numbers.py` (new): 32/34 OK.** The two misses are the same number: the
  primary interaction's upper bound is printed as **0.188** in Sec. 4.1 and in Table 1's J row, while the frozen primary
  artifact `configs/research/xsev_result.json` (`PRIMARY_A.J`) gives **0.187**. This is the round-2 camera-ready item B1,
  applied to Draft 13 at `f2a5d0c` and lost in the Draft 14 rewrite (the 0.188 comes from the re-bootstrapped `released_J`
  of `r2_E3_result.json`, a different seed namespace). Camera-ready: two occurrences → `[+0.131, +0.187]`. All other 32
  printed numbers (abstract, Secs. 4.1–4.4, Table 1 both blocks, ΔJ values, hip-hop pooled row, severity-1 heterogeneity,
  crop analysis) reproduce from the committed artifacts.
* **Generator manifest label.** `reversal_xsev_gen.py` wrote `recipe.weight_convention = "ema"` for every system,
  including the raw-export systems (`shortft`, `longft`, `denseft`); the truth is in `provenance.checkpoint_convention`
  (`longft_raw:b4b1bddf…`, etc.). Fixed today for all future manifests (label is now `"raw"` for `shortft/longft/denseft/
  dense_raw`); the frozen manifests are left untouched and this note is the record.

## 8. Constraints re-tested and still binding

* **Singh's dense fine-tuned checkpoint:** deleted (author confirmation 2026-08-31, DENSE-FT-CLOSURE). Nothing new; not
  recoverable. The million-step dense control stays the declared missing experiment.
* **EMA weights of the 20 000-step runs:** never kept (§1a). Not recoverable.
* **Human listening:** cancelled pre-launch (no ethics clearance). Cannot be corrected by compute.
* **Credit:** `total_spent` 135.56 cr; Gabriel's 20-cr pool exhausted; Lightning returned OUT_OF_FUNDS. Every GPU item is
  conditional on a top-up and an explicit GO; nothing has been launched.

## 9. Summary

| Item | Nature | Correctable? | Cost | Action |
|---|---|---|---:|---|
| 1a — evaluate the saved EMA weights | methodological | **No** (never saved; not reconstructible) | — | state it in Sec. 3.2 |
| 1b — evaluate the raw dense baseline | methodological | **Yes** (raw tensors exist; `dense_raw` path dry-run PASS) | ≈ 2.3 cr, cap 2.7 | freeze addendum; launch `r3-denseraw` on GO + top-up |
| 1c — P EMA-derived vs raw fine-tunes | wording + (1b) | Yes | 0 (+1b) | one sentence now; sign/scale from 1b |
| 2 — text-FT reference placement | presentation | Yes | 0 | Gabriel's choice; numbers in `r2_B_result.json` |
| 3 — severity-2 prompt sensitivity | analysis | **Done** (split-half +0.159 / +0.160; same-prompt sev-1 vs sev-2 +0.010) | 0 | corrected sentence; second draw optional (≈ 3.4 cr) |
| 4 — hip-hop has no interaction | analysis | **Done** (J_hh +0.001 [−0.026, +0.028]; AC − HH +0.158 [+0.120, +0.197]) | 0 | one sentence in Sec. 4.3 or 5 |
| 5 — soften the CLAP-only claim | wording | Yes (wording); human evidence **no** | 0 | reviewer's phrasing; optional KL/PANNs view (0 cr) |
| 6 — ρ clause, URL, page budget | presentation | Yes | 0 | clause; URL allowed; Draft 14 fits 4 pages |
| provenance — J upper bound 0.188 → 0.187 | camera-ready | Yes | 0 | two occurrences |
