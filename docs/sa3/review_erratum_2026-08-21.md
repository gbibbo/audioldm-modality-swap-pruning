# Erratum + methodological corrections (Gabriel's review, 2026-08-21)

Gabriel's morning review found several methodological problems in the overnight adversary/RQ1
analysis. **No new GPU was spent** — all corrections use the 464 wavs + pilot JSON already on disk
(plus cheap CPU field-norm / OpenL3 recomputes). The overnight conclusion **changes materially**.

## E1 — Seed-resolution floor was on the wrong statistical scale (task 1)

`analyze_adversary.py` built `r_CLAP` from the 95th pct of **prompt-to-prompt** CLAP differences
between seed streams, then compared it against differences of **system panel means** — two different
scales. **Corrected** (`reanalyze_adversary.py`): `r_CLAP` = 95th pct of the **10 pairwise
panel-mean CLAP differences** among the R=5 dense-8 streams.

* OLD (invalid): `m_CLAP = 0.164`. **Corrected: `m_CLAP = 0.0399`** (4× tighter). The old floor was
  far too permissive. `m_KL` similarly: OLD 3.50 → corrected 1.09 (4-sample panel-mean null;
  a full 10-pair pairwise KL floor needs re-scored posteriors — `kl_floor_pairwise.py`, pending).

## E2 — Latency comparator was hardcoded to dense-7 (task 3)

The rule is *nearest measured latency*. Smoke latencies: `dense7 = 0.514 s`, `dense8 = 0.600 s`,
`skip@8 = 0.560 s`. **0.560 is nearest to dense-8 (Δ0.040 < 0.046)**, not dense-7. `reanalyze_adversary.py`
now reports the bracket {dense7, dense8}, marks dense-8 as nearest, and gives deficits vs BOTH.
(Latency is only measured for skip-5@8 in the smoke; treated as representative for all single-block
removals — a small approximation, flagged; per-block latency would need a GPU pass.)

## E3 — Corrected single-block verdict (CLAP, point estimates, no CI)

With `m_CLAP = 0.040` and the nearest comparator (dense-8), **9/20 blocks are inferior on CLAP**:
boundary **0, 1, 19** (CLAP → ≈0) **plus interior 5, 6, 7, 9, 17, 18** (deficits 0.05–0.10 > 0.040).
Blocks still within margin: 2, 3, 4, 8, 10–16 (deficits ≤ 0.04, several borderline: 2/8/12 ≈ 0.03–0.04).

**This overturns the overnight read** ("only 0/1/19 inferior; middle blocks within margin"), which was
an artifact of the 4×-too-loose floor. **CASE-E direction is stronger than reported** — 6 interior
blocks fail on CLAP alone against the latency-matched comparator. Still: point estimates, no bootstrap
CI (borderline blocks unresolved), CLAP-only, N=16 pilot — NOT a main-panel decision.

## E4 — FD-OpenL3 at N=16 is rank-deficient; use paired per-prompt drift (task 4)

`FD_openl3` estimates a 512-dim covariance from 16 clips → `rank(cov) ≤ 15`: a degenerate
small-sample Frechet. Not deleted (smoke FD kept, descriptive). Replaced for the pilot by a
**paired per-prompt cosine drift** (`paired_openl3_drift.py`, pre-registered: 1−cos of mean OpenL3
env/mel256/512 hop=1.0s embeddings, skip-g vs dense-8 per prompt, bootstrap CI, null = dense-stream
seed drift). [RESULT PENDING — running on CPU.]

## E5 — OpenL3 device/timing (task 5)

* `.venv-metrics` torch is **CPU-only** (`torch 2.2.2+cpu`, `cuda_avail=False`); a "GPU OpenL3 pass"
  would need a CUDA rebuild of the metrics venv.
* `torchopenl3.get_audio_embedding` takes **no device arg** and does not move the model to CUDA — a
  GPU pass would also need code changes. `score_e_metrics.ol3_embed()` never sets a device (unlike
  `passt` which does).
* CPU timing: **29.5 s/clip** at default hop (0.1 s) → ~3.8 h for 464 clips (impractical). At
  **hop=1.0 s → 4.5 s/clip** → the paired drift over ~400 clips is ~30 min (feasible, free).
* **Reproducibility fix:** `.venv-metrics` had lost `setuptools`/`pkg_resources` (resampy needs it);
  reinstalled `setuptools==75.8.0` (setuptools ≥ 81 removed `pkg_resources`). Requirements re-frozen.

## E6 — RQ1: the "10×" and the I_PT≈D_P collapse (tasks 6, 7)

* **10× caveat (task 6):** `D_P` and `D_B` are normalized by DIFFERENT denominators (`‖F_P‖²`,
  `‖F_B‖²`). The non-normalized damage ratio is `(D_P/D_B)·(‖F_P‖²/‖F_B‖²)`. Measuring the field-norm
  ratio (`rq1_field_norms.py`, CPU, no block removal) settles whether the 10× is real amplification
  or a normalization artifact. [RESULT PENDING.] The pilot did not save raw numerators; `pilot_fields.py`
  is fixed to save them going forward.
* **I_PT≈D_P (task 7, `rq1_reanalysis.py`, from disk):** ρ(D_P,I_PT)=+0.93, ρ(D_P,D_B)=+0.82,
  ρ(I_PT,W)=−0.34, ρ(D_P,W)=−0.21. **LOO-ranking-induced removal sets** (least-damage-to-remove;
  NOT confirmatory greedy, NO bootstrap floor): R_DP vs R_IPT disagree by **≤1 block** at k∈{2,4,6}
  (0 at k=4). → I_PT nearly collapses onto D_P; whether the ≤1 disagreement is real needs the floor
  (not computed). **Weak evidence against a separate PT-aware criterion.** R_DP vs R_DB disagree by
  2–3 blocks (base/post reorganization shows in the removal ranking).

## E7 — Not proceeding to main; pilot/sizing not finished (task 8)

`N_main` was never frozen (no bootstrap set-floor on the main panel) and `n_u` is not frozen
(`A_tan` not piloted). Per rc3.1 §6.0, the pilot/sizing must finish first. No main-panel work.

## E8 — Result-doc phrasing (task 9)

`pilot_fields_result.md` corrected: "post-training changed where structural importance sits" →
**"few-step post-training strongly amplifies relative interior-block sensitivity while much of the
global block-importance ranking remains shared with the base."** (Pending the field-norm check that
the amplification is not a normalization artifact.)

## What this means for the direction

If, after these corrections, interior blocks turn out **removable end-to-end and D_P already
identifies which**, RQ1 is interesting but does not by itself justify a PT-aware pruning method
(the I_PT≈D_P collapse points that way). The potential contribution then shifts to **RQ2**: `A_tan`
must say something **different from D_P/E** and predict real held-out adapter survival (case C). That
is the next scientific question — not a fast FD number.
