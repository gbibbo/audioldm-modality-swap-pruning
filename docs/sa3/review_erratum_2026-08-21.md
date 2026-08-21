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

## E3 — Corrected single-block verdict (CLAP, PAIRED BOOTSTRAP CI, protocol 3-way rule)

With `m_CLAP = 0.040`, the nearest comparator (dense-8), and the protocol §9.2 rule applied to the
paired CLAP-deficit bootstrap CI (B=2000 over the 16 prompts) — inferior iff **lower** CI > margin;
non-inferior iff **upper** CI ≤ margin; else indeterminate:

* **INFERIOR: 0, 1, 6, 19** (3 boundary + interior block 6).
* **NON-INFERIOR (removable): 13, 14, 16.**
* **INDETERMINATE: 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 15, 17, 18 (13 blocks).**

**Both prior reads were wrong.** The overnight read ("only 0/1/19 inferior") used a 4×-too-loose
floor; my first correction ("9/20 inferior incl. 6 interior") used point estimates and ignored
sampling uncertainty. **The honest answer with CIs: N=16 is UNDERPOWERED to decide CASE E for the
middle blocks** — most are indeterminate. Only boundary blocks + interior block 6 are confidently
inferior; blocks 13/14/16 are confidently removable. This is direct evidence for task 8 (size
`N_main` before any CASE-E call). **Coherent cross-check:** block 6 is the single most field-amplified
interior block (non-normalized 10.2×, E6) and it is the one confidently-inferior interior block;
the confidently-removable 13/14/16 have low D_P/D_B — so D_P/D_B tracks the end-to-end verdict where
the pilot has resolution. CLAP-only; FD via the paired drift (E4); NOT a main-panel decision.

## E4 — FD-OpenL3 at N=16 is rank-deficient; use paired per-prompt drift (task 4)

`FD_openl3` estimates a 512-dim covariance from 16 clips → `rank(cov) ≤ 15`: a degenerate
small-sample Frechet. Not deleted (smoke FD kept, descriptive). Replaced for the pilot by a
**paired per-prompt cosine drift** (`paired_openl3_drift.py`, pre-registered: 1−cos of mean OpenL3
env/mel256/512 hop=1.0s embeddings, skip-g vs dense-8 per prompt, bootstrap CI, null = dense-stream
seed drift). **RESULT (`openl3_drift.json`):** null seed-drift p95 = **0.0148**; dense 4-step drift
(0.0048) is *below* seed noise (robust to step reduction). Using the lower-CI-vs-p95 rule, **only
boundary blocks 0, 1, 19 drift beyond seed noise** (0.021–0.031); **every interior block 2–18 is
within/below the floor** (0.003–0.012; block 5 = 0.0066). **This confirms the FD rank-deficiency
concern:** the smoke's alarming set-level `FD=99.9` for skip-5 was a small-sample Frechet artifact —
the trustworthy paired per-prompt drift says block-5 removal barely moves the acoustic embedding.

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
  ratio (`rq1_field_norms.py`, CPU, no block removal) settles this. **RESULT: `‖F_P‖²/‖F_B‖² = 0.889`**
  (panel; the two fields have nearly equal magnitude), so the normalization is NOT distorting the picture.
  The **non-normalized** damage ratio `‖F_P−F_P^−g‖² / ‖F_B−F_B^−g‖²` is **9–10× for interior blocks 5/6/7/9**
  (median 5.3× across all blocks, range 0.8–10.2×) and ~1.0–1.3× at the boundaries. **⇒ the 10× is REAL
  functional amplification, not a normalization artifact.** The RQ1 finding survives this falsifier.
  (`pilot_fields.py` now saves raw numerators for future runs.)
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

## Synthesis of the corrected pilot (the honest combined read)

Two independent end-to-end metrics after correction, N=16 pilot:

* **OpenL3 (acoustic paired drift):** only boundaries 0/1/19 beyond seed noise; **all interiors
  removable-looking**.
* **CLAP (semantic, bootstrap CI):** boundaries 0/1/19 + **block 6** confidently inferior; 13/14/16
  confidently removable; **13 blocks indeterminate** (underpowered).

⇒ **Single-block removal of interior blocks looks largely tolerable end-to-end at N=16; this is NOT
a clean CASE E.** The only interior flag is block 6 (CLAP alone) — the most field-amplified block
(non-normalized 10.2×), so `D_P/D_B` and the end-to-end verdict agree where the pilot can resolve.

**Consequence for the direction (matches Gabriel's "qué me interesa decidir después").** If interior
blocks are removable and `D_P` already identifies which (and `I_PT ≈ D_P`), RQ1 is scientifically
interesting but does **not by itself justify a PT-aware pruning method**. The potential contribution
is now RQ2: **`A_tan` must (a) differ from `D_P`/`E` and (b) predict real held-out adapter survival
(case C)**. That — not a faster FD number — is the decision-relevant next experiment. Before it,
finish sizing (`N_main` from the D_P/I_PT bootstrap; `n_u` from an `A_tan` pilot) per rc3.1 §6.0.

## What this means for the direction

If, after these corrections, interior blocks turn out **removable end-to-end and D_P already
identifies which**, RQ1 is interesting but does not by itself justify a PT-aware pruning method
(the I_PT≈D_P collapse points that way). The potential contribution then shifts to **RQ2**: `A_tan`
must say something **different from D_P/E** and predict real held-out adapter survival (case C). That
is the next scientific question — not a fast FD number.
