# RECOVERY-REVERSAL-V1 — prospective cross-domain ranking test

```
STATUS: FROZEN / PREREGISTERED
FROZEN BEFORE AUDIOCAPS BATTERY SELECTION
NO AUDIOCAPS OUTCOME DATA OBSERVED
```

Frozen 2026-08-29 (MVD), supervisor decision GO-to-freeze at 96×2×3. This preregistration is the
binding scientific/statistical contract for the RECOVERY-REVERSAL-V1 AudioCaps experiment. It is
frozen **before** the 96 AudioCaps ytids are selected (git history: freeze commit precedes the
manifest commit). A SHA256 of this file is recorded in `docs/recovery_reversal_v1.md.sha256` and
the ledger entry `RECOVERY-REVERSAL-V1-FREEZE`. Supersedes the AUDIT-1 `64×3` sizing sketch.

The sensitivity preflight (ledger RECOVERY-REVERSAL-V1-PREFLIGHT) did not trigger the predefined
STOP. We accept limited power to satisfy the complete decision rule for true effects very near the
SESOI, while being adequately powered for the larger effects that would make the reversal
scientifically compelling.

---

## 1. Scientific status (bounded)

A **post-hoc music-domain observation motivated a prospectively specified, held-out AudioCaps test
of a recovery-reversal hypothesis.** The published post-pruning "recovered" AudioLDM checkpoint
scores *worse* than the pruned-only checkpoint on a held-out 100%-music battery (frozen
`R_music = C_recovered − C_pruned = −0.0941`, CI95 `[−0.1241, −0.0646]`, n=64). That music result
is historical/post-hoc — the *motivation*. The **prospectively specified validation arm** is the
future AudioCaps-test experiment: does `recovered` *beat* `pruned` on held-out **in-domain**
(AudioCaps) captions under the **same** operating point that produced the music failure?

Catastrophic forgetting is **not** an established mechanism and is not claimed. Candidate
contribution stays narrow (§10).

---

## 2. Binding design

| Item | Value |
|---|---|
| Test prompts | **96 unique AudioCaps-test ytids** (canonical universe = `audiocaps_test_label.json`) |
| Caption | **1 deterministic caption per ytid** |
| Replicates | **2 paired generation replicates** per prompt |
| Standalone backbones | **dense EMA** (descriptive), **`p1_pruned_ema_reconstructed`**, **`p1_recovered`** |
| Total generations | **96 × 2 × 3 = 576 WAVs** (192 per system) |
| Dense EMA | **descriptive anchor only — cannot change PASS** |

**Operating point (identical to the historical music arm):**

```
clip length 3.84 s (latent_t 96) · DDIM 50 steps · eta 0 · guidance 2.5 · FP32
single generation per seed · no adapter · no best-of-3 · no generation-time CLAP selection
```

**Frozen constants (distinct namespaces — do not conflate):**

```
SELECTION_SALT  = "RECOVERY-REVERSAL-V1|AUDIOCAPS-TEST|2026-08-27"   # which ytids/captions
GENERATION_SALT = "RECOVERY-REVERSAL-V1|GENERATION|2026-08-27"       # generation noise seeds
BOOTSTRAP_SEED  = 20260827                                           # prospective statistics (PCG64)
B               = 10000
# The primary CLAP scorer RNG convention is UNCHANGED from the historical scorer:
# np.random.seed(20260826) reset once per 192-item system. Do NOT replace it with 20260827.
```

**Selection rule (frozen):** eligible unique ytids sorted ascending by
`selection_key = sha256(SELECTION_SALT | ytid)`, take the first 96.
**Caption rule (frozen):** per ytid, unique captions ordered UTF-8 bytewise; choose the caption
minimising `sha256(SELECTION_SALT | ytid | caption)`; record its index in that canonical order.
**Exclusions (frozen, in order):** (1) in canonical AudioCaps TEST; (2) not in AudioCaps TRAIN;
(3) not in the frozen 64 music-battery ytids; (4) not in the 44 Kim training-source ytids. No
semantic/Music/length/difficulty filtering; no manual replacement; counts recorded after each step.

**Generation seed convention (frozen):**
`generation_seed(ytid, r) = int.from_bytes(sha256(GENERATION_SALT | ytid | r)[:8], "big")`
(the frozen `derive_paired_seed` CRN convention). One seed per `(ytid, replicate)`; the **same**
initial latent `x_T` (shape `(1, 8, 96, 16)`) is used across dense/pruned/recovered — no
backbone-specific transformation. Replicates 0 and 1 are distinct. Implemented and tested in
`research_pruning/eval/reversal.py` (`generation_seed`) and `scripts/research/reversal_v1_gen_preflight.py`.

**Primary CLAP scorer convention (frozen, unchanged):** LAION `clap-htsat-fused` rev `365dea6e`,
SR 48000, `truncation="fusion"`, one 192-item seed-once call per system in canonical
`(prompt_index, replicate_index)` order, `np.random.seed(20260826)` reset once before each system,
OpenBLAS Haswell guard (`scripts/research/gate0_clap_scorer.py --score-groups`, reused unchanged).

---

## 3. Primary quantities & PASS gate

```
R_AC    = C_recovered,AC − C_pruned,AC          (per prompt: mean over the 2 replicates; paired)
R_music = −0.0941   [frozen historical, CI95 −0.1241 … −0.0646]   (durable baseline, §5)
I       = R_AC − R_music                          (cross-domain interaction)
```

Estimation: reduce each prompt to its paired mean FIRST, then prompt-cluster percentile bootstrap
(B=10000, PCG64(20260827)). For `I`, a joint two-sample bootstrap independently resamples the 96
AC per-prompt contrasts and the frozen 64 music per-prompt contrasts: `I* = mean(AC*) − mean(music*)`.

**PASS requires ALL of:**

```
1.  R_AC point            >= +0.025        # SESOI on the POINT estimate (practical floor)
2.  lower_CI95(R_AC)      >  0
3.  lower_CI95(I)         >  0
```

`0.025` is the predefined practical SESOI on the **point** estimate; it does **not** assert
`R_AC ≥ 0.025` at 95% confidence. Dense has **no** influence on PASS. Verdict code frozen now:
`scripts/research/reversal_v1_verdict.py` / `research_pruning/eval/reversal.py::primary_verdict`.

Also reported descriptively (no gate role): prompt sign fraction, median prompt contrast, the
ECDF-ready prompt-level contrast vector, and dense gaps for each standalone system.

---

## 4. FAIL interpretation (no automatic rescue)

If PASS fails — including the case where `recovered` also fails on AudioCaps — **V1 fails**. We do
**not** conclude operating-point brittleness, do **not** launch an alternate-operating-point
(10.24 s / DDIM 200 / guidance 3.5 / best-of-3) rescue, do **not** add a second pruning severity,
and do **not** invent post-outcome metrics. A negative is a valid, reportable result.

---

## 5. Historical R_music provenance (frozen; reconstructed without rescoring)

Rebuilt from the persisted phenom artifacts (no WAV rescored):
`scripts/research/reversal_reconstruct_music_contrast.py`; durable per-prompt baseline (tracked)
`configs/research/reversal_v1_r_music_clap.json`. `R_music = −0.0941`, CI95 `[−0.1241, −0.0646]`
(seed 20260826), 79.7% of prompts recovered<pruned. `phenomenon_verdict.json` md5 `326eb639`
confirmed; scorer rev `365dea6e`, scoring git `3d499dd`, generation git `22910045`. Regression
guarded by `tests/research/test_reversal.py` T3.

---

## 6. Robustness of the motivating music result (diagnostic; does NOT redefine R_music)

**Physical waveform panel** (`scripts/research/reversal_waveform_panel.py`, historical OFF WAVs,
3×192): the semantic-score deficit co-occurs with an objective shift in waveform statistics,
particularly elevated RMS (median 0.163 vs dense 0.135; recovered > dense in 75% of prompts, >
pruned in 100%) and elevated spectral centroid (2232 Hz vs dense 1655 Hz). **These signal-level
changes establish that recovered and pruned differ beyond the primary scorer, but do not by
themselves establish perceptual degradation.** Systematic clipping was **not** observed (peak
median 0.898, near-clip fraction ≈ 0). Elevated loudness / high-frequency tilt are **not** called
degradation absent perceptual evidence. Diagnostic only, not a gate; R_music unchanged.

**Human-CLAP corroboration** (`scripts/research/reversal_humanclap.py`): recovered < pruned
confirmed — `R_music_HC = −0.1446`, CI95 `[−0.1781, −0.1113]`, 85.9% of prompts negative. Human-CLAP
is a CLAP-family model, not an independent human evaluation; agreement supports **scorer
robustness**, not human preference.

---

## 7. SECONDARY prospective analyses for AudioCaps (frozen now, no PASS influence)

Both are decided **before** seeing AudioCaps, reported **regardless of direction**, and can **never**
change the primary PASS.

### 7a. Human-CLAP secondary (corroborative)

Score the **same** future AudioCaps WAVs (`p1_pruned_ema_reconstructed`, `p1_recovered`; dense
descriptively if convenient) with the exact preflight-validated Human-CLAP implementation. Define:

```
R_AC_HC = HC_recovered,AC − HC_pruned,AC
I_HC    = R_AC_HC − R_music_HC        (joint two-sample bootstrap vs the frozen historical HC music
                                       contrasts, configs/research/reversal_v1_r_music_humanclap.json)
```

Report: point estimate, prompt-clustered 95% CI, fraction of prompt-level contrasts > 0, and
`I_HC` point/CI. SECONDARY, CORROBORATIVE, **no SESOI**, **no multiple-metric rescue**. The
Human-CLAP result MUST be reported even if it disagrees with primary CLAP.

**Pinned provenance (frozen):**
```
model         sarulab-speech/human-clap-wsce-mae
revision      06788887d254df15db5c0ca9d54da39d46188063
safetensors   sha256 09357f504d52900cb1bc3bf2fe1f3173dd1702781ef0bdedb122a6e47d4c5c61
processor     laion/clap-htsat-fused ; sample rate 48000
convention    truncation="fusion", get_*_features cosine, np.random.seed(20260826) once per 192-item system
ordering      canonical (prompt_index, replicate_index), one 192-item call per system
environment   .venv-metrics (transformers 4.30.2, torch 2.2.2 CPU, librosa 0.11.0); OPENBLAS_CORETYPE=Haswell
```
Human-CLAP remains CLAP-family; agreement supports scorer robustness, NOT human preference.

### 7b. Waveform panel secondary (descriptive)

Apply the **same** frozen waveform-statistic definitions (`scripts/research/reversal_waveform_panel.py`
`clip_stats`: RMS, peak, near-clip fraction @ |x|≥0.99, crest_db, spectral centroid) to the future
AudioCaps systems. Purpose: whether the recovered loudness / spectral shift observed in music is
also present in-domain. Descriptive only, no PASS gate, no perceptual-quality interpretation without
perceptual evidence. Metric definitions are pinned by that versioned script; no new metrics after
seeing AudioCaps.

---

## 8. Denoiser functional drift is OUTSIDE the binding V1 contract

Denoiser-drift is **not** part of this preregistration and its execution is **not** part of V1
PASS. The earlier reference-free `N(0,I)`-at-arbitrary-`t` construction is withdrawn: at a finite
diffusion timestep a valid noisy state follows the timestep-dependent diffusion marginal / sampler
trajectory, not an arbitrary standard normal at every `t`, which would make low/intermediate-noise
interpretation questionable.

Recorded future preferred direction only (non-binding, `NOT PREREGISTERED`, review before any
implementation): *If pursued later, compare dense/pruned/recovered vector fields at identical states
taken from a common realistic generation trajectory — preferably the deterministic dense DDIM
trajectory for the same prompt and initial latent. Capture selected `x_t` states from that common
reference trajectory and evaluate all backbones at identical `(x_t, t, conditioning)`. Separate
prediction direction (cosine) from norm/scale. No gate, and no implication that local vector-field
similarity guarantees final semantic quality.* Do not implement now.

---

## 9. Sensitivity preflight summary (historical data only)

`scripts/research/reversal_sensitivity.py` (PCG64(20260827), 2000×2000): plug-in σ_between 0.097 /
σ_within 0.131. The interaction never binds (`P(lo95(I)>0)=1.000`, since R_music≪0), so V1 reduces
to detecting `R_AC`. P(all three PASS) at plug-in / 2× total-variance: 0.95/0.73 @R_AC 0.05,
1.00/0.96 @0.075, 0.83/0.54 @0.04, 0.43/0.28 @SESOI 0.025. Well-powered for R_AC ≥ 0.05;
underpower at the SESOI is intrinsic to `lo95>0` at effect=threshold. N and gate unchanged.

---

## 10. Publication framing (bounded — verify against primary sources)

Candidate contribution, narrowly: *evidence that the published post-pruning recovery artifact can
change the ordering between pruned and recovered AudioLDM models across domains under a controlled
common generation operating point.* Call it a **"recovery-induced cross-domain ranking reversal"**
only if the prospective AudioCaps arm establishes the opposite ordering. Precedent (check primary
sources, do not cite from memory): Singh et al. 2026 (studied artifact); Kumar et al. ICLR 2022
(ID↑/OOD↓); Thangarasa et al. MLSys 2025 (post-pruning SFT shifts behaviour); Takano et al. 2025
(Human-CLAP).

---

## 11. Budget / compute governance (block active)

Current Lightning balance ≈ **0.72 credits** (2026-08-28). The `0.5332 cr` central-chain "remaining"
in older ledger text is HISTORICAL, not the account balance. Projected V1 GPU envelope for the
576-generation arm ≈ **1.50 cr** > 0.72 → **GPU launch BLOCKED**. All V1 work to date is CPU-only,
0 credits, no GPU. Freezing this contract does not authorise generation.
