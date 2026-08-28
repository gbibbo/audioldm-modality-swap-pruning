# RECOVERY-REVERSAL-V1 — prospective cross-domain ranking test (DRAFT)

```
STATUS: DRAFT / NOT FROZEN
NO AUDIOCAPS OUTCOME DATA OBSERVED
NO GPU GENERATION AUTHORIZED
```

Authored 2026-08-28 (MVD). Supersedes the stale `64×3×2` sizing sketch in the
RECOVERY-REVERSAL-AUDIT-1 ledger entry wherever the two conflict; the sizing and gate below are
the later supervisor-confirmed specification. This document is a **draft preregistration**: it is
NOT frozen, the 96 AudioCaps ytids are NOT selected, and no generation/scoring is authorized.

---

## 1. Scientific status (bounded)

A **post-hoc music-domain observation motivated a prospectively specified, held-out AudioCaps
test of a recovery-reversal hypothesis.** The published post-pruning "recovered" AudioLDM
checkpoint scores *worse* than the pruned-only checkpoint on a held-out 100%-music battery
(frozen `R_music = C_recovered − C_pruned = −0.0941`, CI95 `[−0.1241, −0.0646]`, n=64). That
music result is **historical / post-hoc** and is the *motivation*, not the confirmation.

The **prospectively specified validation arm** is the future AudioCaps-test experiment: does
`recovered` *beat* `pruned` on held-out **in-domain** (AudioCaps) captions under the **same**
controlled operating point that produced the music failure? If yes, the ordering between the two
models reverses across domains. If `recovered` also fails on AudioCaps, **V1 fails** — we do not
conclude operating-point brittleness and do not launch a rescue experiment (see §8).

**Catastrophic forgetting is NOT an established mechanism here** and must not be asserted. The
candidate contribution is narrow (§7).

---

## 2. Design (intended, not frozen)

| Item | Value |
|---|---|
| Test prompts | **96 unique AudioCaps-test ytids** (deterministic selection; NOT yet drawn) |
| Caption | **1 deterministic caption per ytid** |
| Replicates | **2 paired generation replicates** per prompt |
| Backbones (standalone) | **dense EMA**, **`p1_pruned_ema_reconstructed`**, **`p1_recovered`** |
| Total generations | **96 × 2 × 3 = 576 WAVs** |
| Dense EMA | **descriptive anchor only** — NOT part of the PASS gate |

**Operating point — identical to the historical music arm** (the whole point is a common
controlled operating point):

```
clip length     3.84 s      (latent_t = 96)
sampler         DDIM, 50 steps, eta = 0
guidance        2.5
precision       FP32
generation      single generation per seed (NO best-of-3, NO CLAP selection)
```

**Selection & bootstrap constants (later decisions, not reconciled to the old audit):**

```
SELECTION_SALT = "RECOVERY-REVERSAL-V1|AUDIOCAPS-TEST|2026-08-27"
BOOTSTRAP_SEED = 20260827          # NB: prospective V1 seed; the HISTORICAL music arm keeps 20260826
```

Scorer: the frozen primary convention (LAION `clap-htsat-fused`, rev `365dea6e`, SR 48000,
`truncation="fusion"`, seed-once-per-192-item system, `get_*_features` cosine). Prompt-cluster
percentile bootstrap, **B = 10,000**, generator **PCG64(20260827)**.

---

## 3. Primary quantities & PASS gate

```
R_AC    = C_recovered,AC − C_pruned,AC          (paired, prompt-clustered)
R_music = −0.0941   [frozen historical, CI95 −0.1241 … −0.0646]   (descriptive baseline)
I       = R_AC − R_music                         (cross-domain interaction)
```

**Planned PASS requires ALL of:**

```
1.  R_AC point            >= +0.025        # SESOI on the POINT estimate (practical floor)
2.  lower_CI95(R_AC)      >  0
3.  lower_CI95(I)         >  0
```

`0.025` is the **predefined practical SESOI on the point estimate**. It does **not** claim
`R_AC ≥ 0.025` at 95% confidence. `R_music` enters `I` with its own historical uncertainty
retained (joint two-sample bootstrap), not as a known constant.

---

## 4. Historical music provenance (frozen; reconstructed without rescoring)

Rebuilt from the persisted phenomenon artifacts (no WAV rescored) by
[reversal_reconstruct_music_contrast.py](../scripts/research/reversal_reconstruct_music_contrast.py)
→ `artifacts/icassp_gate0/reversal_music_contrast.json` (regenerable; gitignored):

- 64 prompts × 3 paired replicates; pairing by generation seed across backbones.
- `R_music = −0.0941`, CI95 `[−0.1241, −0.0646]` — **reproduces the frozen ledger value exactly**
  (prompt-cluster percentile bootstrap, seed 20260826).
- 79.7% of prompts have `recovered < pruned`.
- Sources hashed; `phenomenon_verdict.json` md5 `326eb639` confirmed; scorer + generation git
  SHAs captured. Regression assertions fail loudly on any ordering/pairing/value drift
  ([test_reversal.py](../tests/research/test_reversal.py) T3).

---

## 5. Sensitivity preflight (CPU-only; historical data only)

[reversal_sensitivity.py](../scripts/research/reversal_sensitivity.py) →
`artifacts/icassp_gate0/reversal_sensitivity.json`. Random-effects decomposition of the
historical music paired differences, projected onto the future 96×2 design. **No AudioCaps data.**

- Variance plug-in from music: `σ_between = 0.0973`, `σ_within = 0.1314` (per-replicate). The
  3-rep music within-noise maps onto 2-rep AudioCaps as `σ_within²/2`, so the replicate asymmetry
  (music 3, AC 2) is handled explicitly.
- **Inflation interpretation.** *Primary:* scale BOTH components together (total-variance stress)
  at 1.0× / 1.5× / 2.0×. *Alternatives (transparency):* between-only 2× and within-only 2×.
  Justification: cross-domain generalisation can plausibly widen both prompt-level heterogeneity
  and generation noise; the primary stresses both, the alternatives localise the source.
- **The interaction I never binds:** `P(lower_CI95(I) > 0) = 1.000` in every cell, because
  `R_music ≈ −0.094` is strongly negative, so `I = R_AC + 0.094` is comfortably positive. **V1
  therefore reduces to detecting `R_AC` (point ≥ 0.025 AND lower_CI95 > 0).**

**P(all three PASS conditions), 2000 sims × 2000 bootstrap, PCG64(20260827):**

| true R_AC | plug-in (1.0×) | both 1.5× | both 2.0× | between 2× | within 2× |
|---|---|---|---|---|---|
| 0.025 | 0.43 | 0.33 | 0.28 | 0.31 | 0.32 |
| 0.040 | 0.83 | 0.65 | 0.54 | 0.65 | 0.66 |
| 0.050 | **0.95** | 0.85 | 0.73 | 0.84 | 0.85 |
| 0.075 | 1.00 | 0.99 | **0.96** | 0.99 | 1.00 |

Expected CI half-width ≈ 0.027 (plug-in) → 0.038 (2×). Point-estimate SD ≈ 0.014 → 0.020.

**Reading:** 96×2 is well-powered (≥0.95 plug-in; ≥0.73 under 2× total-variance stress) for
`R_AC ≥ 0.05`, and well-powered even under 2× for `R_AC ≥ 0.075`. It is only moderately powered
at `R_AC = 0.04` (0.83 plug-in, 0.54 under 2×) and **intrinsically underpowered right at the
SESOI (`R_AC = 0.025`)** — expected, since `lower_CI > 0` at an effect equal to the point
threshold cannot exceed ~0.5. This is a property of the gate, not a fixable N choice; **N and the
gate are NOT changed** (§8 discipline).

---

## 6. Robustness of the motivating music result (diagnostic; does NOT redefine R_music)

**Physical waveform panel** ([reversal_waveform_panel.py](../scripts/research/reversal_waveform_panel.py),
historical OFF WAVs, 3×192): recovered's poor music behaviour has an **objective signal-level
signature** — elevated loudness (RMS median 0.163 vs dense 0.135; recovered > dense in 75% of
prompts, > pruned in 100%) and a marked upward spectral-centroid shift (2232 Hz vs dense 1655 Hz).
Pruned-only is *quiet* (RMS 0.041). **Correction to the earlier small-sample claim:** on the full
panel recovered does **not** systematically clip (peak median 0.898, near-clip fraction ≈ 0), so
the previous "peak 0.959 / near-clipping" characterisation is downgraded to "elevated loudness and
high-frequency tilt, no systematic clipping." The degradation is physical, not a pure scorer
artefact — but this is diagnostic, **not a gate**.

**Human-CLAP corroboration** ([reversal_humanclap.py](../scripts/research/reversal_humanclap.py),
`sarulab-speech/human-clap-wsce-mae`, CPU): recovered < pruned **confirmed** —
`R_music(Human-CLAP) = −0.1446`, CI95 `[−0.1781, −0.1113]`, 85.9% of prompts negative. Same
direction as the primary CLAP, comparable-or-larger in its own scale. **Caveat:** Human-CLAP is a
CLAP-family model, not an independent human evaluation, and its published validation is not
music-specific; this shows the ordering is **not idiosyncratic to the exact frozen CLAP
checkpoint**, nothing about human preference. The frozen `R_music` (primary CLAP) is unchanged.

---

## 7. Publication framing (bounded — verify against primary sources before writing)

Candidate contribution, **narrowly**:

> evidence that the published post-pruning recovery artifact can change the ordering between
> pruned and recovered AudioLDM models across domains under a controlled common generation
> operating point.

Call it a **"recovery-induced cross-domain ranking reversal"** only if the prospective AudioCaps
arm actually establishes the opposite ordering. Until then it is a hypothesis.

Relevant precedent (do NOT write novelty claims from memory; check primary sources):
Singh et al., *Efficient Text-to-Audio Generation via Pruning* (2026, the studied artifact);
Kumar et al., *Fine-Tuning Can Distort Pretrained Features…* (ICLR 2022, ID↑/OOD↓ precedent);
Thangarasa et al., *Self-Data Distillation…* (MLSys 2025, post-pruning SFT shifts behaviour);
Takano et al., *Human-CLAP* (2025, corroborative scorer motivation).

---

## 8. Explicitly OUT of scope for V1 (do not implement)

- **No textual domain-distance / nearest-neighbour regression** — with two deliberately separated
  domains it mostly re-expresses domain membership through an arbitrary embedding.
- **No second pruning severity** — later replication decision after V1 results + budget review.
- **No alternate operating point** (10.24 s / DDIM 200 / guidance 3.5 / best-of-3). V1 asks first
  whether recovered beats pruned in-domain under the SAME operating point that produced the music
  failure. If recovered also fails on AudioCaps, V1 fails; brittleness is not auto-concluded.
- **No N or gate change** in response to the §5 power table (this is sensitivity, not post-data N
  optimisation).

---

## 9. Denoiser functional drift — DESIGN ONLY (`PROPOSED / NOT YET PREREGISTERED`)

A future CPU-only *descriptive* mechanistic analysis. The naive raw-L2 proposal is **rejected**:
`recovered`'s large gain/norm shift (ledger PHENOM-VALIDITY-GEOM) confounds any raw-L2 distance.
Requirements this design must satisfy: (1) not confounded by the norm shift; (2) separate
prediction *direction* from prediction *scale*; (3) symmetric `x_t` across both domains; (4) no
reference-audio latents (equivalent reference audio is not guaranteed symmetric across the music
battery and AudioCaps); (5) identical deterministic latent/noise construction on both domains;
(6) no inferential gate; (7) no claim that per-timestep closeness to dense implies final quality.

**Recommended exact symmetric construction (one, proposed):**

- **Reference-free noised latent.** For each prompt `p` (in either domain), replicate `r`, and
  timestep `t` from a fixed grid `T = {100, 300, 500, 700, 900}` on the 1000-step schedule, draw a
  standard-normal latent `z_{p,r,t} ~ N(0, I)` in the AudioLDM latent shape, seeded deterministically
  by `derive_paired_seed(SELECTION_SALT, ytid_p, r) ⊕ t` — the **same** RNG scheme on both domains.
  Use `z` directly as the noised sample `x_t` (the diffusion prior; no VAE-encoded reference audio,
  satisfying (3)(4)(5)).
- **Two separated read-outs** of each model's ε-prediction `ε_θ(x_t, t, c_p)` (classifier-free
  guidance OFF, to probe the denoiser itself), comparing `recovered` and `pruned` each against
  `dense`:
  - **Direction:** `cos( ε_model, ε_dense )` per `(p, t)` — invariant to the global scale shift.
  - **Scale:** `‖ε_model‖ / ‖ε_dense‖` per `(p, t)` — the norm/gain channel, reported separately.
- **Aggregation:** per-prompt means, prompt-clustered percentile bootstrap **for descriptive CIs
  only** (no threshold, no PASS). The mechanistic question: is `recovered`'s **direction** drift
  from `dense` larger in the **music** domain than in **AudioCaps** (domain-dependent functional
  drift), after the norm shift is separated into the scale channel?
- **No claim** that direction-closeness at any `t` implies final semantic quality (7).

Rejected alternative: VAE-encoding each clip's reference audio to build `x_t` — asymmetric
(availability differs across domains) and reintroduces a reference-latent confound. **Review
required before implementation.**

---

## 10. Budget / compute governance

- Current Lightning balance ≈ **0.72 credits** (Gabriel, 2026-08-28). The `0.5332 cr` figure in
  the older ledger is *historical remaining capacity from the retired 5-credit central-chain
  accounting*, NOT the current account balance.
- Planned GPU extension envelope for the 576-generation V1 arm: **≈ 1.50 cr**. This EXCEEDS the
  current balance, so **GPU launch is presently BLOCKED**. This block does not require any design
  change; it defers execution until the balance is raised.
- Everything in this document to date is **CPU-only, 0 credits, no GPU**.

---

## 11. Do-not-do (this block)

Do NOT freeze this document, select the 96 AudioCaps ytids, generate AudioCaps audio, run paid
GPU/scoring, or implement §9. STOP and report before any of those.
