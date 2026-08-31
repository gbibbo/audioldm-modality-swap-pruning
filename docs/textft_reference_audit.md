# Part B0 — Public Dense Text-FT Reference (`audioldm-m-text-ft`) — CPU Audit + Verdict

**Type:** CPU-only provenance / structural-compatibility / comparability / power audit. 0 GPU, no
generation, no launch. Result artifact `configs/research/textft_checkpoint_audit.json` +
`configs/research/partb_power_analysis.json`.

**Role (frozen):** PUBLIC dense text-fine-tuning **reference** only. **NOT** Singh's deleted dense-FT
checkpoint, **NOT** a matched causal control. Forbidden language: "generic fine-tuning ruled out",
"matched dense control", "pruning causality established", "generic FT explains Singh recovery".

## B0.1 Provenance — **PASS**

Downloaded by immutable URL (Zenodo record 7813012). Verified against the official release:

| Field | Value | Match |
|---|---|---|
| bytes | 4,571,676,474 | ✓ |
| md5 | `036bc9b547a50f78b960ef8f14d0e1fb` | ✓ |
| sha256 | `d77d5a61785af82012edb8a72158d52592ac7c76d7f6ed51a048ec2dec8d5eca` | ✓ |

Documented as the medium AudioLDM fine-tuned with **AudioCaps + MusicCaps** audio-text pairs.

## B0.2 Structural compatibility — **PASS**

Compared against `audioldm-m-full.ckpt` (dense pretrained; our pipeline strict-loads it 690/690).

* **Identical layout:** 2299 shared `state_dict` keys, **0 shape mismatches**, **0 text-ft-only keys, 0
  full-only keys**. Architecture `channel_mult [1,2,3,5]`, `model_channels=192` (input conv
  `[192,8,3,3]`), EMA present (692 keys).
* **Strict-load into our production U-Net: 690/690, 0 missing / 0 unexpected.**
* **CPU forward dry-run runs at BOTH operating points:** latent_t 96 → `[1,8,96,16]` (3.84 s) and
  latent_t 256 → `[1,8,256,16]` (10.24 s). The same generation code can run at 96 and 256.
* **Component delta (what the fine-tune changed):**

  | component | changed | identical |
  |---|---:|---:|
  | U-Net (`model.diffusion_model.*`) | 690 | 0 |
  | U-Net EMA (`model_ema.*`) | 691 | 1 |
  | VAE (`first_stage_model.*`) | 0 | 398 |
  | **CLAP conditioner (`cond_stage_model*`)** | **0** | **505** |
  | schedules/other | 0 | 14 |

  Only the U-Net (and its EMA) were fine-tuned; **VAE and the CLAP text conditioner are byte-identical
  to `audioldm-m-full`** ⇒ the **conditioning implementation/config does NOT differ** from our pipeline.
  EMA convention matches the checkpoints we already EMA-reconstruct.

**No incompatibility found.** Part B is not structurally blocked.

## B0.3 Scientific comparability (documented; bounds the claim)

| Axis | text-FT reference | Singh recovery |
|---|---|---|
| training data | AudioCaps **+ MusicCaps** | AudioCaps only |
| starting model | **DENSE** `audioldm-m-full` | **PRUNED** backbone |
| objective / conditioning | full-U-Net text-conditioned FT (FiLM CLAP text; conditioner frozen) — **same pathway as our pipeline** | AudioCaps FT of the pruned U-Net (1M steps) |
| training duration | **UNKNOWN** (not reported identical to Singh) | 1M steps (pruned) |

**Maximum defensible claim:** *whether a duration interaction is ALSO observable in one independently
released dense text-fine-tuned AudioLDM companion.* It CANNOT establish that generic fine-tuning
explains Singh recovery, nor that pruning uniquely causes the interaction.

## B0.4 Power / decision value (existing frozen data; n=80)

From `op_duration_discriminator_1` raw per-ytid CLAP cosines (the SAME Arm-D battery Part B reuses;
`configs/research/partb_power_analysis.json`). Identity: a two-system duration interaction
`J(A,B)=swing(A)−swing(B)`, so the empirical `Var_ytid[J_recovery]` estimates the variance scale of any
two-system duration interaction (incl. `J_dense_textFT`). Empirical `J_recovery`: SD_ytid 0.204,
SE 0.0228 (reproduces op_duration J_CLAP +0.044 [−0.001,+0.087]); corr(rec swing, pru swing)=0.222.

* **J_dense_textFT precision** (central; ±50 % variance band): SE ≈ 0.0228 (0.016–0.028), 95% CI
  half-width ≈ 0.045, **MDE (80 % power) ≈ 0.064**. ⇒ a sev-2-magnitude interaction (~+0.16) is
  detectable; a sev-1-magnitude one (~+0.044) is NOT reliably detectable.
* **Q = J_recovery_sev1 − J_dense_textFT:** SE 0.023–0.040 across ρ ∈ {0.5, 0, −0.5}; **TOST
  equivalence within ±0.025 is NOT attainable at n=80 in ANY scenario** (90% CI half-width
  0.038–0.065 > 0.025). **⇒ remove Q-equivalence as a planned claim** (do not run an underpowered
  equivalence test).
* **Asymmetric value:** a clearly-positive `J_dense_textFT` (≥ ~0.064) would broaden the evaluation
  warning beyond pruning recovery; an unresolved `J_dense_textFT` does NOT establish absence, and a null
  point without powered equivalence is not evidence of no interaction.

## Part B authorization gate

| # | Criterion | Status |
|---|---|---|
| 1 | checkpoint provenance PASS | ✓ |
| 2 | structural compatibility PASS | ✓ (strict-load 690/690; runs at 96 & 256) |
| 3 | scientific role interpretable | ✓ (public dense text-FT reference) |
| 4 | n=80 answers ≥1 decision-relevant question | ✓ **weakly** — can detect a clearly-positive `J_dense_textFT` (MDE 0.064); CANNOT establish Q-equivalence |
| 5 | expected spend ≤ 0.70 cr | ✓ (160 WAVs ≈ 0.47–0.60 cr; hard cap 0.70; smallest T4) |
| 6 | no new prompts/systems beyond single textFT arm | ✓ (reuse dense + frozen Arm-D 80) |

### VERDICT: **B-GO-CANDIDATE** — but **NO LAUNCH** (locked pending explicit supervisor GO + top-up)

All six gate criteria are (conditionally) met, so the experiment is a *candidate*. Honest scope, so the
supervisor can weigh it:

* The ONLY cleanly-powered decision (Q-equivalence, "is the recovery interaction distinguishable from a
  dense text-FT interaction?") is **infeasible at n=80 and is removed**. What remains is the one-sided
  question "is there a **clearly-positive** dense-text-FT duration interaction?" — powered only for large
  effects.
* The reference is not a control; the maximum claim is the companion-observability statement above.
* Part A already delivered the high-value additions (independent-evaluator frame-level corroboration +
  a clean negative on late-allocation), which **reduces Part B's marginal value**.

**Recommendation:** legitimate but **low-to-moderate** marginal value. If the supervisor wants a public
dense-text-FT duration anchor for the paper and accepts the asymmetric (positive-only) power, launch the
single textFT arm (160 WAVs, ≤0.70 cr) at 3.84/10.24 s reusing the frozen `x_T` and all dense outputs.
Otherwise B-STOP is fully defensible. **No GPU launched. No top-up requested.**
