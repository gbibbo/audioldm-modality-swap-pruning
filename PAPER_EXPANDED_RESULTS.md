# Expanded results for the ICASSP paper

This document contains numerical detail kept outside the four technical pages. It is an audit layer, not a substitute for the manuscript. All intervals are 95% prompt-level percentile-bootstrap intervals unless noted otherwise.

Notation follows the paper. P is the pruned checkpoint, P+FT the recovered checkpoint, R(d) the paired recovery gain at requested duration d, J = R(10.24 s) - R(3.84 s), and Delta J = J(train@10.24) - J(train@3.84).

## Released recovery across duration

| Duration | CLAP R | Human-CLAP R | KL recovery | PANNs top-10 capture R |
|---|---:|---:|---:|---:|
| 3.84 s | +0.085 [0.066,0.105] | +0.189 [0.162,0.217] | +0.66 [0.43,0.92] | +0.19 [0.06,0.32] |
| 5.12 s | +0.139 [0.115,0.164] | +0.278 [0.243,0.314] | +1.16 [0.91,1.41] | +0.38 [0.23,0.53] |
| 7.68 s | +0.201 [0.175,0.227] | +0.371 [0.336,0.405] | +1.95 [1.68,2.23] | +0.76 [0.61,0.91] |
| 10.24 s | +0.244 [0.215,0.273] | +0.375 [0.340,0.409] | +2.22 [1.92,2.52] | +0.86 [0.70,1.02] |
| J | +0.159 [0.131,0.188] | +0.185 [0.151,0.220] | +1.56 [1.20,1.92] | +0.67 [0.49,0.85] |

The published DDIM-200, guidance-3.5 recipe gives J = +0.184 [0.126,0.243]. The severity-2 registered family survives Holm correction.

### Extension beyond 10.24 s

On the matched first 96 AudioCaps prompts, R(10.24 s) = +0.242 [0.198,0.285] and R(15.36 s) = +0.264 [0.216,0.310]. The matched step is +0.021 [-0.023,+0.067]. This does not establish a plateau and is reported in the paper only as no clear increase beyond 10.24 s.

## Symmetric 20k-step training-duration intervention

Both checkpoints start from the same severity-2 P baseline and use the same 20,000-step full-U-Net recipe apart from training duration.

| Training duration | R(3.84 s) | R(10.24 s) | J |
|---|---:|---:|---:|
| 3.84 s | +0.009 [-0.006,+0.024] | +0.075 [0.053,0.097] | +0.065 [0.044,0.087] |
| 10.24 s | +0.017 [0.001,0.033] | +0.048 [0.025,0.070] | +0.031 [0.010,0.052] |

Delta J = J(train@10.24) - J(train@3.84) = **-0.035 [-0.064,-0.004]**.

Training-duration specialization predicts Delta J > 0. The observed contrast is resolved in the opposite direction at this matched 20k-step budget.

## Dense diagnostic 2x2

The dense AudioLDM-M-Full baseline was fine-tuned for 20,000 steps at each duration.

| Dense fine-tune | G(3.84 s) | G(10.24 s) | J |
|---|---:|---:|---:|
| train@3.84 s | -0.182 [-0.207,-0.157] | -0.235 [-0.271,-0.199] | -0.051 [-0.090,-0.013] |
| train@10.24 s | -0.166 [-0.190,-0.141] | -0.223 [-0.255,-0.191] | -0.057 [-0.091,-0.023] |

Delta J_dense = -0.005 [-0.034,+0.024]. The full-parameter short fine-tunes degrade the EMA dense baseline. This diagnostic therefore cannot replace the unavailable dense checkpoint after the released million-step recovery. The follow-up audit attributes the mismatch to the pre-declared raw-weight-versus-EMA issue together with limited AudioCaps headroom, not optimizer divergence.

## Domain transfer

| Domain | n | R(3.84 s) | R(10.24 s) | rho_dense at 10.24 s |
|---|---:|---:|---:|---:|
| AudioCaps | 192 | +0.085 [0.066,0.105] | +0.244 [0.215,0.273] | 0.82 |
| Clotho | 96 | +0.098 [0.072,0.125] | +0.210 [0.176,0.243] | 0.74 |
| Hip-hop | 127 | +0.026 [0.007,0.044] | +0.027 [0.003,0.050] | 0.119 [0.015,0.215] |

For hip-hop at n=127, dense above shuffled-caption chance is +0.110 [0.091,0.128] at 3.84 s and +0.108 [0.087,0.130] at 10.24 s. The pooled rho_dense values are 0.106 [0.031,0.177] and 0.119 [0.015,0.215], respectively.

## Severity-1 prompt-set heterogeneity

| Prompt set | J |
|---|---:|
| original Arm-D 80 | +0.044 [-0.000,0.088] |
| new disjoint 96 | +0.169 [0.115,0.222] |
| pooled 176 | +0.112 [0.076,0.149] |

The unpaired new-minus-original difference is +0.124 [0.058,0.194]. Both samples were selected without outcome access, but they use different source pools, hash salts and selection rules. The pooled result is therefore evidence that the duration interaction resolves at severity 1, not evidence that its magnitude is sample-invariant.

## Short-generation diagnostics

The dense model's floor-corrected duration response is +0.142 [0.111,0.172], close to the real-audio crop response +0.150 [0.133,0.166]. This does not support a strong claim that the dense base model is broken at 3.84 s under CLAP.

The crop analysis remains informative for recovery itself. R_crop = +0.172 [0.150,0.194] when scoring the first 3.84 s of a 10.24 s generation, which exceeds recovery from a separately generated 3.84 s clip by +0.087 [0.065,0.110].

## Provenance

Second-round protocol: `docs/reviewer2_followup_ext.md`.

Main new artifacts: `configs/research/r2_EXT2x2_result.json` and `configs/research/r2_posthoc_pooled_anchors.json`.

Consolidated evidence report: `docs/reviewer2_scientific_report.md`.
