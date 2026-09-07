# Expanded results for Draft 15

This file retains numerical detail that is useful for audit but too dense for the four technical pages. Intervals are 95% prompt-level percentile-bootstrap intervals unless stated otherwise.

## Released severity-2 duration response

| Duration | CLAP recovery R |
|---|---:|
| 3.84 s | +0.085 [+0.066,+0.105] |
| 5.12 s | +0.139 [+0.115,+0.164] |
| 7.68 s | +0.201 [+0.175,+0.227] |
| 10.24 s | +0.244 [+0.215,+0.273] |

Primary endpoint interaction: **J = +0.159 [+0.131,+0.187]**.

Matched n=96 extension: R(15.36)-R(10.24) = +0.021 [-0.023,+0.067]. The supported reading is no clear increase beyond 10.24 s, not a demonstrated plateau.

## Symmetric 20k training-duration intervention

| Training duration | R(3.84 s) | R(10.24 s) | J |
|---|---:|---:|---:|
| 3.84 s | +0.009 [-0.006,+0.024] | +0.075 [+0.053,+0.097] | +0.065 [+0.044,+0.087] |
| 10.24 s | +0.017 [+0.001,+0.033] | +0.048 [+0.025,+0.070] | +0.031 [+0.010,+0.052] |

Directional specialization contrast: **Delta J = -0.035 [-0.064,-0.004]**.

The P baseline is EMA-derived and the two experimental fine-tunes are raw exports. This convention cancels exactly in Delta J. The dense step-zero raw-vs-EMA offset has duration interaction -0.019 [-0.050,+0.013], providing a scale for the convention caveat on individual J values.

## Dense raw baseline and short-budget dense 2x2

Dense raw vs EMA:

| Duration | O = raw - EMA |
|---|---:|
| 3.84 s | +0.001 [-0.017,+0.020] |
| 10.24 s | -0.017 [-0.044,+0.010] |

The raw-vs-EMA convention cannot explain the approximately 0.2 degradation after the 20k dense fine-tunes.

Against the raw dense start:

| Dense arm | G'(3.84 s) | G'(10.24 s) | J' |
|---|---:|---:|---:|
| train@3.84 | -0.183 [-0.206,-0.160] | -0.220 [-0.258,-0.181] | -0.034 [-0.071,+0.002] |
| train@10.24 | -0.168 [-0.190,-0.144] | -0.206 [-0.235,-0.177] | -0.038 [-0.069,-0.008] |

Dense training-duration contrast: Delta J_dense = -0.005 [-0.033,+0.025]. The experiment is a negative diagnostic and does not replace the unavailable matched dense 10^6-step recovery checkpoint.

## Public dense text-fine-tuned reference

G(3.84) = -0.022 [-0.061,+0.017], G(10.24) = +0.091 [+0.042,+0.141], J = +0.113 [+0.051,+0.173]. This checkpoint is not recipe-matched and is used as contextual evidence only.

## Domain and duration

| Domain | R(3.84 s) | R(10.24 s) | J |
|---|---:|---:|---:|
| AudioCaps | +0.085 [+0.066,+0.105] | +0.244 [+0.215,+0.273] | +0.159 [+0.131,+0.187] |
| Clotho | +0.098 [+0.072,+0.125] | +0.210 [+0.176,+0.243] | +0.112 [+0.079,+0.146] |
| Hip-hop | +0.026 [+0.007,+0.044] | +0.027 [+0.003,+0.050] | +0.001 [-0.026,+0.028] |

AudioCaps minus hip-hop J = +0.158 [+0.120,+0.197]. AudioCaps minus Clotho J = +0.047 [+0.003,+0.090].

At n=127, the dense hip-hop anchor remains about +0.11 above shuffled-caption chance. Native rho_dense = 0.119 [0.015,0.215]. Its interval is wide because rho is a jointly bootstrapped ratio with a small numerator and an estimated denominator.

## Prompt-sample stability

Severity 1:

- original n=80: J = +0.044 [-0.000,+0.088]
- disjoint n=96: J = +0.169 [+0.115,+0.222]
- between-set difference = +0.124 [+0.058,+0.194]

Severity 2 split-half of the outcome-blind n=192 draw:

- first n=96: J = +0.159 [+0.116,+0.200]
- second n=96: J = +0.160 [+0.122,+0.198]
- difference = +0.001 [-0.056,+0.057]

The published sampler and secondary scorer checks reuse the same AudioCaps prompts and are robustness checks, not independent prompt replications. Clotho supplies an independent prompt set.
