# Expanded results for the ICASSP paper

This document contains the numerical detail intentionally removed from the four technical pages of the manuscript. It is supplementary audit material, not a substitute for the paper. The paper states the scientific argument and the values needed to support it. This page exposes the denser numerical layer for readers who want to inspect or reproduce individual contrasts.

Notation follows the paper. P is the pruned checkpoint, P+FT is the same pruned architecture after released recovery fine-tuning, R is the paired recovery gain, and J is the difference in recovery gain between 10.24 s and 3.84 s. Intervals are 95% prompt-level percentile bootstrap intervals unless stated otherwise.

## Section 4.1 Duration

### Complete CLAP table

| Setting | P | P+FT | R [95% CI] | W |
|---|---:|---:|---:|---:|
| **Severity 1, (1,2,3,1)** | | | | |
| AudioCaps 3.84 s | 0.104 | 0.111 | +0.008 [-0.023, +0.039] | 0.44 |
| AudioCaps 10.24 s | 0.253 | 0.304 | +0.052 [+0.009, +0.093] | 0.64 |
| music 3.84 s | 0.117 | 0.023 | -0.094 [-0.124, -0.065] | 0.20 |
| duration response, s(P), s(P+FT), J | +0.149 | +0.193 | +0.044 [-0.001, +0.087] | |
| domain contrast at 3.84 s, n=96 | | | +0.092 [+0.054, +0.131] | |
| **Severity 2, (1,2,1,1)** | | | | |
| AudioCaps 3.84 s | 0.015 | 0.100 | +0.085 [+0.066, +0.105] | 0.72 |
| AudioCaps 5.12 s | 0.015 | 0.154 | +0.139 [+0.115, +0.164] | 0.79 |
| AudioCaps 7.68 s | 0.029 | 0.231 | +0.201 [+0.175, +0.227] | 0.89 |
| AudioCaps 10.24 s | 0.055 | 0.299 | +0.244 [+0.215, +0.273] | 0.87 |
| music 3.84 s | 0.005 | 0.014 | +0.009 [-0.013, +0.032] | 0.48 |
| music 10.24 s | 0.089 | 0.094 | +0.005 [-0.028, +0.039] | 0.53 |
| duration response, s(P), s(P+FT), J | +0.040 | +0.200 | +0.159 [+0.131, +0.187] | |
| domain contrast at 3.84 s | | | +0.076 [+0.047, +0.105] | |
| domain contrast at 10.24 s | | | +0.239 [+0.195, +0.283] | |

W is the fraction of prompts for which P+FT scores above P.

Primary severity-2 replication artifact

* [`configs/research/xsev_result.json`](configs/research/xsev_result.json)

### Duration sweep

At severity 2, R is +0.085, +0.139, +0.201 and +0.244 at 3.84, 5.12, 7.68 and 10.24 s. Adjacent increases are

| Step | Increase in R [95% CI] |
|---|---:|
| 3.84 to 5.12 s | +0.054 [+0.032, +0.077] |
| 5.12 to 7.68 s | +0.062 [+0.037, +0.087] |
| 7.68 to 10.24 s | +0.043 [+0.014, +0.073] |

P is flat through 5.12 s while P+FT rises at every step under the primary scorer.

Artifact

* [`configs/research/draft5_opsweep_result.json`](configs/research/draft5_opsweep_result.json)

### Anchors and recovery ratios

| Setting | chance floor P/P+FT | real audio | gap to dense closed | gap to real audio closed |
|---|---:|---:|---:|---:|
| sev. 1 AudioCaps 3.84 s | -0.031 / -0.033 | 0.264 | 8% [-30, 36] | 5% [-16, 24] |
| sev. 1 AudioCaps 10.24 s | -0.012 / -0.036 | 0.442 | 52% [11, 103] | 27% [6, 46] |
| sev. 2 AudioCaps 3.84 s | -0.005 / -0.015 | 0.274 | 44% [36, 53] | 33% [26, 39] |
| sev. 2 AudioCaps 5.12 s | -0.005 / -0.025 | 0.307 | 63% [52, 74] | 48% [40, 55] |
| sev. 2 AudioCaps 7.68 s | +0.003 / -0.037 | 0.333 | 78% [68, 88] | 66% [59, 74] |
| sev. 2 AudioCaps 10.24 s | +0.020 / -0.022 | 0.440 | 82% [72, 92] | 63% [56, 71] |
| sev. 1 music 3.84 s | +0.055 / +0.001 | not available | not defined | not defined |
| sev. 2 music 3.84 s | -0.013 / -0.004 | not available | not defined | not defined |
| sev. 2 music 10.24 s | +0.070 / +0.061 | not available | not defined | not defined |

Artifact

* [`configs/research/draft5_floor_ceiling_result.json`](configs/research/draft5_floor_ceiling_result.json)

### Dense duration control

At severity 2, matched dense generation changes from 0.207 at 3.84 s to 0.354 at 10.24 s, giving a duration response of +0.147 [+0.117, +0.177]. P responds -0.107 [-0.137, -0.076] relative to dense, whereas P+FT responds +0.053 [+0.017, +0.089] more than dense. The duration result is therefore not the statement that every system obtains a larger CLAP value on longer clips.

Relevant artifacts include

* [`configs/research/xsev_dense_192_control_result.json`](configs/research/xsev_dense_192_control_result.json)
* [`configs/research/draft4_dense_duration_control_result.json`](configs/research/draft4_dense_duration_control_result.json)

### Corroboration

The published sampling recipe, DDIM 200 with guidance 3.5, reproduces the severity-2 duration interaction on the pre-specified 64-prompt subset with J = +0.184 [+0.126, +0.243]. The frozen primary recipe gives +0.168 on the same subset. Their difference is +0.016 [-0.040, +0.072].

Human-CLAP at the severity-2 native point gives R = +0.375 [+0.340, +0.408] and J = +0.185. Two event-level measures outside the CLAP family also show a larger native recovery effect. KL to real references improves by +2.22 [+1.93, +2.53] at 10.24 s and +0.66 [+0.42, +0.92] at 3.84 s. PANNs top-10 captured labels improve by +0.86 [+0.70, +1.02] at 10.24 s and +0.19 [+0.06, +0.32] at 3.84 s. The corresponding interactions are +1.56 [+1.19, +1.92] for KL and +0.67 [+0.49, +0.86] for PANNs.

Post-hoc sweep rescoring also rises at every duration for Human-CLAP, KL and PANNs. The last 7.68 to 10.24 s step is not resolved by these secondary scorers. Human-CLAP is +0.004 [-0.033, +0.041], KL is +0.27 [-0.04, +0.57], and PANNs is +0.10 [-0.07, +0.27].

Artifacts

* [`configs/research/draft5_pubrecipe_result.json`](configs/research/draft5_pubrecipe_result.json)
* [`configs/research/draft5_sweep_hc.json`](configs/research/draft5_sweep_hc.json)
* [`configs/research/draft5_sweep_secondary_metrics.json`](configs/research/draft5_sweep_secondary_metrics.json)
* [`configs/research/draft5_holm_extension.json`](configs/research/draft5_holm_extension.json)

The Holm correction over the registered severity-2 family leaves the paper's severity-2 conclusions unchanged. Severity 1 is directionally consistent but its raw J interval includes zero, +0.044 [-0.001, +0.087], so the paper bases the duration conclusion on the replicated severity-2 evidence.

## Section 4.2 Domain

At severity 2, AudioCaps recovery is +0.085 at 3.84 s and +0.244 at 10.24 s. Music recovery is +0.009 [-0.013, +0.032] and +0.005 [-0.028, +0.039] at the same durations. The resulting domain contrasts are +0.076 [+0.047, +0.105] and +0.239 [+0.195, +0.283]. The music duration interaction is -0.004.

Chance floors show that after fine-tuning, the music cells sit only 0.022, 0.018 and 0.033 CLAP units above chance for severity 1 at 3.84 s and severity 2 at 3.84 and 10.24 s. Severity-2 AudioCaps sits 0.115 above chance at 3.84 s and 0.321 above chance at 10.24 s.

Severity 1 produced a negative music gain of -0.094 [-0.124, -0.065], but this penalty did not replicate at severity 2. In the original frozen 64-prompt battery the severity-2 hip-hop gain was unresolved. Draft 13 does not interpret that result as absence of recovery because the higher-power reviewer follow-up below resolves a small positive gain.

The held-out battery differs from AudioCaps in both content and caption style. Within AudioCaps, native recovery is uncorrelated with caption length, Spearman rho = +0.04 [-0.12, +0.18], and no AudioCaps caption is truncated by the conditioner. In the music battery, 47% of captions exceed CLAP's 77-token pre-training length. Content and caption style therefore remain inseparable in this domain test.

### Author listening

The author listening check was blinded with opaque identifiers, a sealed key and a seeded prompt draw. At 10.24 s on AudioCaps, P+FT was preferred on 6 of 8 pairs and P was mostly noise. At 3.84 s, both systems sounded like noise and P+FT was preferred on 0 of 8. On music, P+FT was preferred on 8 of 8 pairs and was heard as music on 5 of 8, compared with 1 of 8 for P, although neither system followed the long captions well.

Artifact

* [`configs/research/author_listening_1_result.json`](configs/research/author_listening_1_result.json)

This check is descriptive and informal. It is not used as a powered perceptual endpoint.

## Section 4.3 Where the duration effect arises

FineLAP tests whether the native-duration recovery gain is concentrated late in the clip. Outcome-blind eligibility leaves 110 severity-2 and 49 severity-1 prompts. At severity 2, semantic mass increases by +0.27 [+0.22, +0.33], but early and late gains are effectively equal. Their difference is T = -0.002 [-0.024, +0.020]. The data therefore reject the proposed late-allocation account.

The post-hoc crop analysis separates generation length from scoring-window length. At severity 2, scoring the first 3.84 s of the native 10.24 s generation gives R_crop = +0.172 [+0.150, +0.194]. This exceeds recovery from a separately generated 3.84 s clip by +0.087 [+0.065, +0.110]. The remaining difference from the complete 10.24 s generation is +0.072. At severity 1, crop recovery exceeds separately generated short recovery by +0.043 [+0.002, +0.085].

The short-duration deficit therefore arises primarily from generating a short clip rather than from CLAP seeing a short scoring window.

## Pre-specified negatives retained in the project record

Several hypotheses were not supported and are intentionally preserved.

1. A proposed trade of in-domain gain for out-of-domain alignment fails at severity 1 because the required AudioCaps gain at 3.84 s is absent.
2. The severity-1 music penalty does not replicate at severity 2.
3. The proposed late-allocation explanation is rejected by FineLAP.
4. P+FT is not restored to dense. At 10.24 s, dense leads by +0.048 [-0.000, +0.096] at severity 1 and +0.055 [+0.021, +0.088] at severity 2. At 3.84 s, the severity-2 dense lead is +0.107.

The broader project's earlier hypotheses that were rejected at their registered gates remain documented in [`docs/claims_matrix.md`](docs/claims_matrix.md) and the root [`README.md`](README.md). They are not converted into positive claims in the ICASSP paper.

## Analysis status

The severity-2 duration interaction, seam sensitivity, domain tests, secondary metrics and published-sampler check were prospectively specified before their corresponding scores. FineLAP and the four-duration sweep were registered after the primary result but before their own scoring. Crop analysis, anchor decomposition, the multiple-comparison extension and author listening are post-hoc. The committed artifacts preserve their protocol hashes and status.

## Numerical provenance

The values on this page come from committed result artifacts under [`configs/research/`](configs/research/). The existing verification and figure-building code is under [`scripts/research/paper_figs/`](scripts/research/paper_figs/). The purpose of this page is to expose the complete reporting layer while the manuscript keeps only the values required for scientific comprehension.


---

## Reviewer-2 follow-up experiments

These experiments were pre-specified in `docs/reviewer2_followup.md` after the ICASSP review and before generation. They do not retroactively alter the frozen primary verdicts. They do change the interpretation used in Draft 13. Full machine-readable values live in `configs/research/r2_*_result.json`.

### Duration mechanism and range

| Experiment | Contrast | Estimate | 95% CI | Interpretation |
|---|---|---:|---:|---|
| E3, pruned checkpoint fine-tuned at 3.84 s | R at 3.84 s | +0.009 | [-0.006, +0.024] | unresolved at its own training duration |
| E3 | R at 10.24 s | +0.075 | [+0.053, +0.096] | resolved positive |
| E3 | J = R(10.24)-R(3.84) | +0.065 | [+0.043, +0.087] | contradicts training-duration specialization |
| B, public dense text-FT reference | J | +0.113 | [+0.051, +0.173] | duration dependence is not pruning-specific |
| E1c, beyond native duration | R at 15.36 s | +0.264 | [+0.216, +0.310] | recovery remains large |
| E1c | R(15.36)-R(10.24) | +0.021 | [-0.023, +0.067] | plateau beyond 10.24 s |
| E8, severity 1 pooled | J, n=176 | +0.112 | [+0.076, +0.149] | duration interaction resolves at the lower pruning severity |

The E3 intervention is the direct test of the old title's specialization implication. A 3.84-s recovery fine-tune does not move the gain toward 3.84 s. Its gain remains larger when evaluated at 10.24 s. The public dense text-FT reference shows the same direction, so Draft 13 treats duration dependence as an operating-point property of fine-tuning gain rather than a pruning-specific mechanism.

### Domain transfer and anchors

| Battery | Duration | R | 95% CI | rho_dense | rho_real |
|---|---:|---:|---:|---:|---:|
| Clotho, n=96 | 3.84 s | +0.098 | [+0.072, +0.125] | 0.49 | 0.28 |
| Clotho, n=96 | 10.24 s | +0.210 | [+0.176, +0.243] | 0.74 | 0.59 |
| hip-hop, pooled n=127 | 3.84 s | +0.026 | [+0.008, +0.044] | -- | -- |
| hip-hop, pooled n=127 | 10.24 s | +0.027 | [+0.004, +0.051] | -- | -- |

For Clotho, the AudioCaps-minus-Clotho recovery contrast at 10.24 s is +0.032 [-0.023, +0.088], so a difference from AudioCaps is unresolved. Hip-hop is different. The dense checkpoint lies above its shuffled-caption floor at every tested cell, including +0.106 [+0.079, +0.135] at severity 2 and 10.24 s, which rules out a simple floor explanation. Yet recovery closes only about 2% to 4% of the dense gap in the original anchored hip-hop cells. With 127 prompts the recovery gain becomes statistically resolvable but remains roughly an order of magnitude smaller than AudioCaps.

### Consequence for the paper framing

The evidence supports `recovery gain is operating-point dependent`. It does not support `recovery is largest where it was trained`. The manuscript therefore reports duration dependence as a robust empirical property, explicitly rejects training-duration specialization as the explanation tested by E3, and describes domain transfer as graded rather than as an in-domain/out-of-domain binary.
