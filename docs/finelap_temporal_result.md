# Part A — FineLAP Temporal-Semantic Recovery Profile — RESULT

**Type:** CPU-only frozen post-result diagnostic. 0 GPU, 0 new generation. Executed strictly AFTER the
protocol + eligibility manifests were committed (`13bd3ac`). Result artifact
`configs/research/finelap_temporal_result.json` (sha `ce5519c8…`). Protocol
`docs/finelap_temporal_protocol.md` (sha `278d2e0d…`).

## Headline — Branch **A2** (frame-level gain, NO late redistribution)

The primary hypothesis — a preferential **late** allocation of the recovered−pruned semantic contrast
(`T_2 > 0`) — is **NOT supported**. Instead, recovery produces a **large, temporally BROAD** frame-level
requested-event grounding gain (early ≈ late), independently corroborated across four fixed secondary
endpoints, at BOTH pruning severities and robust to the seam convention.

## Primary endpoint (severity 2, recovered2 − pruned2_A, n=110 prompts / 131 event occ)

| Statistic | Point | 95% CI | Reading |
|---|---:|---|---|
| **T_2 = D_late − D_early** | **−0.0017** | [−0.0236, +0.0201] | **gate `lo95>0` FAILS** — no late redistribution |
| D_early2 | +0.2746 | [+0.2191, +0.3336] | recovered ≫ pruned early |
| D_late2 | +0.2729 | [+0.2160, +0.3327] | recovered ≫ pruned late (≈ early) |

Per-system window means: recovered early **0.286** / late **0.287** (flat); pruned2_A early **0.012** /
late **0.014** (near-floor, flat). D_early ≈ D_late ⇒ T_2 ≈ 0. `frac(T>0) = 0.45`.

**Seam sensitivity (pruned2_B):** T_2^B = −0.0011 [−0.0229, +0.0207]; D_early +0.2757 / D_late +0.2746.
Identical null — **seam-robust**.

## Cross-severity replication support (severity 1, recovered − pruned, n=49 / 63)

`T_1 = −0.0203 [−0.0532, +0.0112]` — **point NOT > 0** (CI crosses 0). The directional support condition
fails; if anything the sev-1 gap narrows slightly late (pruned grounding rises 0.112→0.135 late while
recovered stays flat 0.302→0.305). D_early +0.1902 [+0.106, +0.280]; D_late +0.1699 [+0.086, +0.256].
**Consistent with sev-2: the recovered advantage is temporally uniform, not late-preferential.**

## Secondary endpoints (severity 2, recovered − pruned2_A; descriptive; all CIs exclude 0)

| Endpoint | recovered | pruned2_A | Contrast (95% CI) |
|---|---:|---:|---|
| semantic mass (mean frame score) | 0.286 | 0.013 | **+0.2735** [+0.218, +0.331] |
| occupancy (frac frames ≥ 0.5) | 0.271 | 0.003 | **+0.2680** [+0.201, +0.340] |
| quarter coverage (of four 2.56-s quarters) | 0.385 | 0.009 | **+0.3761** [+0.294, +0.460] |
| peak frame score | 0.446 | 0.039 | **+0.4065** [+0.340, +0.474] |

Severity 1 mirrors this (recovered mass 0.304 vs pruned 0.127; occupancy 0.285 vs 0.110; qcov 0.384 vs
0.156; peak 0.459 vs 0.243). Both **occupancy/coverage** AND **peak** improve strongly, so the gain is a
general frame-level semantic improvement (not specifically coverage/persistence-dominant vs
peak-dominant). No onset/timing claim (captions lack timestamps). `tau=0.5` is the frozen sigmoid/BCE
decision geometry, not a calibrated probability.

Sanity: no score saturation (frac ≥ 0.99 ≈ 0.01–0.1%), so the flat early/late is genuine, not a
ceiling. sev-2 pruned is near-floor (22/131 all-zero event vectors) — the (1,2,1,1) model genuinely
fails to ground requested events — but this floor is EQUAL early and late, so it does not bias the T
contrast (which measures recovered's own late-vs-early balance against a flat baseline).

## Interpretation (frozen branch A2)

> Recovery improves FineLAP-localized requested-event evidence, but the long-duration advantage is not
> explained by preferential late allocation.

The interpretation guard is respected: `D_late_2 = +0.273 > 0`, but because the **primary `T_2` gate
fails**, "late semantic gain" wording is NOT triggered (the late gain is matched by an equal early
gain). The clip-level duration interaction (recovered−pruned advantage larger at 10.24 s than 3.84 s,
established at clip level by CLAP) is **NOT** explained by within-clip late semantic reallocation under
FineLAP.

## Part A scientific value (no causal overclaim)

1. **Independent-evaluator frame-level corroboration.** FineLAP (EAT audio encoder + RoBERTa text,
   frame-level grounding supervision) is a genuinely different evaluator family from CLAP / Human-CLAP
   (both CLAP-derived). It strongly corroborates the native recovered advantage at the frame level —
   recovery makes the requested event far more grounded throughout the generation (mass, occupancy,
   coverage, peak; CIs exclude 0; both severities; seam-robust). This directly addresses the post-result
   audit's "single scorer family" weakness (`docs/xsev_postresult_adversarial_audit.md` §12).
2. **Clean prospectively-frozen negative on the late-allocation mechanism.** The specific "late temporal
   semantic recovery" hypothesis motivating this reopening is not supported (T_2 ≈ 0 seam-robust; T_1
   not positive). This prevents an over-claimed "restores late temporal coverage" narrative and bounds
   what the duration finding can mean.
3. **Characterization.** The recovered gain is a temporally broad increase in requested-event grounding
   (present early and late alike), not a back-loaded reallocation and not an onset/timing effect.

## Validity boundary (frozen A5)

Prospectively frozen **POST-RESULT** diagnostic, motivated by already-observed global duration
interactions — NOT an independent preregistered confirmation of the original hypothesis. FineLAP frame
scores are contextualized frame-level audio–phrase **grounding evidence** (the EAT encoder is fully
self-attended; §A0.7), not calibrated event probabilities or causal local activations. Mechanism
attribution (pruning-trajectory vs generic fine-tuning) remains blocked (the dense-FT control is
unavailable). No new temporal metric was introduced after seeing results.
