# Paper companion

This page maps **Recovery Gain Is Operating-Point Dependent in Pruned Text-to-Audio Diffusion** to the repository evidence that supports each part of the argument.

The ICASSP manuscript is designed to stand on its own. It contains the scientific question, the experimental design, the central effect sizes, the robustness table, the main limitations and the intended figure footprint. This companion is the audit and reproduction layer.

## Paper files

* [`icassp_operating_point.tex`](icassp/icassp_operating_point.tex), canonical Draft 14 source
* [`sections/draft14_1.tex`](icassp/sections/draft14_1.tex) through [`sections/draft14_4.tex`](icassp/sections/draft14_4.tex), modular source
* [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md), complete numerical layer

## Section map

| Paper section | Question | Repository evidence |
|---|---|---|
| Sec. 1 | Why is one post-recovery score insufficient? | Root README and `docs/claims_matrix.md` distinguish final checkpoint quality from the paired recovery gain. |
| Sec. 2 | How does recovery fit into compression and TTA evaluation? | Upstream provenance, cited releases and the project claims matrix. |
| Sec. 3.1 | Which systems and operating points are compared? | Frozen manifests and result artifacts under `configs/research/`. |
| Sec. 3.2 | Does changing the fine-tuning duration move the favorable evaluation duration? | `docs/reviewer2_followup_ext.md`, `configs/research/r2_EXT2x2_result.json`, and the earlier short-duration result `configs/research/r2_E3_result.json`. |
| Sec. 3.3 | How are R, J, Delta J and recovery ratios estimated? | Result artifacts record the scorer revision, common-noise pairing, prompt-level bootstrap and seed namespace. |
| Sec. 4.1 | How does released recovery vary with requested duration? | Original severity-2 result, four-duration sweep, Human-CLAP, secondary metrics, published sampler and 15.36 s extension. |
| Sec. 4.2 | Does the duration effect follow training duration? | The symmetric pruned 20k-step control and dense diagnostic 2x2 in `r2_EXT2x2_result.json`. |
| Sec. 4.3 | How does recovery transfer across domains? | Clotho follow-up, extended hip-hop battery and full-coverage dense anchors in `r2_posthoc_pooled_anchors.json`. |
| Sec. 4.4 | How stable is the result across severity and short-generation diagnostics? | Severity-1 original/new split, dense-vs-real duration response and crop analysis. |
| Sec. 5 | Which explanation survives and what remains unresolved? | `docs/reviewer2_scientific_report.md`, `docs/claims_matrix.md`, and the audit sections of the follow-up reports. |

## Figure 1

Draft 14 reserves the final figure footprint but intentionally does not generate the graph. The complete production specification is embedded as comments in `icassp/sections/draft14_2.tex` immediately before `figure*`.

The plotting implementation should consume committed per-prompt artifacts rather than values copied from the paper. Panel (a) must preserve the matched n=96 comparison for the 10.24 to 15.36 s extension. Panel (b) must compare the two symmetric 20k-step pruned fine-tunes and display the uncertainty on J and Delta J.

## Expanded results

Use [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md) when an exact interval, subgroup result, dense diagnostic or follow-up status is needed. The paper retains only numbers required to understand the scientific argument.
