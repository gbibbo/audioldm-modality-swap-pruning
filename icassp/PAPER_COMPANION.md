# Paper companion

This page accompanies **Recovery Gain Is Operating-Point Dependent in Pruned Text-to-Audio Diffusion**.

The manuscript is self-contained. The repository keeps the numerical and provenance layer needed to audit individual contrasts, inspect follow-up protocols, and reproduce the figures.

## Draft 16 map

| Paper section | Main question | Repository evidence |
|---|---|---|
| Sec. 3.1 | Which checkpoints and operating points are compared? | frozen manifests and checkpoint provenance under `configs/research/` and `docs/` |
| Sec. 3.2 | Does the duration effect follow training duration? | `configs/research/r2_EXT2x2_result.json` and the symmetric short/long fine-tuning protocol |
| Sec. 3.2 and 4.2 | Is the dense 20k degradation a raw-vs-EMA artifact? | `configs/research/r3_denseraw_result.json`, `docs/review3_denseraw_results.md` |
| Sec. 4.1 | How does released recovery vary with requested duration? | `configs/research/xsev_result.json`, duration sweep and 15.36 s extension artifacts |
| Sec. 4.3 | How do duration and domain interact? | Clotho `r2_E5_result.json`, hip-hop `r2_E7_result.json`, full dense anchors |
| Sec. 4.4 | Is severity 2 sensitive to prompt sampling? | `configs/research/r2_posthoc_review3.json` split-half and cross-severity analysis |
| Sec. 5 | What causal claims remain open? | third-review methodological audit and dense-raw follow-up |

## Figure 1

The figure is generated from the frozen artifacts by `scripts/research/paper_figs/make_draft16_fig1.py` (CPU, 0 cr); the plotting specification it implements is kept as TeX comments above the figure in `icassp/icassp_operating_point.tex` (and in the archived `icassp/archive/draft16_sections/draft16_2.tex`). Panel (a) shows the released severity-2 duration sweep with the 15.36 s point connected only to its matched n=96 10.24 s subset. Panel (b) shows the symmetric 20k intervention and the directional specialization contrast.

## Numerical audit

The paper keeps only values required to understand the claims. The denser numerical layer is in `PAPER_EXPANDED_RESULTS.md` and the committed JSON artifacts. In particular, Draft 16 corrects the primary CLAP interaction interval to `+0.159 [+0.131,+0.187]` and reports the dense raw-vs-EMA control, severity-2 split-half stability, and the hip-hop duration interaction.
