# Operating-point evaluation of recovery fine-tuning

This branch accompanies the ICASSP manuscript **Recovery Gain Is Operating-Point Dependent in Pruned Text-to-Audio Diffusion**.

The study evaluates released pruned and recovered AudioLDM-M checkpoints and asks a specific question: how stable is the incremental gain from recovery fine-tuning when inference duration or prompt domain changes?

## Main findings

- At 82.9% pruning, released recovery gain rises from +0.085 CLAP at 3.84 s to +0.244 at 10.24 s.
- A symmetric 20k intervention falsifies the directional prediction of training-duration specialization at that budget: Delta J = -0.035 [-0.064,-0.004].
- A dense raw-baseline control shows that the short dense 2x2 degradation is real under the 20k recipe, not a raw-vs-EMA artifact.
- Recovery transfers strongly to Clotho but remains small on hip-hop. The duration interaction is +0.112 on Clotho and +0.001 on hip-hop.
- Severity-2 prompt split-halves are stable, while severity 1 shows substantial between-draw heterogeneity.

## Manuscript and audit layer

- `icassp/icassp_operating_point.tex` is the canonical manuscript source on the review branch.
- `PAPER_COMPANION.md` maps paper sections to result artifacts.
- `PAPER_EXPANDED_RESULTS.md` contains numerical detail intentionally omitted from the four technical pages.
- `docs/review3_denseraw_results.md` records the dense raw-baseline experiment.
- `docs/review/2026-09-06_review_round3_methodological_response.md` records the third-review methodological audit.

The final paper figure is intentionally not generated in the editorial draft. The exact two-panel plotting specification is embedded in the LaTeX source so that it can be rendered later from the committed per-prompt artifacts without changing the page layout.


Camera-ready companion landing page: [`paper-operating-point-recovery/`](paper-operating-point-recovery/).
