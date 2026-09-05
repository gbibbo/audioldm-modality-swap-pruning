# ICASSP 2027 Overleaf package

Main file is `icassp_operating_point.tex`.

This package uses the ICASSP 2027 LaTeX format based on `spconf.sty` and `IEEEbib.bst`. The manuscript uses `\documentclass{article}`, loads `spconf`, invokes `\ninept`, and retains the official two-column geometry and empty page style.

Draft 13 incorporates the ICASSP reviewer follow-up experiments. The central claim is now that recovery gain is operating-point dependent. The manuscript explicitly does not attribute the duration interaction to pruning or to specialization at the recovery-training duration. It adds the 3.84-s fine-tuning intervention, the 15.36-s duration point, Clotho transfer, dense hip-hop anchors, increased-power severity-1 and hip-hop analyses, and an in-paper compact robustness table.

The paper remains self-contained. `PAPER_COMPANION.md` maps paper sections to repository evidence, `PAPER_EXPANDED_RESULTS.md` contains the denser numerical layer, and `docs/reviewer2_response_manuscript.md` records how each reviewer concern was addressed.

The compiled manuscript is five pages. Pages 1 to 4 contain technical content; page 5 contains references only.
