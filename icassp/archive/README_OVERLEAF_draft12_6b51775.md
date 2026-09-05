# ICASSP 2027 Overleaf package

Main file is `icassp_operating_point.tex`.

This package uses the ICASSP 2027 LaTeX format based on `spconf.sty` and `IEEEbib.bst`. The manuscript uses `\documentclass{article}`, loads `spconf`, invokes `\ninept`, and retains the conference two-column geometry and empty page style.

Draft 12 retains the information architecture introduced in Draft 11 and rewrites the prose from scratch without changing the scientific result. The approved Draft 10 abstract is unchanged. Every paragraph from the Introduction through the Conclusion was regenerated from its scientific purpose rather than edited from Draft 11. The Introduction and Background explain what recovery means, how it differs from final checkpoint quality, and why operating-point coverage matters. The methodology introduces each estimand only after motivating the question it answers.

The two complete numerical tables from the previous draft are no longer printed in the four-page body. Figure 1 carries the main visual evidence and the prose retains the effect sizes required to establish each conclusion. Complete table values, sensitivity intervals, intermediate operating points and artifact links live in `PAPER_EXPANDED_RESULTS.md` in the companion GitHub repository. The manuscript points directly to `PAPER_COMPANION.md`, which maps every paper section to the corresponding repository evidence.

The compiled manuscript has four technical pages followed by a references-only fifth page. The approved Draft 10 abstract is unchanged.

Compile with pdfLaTeX. Two passes are recommended because references are defined inside the main file.

See `TEMPLATE_VERIFICATION.txt` and `EDITORIAL_AUDIT.txt` for the checks performed before packaging.
