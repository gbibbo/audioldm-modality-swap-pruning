# ICASSP 2027 Overleaf package

Four files, no subdirectories. Upload the zip to Overleaf and it compiles as delivered.

| File | What it is |
|---|---|
| `icassp_operating_point.tex` | the whole manuscript, one flat file |
| `spconf.sty` | the ICASSP class file (not on CTAN, must ship) |
| `fig1a_duration.pdf` | Figure 1(a) |
| `fig1b_intervention.pdf` | Figure 1(b) |

`IEEEbib.bst` is not shipped: the references are an inline `thebibliography`, so BibTeX never runs.

## Why one file

Draft 16 was first delivered as `icassp_operating_point.tex` plus `sections/draft16_*.tex`. Overleaf
picks the main file by looking for `\documentclass`, found it in `sections/draft16_1.tex`, compiled
that section alone and aborted with `(job aborted, no legal \end found)`, because the matching
`\end{document}` was in `draft16_4.tex`. The single file removes the ambiguity by construction. The
four sections are archived verbatim under `archive/draft16_sections/`; the flat file is their
concatenation in order, and differs from Gabriel's delivery only in the two `\includegraphics` lines.

## Figure 1

Generated from the committed result artifacts by `scripts/research/paper_figs/make_draft16_fig1.py`,
replacing the two reserved boxes. The production specification is kept as comments immediately above
the figure. Each panel is exactly 0.49\textwidth by 4.80 cm, the footprint the source reserved, so
the `figure*` footprint and the four-page technical budget are unchanged.

## Rebuild

`OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/build_draft16_zip.py` —
it unpacks the bundle into a scratch directory and compiles it there before writing the zip, so the
bundle is never published unless exactly those four files build on their own.
