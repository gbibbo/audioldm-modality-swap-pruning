ICASSP manuscript (Draft 3) - Overleaf upload (official ICASSP/ICIP LaTeX template)
===================================================================================

Main document : icassp_operating_point.tex
Compiler      : pdfLaTeX  (Overleaf menu -> Compiler -> pdfLaTeX)   [recommended; proper Times]
Template      : OFFICIAL ICASSP/ICIP style, bundled here:
                  spconf.sty   - ICASSP/ICIP LaTeX style (page geometry, Times, headings)
                  IEEEbib.bst  - IEEE bibliography style (only needed if you switch to BibTeX;
                                 the draft uses an inline \thebibliography, so no .bib is required)
Font size     : 9 pt via \ninept (the template's sanctioned option) so the full content fits the
                4-page limit (verified locally with tectonic: 4 pages incl. references).

Figures (both embedded in the body):
  figs/fig1_interaction.pdf  - MAIN: P vs P+FT CLAP at short/native, both severities, 95% CI
                               whiskers on the paired gain, dense reference, J annotated.
  figs/fig2_where.pdf        - two-panel: (a) FineLAP frame-level grounding gain vs. time (uniform,
                               not back-loaded); (b) generation length vs scoring window (R on the
                               generated 3.84 s clip / first 3.84 s of the 10.24 s clip / full clip).
Spares on disk (NOT embedded): figs/fig_summary.pdf, figs/fig2_forest.pdf, figs/fig3_finelap.pdf.

Authors : Gabriel Bibbo (Independent researcher, gabobibbo@gmail.com);
          Arshdeep Singh and Mark D. Plumbley (King's College London, UK,
          {arshdeep.singh, mark.plumbley}@kcl.ac.uk).

To open in Overleaf: New Project -> Upload Project -> select this .zip (files are at the zip root).
See MANUSCRIPT_NOTES.md for the Draft-2 -> Draft-3 change log, number provenance, and the list of
references whose bibliographic details still need a final check.
