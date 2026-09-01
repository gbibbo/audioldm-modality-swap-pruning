ICASSP manuscript (Draft 2) - Overleaf upload (official ICASSP/ICIP LaTeX template)
===================================================================================

Main document : icassp_operating_point.tex
Compiler      : pdfLaTeX  (Overleaf menu -> Compiler -> pdfLaTeX)   [recommended; proper Times]
Template      : OFFICIAL ICASSP/ICIP style, bundled here:
                  spconf.sty   - ICASSP/ICIP LaTeX style (page geometry, Times, headings)
                  IEEEbib.bst  - IEEE bibliography style (only needed if you switch to BibTeX;
                                 the draft uses an inline \thebibliography, so no .bib is required)
Font size     : 9 pt via \ninept (the template's sanctioned option) so the full content fits the
                4-page limit. Remove/comment the \ninept line for 10 pt (then trim to fit).

Figures (all embedded in the body):
  figs/fig1_interaction.pdf  - MAIN: pruned vs post-FT CLAP at short/native, both severities,
                               95% CI whiskers on the paired contrast, dense reference, J annotated.
  figs/fig_summary.pdf       - two-panel: (a) forest of paired contrasts w/ 95% CI; (b) FineLAP
                               frame-level grounding gain vs. time (broad, not late).
Standalone spares (NOT embedded; single-panel versions for a longer/journal version):
  figs/fig2_forest.pdf, figs/fig3_finelap.pdf.

Authors : Gabriel Bibbo (Independent researcher, gabobibbo@gmail.com);
          Arshdeep Singh and Mark D. Plumbley (King's College London, UK,
          {arshdeep.singh, mark.plumbley}@kcl.ac.uk).

To open in Overleaf: New Project -> Upload Project -> select this .zip (files are at the zip root).
See MANUSCRIPT_NOTES.md for the Draft-1 -> Draft-2 change log, claim classes, and provenance.
