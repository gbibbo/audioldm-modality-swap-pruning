ICASSP manuscript - Overleaf upload (official ICASSP/ICIP LaTeX template)
=========================================================================

Main document : icassp_operating_point.tex
Compiler      : pdfLaTeX  (Overleaf menu -> Compiler -> pdfLaTeX)   [recommended; proper Times]
Template      : OFFICIAL ICASSP/ICIP style, bundled here:
                  spconf.sty   - ICASSP/ICIP LaTeX style (page geometry, Times, headings)
                  IEEEbib.bst  - IEEE bibliography style (only needed if you switch to BibTeX;
                                 the draft uses an inline \thebibliography, so no .bib is required)
Font size     : 9 pt via \ninept (the template's sanctioned option, enabled right after
                \begin{document}) so the full content fits the 4-page limit. Remove/comment the
                \ninept line for 10 pt (then trim to fit 4 pages).
Figures       : figs/fig1_interaction.pdf (referenced). figs/fig2_forest.pdf is SUPPLEMENTARY
                (forest plot of the six context x severity contrasts) and is NOT in the body.

To open in Overleaf:  New Project -> Upload Project -> select this .zip (files are at the zip root).

Authors: affiliations are placeholders derived from the supplied e-mail domains
(gabobibbo@gmail.com; {arshdeep.singh,mark.plumbley}@kcl.ac.uk) -- confirm before submission.
See MANUSCRIPT_NOTES.md for narrative decisions, claim classes, and open editorial items.
