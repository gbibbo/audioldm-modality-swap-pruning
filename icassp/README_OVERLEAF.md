ICASSP manuscript (Draft 4) - Overleaf upload (official ICASSP/ICIP LaTeX template)
===================================================================================

Main document : icassp_operating_point.tex
Compiler      : pdfLaTeX  (Overleaf menu -> Compiler -> pdfLaTeX)   [recommended; proper Times]
Template      : OFFICIAL ICASSP/ICIP style, bundled here:
                  spconf.sty   - ICASSP/ICIP LaTeX style (page geometry, Times, headings)
                  IEEEbib.bst  - IEEE bibliography style (only needed if you switch to BibTeX;
                                 the draft uses an inline \thebibliography, so no .bib is required)
Font size     : 9 pt via \ninept (the template's sanctioned option).
Length        : 4 pages of content + references on page 5 (verified locally with tectonic). ICASSP
                allows a 5th page containing ONLY references -- confirm against the ICASSP 2027 CFP.
                If 4 pages total are required: cut Sec. 2 (Background) by ~8 lines, or switch the
                bibliography to BibTeX/IEEEbib with abbreviated venues.

Figures (both embedded in the body):
  figs/fig1_interaction.pdf  - MAIN: P vs P+FT CLAP at short/native, both severities, 95% CI
                               whiskers on the paired gain, MATCHED dense control in (a) with the
                               three duration responses annotated, J annotated.
  figs/fig2_where.pdf        - two-panel: (a) FineLAP frame-level grounding gain vs. time (uniform,
                               not back-loaded); (b) generation length vs scoring window (R on the
                               generated 3.84 s clip / first 3.84 s of the 10.24 s clip / full clip).
Spares on disk (NOT embedded): figs/fig_summary.pdf, figs/fig2_forest.pdf, figs/fig3_finelap.pdf.

Authors : Gabriel Bibbo (Independent researcher, gabobibbo@gmail.com);
          Arshdeep Singh and Mark D. Plumbley (King's College London, UK,
          {arshdeep.singh, mark.plumbley}@kcl.ac.uk).

Title options (Draft 4 uses the first): (1) "Recovery Fine-Tuning Recovers Where It Was Trained:
Duration- and Domain-Dependent Gains in Pruned Text-to-Audio Diffusion"; (2) "How Much Does Recovery
Fine-Tuning Recover? ..." (Draft 3); (3) "Post-Pruning Recovery in AudioLDM Restores Native-Duration
Alignment In-Domain Only".

To open in Overleaf: New Project -> Upload Project -> select this .zip (files are at the zip root).
See MANUSCRIPT_NOTES.md for the Draft-3 -> Draft-4 change log, number provenance, and the list of
references whose bibliographic details still need a final check.
