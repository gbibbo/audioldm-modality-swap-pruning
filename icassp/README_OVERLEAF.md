ICASSP manuscript (Draft 5) - Overleaf upload (official ICASSP/ICIP LaTeX template)
===================================================================================

Main document : icassp_operating_point.tex
Compiler      : pdfLaTeX  (Overleaf menu -> Compiler -> pdfLaTeX)   [recommended; proper Times]
Template      : OFFICIAL ICASSP/ICIP style, bundled here:
                  spconf.sty   - ICASSP/ICIP LaTeX style (page geometry, Times, headings)
                  IEEEbib.bst  - IEEE bibliography style (only needed if you switch to BibTeX;
                                 the draft uses an inline \thebibliography, so no .bib is required)
Font size     : 9 pt via \ninept (the template's sanctioned option).
Length        : 4 pages of technical content + references on page 5. ICASSP 2027 paper kit: "4 pages
                of technical content ... and one additional optional 5th page containing only references"
                (confirmed 2026-09-02 on the ICASSP 2027 paper-kit page). Full-paper deadline per the
                CFP: 16 September 2026.

Figures (both embedded in the body):
  figs/fig1_interaction.pdf  - MAIN: P vs P+FT CLAP at short/native, both severities, 95% CI
                               whiskers on the paired gain, matched dense control in (a), and the
                               Draft-5 anchors: real-audio ceiling (triangles) and per-cell
                               shuffled-caption chance floor (short ticks); J annotated.
  figs/fig2_where.pdf        - two-panel: (a) FineLAP frame-level grounding gain vs. time (uniform,
                               not back-loaded); (b) generation length vs scoring window.
Spares on disk (NOT embedded): figs/fig_summary.pdf, figs/fig2_forest.pdf, figs/fig3_finelap.pdf.

Tables: Table 1 recovery gains (all CIs); Table 2 anchors and recovery ratios (chance floor P / P+FT,
real audio, rho vs real audio; rho vs dense at severity 1 in the caption). The analysis-plan table of
Draft 4 was folded into two sentences in Sec. 3.4.

Authors : Gabriel Bibbo (Independent researcher, gabobibbo@gmail.com);
          Arshdeep Singh and Mark D. Plumbley (King's College London, UK,
          {arshdeep.singh, mark.plumbley}@kcl.ac.uk).

Title options (Draft 5 keeps the first): (1) "Recovery Fine-Tuning Recovers Where It Was Trained:
Duration- and Domain-Dependent Gains in Pruned Text-to-Audio Diffusion"; (2) "How Much Does Recovery
Fine-Tuning Recover? Operating-Point-Dependent Gains in Pruned Text-to-Audio Diffusion".

To open in Overleaf: New Project -> Upload Project -> select this .zip (files are at the zip root).
See MANUSCRIPT_NOTES.md for the Draft-4 -> Draft-5 change log, number provenance, and the list of
references whose bibliographic details still need a final check.
