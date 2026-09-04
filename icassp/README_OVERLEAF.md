ICASSP manuscript (Draft 6) - Overleaf upload (official ICASSP/ICIP LaTeX template)
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

Figure (one, full text width, five panels; Draft 6 replaced the two column-width figures of Draft 5):
  figs/fig1_operating_points.pdf - (a),(b) duration: P vs P+FT CLAP against generated clip duration at
                               severity 1 and 2, with 95% CI whiskers on the paired gain, the matched
                               dense control, the real-audio ceiling and the per-cell shuffled-caption
                               chance floor; J annotated. (c) NEW in Draft 6, prompt domain: the same
                               two checkpoints in-domain and on held-out hip-hop captions, per
                               (severity, duration) cell, against their own chance floors.
                               (d) FineLAP frame-level grounding gain vs. time (uniform, not
                               back-loaded). (e) generation length vs. scoring window.
Superseded/spare figures live in figs/archive/ (the Draft-5 pair fig1_interaction + fig2_where, and the
never-embedded fig_summary / fig2_forest / fig3_finelap); see figs/archive/README.md. Only
figs/fig1_operating_points.pdf is in the zip.

Tables: Table 1 recovery gains (all CIs); Table 2 anchors and recovery ratios (chance floor P / P+FT,
real audio, rho vs the matched dense model and vs real audio, both severities). The analysis-plan table of
Draft 4 was folded into two sentences in Sec. 3.4.

Authors : Gabriel Bibbo (Independent researcher, gabobibbo@gmail.com);
          Arshdeep Singh and Mark D. Plumbley (King's College London, UK,
          {arshdeep.singh, mark.plumbley}@kcl.ac.uk).

Title options (Draft 6 keeps the first): (1) "Recovery Fine-Tuning Recovers Where It Was Trained:
Duration- and Domain-Dependent Gains in Pruned Text-to-Audio Diffusion"; (2) "How Much Does Recovery
Fine-Tuning Recover? Operating-Point-Dependent Gains in Pruned Text-to-Audio Diffusion".

LOCAL PREVIEW CAVEAT: `icassp_operating_point.pdf` in the repo is built locally with tectonic plus
Liberation Serif (metric-compatible with Times) because plain tectonic has no `TUptm.fd` and silently falls
back to the wider Latin Modern, which adds about 0.6 of a column and makes the paper look over-length.
Overleaf's pdfLaTeX build (NimbusRomNo9L = Times) is the authoritative one; use
`scripts/research/paper_figs/pagecheck_times.py` for any local page-budget decision.

To open in Overleaf: New Project -> Upload Project -> select this .zip (files are at the zip root).
See MANUSCRIPT_NOTES.md for the Draft-5 -> Draft-6 change log, number provenance, and the list of
references whose bibliographic details still need a final check.
