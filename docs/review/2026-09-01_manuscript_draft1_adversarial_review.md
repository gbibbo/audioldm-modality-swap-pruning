# Adversarial review of MANUSCRIPT-DRAFT-1B (`icassp/icassp_operating_point.tex`)

**Date:** 2026-09-01. **Scope:** editorial/presentation + reviewer-risk audit of the built 4-page PDF.
**No science reopened; no number changed.** Every defect below was verified against the built PDF
(rendered pages), the `.tex` source, and the frozen artifacts. Fixes are classified:
**[P] presentation-only** (no GO needed, numbers untouched) vs **[R] re-analysis of committed raw
data** (new derived number → needs explicit Gabriel GO under the freeze).

## A. Verified presentation defects

1. **Fig. 1 lands on page 4** — after every reference to it (§4.3, §4.4) and above
   Limitations/Conclusion/References. The paper's only figure is orphaned on the last page. [P]
2. **Only one figure in 4 pages; pages 1–3 are text walls.** `fig2_forest.pdf` (all six
   context×severity contrasts with CIs) is generated and committed but not embedded. [P]
3. **Fig. 1 shows no uncertainty**: no error bars on means or on the R gaps; the J CI lives only in
   the panel title. Self-defeating for a paper whose entire contribution is inferential. [P]
4. **FineLAP has no figure** although `artifacts/finelap_temporal/scores_sev2.json` holds 110
   prompts × 3 systems × 64-frame grounding curves. "Broad, not late" is literally a picture
   (mean grounding vs frame, boundary at frame 24, flat gap); the draft delivers it as a
   six-number paragraph. [P]
5. **Heterogeneity (a stated core-story item) has zero visualization**, though per-prompt data is
   committed: `op_duration_discriminator_1_result.json` raw_cosines (80×4),
   `reversal_v1_1_result.json` prompt_contrast_vector (96) + raw_clap_scores (3×192). [P]
6. **Related work renders as one run-on block**: no blank line before
   `\textbf{Compression and inference regime.}` and `\textbf{TTA evaluation.}` (verified in
   source), so the three run-in headings collapse into a single paragraph. [P]
7. **`\resizebox{\columnwidth}{!}` on both tables** → arbitrary font scaling, Table 1 (the table
   carrying every primary number) set smaller than body text. [P]
8. **Abstract = 233 words** of CI-laden hedging (ICASSP norm ≈ 120–150). [P]
9. **Bibliography defects**: `kim2019audiocaps` is a bibitem but never `\cite`d; ref [2] — the
   paper under study — prints "(identifier to verify)"; all refs are truncated "et al."; no DDIM
   citation; the compiled PDF prints "Affiliation to be confirmed". [P]
10. **The two secondary evaluators are unnamed and uncited** ("a second CLAP-family scorer",
    "a non-CLAP audio–text grounding model") → irreproducible as printed. [P]

## B. Reviewer-risk findings (scientific presentation, not new science)

11. **No efficiency numbers in a pruning-adjacent paper**: no latency/RTF/MACs/memory anywhere;
    only parameter counts. `docs/compute_budget.md` holds measured T4 latencies (different
    config) and Singh et al. report efficiency — cite or tabulate. [P]
12. **Absolute levels are hidden.** Sev-2 pruned native CLAP = 0.055 vs dense 0.352: the headline
    +0.244 is recovery from near-collapse, visible only by squinting at Fig. 1b's axis. A reviewer
    who finds it will read concealment. Disclose absolute means in Table 1 (they are already in
    the frozen story) and reframe: fine-tuning matters most where pruning hurts most. [P]
13. **J is a difference-in-differences on the raw cosine scale**; scale-dependence/floor effects
    unaddressed; a rank/robust sensitivity check on committed raw scores would harden it. [R]
14. **Two duration points ≠ a temporal trend**; no sweep; not acknowledged in Limitations. [P]
15. **The duration-extrapolation alternative is never separated**: 3.84 s is off-native for every
    system; dense@3.84 s (0.204, n=96) and dense@10.24 s (0.352, n=80) exist in committed
    artifacts and are never used to anchor this discussion. Subset-matched dense means would be a
    re-analysis. State the alternative explicitly in Limitations at minimum. [P]/[R]
16. **Multiplicity**: many gates/contrasts/scorers/seams with no stated correction or hierarchy
    beyond primary/secondary labels. State the hierarchy explicitly. [P]
17. **Negative-first framing**: contribution bullet 3 is a list of failures; ≈25–30% of the body
    is caveat/taxonomy (§3.6 + Limitations + negatives). Keep every negative, but lead with the
    confirmed positive and concentrate negatives in one labeled subsection. [P]
18. **Discussion is non-actionable**: "evaluate across operating points" without saying which or
    what to report. Deliver a minimal reporting checklist (native + one off-native point, paired
    common-noise design, per-prompt CIs, absolute levels of both endpoints). [P]
19. **Question title with the answer absent**; prefer a declarative title. [P]

## C. Suggested figure set (all CPU, from committed artifacts)

* Fig. 1 (page 2, with CI whiskers on the R gaps): interaction panels, as now but with uncertainty.
* Fig. 2: forest plot (already built, un-embedded).
* Fig. 3: FineLAP temporal profile (mean Δgrounding vs frame; early/late boundary; flat gap ⇒ the
  "broad, not late" negative becomes visible).
* Fig. 4 (optional, quarter-column): per-prompt ΔCLAP strip/scatter (sev-1 committed vectors) to
  *show* heterogeneity.

## D. Verdict

As built, the draft under-delivers the project's own evidence: one figure (misplaced, no error
bars), no efficiency context, hidden absolute levels, unnamed evaluators, and caveat-first
framing. The underlying science is publishable; the artifact is not yet. All category-A fixes and
most category-B fixes are presentation-only and respect the freeze.
