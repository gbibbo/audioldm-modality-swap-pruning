# MANUSCRIPT_NOTES — ICASSP first draft

**Project folder:** everything lives in `icassp/` (this folder): `icassp_operating_point.tex`, the
official style files `spconf.sty` + `IEEEbib.bst`, `figs/`, the built `icassp_operating_point.pdf`,
this notes file, `README_OVERLEAF.md`, and the ready-to-upload `icassp_operating_point_overleaf.zip`
(gitignored; regenerable). The paper is **4 pages total** (content + references). Written from the
frozen scientific state at repository HEAD `e3d761c` (`docs/final_scientific_story.md` +
`configs/research/final_claim_registry.json`, self-sha `ff023016…`). Every number is read from a
committed frozen artifact under `configs/research/`; none was copied from chat or an old draft.
All work here is CPU-only (LaTeX + Matplotlib), 0 GPU, 0 credits.

## Template and how to build

The manuscript is on the **official ICASSP/ICIP LaTeX template** — `\documentclass{article}` +
`\usepackage{spconf,...}`, with `spconf.sty` and `IEEEbib.bst` bundled in this folder (fetched from
the public ICASSP paper kit; these files are stable across ICASSP years incl. 2027). spconf sets
Times and the ICASSP page geometry and centered numbered headings. The body uses the template's
sanctioned **9 pt mode** (`\ninept`, enabled right after `\begin{document}`) so the full content
fits the 4-page limit; comment that line out for 10 pt (then trim to fit). The bibliography is an
inline `\thebibliography` (no `.bib`/BibTeX pass needed; `IEEEbib.bst` is bundled only in case you
switch to BibTeX).

```bash
# figures (reads durable artifacts, writes icassp/figs/*.pdf)
OPENBLAS_CORETYPE=Haswell python scripts/research/paper_figs/make_manuscript_figs.py
# manuscript (pdfLaTeX on Overleaf; locally, tectonic -- spconf.sty must sit beside the .tex)
cd icassp && tectonic -X compile icassp_operating_point.tex --outdir build
```

Overleaf: upload `icassp_operating_point_overleaf.zip` (New Project → Upload Project; files are at
the zip root), set the compiler to **pdfLaTeX** (proper Times; no font warnings). On tectonic/XeTeX
locally the paper compiles but emits harmless Times-shape substitution warnings.

## Narrative decisions

* **Framing = evaluation/audit of post-pruning fine-tuning.** Not a new pruning or fine-tuning
  method; not a mechanistic explanation. The released pruned and post-fine-tuning checkpoints are
  the object of study.
* **Central claim = temporal operating-point sensitivity.** The confirmatory quantity is the
  temporal interaction `J = R_native − R_short`. Severity 1 is directional evidence; severity 2 is
  the prospective confirmation (disjoint set, stronger severity, frozen estimands/gates).
* **Terminology.** "post-fine-tuning checkpoint" / "pruned checkpoint" throughout; the word
  "recovered model" is avoided in the body per the registry. "Recovery" is used only as the name of
  the fine-tuning stage / as a general concept, never as an achieved restoration.
* **The failed reversal is the opening move.** The pre-registered severity-1
  domain-specialization/reversal hypothesis (`R_AC ≈ 0`, gate FALSE) motivates and reframes the
  paper; it is presented as a genuine negative, not hidden.
* **`K` is a bundled context contrast (domain + duration), never a pure-domain effect.** Stated in
  Methods and in the Table I footnote. The matched-duration domain residual (a post-hoc descriptive
  decomposition) is deliberately NOT promoted to a confirmatory domain claim.
* **CASE C is reported honestly:** context (`K`) and duration (`J`) dependence replicate at
  severity 2, but the specific severity-1 native-positive/music-negative *sign pattern* does not
  (`R_music ≈ 0`; `H_music` FALSE). The short-operating-point advantage is explicitly **present, not
  absent**, at severity 2 (`R_short` resolved > 0; equivalence FAILS).
* **A′/B′ seam.** A′ is the method-consistent primary pruning convention; B′ reproduces the
  published `dp1` decoder-seam convention and is a pre-specified sensitivity analysis. The paper
  states the main result (`J`, `K`) is robust to the convention and never describes A′ as the proven
  exact parent of the public post-fine-tuning checkpoint.
* **Secondaries are conservative.** CLAP is primary. Human-CLAP is labeled a "second CLAP-family
  scorer … not human evaluation." KL and the PANN top-10 *captured-label count* (a count, not a
  rate) are pre-specified secondaries implemented after the primary result, reported with paired
  CIs. FD/FAD are descriptive (no paired CI).
* **FineLAP** is labeled a prospectively frozen **post-result diagnostic**: a broad native-duration
  frame-level grounding gain, but the late-redistribution hypothesis is rejected (`T ≈ 0`, gate
  fails). It is described as frame-level grounding evidence, not calibrated probability, causal
  activation, or perceptual quality. To save space it has no figure and is one paragraph.
* **Discussion** keeps the "evaluate across operating points" message as a methodological
  implication, explicitly *not* a universal empirical law.

## What was omitted for the ICASSP page budget

* **Figure 2 (forest plot of all six context×severity contrasts).** Generated and retained at
  `icassp/figs/fig2_forest.pdf` (+ `.png`) by the same script, but **not embedded** in the
  4-page body; its information is fully carried by Table I and the results text. It is a ready
  drop-in for a longer/journal version.
* **Cross-severity magnitude deltas** (post-hoc, CI-excluding-0 but confounded) — reduced to a
  single qualitative "exploratory only" sentence; the numeric Δ values live in
  `docs/xsev_postresult_adversarial_audit.md` §5.
* **Provenance detail** — per-experiment SHAs, seeds, oracle proofs, the music eligibility funnel,
  EMA reconstruction specifics, and the exact FineLAP window/τ definitions — left to the repository.
* **Dense secondary numbers** (`G_pruned`, `G_post-FT`) compressed to "neither resolves a residual
  gap to dense nor establishes equivalence."
* **The public-examples heterogeneity illustration** (e.g. Example 4) is referenced as
  "heterogeneous across prompts" without enumerating a sample, to avoid elevating an illustration to
  evidence.

## Claim inventory (class per the registry)

| Result in paper | Class |
|---|---|
| Severity-2 `J`, `K`, `R_native`, `R_short`, `R_music` (CASE C) + seam B′ | prospectively frozen **primary** |
| Severity-1 duration interaction `J` (directional; CI includes 0) | prospectively specified **follow-up** |
| Severity-1 reversal `R_AC` (PASS=FALSE) | prospectively frozen **primary NEGATIVE** |
| Human-CLAP contrasts | **secondary**, corroborative, not human eval |
| KL / PANN captured-label count | **secondary** (pre-specified, implemented after primary), inferential |
| FD / FAD | **descriptive** (no CI) |
| FineLAP grounding + `T` | prospectively frozen **post-result diagnostic** |
| Cross-severity magnitude | **exploratory/confounded** (not in the main claims) |

## Items that still need Gabriel's editorial decision

1. **Authors set; affiliations to CONFIRM.** Byline is now Gabriel Bibb\'o
   (`gabobibbo@gmail.com`), Arshdeep Singh and Mark D. Plumbley
   (`{arshdeep.singh,mark.plumbley}@kcl.ac.uk`). Affiliations were **inferred from the e-mail
   domains**: the two `@kcl.ac.uk` authors are set to *King's College London, UK*, and Gabriel's is a
   visible placeholder ("Affiliation to be confirmed"). **Confirm both** — note these researchers are
   frequently associated with the University of Surrey / CVSSP, so verify KCL vs Surrey and fill in
   Gabriel's affiliation. (The `\twoauthors` block in the `.tex` is a one-line edit.)
2. **Reference `[2]` (Singh et al.).** Title/author list/venue and the arXiv identifier are marked
   "(identifier to verify)". Confirm the exact citation (the prior draft used arXiv:2607.13330).
3. **Naming the secondary scorers/evaluators.** The draft calls Human-CLAP "a second CLAP-family
   scorer" and FineLAP "a non-CLAP audio–text grounding model" without naming them, to avoid
   under-explained acronyms in 4 pages. Decide whether to name Human-CLAP
   (`sarulab-speech/human-clap-wsce-mae`) and FineLAP (EAT audio encoder + text encoder) and add
   their citations.
4. **References** are compact (first-author "et al.", short venues). Expand author lists and verify
   every venue/year before submission; add DDIM/EAT/RoBERTa/AudioCaps-vs-MusicCaps citations if the
   named-scorer decision (item 3) is taken.
5. **Title.** "When Does Post-Pruning Fine-Tuning Help? Temporal Operating-Point Sensitivity in
   Text-to-Audio Diffusion Evaluation" — confirm or replace.
6. **Whether to reinstate Figure 2** (forest) for a longer/journal version.
7. **Camera-ready boilerplate** not included in this draft: acknowledgements, a data/repro
   availability statement, and (if desired) an explicit companion-page footnote beyond the
   Conclusion link.

## Verification performed

* Compiled on the official ICASSP `spconf` template with tectonic; inspected the rendered PDF —
  **4 pages total, no overfull-hbox overflow** (one negligible 0.9pt line remains; the contributions
  list is set `\raggedright` so `domain-specialization`, which TeX cannot re-hyphenate, no longer
  overflows the narrow 9pt column). `\emergencystretch` absorbs other tight lines.
* On tectonic/XeTeX, `spconf` requests Times (`\rmdefault=ptm`), so XeTeX emits harmless
  `Font shape TU/ptm ... undefined` substitution warnings; the serif renders clean and **pdfLaTeX on
  Overleaf uses real Times with no warnings**. No action needed.
* Numbers, forbidden-language, evidence-class, figure/table consistency, and required-negatives
  completeness were re-checked against the frozen artifacts by a 5-agent adversarial pass
  (Workflow tool) plus a manual scan. Result: **PASS**. Three minor issues were found and fixed:
  (i) the seam-B′ interaction was printed `+0.162` (double-rounding of the story's 4-dp `0.1615`);
  corrected to `+0.161` (artifact `0.16145…`). (ii) Table I's severity-1 `R_music` (−0.094,
  CI excludes 0) was missing the `†` its own legend implies; added. (iii) §Corroboration now
  discloses that the second CLAP-family scorer shows a small severity-2 music-negative (−0.037)
  that the primary CLAP does not, with no gate role (it was previously one-sided). The manual
  forbidden-language scan confirmed every "restore/pure-domain/perceptual/causal" token appears only
  inside an explicit negation, and "recovery" is used only for the fine-tuning stage, never
  "recovered model."
