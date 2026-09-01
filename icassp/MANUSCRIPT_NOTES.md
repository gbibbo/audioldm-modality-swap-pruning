# MANUSCRIPT_NOTES - ICASSP Draft 2

**Everything lives in `icassp/`:** `icassp_operating_point.tex`, official style files
`spconf.sty` + `IEEEbib.bst`, `figs/`, the built `icassp_operating_point.pdf`, this file,
`README_OVERLEAF.md`, and the ready-to-upload `icassp_operating_point_overleaf.zip`
(gitignored; regenerable). **4 pages total** (content + references), 0 overfull boxes.
Written from the frozen scientific state at HEAD `e3d761c` (`docs/final_scientific_story.md`
+ `configs/research/final_claim_registry.json`, self-sha `ff023016...`). Every number is read
from a committed frozen artifact under `configs/research/`. CPU-only (LaTeX + Matplotlib), 0 GPU,
0 credits. **No new experiment, GPU run, scoring, or inferential analysis was performed for Draft 2.**

## Build

```bash
# figures (reads durable artifacts, writes icassp/figs/*.pdf)
OPENBLAS_CORETYPE=Haswell python scripts/research/paper_figs/make_manuscript_figs.py
# manuscript (pdfLaTeX on Overleaf; locally, tectonic -- spconf.sty must sit beside the .tex)
cd icassp && tectonic -X compile icassp_operating_point.tex --outdir build
```

## Draft 1 -> Draft 2: major changes

* **Positive-first reframe.** Results now LEAD with the severity-2 prospective replication
  (J, K, R_native) and the "advantage present at the short point too", then fold the failed
  domain-specialisation hypothesis into the Results as the scientific *path* (Sec. 4.3), not a
  headline. The Introduction's contribution bullets are three positive items (framework;
  prospective cross-severity interaction + seam sensitivity; multi-metric/frame-level
  characterisation with boundary conditions) - no "list of failures" bullet.
* **Declarative title:** "Post-Pruning Fine-Tuning Gains Depend on the Temporal Evaluation
  Operating Point in AudioLDM" (scoped to AudioLDM; no universal-law claim).
* **Abstract** rewritten to ~155 words (was 233): problem -> operating-point finding -> sev-2 J
  as the headline number -> prospective replication + seam robustness -> short-point still
  positive -> methodological implication + one-line limitation. No caveat pile-up.
* **Absolute levels exposed.** Table 1 (core) now shows, per severity/operating point, the
  absolute Pruned and Post-FT mean CLAP cosine AND the paired contrast R with 95% CI; the text
  states the sev-2 native pruned score (0.055) vs post-FT (0.299) explicitly ("recovery matters
  most where pruning hurts most"). No "restoration to dense" claim.
* **Uncertainty is visible.** Fig. 1 draws 95% CI whiskers of the paired contrast about the
  post-FT mean at each operating point; Fig. 2(a) is a forest plot of all six contrasts + J with
  95% CI (filled = CI excludes 0).
* **FineLAP figure added.** Fig. 2(b) is a descriptive time-course of the frame-level
  Delta-grounding (post-FT - pruned) vs. time with the 3.84 s early/late boundary and the
  D_early/D_late window means, making "broad, not late" (T ~ 0) visible.
* **Efficiency/pruning context.** Table 1 (systems) reports exact U-Net parameter counts
  (Dense 415.96 M -> sev-1 145.67 M, -65.0% -> sev-2 71.08 M, -82.9%) with the channel
  multipliers. Counts computed on CPU from the materialised checkpoints (sum of `numel` over
  tensors keyed `model.diffusion_model.*`); validated bit-exact against the two independently
  documented values (415.955 M dense, 145.674 M sev-1). No FLOPs/MACs/latency/RTF/memory were
  fabricated; no Singh efficiency number was borrowed.
* **Tables de-scaled.** `\resizebox` removed from all tables; they are redesigned to fit the
  column at normal 9 pt (shorter labels, `\multirow` severity groups). The secondary
  corroboration table was folded into one prose sentence to save space.
* **Related Work** uses `\rwhead` (bold run-in headings on their own paragraph break) so the
  three sub-topics are skimmable, not a wall of text.
* **Named + cited evaluators.** CLAP `laion/clap-htsat-fused` (rev 365dea6e); Human-CLAP
  (`sarulab-speech/human-clap-wsce-mae`, Takano et al., arXiv:2506.23553) explicitly labelled a
  CLAP-family *model* fine-tuned on human scores, NOT a listening test; FineLAP (Li et al.,
  arXiv:2604.01155, ACL 2026) named with its EAT audio encoder; DDIM, AudioCaps, MusicLM cited.
* **Discussion is actionable:** a concrete 7-point reporting checklist for post-pruning recovery
  studies (native + >=1 off-native operating point; absolute levels; paired common-noise
  contrasts; prompt-unit uncertainty; interaction contrast; automatic-vs-human separation).
* **Limitations** compressed to one section (mechanism blocked; only two temporal points, no
  continuous trend; short is off-native; CLAP-family primary; no human eval; off-recipe DDIM50;
  cross-severity exploratory; FineLAP post-result).
* **Placeholders removed:** author affiliations resolved (Gabriel Bibbo = Independent researcher;
  Singh & Plumbley = King's College London, per Gabriel 2026-09-01); Singh et al. citation
  resolved ("Efficient text-to-audio generation via pruning," DCASE 2026, arXiv:2607.13330);
  evaluators named/cited; no "to be confirmed" / "identifier to verify" remain.

## External review criticisms -> disposition

Addressed (presentation-only, freeze respected): A1 (fig on p4 -> Fig 1 on p3, before p4);
A2 (only one figure -> two figures incl. the un-embedded forest, now Fig 2a); A3 (no uncertainty
-> CI whiskers + forest); A4 (no FineLAP figure -> Fig 2b); A5 (heterogeneity now shown as a
traceable 64%/44% native-vs-short win-rate); A6 (Related Work headings); A7 (`\resizebox`
removed); A8 (abstract ~155 w); A9/A10 (citations + named evaluators; no placeholders);
B11 (efficiency: exact U-Net params); B12 (absolute levels exposed); B14 (two-points-not-a-trend
stated in Limitations); B15 (duration-extrapolation alternative: dense@3.84 vs dense@10.24 shown
in Fig 1a and discussed as "short is off-native for the whole family"); B17 (positive-first);
B18 (actionable checklist); B19 (declarative title).

**Intentionally NOT addressed (would require NEW post-result inferential analysis, forbidden by
the current freeze; flagged for a separate authorised pass):**
* B13 - rank/robust-scale sensitivity check on J (a new estimand on the raw scores).
* B15 (inferential part) - subset-matched dense@3.84 vs dense@10.24 *paired* means as a formal
  duration-extrapolation control (a new paired contrast). Only the existing descriptive dense
  scores are shown.
* B16 - a formal multiplicity correction / stated testing hierarchy beyond the existing
  primary/secondary + prospectively-frozen labels.
These are recorded here per the instruction to stop any presentation change that would need new
analysis; they need explicit approval to implement.

## Figures / tables retained (and why)

| Element | Content | Why |
|---|---|---|
| Fig. 1 (single col, stacked) | pruned vs post-FT CLAP at short/native, sev 1 & 2; CI whiskers; dense ref | the central operating-point interaction + absolute levels + uncertainty, before p4 |
| Fig. 2 (single col, 2-panel) | (a) forest of 6 contrasts + J w/ 95% CI; (b) FineLAP Delta-grounding vs time | inferential summary + the "broad, not late" negative, next to their arguments |
| Table 1 | systems: channel_mult, exact U-Net params, reduction | efficiency/pruning context, reader-immediate |
| Table 2 | absolute Pruned/Post-FT + R [95% CI] per severity/op-point + J, K rows + reversal footnote | absolute levels + primary contrasts + the pre-registered negative in one place |

Both severity panels of Fig. 1 were made single-column (stacked) so all four floats are
single-column; this resolved a float-placement logjam that otherwise forced a 5th page.
Fig. 2 merges the forest and FineLAP into one two-panel float (one caption) for the page budget;
single-panel spares `fig2_forest.pdf` / `fig3_finelap.pdf` are generated and committed for a
longer/journal version but not embedded.

## Provenance guard on the FineLAP figure

Fig. 2(b) reconstructs the per-frame Delta-grounding curve from the raw frame scores
`artifacts/finelap_temporal/scores_sev{1,2}.json` (gitignored). The figure script ASSERTS their
`scores_sha256` equals the value recorded in the committed frozen verdict
`configs/research/finelap_temporal_result.json`, and that the reconstructed window means reproduce
the frozen `D_early`/`D_late` to < 1e-6 (they do, bit-exact). The plotted curve is therefore
provably the same object as the frozen post-result diagnostic - a visualisation, not a new
statistic.

## Items that still need Gabriel's editorial decision

1. **Author affiliations** are now filled (Independent researcher; King's College London) per the
   2026-09-01 decision. Note the paper under study (arXiv:2607.13330) lists the CVSSP/University of
   Surrey group (Singh, Yuan, Chen, W. Wang, Plumbley); the @kcl.ac.uk / KCL byline was Gabriel's
   explicit instruction - confirm once more before submission if a Surrey affiliation is intended.
2. **Reference completeness.** Long author lists were shortened to "et al." after three names to
   fit four pages (AudioLDM, BK-SDM, CLAP, PANNs, FAD, MusicLM, Human-CLAP, FineLAP). Expand for a
   camera-ready/journal version if space allows. EAT (FineLAP's audio encoder) is named in text but
   not separately cited (page budget); add the EAT citation for the journal version.
3. **Camera-ready boilerplate** not included: acknowledgements, data/reproducibility statement.

## Verification performed (Draft 2)

* Compiled with tectonic (official spconf template): **4 pages, 0 overfull/underfull-hbox** beyond
  the harmless XeTeX `TU/ptm` Times-substitution font warnings (pdfLaTeX on Overleaf uses real
  Times, no warnings).
* Both figure PDFs checked for out-of-bounds text blocks (clipping): **none**; all labels present.
* Every numeric value cross-checked against the frozen artifacts (Table 1/2, corroboration prose,
  FineLAP paragraph, dense reference, parameter counts). The 64%/44% heterogeneity figures are the
  committed per-prompt win-rates (recovered > pruned) at native/short (sev-1, n=80).
* Forbidden-language scan: every "restor*/pure-domain/causal/perceptual/single restored score"
  token appears only inside an explicit negation; no "recovered model", "consistently improves",
  "advantage disappears at 3.84 s", or "human listeners confirmed".
* Placeholder scan: no "to be confirmed" / "identifier to verify" / "TODO" / `\resizebox`.
* Bibliography: all 12 `\cite` keys have a `\bibitem` and every `\bibitem` is cited (no unused).
* Author lists for the two newest citations verified against arXiv (Human-CLAP = Takano et al.,
  NOT the earlier guessed "Yamamoto"; FineLAP = Li et al.).

## NOTE on visual inspection

Page-by-page image rendering to the operator was unavailable this session (the IDE display host was
unreachable), so the compiled PDF was verified by text/geometry extraction (page count, float page
assignment, per-block bounding boxes for clipping, and full text of both figures) rather than by
eye. A final human glance on Overleaf is advisable before submission.
