# ICASSP reviewer simulation of the Draft-5 manuscript (`icassp/icassp_operating_point.tex`, dense-192 integrated)

**Date:** 2026-09-03 (10:23 →, America/Montevideo). **Trigger:** Gabriel asked to identify the latest
Overleaf-ready ICASSP manuscript and review it as a busy ICASSP reviewer who cannot accept every paper: a
concrete score, what the score weighs, and the written review. **Object reviewed:** the Draft-5 manuscript
with the XSEV-DENSE-192-CONTROL integration, as committed at `73ce72a` (tex 2026-09-03 05:36 UTC; built PDF 5
pages = 4 content pages + references page; `icassp_operating_point_overleaf.zip` rebuilt at the same time;
22 references; Fig. 1 interaction plot, Fig. 2 FineLAP/crop; Tables 1-2). **Compute:** CPU only, 0 cr,
read-only on every frozen artifact. **Scope note:** this is a simulation of an *external* ICASSP reviewer
(not the co-author-level reviews of Drafts 1-4); it does not re-derive numbers (the verifier reports
108/108) and does not reopen closed experiments.

## 0. Two defects found in the "ready" package (fixed in this pass, no number changed)

* **D1 — literal cross-reference in the compiled PDF.** Sec. 4.1, second paragraph, printed
  "82 % at 10.24 s (Table eftab:anchors)." The `.tex` at `73ce72a` held `(Table~` + newline + `ef{tab:anchors})`
  (a lost `\r` from a manual post-integration edit; the integration script itself writes `\ref` correctly).
  Fixed to `(Table~\ref{tab:anchors})`; PDF and zip rebuilt.
* **D2 — stale Fig. 1 caption.** After the dense-192 integration both panels carry the matched dense line, but
  the caption still said "Panel (a) includes the matched dense control (stars; same 80 prompts ...)".
  Fixed to "Both panels include the matched dense control (stars; same prompts and scoring convention:
  80 at severity 1, 192 at severity 2)."

Both are exactly the kind of blemish that costs a borderline paper its "clarity" point. Neither affects a number.

## 1. Score (ICASSP review form, 1-5)

| Criterion | Score | One-line reason |
|---|---|---|
| Importance / relevance to ICASSP | 3 | Evaluation practice for pruned TTA models; relevant to the AASP track, narrow audience. |
| Novelty / originality | 2.5 | The finding (fine-tuning helps most at its own duration/domain) is what a reader expects a priori; the novelty is the anchored, paired, pre-specified measurement protocol and the recovery-ratio quantity, not the phenomenon. |
| Technical correctness | 4 | Paired design under common random numbers, prompt-level bootstrap, Holm, seam sensitivity, chance floor and real-audio ceiling, pre-specification stated; no error found. |
| Experimental validation | 3 | Thorough within its box, but the box is small: one model family, one checkpoint pair per severity, two durations (a two-point "interaction"), a domain axis confounded with caption style, no training, no human ratings; the headline interaction is corroborated only by CLAP-family scorers. |
| Clarity of presentation | 2.5 | Results are a wall of numbers in prose; two figures at 0.74/0.58 column width with 6-7 pt lettering; a literal `ef{tab:anchors}` and a stale caption in the submitted PDF (D1, D2). |
| Reference to prior work | 4 | Adequate and current; no obvious omission for a 4-page paper. |
| **Overall** | **3 / 5 — borderline, weak accept if the session has room** | Honest, careful, reproducible; incremental and hard to read. A busy reviewer would not fight for it; two of three reviewers might land at 3, one at 2. |
| Reviewer confidence | 4 | Familiar with TTA evaluation and pruning; did not re-run anything. |

**What would move it to 4:** (i) remove the three overclaims listed in §3 (cost: minutes); (ii) event-level
metrics (KL, PANNs) at the 3.84 s point on the WAVs already on disk, so that at least one non-CLAP metric
tests the duration dependence (CPU, 0 cr); (iii) one extra duration or one sampler setting from the published
recipe (GPU; conflicts with the project's closure of new GPU generation — rebuttal answer instead);
(iv) a readability pass that moves half of the numbers in Secs. 4.1-4.2 into the tables and states the
"so what" of each paragraph first; (v) figures at full column width.

## 2. What weighed positively

1. **A pre-specified primary endpoint with a replication.** The severity-2 test (192 disjoint prompts,
   estimand and gate frozen before scoring) is the kind of design ICASSP evaluation papers rarely have; the
   severity-1 "underpowered" statement and the Holm result are reported instead of hidden.
2. **Anchors.** The shuffled-caption chance floor per cell and the real-audio ceiling of the *same prompts*
   turn CLAP cosines into interpretable quantities; the recovery ratio ρ is the number a practitioner wants.
3. **The crop analysis** (first 3.84 s of the 10.24 s generation vs. a generated 3.84 s clip, same scorer
   convention) is the single best argument that the short-point deficit is a generation effect, not a
   repeat-padding artefact of CLAP. This paragraph carries the paper.
4. **The matched dense control at both severities** (s(dense) ≈ +0.15 on both prompt sets; s(P) below and
   s(P+FT) above it at severity 2) separates the scorer's duration response from the systems'.
5. **Negative results kept** (reversal hypothesis, music-penalty replication, late-allocation account),
   and "not restored to dense" stated with a TOST-style reading.
6. **Reproducibility statement is unusually concrete** (scorer revision, seeds, checksums, pre-specification
   in a version-controlled repository, audio page).

## 3. What weighed negatively (as a reviewer would write it)

1. **Incremental contribution.** The title is nearly a tautology for the reader who knows Kumar et al. (2022):
   fine-tuning is best evaluated where it was trained. The paper's real contribution is the *measurement*
   (how much, anchored how), and the abstract/intro sell the phenomenon instead of the protocol.
2. **Observational study of two released checkpoints; nothing trained.** With one fine-tuning run per severity,
   the duration/domain dependence cannot be separated from idiosyncrasies of that run. The obvious experiment
   (fine-tune P briefly at 3.84 s, or on a music subset, and show the gain move) is absent; without it
   "recovers where it was trained" is a description, not a tested claim.
3. **Two durations only.** An "interaction" measured at two points is a slope; a 3-4 point duration sweep
   (5 s, 7.5 s, 15 s) would cost only generation and would show whether the gain is monotone, peaked at the
   training duration, or saturating.
4. **The domain axis is confounded.** Hip-hop/rap MusicCaps captions are 7× longer than AudioCaps captions
   and 47 % exceed CLAP's 77-token text window; the paper concedes "inseparable from content". A reviewer
   asks why a held-out domain with AudioCaps-style captions (e.g., ESC-50 / FSD50K class captions) was not
   used, which would have separated content from caption style at zero training cost.
5. **Corroboration of the headline is CLAP-family only.** The abstract says the duration dependence "is
   corroborated by a second scorer, event-level metrics and a frame-level grounding model". Only the second
   scorer (Human-CLAP, itself a CLAP fine-tune) was run at both durations; KL, PANNs, FAD and FineLAP were run
   at the native point only and corroborate the *native gain*, not its *duration dependence*. This is an
   overclaim as written. (Fix: reword; better, score the 3.84 s WAVs with KL/PANNs.)
6. **"At both severities and durations" (intro, Domain bullet) is not what was measured.** The music cells are
   severity 1 @ 3.84 s, severity 2 @ 3.84 s and @ 10.24 s; there is no severity-1 @ 10.24 s music cell.
7. **Inconsistent labelling of the severity-2 dense control.** Sec. 3.4 lists "dense control" among the
   post-hoc analyses; Table 2 calls the severity-2 control "pre-specified design completion"; Limitations
   says "severity-1 dense-control analyses are post-result". A reviewer who checks pre-registration claims
   will see three labels for one thing. (It was registered before its own scores were seen but after the
   primary result: say exactly that, once.)
8. **Sampler settings off the published recipe** (DDIM 50 vs 200, guidance 2.5 vs 3.5, single vs best-of-3).
   The gain of recovery fine-tuning plausibly depends on guidance; one spot check at the published recipe
   would remove the objection. The intro also names "a sampler configuration" as an operating-point axis and
   then never varies it: either test it or drop it from the definition.
9. **No human evaluation** for a text-to-audio paper whose primary metric is a learned proxy; acknowledged,
   but a reviewer for this track expects at least a small MOS/preference test or a clear statement of why
   CLAP suffices for a *paired* contrast.
10. **Readability.** Secs. 4.1-4.2 read as a ledger: 40+ numbers with intervals in running text although the
    tables already hold them; the "so what" of each paragraph comes last. Fig. 1 at 0.74 and Fig. 2 at 0.58
    column width have 6-7 pt lettering and overlapping labels (R_short label vs the chance-floor tick in
    panel (a)). At ICASSP, clarity is a scored criterion.
11. **Self-evaluation.** Two co-authors are authors of the evaluated checkpoints and of the arXiv preprint that
    reports their in-domain FAD. Not a flaw per se, but the paper should say so and should not lean on
    "their fine-tuned pruned model even surpasses the unpruned model" (FAD 1.57 vs 3.95) as external evidence.
12. **Anonymity / policy check.** The PDF carries author names, e-mail addresses and a GitHub Pages URL
    under the first author's handle. Verify the ICASSP 2027 review-anonymity policy before upload; if the
    track is double-blind, this is a desk-reject risk, not a comment.

## 4. Review text (as it would be returned to the authors)

**Summary.** The paper evaluates the released pruned (65 % / 83 % of U-Net parameters removed) and
recovery-fine-tuned AudioLDM-M checkpoints of Singh et al. at two clip durations (3.84 s, 10.24 s) and two
prompt domains (AudioCaps, MusicCaps hip-hop/rap), measuring the paired CLAP gain of fine-tuning over the
pruned checkpoint against a shuffled-caption chance floor, the real audio of the same prompts and the
unpruned model. It finds that the gain is several times larger at the fine-tuning duration than at 3.84 s
(closing 63 % vs 33 % of the gap to real audio at 83 % pruning), absent on the held-out music captions, and
that the short-duration deficit is a generation-length rather than a scoring-window effect (crop analysis).
The authors train nothing and frame the claims as evaluation, not mechanism.

**Strengths.** Pre-specified primary endpoint with a disjoint-prompt replication; paired design under common
random numbers with prompt-level bootstrap and a family-wise correction; chance floor and real-audio ceiling
that make CLAP cosines interpretable; matched dense controls at both severities; a crop analysis that
cleanly separates generation length from scoring window; negative results reported; complete
reproducibility details.

**Weaknesses.** (1) Limited novelty: that fine-tuning helps most at its own duration and domain is expected
(the authors cite Kumar et al. 2022 for exactly this); the contribution is the measurement protocol, which
the abstract and title under-sell. (2) Observational scope: one model family, one fine-tuning run per
severity, two durations, no training experiment that would test "recovers where it was trained" causally
(e.g., fine-tuning at 3.84 s). (3) The domain axis bundles content with caption length (7× longer captions,
47 % beyond CLAP's 77-token window); a held-out domain with AudioCaps-style captions would have separated the
two at no training cost. (4) The duration dependence is corroborated only by CLAP-family scorers; KL, PANNs,
FAD and FineLAP were run at the native point only, so the abstract's "corroborated by ... event-level
metrics and a frame-level grounding model" overstates. (5) Sampler settings differ from the published
recipe (DDIM 50 vs 200, guidance 2.5 vs 3.5) without a spot check. (6) No human evaluation.
(7) Presentation: the results sections are hard to read (dozens of numbers with intervals in prose that the
tables already contain), figures are small, and the submitted PDF contains a literal "eftab:anchors" and a
figure caption that describes only panel (a) although both panels show the dense control.

**Questions for the rebuttal.**
1. Is the recovery gain monotone in duration, or peaked at 10.24 s? Even two more durations would answer.
2. Does the domain result survive with a held-out domain whose captions match AudioCaps in length?
3. Do KL / PANNs at 3.84 s agree with CLAP on the duration dependence? These require scoring only.
4. Does the interaction hold at the published sampler recipe (guidance 3.5, DDIM 200)?
5. Please state once, precisely, the status of each analysis (pre-specified before any score / registered
   after the primary result / post-hoc); Sec. 3.4, Table 2 and the Limitations currently disagree on the
   severity-2 dense control.
6. Two co-authors released the evaluated checkpoints; please say so.

**Minor.** "at both severities and durations" (Sec. 1, Domain bullet) overstates: there is no severity-1
music cell at 10.24 s. Table 2: give n per row or a pointer. Fig. 1: full column width; the R_short label
collides with the chance-floor tick in (a). Fig. 2: full column width; move the T annotation out of the
curve. Consider a title that names the measurement rather than the expected finding, e.g. "How much does
recovery fine-tuning recover? Anchored recovery ratios across operating points for pruned text-to-audio
diffusion". Check every reference's venue/pages; several are arXiv-only.

**Overall recommendation.** 3 — borderline / weak accept. Technically sound and unusually honest, but
incremental in finding, narrow in scope, and harder to read than it needs to be.

## 5. Actions before submission, ordered by cost (Gabriel's call)

| # | Action | Cost | Status |
|---|---|---|---|
| A1 | Fix D1 (literal ref) and D2 (Fig. 1 caption) | 0 | **done in this pass** |
| A2 | Reword the abstract's corroboration sentence | minutes | **done 2026-09-03**; A8 then made it stronger, not weaker: "reproduced at both durations by a second scorer and by two event-level metrics outside the CLAP family" |
| A3 | Intro Domain bullet: "at both severities and durations" → what was actually measured | minutes | **done**: "at both severities at 3.84 s and at both durations at severity 2" |
| A4 | One consistent status label for the severity-2 dense control in Sec. 3.4 / Table 2 / Limitations | minutes | **done**: three labels defined once in Sec. 3.4 (pre-specified / registered after the primary result / post-hoc); Table 2 and Limitations now point at it |
| A5 | Add a one-clause disclosure that two co-authors released the evaluated checkpoints | minutes | **done** (Sec. 3.1, one clause) |
| A6 | Figures at full column width, R_short label moved; page budget rechecked | ~30 min | **done**: both figures now `\columnwidth` and drawn at 3.35 in natural size, so their lettering renders at its stated point size instead of 0.74x/0.58x. R_short moved off the zero line and the floor ticks; Fig. 2 "early/late" moved off the T annotation; float/caption skips tightened. Still 4 content pages + references page |
| A7 | Readability pass on Secs. 4.1-4.2 | 1-2 h | **done**: each paragraph now leads with its claim; nine intervals that Tables 1-2 already carry removed from the prose (the verifier now checks them in the table form) |
| A8 | KL / PANNs at the 3.84 s point on the frozen sev-2 WAVs, CPU | hours, 0 cr | **done, 0 cr** — `scripts/research/xsev_secondary_metrics_short.py` → `configs/research/xsev_secondary_metrics_short.json`. Both event-level metrics reproduce the interaction: J_KL = +1.56 [+1.19, +1.92], J_PANN = +0.67 [+0.49, +0.86], seam-robust. Guard: the native point recomputed against the original references reproduces the frozen artifact exactly (max |diff| 0.0) |
| A9 | Verify ICASSP 2027 anonymity policy and every reference's venue/pages | 30 min | **done**: ICASSP 2027 is **single-anonymous** ("ICASSP does not perform blind reviews, so be sure to include the author list"), so names, e-mails and the GitHub Pages URL stay. Three references were published since the preprint and were corrected: Tango → ACM MM 2023 pp. 3590-3598; Human-CLAP → APSIPA ASC 2025 pp. 131-136; FineLAP → ACL 2026 pp. 10393-10408. The rest verified as stated |
| A10 | Duration sweep / published-recipe spot check / short-duration fine-tune | GPU | **costed, not launched** — `docs/compute_budget.md` §A10 and `scripts/research/a10_gpu_cost_estimate.py`. Duration sweep (2 extra points, 3 systems) ≈ 3.33 cr point / 4.00 cap; published-recipe spot check on a 64-prompt subset ≈ 2.45 / 2.95; both together ≈ 5.8 / 7.0. The short-duration fine-tune is not estimable from measured data (no training throughput has ever been measured here); under unverified assumptions 20 k steps ≈ 10 cr and the released 10^6 steps ≈ 508 cr. Gabriel's call |

No frozen number, gate or verdict was touched. Verifier after the fixes: 108/108; PDF: 5 pages (4 content +
references); Overleaf zip rebuilt and compiles standalone from a fresh extraction.

## 6. Implementation pass (2026-09-03, 11:46 → , CPU only, 0 cr)

Gabriel asked for A2-A9 to be implemented and A10 to be costed rather than run. All of A2-A9 are done
and the status column above records exactly what changed. Notes:

* **A8 is the one that changes the paper's standing.** The reviewer's weakness (4) was that only
  CLAP-family scorers had been measured at both durations. They now are not: on the frozen severity-2
  WAVs, the KL gain falls from +2.22 [+1.93, +2.53] at 10.24 s to +0.66 [+0.42, +0.92] at 3.84 s and
  the PANNs top-10 capture gain from +0.86 [+0.70, +1.02] to +0.19 [+0.06, +0.32]; both interactions
  exclude zero and hold under the B$'$ seam convention. The fine-tuned checkpoint also *improves* with
  duration on both metrics while the pruned one gets *worse*, mirroring the CLAP duration responses.
* **Page budget.** Full-column figures plus the A2-A5/A9 additions cost about 20 lines. They were paid
  for by the A7 pass, by tightening float and caption skips, by trimming caption and Discussion text
  that repeated Table 2, and by two figure-height reductions. No result, interval or negative finding
  was dropped. The verifier still reports 108/108 and the compile has zero overfull boxes.
* **Not done, by instruction:** every GPU run (A10). Nothing was launched and no credit was spent.
