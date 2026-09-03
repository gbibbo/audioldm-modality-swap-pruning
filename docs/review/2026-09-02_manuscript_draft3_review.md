# Scientific review of MANUSCRIPT-DRAFT-3 and Draft-4 rewrite (`icassp/icassp_operating_point.tex`)

**Date:** 2026-09-02/03 (America/Montevideo). **Reviewer role:** adversarial ICASSP reviewer + rewriting
co-author, on Gabriel's request ("find the scientific and methodological gaps, the cheap real improvements,
and a better, more sellable ICASSP-grade write-up"). **Scope:** Draft 3 as built (4 pages, 59/59 numbers
verified), every frozen artifact it cites, the persisted per-prompt scorer outputs, the generation
directories still on disk, and the two prior reviews (Draft 1, Draft 2). **Compute:** CPU only, 0 GPU,
0 cr. **Freeze:** no frozen artifact, gate, verdict or raw score was modified; the new analyses below read
committed raw scores (and, for the dense control, re-score 80 *existing* WAVs), are labelled post-hoc in
the paper, and change no frozen conclusion.

Verdict up front: **Draft 3 is scientifically sound and provenance-perfect, but it leaves its strongest
argument unmade (a matched dense duration control that was computable from data already on disk), carries
three real reporting gaps a competent reviewer would catch (multiplicity, the unstated FineLAP n, two
different severity-1 short-duration numbers), and tells its story in the order the project happened rather
than in the order a reader needs.** All of it is fixed in Draft 4 at zero credit cost.

---

## A. Scientific / methodological findings (new relative to the Draft-1 and Draft-2 reviews)

### A1. The dense duration control was missing — and computable for free [FIXED, new analysis]
Draft 3 wrote "every system scores higher at 10.24 s (dense: 0.204→0.352)". Those two dense means come from
**different prompt sets and different scorer batches** (V1.1: 96 prompts × 2 replicates at 3.84 s; Arm-D
dense union: 80 prompts at 10.24 s). The paper's central objection — "is $J$ just everything gaining from the
longer clip?" — can only be answered with the dense model measured at both durations on the *same* prompts
under the *same* scoring convention. The Arm-D 80 prompts are a strict subset of the V1.1 96
(verified: 80/80 ytids), the V1.1 dense r0 3.84 s WAVs are still on disk
(`/teamspace/jobs/reversal-v11-gen-1/…/reversal_v1_1_gen/dense_noadapter_p*_r0.wav`, 80/80 present), and
the Arm-D short groups were themselves produced by re-scoring those V1.1 WAVs as one 80-item call. So one
80-item CLAP call (CPU, ~2 min) completes a paired 3 systems × 2 durations design at severity 1.

`scripts/research/draft4_dense_duration_control.py` → `configs/research/draft4_dense_duration_control_result.json`
(seed namespace `DRAFT4-DENSE-DURATION-CONTROL|BOOTSTRAP|2026-09-02`, PCG64 seed derived from it, B=10000;
consistency guards reproduce the frozen Arm-D R_short/R_native/J and the frozen DENSE_CONTROL to 0.0):

| Quantity (n=80, paired) | Point | 95% CI |
|---|---:|---|
| dense 3.84 s → 10.24 s | 0.202 → 0.352 | |
| duration response s(dense) | **+0.150** | [+0.100, +0.198] |
| s(P), severity 1 | +0.149 | [+0.119, +0.178] |
| s(P+FT), severity 1 | +0.193 | [+0.152, +0.232] |
| s(P) − s(dense) | −0.001 | [−0.056, +0.055] |
| s(P+FT) − s(dense) | +0.043 | [−0.020, +0.109] |
| dense gap closed by FT at 3.84 s (R/G) | 8 % | [−30 %, +36 %] |
| dense gap closed by FT at 10.24 s | 52 % | [+11 %, +103 %] |

Reading: at 65 % pruning the pruned checkpoint responds to duration **exactly like the dense model**; the
fine-tuned checkpoint responds more, but its excess over dense is not resolved at n=80. At 83 % pruning the
frozen sensitivity file already gives s(P)=+0.040 [+0.029,+0.052] vs s(P+FT)=+0.200 [+0.173,+0.226]: the
pruned checkpoint has lost most of the dense-magnitude duration response, and fine-tuning restores one of
dense magnitude (a cross-set, descriptive comparison — dense was never generated on the severity-2 192
set). This is the sharpest, most reference-anchored statement in the paper and it was sitting in the data.

Side result: the **batch-composition diagnostic** (identical WAVs scored inside the 192-item V1.1 call vs
the 80-item call) shows means shift by ≤0.002 (per-clip up to 0.10 for the one clip that receives the
fused-CLAP `is_longer` draw). This is now stated in Sec. 3.3 and closes the "why not one big batch?" question.

### A2. No multiplicity statement [FIXED, new analysis]
Draft 3 marks nine contrasts † (CI excludes 0) and one ‡ (gate) with no family-wise consideration.
`scripts/research/draft4_robustness.py` (R1) computes two-sided bootstrap achieved significance levels for all
13 reported contrasts and applies Holm (whole family and per severity). Result: **every severity-2 †
survives (p < 10⁻⁴)**; at severity 1, **R_nat (p = 0.016) and J (p = 0.052) do not survive** — only the
music contrast and the domain contrast do. Draft 4 states this in Sec. 4.1, the Table 2 caption and the
Limitations. This is a more honest severity-1 statement than Draft 3's bare †.

### A3. Rank-scale sensitivity of J was listed as "not done" [FIXED, new analysis]
R2 of the same script: median per-prompt interaction **+0.051 [+0.012, +0.077]** (sev-1) and
**+0.172 [+0.153, +0.214]** (sev-2); Wilcoxon signed-rank p = 0.020 and p < 10⁻¹⁷; J on pooled-rank-transformed
scores +0.063 [−0.015, +0.143] (sev-1, not resolved) and +0.199 [+0.153, +0.244] (sev-2). Together with the
win-rate interaction (Draft 3) and Human-CLAP, the severity-1 effect is *directionally consistent on every
scale but resolved only on some*; Draft 4 says exactly that and lets the severity-2 replication carry the claim.

### A4. The domain contrast bundles content with caption style — unexamined [FIXED, new analysis]
MusicCaps captions are ~7× longer than AudioCaps captions (median 56.5 vs 8 words; CLAP's text tower
truncates long captions). R3: within AudioCaps the per-prompt gain is **uncorrelated with caption length**
(Spearman ρ = +0.04 [−0.12, +0.18] at sev-2 native; long-quartile gain 0.262 vs short-quartile 0.228), so
caption length does not explain the music null; the bundling is now stated as a limitation of the domain axis.

### A5. The "music floor" defence was weaker than the data allow [FIXED, no new inference]
Draft 3 relied on "full spread" to argue the severity-2 music null is not a floor. A stronger, free
observation was unused: at 10.24 s the **pruned checkpoint scores higher on music (0.089) than in-domain
(0.055)**; only the fine-tuned checkpoint's gain is domain-specific (music 0.094 vs in-domain 0.299). Now in
the Intro bullet and Sec. 4.2.

### A6. Two different numbers for "severity-1 AudioCaps at 3.84 s" with no bridge [FIXED, wording]
Table 2 gives R_short = +0.008 (n=80 duration subset), Sec. 4.2 gives R_AC = −0.002 (n=96 pre-specified
set). Both are right; Draft 4's caption and Sec. 3.2 say which is which.

### A7. FineLAP sample sizes were not reported [FIXED]
110 (sev-2) and 49 (sev-1) eligible prompts, selected outcome-blind — now stated.

### A8. "Pre-registered" is an over-claim for git-committed protocols [FIXED, terminology]
There is no external registry. Draft 4 uses **pre-specified** and defines it once ("estimand, gate and
prompt set committed to the version-controlled repository before any score was seen"). Reviewers punish
"pre-registered" without an OSF/AsPredicted link; they accept a defined "pre-specified".

### A9. "Absent on held-out music" understated severity 1 [FIXED]
At severity 1 the music contrast is *negative* (−0.094, CI excludes 0), not absent. Abstract, intro bullet
and conclusion now say "absent or negative".

### A10. Process narrative in Results [FIXED]
"…which reframed the work from mechanism to evaluation" is project history, not science. Rewritten as a
plain negative result: the loss on music is real, the *trade* is not.

### A11. Things verified correct and kept
Both music batteries are hip-hop/rap-filtered MusicCaps captions under identical eligibility rules, disjoint
(0/64 overlap) — Sec. 3.2 wording is accurate. Severity-2 music at 10.24 s uses one replicate (now stated).
Every Draft-3 number reproduces (verifier 59/59 before; 82/82 after the rewrite).

---

## B. Cheap improvements — what was run (all CPU, 0 cr, this pass)

| # | Item | Artifact | Label in paper |
|---|---|---|---|
| B1 | Matched dense duration control (A1) — 80 existing WAVs re-scored, one 80-item frozen-convention call | `configs/research/draft4_dense_duration_control_result.json` | post-hoc control |
| B2 | Holm over all 13 reported contrasts (A2) | `configs/research/draft4_robustness_result.json` (R1) | reported in text/caption |
| B3 | Median/Wilcoxon/pooled-rank J (A3) | same (R2) | post-hoc sensitivity |
| B4 | Caption-length check + pruned music-vs-in-domain observation (A4, A5) | same (R3) | descriptive |
| B5 | Draft-4 rewrite, Fig. 1 with the matched dense line + duration responses, verifier 82/82 | `icassp/` | — |

Not run (and still not recommended): DDIM-200, third severity, more prompts, dense-FT reconstruction,
listening study. One remaining GPU-cheap option if a reviewer insists: dense at both durations on the
severity-2 192 set (≈192 native WAVs ≈ 0.7 cr; the 3.84 s half could be CPU) — it would turn the cross-set
"dense magnitude" comparison at severity 2 into a paired one. Not needed for the claims as written.

---

## C. The rewrite (Draft 4): what changed and why it sells better

* **Title:** *Recovery Fine-Tuning Recovers Where It Was Trained: Duration- and Domain-Dependent Gains in
  Pruned Text-to-Audio Diffusion.* Declarative, states the finding, keeps the searchable keywords.
  Alternatives kept for Gabriel: Draft-3's question title; "Post-Pruning Recovery in AudioLDM Restores
  Native-Duration Alignment In-Domain Only".
* **One concept carries the paper — the duration response s(·)** = a system's mean score change from
  3.84 s to 10.24 s. J = s(P+FT) − s(P) is then an identity, the dense model gets its own s, and the
  severity-2 story becomes one sentence: s(P)=+0.04, s(P+FT)=+0.20, dense ≈ +0.15.
* **Findings-first bullets**, each ending with the one fact a reviewer will remember (dense control; not
  a floor; crop carries the gain).
* **Honesty made structural, not apologetic:** the analysis-plan table stays; multiplicity, rank
  sensitivity and the sev-1 borderline are stated once, in Results, with numbers; Limitations lists what
  is cross-set or post-result.
* **Abstract** keeps the vivid absolute numbers and adds the dense-control sentence; 160 words.
* **Terminology fixed:** recovery fine-tuning · P / P+FT · recovery gain R · duration response s ·
  native/short · in-domain/held-out music · pre-specified (defined) · post-hoc.
* **Layout:** 4 content pages + references on page 5 (ICASSP permits a 5th page containing only
  references — Gabriel to confirm against the ICASSP 2027 CFP; if the rule is 4 pages total, the
  bibliography can be compacted with `\bibliographystyle{IEEEbib}` + abbreviations, or Sec. 2 cut by ~8 lines).

---

## D. Remaining reviewer risks (unchanged, disclosed in the paper)
1. No matched dense fine-tuned control → evaluation, not mechanism.
2. CLAP-family primary; no human ratings.
3. Off-recipe sampler (DDIM 50 / guidance 2.5 / single generation).
4. Two durations only; dense control at severity 1 only.
5. Severity-1 effects are borderline and do not survive Holm — the paper now says so itself.
