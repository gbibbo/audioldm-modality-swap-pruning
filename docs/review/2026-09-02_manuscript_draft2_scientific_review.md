# Scientific + editorial review of MANUSCRIPT-DRAFT-2 (`icassp/icassp_operating_point.tex`)

**Date:** 2026-09-02 (America/Montevideo). **Reviewer role:** adversarial ICASSP reviewer + co-author
rewrite proposal. **Scope:** the built Draft 2 (4 pages), every frozen protocol/result it cites, the raw
per-prompt score artifacts, and the prior Draft-1 review. **Freeze respected:** no frozen artifact was
modified, no GPU was used, no number in the manuscript was changed. Section C reports *scratch*
re-analyses of committed raw scores; they are diagnostic for this review only and enter the paper only
after an explicit GO (they are new derived numbers under the freeze rule).

Verdict up front: **the science is sound and the provenance is exemplary; the paper under-sells it and
carries a handful of real methodological soft spots that are cheap to close.** Nothing found here changes
a frozen verdict. Everything below is classified **[P]** presentation-only, **[R]** CPU re-analysis of
committed raw data (needs GO), or **[G]** small GPU arm (needs GO + credits).

---

## A. Verified scientific / methodological findings (ordered by reviewer risk)

### A1. The headline interaction $J$ is scale-dependent, and the paper never checks a scale-free version [R]
$J=R_{\mathrm{nat}}-R_{\mathrm{short}}$ is a difference-in-differences on the raw CLAP-cosine scale. Every
system scores higher at 10.24 s (dense 0.204→0.352; sev-1 pruned 0.104→0.253), so a reviewer will ask
whether $J>0$ is just "the same relative advantage on a larger scale". The paper does not address this.
Verified from committed raw per-prompt cosines (scratch, not in the paper):

| Severity | win-rate P+FT>P short → native | Δ win-rate [95% CI]* | paired $d$ short → native |
|---|---|---|---|
| 1 (n=80, `op_duration…raw_cosines`) | 0.438 → 0.637 | **+0.200 [+0.06, +0.34]** | 0.05 → 0.27 |
| 2 (n=192, `_score_tmp/xsev_sev2_groups_out.json`) | 0.724 → 0.870 | **+0.146 [+0.08, +0.22]** | 0.62 → 1.18 |

*Prompt bootstrap, B=5000, scratch seed — sensitivity only. The scale-free interaction is resolved at
**both** severities (including severity 1, where the raw-scale $J$ is borderline). This is a free
hardening of the weakest primary number, provided it is labelled *post-hoc sensitivity analysis*. Also
note the duration *slopes*: sev-2 pruned +0.040 vs fine-tuned +0.200 — the pruned (1,2,1,1) checkpoint
barely benefits from the longer generation at all. That is a clearer way to state the finding than $J$.

### A2. $K$ bundles domain and duration; the clean matched-duration domain contrast exists and is unused [R]/[P]
The paper reports $K=R_{\mathrm{nat}}-R_{\mathrm{music}}$ and then has to explain that it confounds two
factors. The clean contrast is the **matched-3.84 s domain gap** $R_{\mathrm{short}}-R_{\mathrm{music}}$:
* severity 1: already **frozen** as $I=R_{AC}-R_{\mathrm{music}}=+0.092$ [+0.054, +0.131]
  (`reversal_v1_1_result.json`, n=96/64) — not in the paper;
* severity 2: point +0.076, scratch independent bootstrap CI [+0.046, +0.104] (needs GO).
Recommendation: **drop $K$ from the paper**, report the matched-duration domain contrast at both
severities. The story then has two orthogonal axes, each isolated: duration (same prompts) and domain
(same duration).

### A3. "Generation length" vs "scoring window" is not separated [R] — the most informative zero-cost analysis
The short point differs from the native point in two ways at once: the model *generates* 3.84 s, and the
scorer *sees* 3.84 s (repeat-padded to 10 s by the LAION fused-CLAP feature extractor, `padding=repeatpad`;
the paper does not say this). FineLAP already hints that the first 3.84 s of a native generation carry the
full fine-tuning gain ($D_{\mathrm{early},2}=+0.275$ vs $D_{\mathrm{late},2}=+0.273$). A direct CLAP test
is free: **score the first 3.84 s crop of the existing native WAVs** with the identical scorer convention
(sev-1: 80×2 crops; sev-2: 192×2 (+192 B′) crops; all WAVs present locally under
`/teamspace/jobs/reversal-{armd,xsev}-gen-1/…`) and compare the crop contrast $R_{\mathrm{crop}}$ with
$R_{\mathrm{short}}$ (same prompts, same scorer). If $R_{\mathrm{crop}}\approx R_{\mathrm{nat}}\gg
R_{\mathrm{short}}$, the interaction is a **generation-length** effect (the fine-tuned model conditions on
latent length), not a scoring artefact — a much stronger, mechanism-adjacent statement that is still an
evaluation claim. If $R_{\mathrm{crop}}\approx R_{\mathrm{short}}$, the effect is a content-window effect
and the paper must say so. Either outcome is publishable; not knowing is the weak spot. CPU cost:
~550 clips at ~1 s/clip. Must be labelled post-hoc diagnostic (the frozen FineLAP protocol explicitly
forbids *assuming* crop ≡ short OP; testing it is fine).

### A4. Music was generated only at 3.84 s, so the design is not a factorial [G]
This is why $K$ exists. One small arm completes the 2×2 (domain × duration) at severity 2:
music-64 at 10.24 s for P and P+FT (A′; B′ optional). Pre-registrable hypothesis with two clean
outcomes: (a) $R_{\mathrm{music,nat}}\approx R_{\mathrm{music,short}}\approx 0$ → the native gain is
**domain-specific** (fine-tuning specialises to AudioCaps content *and* duration); (b)
$R_{\mathrm{music,nat}}\gg 0$ → the duration gain is **generic** (length conditioning restored
regardless of domain). Cost at the measured native rate (~0.0036 cr/WAV): 64×1×2 = 128 WAVs ≈ 0.5–0.6 cr;
64×3×2 = 384 WAVs ≈ 1.4–1.5 cr (matches the frozen 3-seed music convention; preferred if funds allow).
This is the only GPU spend I would recommend; DDIM-200 / third severity / more prompts remain
not-worth-it (agree with `xsev_postresult_adversarial_audit.md` §11).

### A5. Pruned baselines are described inaccurately [P — correctness]
§3.1 says "we use the publicly released L1 structurally pruned checkpoints". Per the frozen provenance:
the released p1 checkpoint's EMA is broken, so the pruned systems are **the released L1 channel selection
applied to the dense EMA weights**; at sev-1 the operator reproduces the released raw weights
bit-exactly (690/690), at sev-2 the released dp1 checkpoint encodes a different seam convention in three
decoder tensors (hence A′/B′). The current wording invites a "which weights exactly?" question that the
provenance answers perfectly — say it in one sentence.

### A6. Frozen secondary evidence that hardens severity 1 is missing from the paper [P]
Human-CLAP at severity 1 gives $J_{HC}=+0.075$ [+0.012, +0.137] (CI excludes 0;
`op_duration_discriminator_1_result.json:SECONDARY_humanclap`). The paper reports HC only for
severity 2. Adding one clause turns "borderline at severity 1" into "borderline on the primary scorer,
resolved on the second scorer and on the win-rate scale" (with A1).

### A7. Singh et al.'s own numbers support the paper's reading and are not used [P]
Singh et al. report the fine-tuned pruned (1,2,3,1) model **beating their unpruned model in-domain**
(FAD 1.57 vs 3.95; KL 1.678 vs 2.16), and M-Full itself was AudioCaps-fine-tuned 0.25 M steps by the
AudioLDM authors. Both facts are in `docs/recovery_metric_audit_1_literature.md` and neither is in the
paper. They are the natural Discussion anchor: the recovery stage behaves partly as *further in-domain,
native-duration specialisation*, which is exactly what an operating-point-dependent advantage looks
like. Also citable from their abstract: 83 % parameters / 39 % MACs removed — replaces the need for an
efficiency table (A12).

### A8. Cross-duration pairing is weaker than the text implies [P]
"Identical generation noise for the compared systems" holds *within* an operating point. Across
durations the latent shapes differ ((1,8,96,16) vs (1,8,256,16)), so $J$ is prompt-paired but not
noise-paired. One clause; otherwise a reviewer who reads the protocol will flag it.

### A9. CLAP's handling of the two durations is undocumented [P]
Fused LAION-CLAP repeat-pads 3.84 s audio to 10 s and the 10.24 s clips are centre-cropped to 10 s
(`gate0_clap_scorer.py`). Both systems get identical treatment, so contrasts are valid, but the paper
should state it (it also motivates A3).

### A10. Operating-point deviations from the published recipe are under-listed [P]
Limitations mention DDIM 50 vs 200 only. The framework default also uses guidance 3.5 and
best-of-3-by-CLAP selection; we use 2.5 and single generation. Best-of-3-by-CLAP is itself an
alignment-maximising selection — worth one sentence, because it explains why absolute FAD/KL here are
not comparable to the published ones.

### A11. Evidence hierarchy is right but the prose spends ~30 % of the paper on it [P]
"Prospectively frozen primary / prospectively specified follow-up / pre-specified secondary implemented
after / post-result diagnostic" is correct and rare at ICASSP, but it reads as defensive. Compress into
one 5-row table ("Analysis · Registered before · Role") and use "pre-registered" once in the abstract.

### A12. Table 1 (parameter counts) costs 9 lines for three numbers [P]
Put 416 M → 146 M (−65 %) → 71 M (−83 %) in the text, cite Singh's MAC reduction, delete the table; the
recovered column space moves Table 2 (the table carrying every primary number) off page 4.

### A13. Layout defects in the built PDF [P]
* **Table 2 is on page 4**, after every Results paragraph that cites it.
* Fig. 2(a): row labels collide with the "severity 1/2" group headers and with the
  "favours pruned / favours post-FT" annotations; (b): "early | late boundary" text collides with the
  legend.
* **The committed PDF is stale relative to the `.tex`**: PDF reference [11] prints
  "T. Yamamoto, S. Takamichi, H. Saruwatari" while the source says Takano et al. (verified correct
  against arXiv 2506.23553: Takano, Okamoto, Kanamori, Saito, Nagase, Saruwatari). Rebuild before any
  delivery; `tectonic` is not currently on PATH in this Studio.
* Citation venues to fix: Singh et al. is "Submitted to DCASE 2026 Workshop" (cite as arXiv:2607.13330,
  not "in Proc. DCASE Workshop"); Human-CLAP is "Submitted to APSIPA ASC 2025" (cite as arXiv);
  FineLAP "Proc. ACL 2026" is not confirmed by the arXiv record (cite as arXiv:2604.01155 unless
  verified).

### A14. Framing/selling defects [P]
* Contribution bullet 1 sells a "paired common-noise evaluation framework" — common random numbers is
  standard practice; reviewers will discount it. Contributions should be **findings**, not apparatus.
* The abstract leads with method and jargon; the vivid numbers (0.055 → 0.299 at 10.24 s vs
  0.015 → 0.100 at 3.84 s; ≈0 on music) never appear in it.
* Results §4.3 narrates history ("the severity-2 test did not come first"). Chronology belongs in one
  sentence plus the evidence table, not a subsection.
* The Discussion checklist has seven items; ICASSP readers keep three or four.
* The FineLAP result is presented as a failed hypothesis ("not concentrated late"). Presented
  positively it is a strong point: *the fine-tuning gain is present in the first 3.84 s of a native
  generation as much as in the rest* — which, with A3, pins the effect on generation length.
* Heterogeneity numbers (64 % vs 44 %) are severity-1 only; say so, or give both severities (A1).
* Related work has no TTA-evaluation or duration/length-conditioning references, and only two diffusion
  pruning papers; ICASSP reviewers expect ~15–18 references.

### A15. Things that are *right* and must survive the rewrite
Pre-registration with frozen gates; bit-exact reproduction audit; seam sensitivity; disjoint replication
set; prompt-level bootstrap with replicates averaged first; honest negatives (reversal, late
allocation, dense not restored); explicit no-human-eval and no-dense-FT-control limitations; every
number traceable. Keep all of it — but move the machinery to a compact table and let the findings lead.

---

## B. Cheap improvements — recommended package (priority order)

| # | Item | Type | Cost | What it buys | Label in paper |
|---|---|---|---|---|---|
| B1 | Scale-free interaction (win-rate Δ + paired $d$) at both severities (A1) | [R] | 0 cr, <1 h | resolves sev-1 on a scale-free statistic; answers the scale objection | post-hoc sensitivity |
| B2 | Matched-duration domain contrast at sev-2 (A2); drop $K$ | [R] | 0 cr, <1 h | clean domain axis at both severities | pre-specified quantity (sev-1 frozen); sev-2 derived |
| B3 | Crop analysis: CLAP on first 3.84 s of native WAVs (A3) | [R] | 0 cr, ~1–2 h CPU | separates generation-length from scoring-window; strongest interpretive gain | post-hoc diagnostic |
| B4 | Add frozen HC sev-1 $J$ (A6) + Singh in-domain facts (A7) | [P] | 0 | hardens sev-1; anchors Discussion | — |
| B5 | Rewrite (Section D), rebuild PDF, fix figures/citations (A5, A8–A14) | [P] | 0 | the paper reads like its evidence | — |
| B6 | Music-64 at 10.24 s, severity 2, P and P+FT (A4) | [G] | ≈0.6 cr (1 seed) / ≈1.5 cr (3 seeds) | completes the 2×2 factorial; removes the only bundled contrast | pre-registered follow-up |
| B7 | Public dense text-FT reference on Arm-D 80 (`textft_reference_audit.md`, B-GO-CANDIDATE) | [G] | ≈0.5–0.7 cr | a dense+FT duration anchor, powered only for large $J$ | optional; lowest priority |

Not recommended (unchanged from prior audits): DDIM-200, third severity, more AudioCaps prompts,
approximate dense-FT reconstruction, relaunching the listening study (ethics block).

Sequencing that respects the freeze: (1) Gabriel GO on B1–B3 as *post-hoc/sensitivity* analyses →
(2) if B6 is funded, freeze a one-page protocol **before** generation (hypotheses (a)/(b) in A4, A′
primary, B′ sensitivity, 3-seed averaging, PCG64 seed namespace, gate: none needed — both outcomes
reportable) → (3) Draft 3 with all of the above → (4) rebuild, adversarial pass, Overleaf zip.
Deadline 2026-09-16 leaves room for all of it; B1–B5 alone are one working day.

---

## C. Scratch numbers produced for this review (NOT paper numbers until GO)

Source: `configs/research/op_duration_discriminator_1_result.json` (raw_cosines, n=80) and
`artifacts/icassp_gate0/_score_tmp/xsev_sev2_groups_out.json` (n=192; music 64×3 averaged per prompt).
Scratch bootstrap B=5000, `numpy default_rng(1)` — to be re-run under a declared seed namespace on GO.

* sev-1: win-rate 0.438 (short) → 0.637 (native); Δ +0.200 [+0.062, +0.338]; paired $d$ 0.053 → 0.267;
  duration slopes P +0.149, P+FT +0.193.
* sev-2: win-rate 0.724 → 0.870; Δ +0.146 [+0.078, +0.219]; paired $d$ 0.621 → 1.176;
  duration slopes P +0.040, P+FT +0.200; music win-rate 0.484.
* sev-2 matched-3.84 s domain gap $R_{\mathrm{short}}-R_{\mathrm{music}}$ = +0.076 [+0.046, +0.104].

---

## D. Rewrite proposal (Draft 3) — structure, voice, terminology

### D1. Title (choose one; all declarative, ICASSP register)
1. **How Much Does Recovery Fine-Tuning Recover? Operating-Point-Dependent Gains in Pruned Text-to-Audio Diffusion**
2. Recovery Fine-Tuning After Pruning Is Not a Single Number: Duration- and Domain-Dependent Gains in AudioLDM
3. Post-Pruning Recovery in AudioLDM Depends on Clip Duration and Prompt Domain

### D2. Terminology (fixed for the whole paper)
| Use | Instead of |
|---|---|
| *recovery fine-tuning* (the stage, Singh et al.'s term) | "post-pruning fine-tuning stage", "recovery" in quotes |
| *pruned checkpoint* (P) / *fine-tuned checkpoint* (P+FT) | "post-fine-tuning checkpoint", "recovered model" |
| *recovery gain* $R$ = CLAP(P+FT) − CLAP(P) | "advantage", "contrast", "R(context)" interchangeably |
| *native duration* (10.24 s, the fine-tuning duration) / *short duration* (3.84 s) | "operating point" everywhere (keep the term for the concept, not for each factor) |
| *in-domain* (AudioCaps) / *held-out music* | "context" |
| *pre-registered* (once), *sensitivity analysis*, *post-hoc diagnostic* | the four-class taxonomy in prose |
| *automatic evaluation* | "in our (non-human) evaluation" |

### D3. Abstract (draft, ~140 words)
> Structured pruning of text-to-audio diffusion models is followed by recovery fine-tuning, and its
> benefit is usually reported as one score at one inference setting. Using the released pruned and
> fine-tuned AudioLDM-M checkpoints at two pruning severities (65 % and 83 % of U-Net parameters
> removed), we measure the recovery gain in text–audio alignment under a paired, common-noise
> protocol across clip duration and prompt domain. The gain is large at the fine-tuning duration of
> 10.24 s (CLAP 0.055 → 0.299 at 83 % pruning), several-fold smaller at 3.84 s (0.015 → 0.100), and
> absent on held-out music; the duration dependence is pre-registered and replicated on a disjoint
> prompt set, robust to the pruning-seam convention, and corroborated by a second scorer, event-level
> metrics, and an independent frame-level grounding model, which shows the native-duration gain spread
> over the whole clip. Recovery should be reported across inference operating points; lacking a
> matched dense fine-tuned control and human ratings, our claims concern evaluation, not mechanism.

### D4. Section plan (4 pages, 9 pt, official template)
1. **Introduction** (½ col). Pruning + recovery is standard; recovery is scored once. Question: does
   the recovery gain depend on the inference operating point? Answer in one sentence with the two
   numbers. Contributions as three *findings*: (i) duration dependence, pre-registered and replicated;
   (ii) domain dependence at matched duration; (iii) the native gain is temporally broad and
   prompt-heterogeneous; plus a reporting recommendation. One sentence: the original reversal
   hypothesis failed pre-registration and is reported.
2. **Background** (⅓ col). Diffusion pruning + recovery (Fang, BK-SDM, TinyFusion, Singh); TTA
   evaluation and its metric fragility (CLAP, FAD reliability, Human-CLAP, FineLAP); duration as a
   conditioning variable in latent audio diffusion (AudioLDM latent length; Stable Audio timing
   conditioning) — the reason duration is a *mechanistically motivated* factor, not a knob.
3. **Experimental design** (1 col). 3.1 Systems (dense, P, P+FT at two severities; exact weight
   provenance in one sentence, A5; parameter counts in text). 3.2 Operating points and batteries
   (2 durations × 2 domains, which cells exist; DDIM 50/η 0/guidance 2.5/single/EMA; deviation from
   the published recipe in one sentence). 3.3 Paired protocol and scoring (CRN within operating
   point, not across; CLAP repeat-pad/crop convention; prompt-level bootstrap; replicates averaged).
   3.4 Quantities and analysis plan — **one table**: quantity · definition · registered before
   scoring? · role (primary / sensitivity / secondary / post-hoc). This replaces §3.5 prose.
4. **Results** (1⅓ col + Fig. 1 + Table 1). 4.1 *The recovery gain grows with duration* — Fig. 1 +
   Table 1 with absolute levels; sev-2 primary, sev-1 borderline on raw scale, resolved on win-rate
   and Human-CLAP (B1, A6). Slopes sentence (A1). 4.2 *The gain is domain-specific at matched
   duration* — $R_{\mathrm{short}}$ vs $R_{\mathrm{music}}$ at both severities (B2); sev-1 music
   negative does not replicate (kept as a negative). 4.3 *Where in the clip the gain lives* — FineLAP
   (early ≈ late; Fig. 2b) + crop analysis (B3) → generation-length reading. 4.4 *Corroboration* —
   HC, KL, PANN count (paired CIs), FAD/FD descriptive; one paragraph. 4.5 *Pre-registered
   negatives* — reversal hypothesis; late allocation; no restoration to dense; one paragraph.
5. **Discussion** (½ col). Recovery gain tracks the fine-tuning operating point (duration and domain);
   consistent with Singh et al.'s fine-tuned model surpassing the unpruned model in-domain (A7);
   heterogeneity across prompts (both severities). Reporting recommendation reduced to four items:
   native + one off-native point; absolute levels of P and P+FT; paired contrasts with prompt-level
   uncertainty; an explicit interaction. Limitations folded in (no matched dense-FT control → no
   mechanism; CLAP-family primary; no human ratings; off-recipe sampler settings; two durations only).
6. **Conclusion** (4 lines) + companion-page footnote.

### D5. Figures and tables
* **Fig. 1** (keep, page 2): duration × system, both severities, CI whiskers, dense reference in (a).
  Add the duration-slope numbers to the annotation instead of $R$ values.
* **Table 1** (page 2, merge current Tables 1+2): rows = severity × {short, native, music};
  columns = P, P+FT, $R$ [95 % CI], win-rate; footer rows = $J$ and the matched domain gap.
  Delete the parameter table.
* **Fig. 2** (page 3): (a) FineLAP Δgrounding vs time, both severities — fix label collisions;
  (b) if B3 is approved: $R_{\mathrm{short}}$ vs $R_{\mathrm{crop}}$ vs $R_{\mathrm{nat}}$ per
  severity (three points with CIs). The forest plot is then redundant with Table 1 and can go.
* Evidence-hierarchy **Table 2** (⅓ col, §3.4) replaces the taxonomy prose.

### D6. Reference list to reach ICASSP norms (verify each before use)
Keep: Liu 2023 (AudioLDM), Singh 2026 (arXiv), Fang 2023, Kim 2024 (BK-SDM), Wu 2023 (CLAP),
Kong 2020 (PANNs), Kilgour 2019 (FAD), Kim 2019 (AudioCaps), Agostinelli 2023 (MusicLM/MusicCaps),
Song 2021 (DDIM), Takano 2025 (Human-CLAP, arXiv), Li 2026 (FineLAP, arXiv).
Add (all real, verify bibliographic details): TinyFusion (Fang et al., CVPR 2025, arXiv 2412.01199);
Liu et al. AudioLDM 2 (IEEE/ACM TASLP 2024); Evans et al., Stable Audio Open (arXiv 2407.14358,
timing conditioning); Gui et al., "Adapting Frechet Audio Distance for generative music evaluation"
(ICASSP 2024); Chung et al., KAD (arXiv 2502.15602, FAD encoder/sample-size fragility);
Kumar et al., "Fine-tuning can distort pretrained features…" (ICLR 2022) for the in-domain
specialisation logic; Ghosal et al., Tango (2023) or Huang et al., Make-An-Audio (ICML 2023) as TTA
context. Target 16–18 references.

---

## E. Decisions requested from Gabriel
1. GO / NO-GO for B1–B3 as labelled post-hoc/sensitivity re-analyses of committed raw scores (0 cr).
2. Current Lightning balance, and GO / NO-GO for B6 (music at 10.24 s, sev-2): 1 seed (~0.6 cr) or
   3 seeds (~1.5 cr). B7 only if funds remain after B6.
3. Title choice (D1) and permission to adopt the D2 terminology ("recovery fine-tuning", P / P+FT) —
   this supersedes the frozen-story rule against "recovered", by replacing it with a neutral label
   rather than reintroducing the restoration connotation.
4. Rebuild toolchain: install `tectonic` (or compile on Overleaf) — the committed PDF is stale (A13).
