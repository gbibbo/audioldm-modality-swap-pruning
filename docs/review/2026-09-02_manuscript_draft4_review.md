# Scientific review of MANUSCRIPT-DRAFT-4 and Draft-5 rewrite (`icassp/icassp_operating_point.tex`)

**Date:** 2026-09-02 (22:18 →, America/Montevideo). **Reviewer role:** adversarial ICASSP area-chair-level
reviewer + rewriting co-author, on Gabriel's request ("the drafts were written with weaker models; find the
real scientific and methodological gaps, the cheap real improvements, and a better, more sellable,
ICASSP-grade write-up"). **Scope:** Draft 4 as built (4 content pages + references page, 82/82 numbers
verified), every frozen artifact it cites, the persisted per-prompt scorer outputs and embeddings, the
generation directories still on disk, the AudioLDM conditioning code, the ICASSP 2027 CFP/paper kit, and the
three prior reviews (Draft 1, 2, 3). **Compute:** CPU only, 0 GPU, 0 cr. **Freeze:** no frozen artifact,
gate, verdict or raw score was modified; the new analysis reads committed raw scores and re-embeds existing
WAVs under the frozen convention with a bit-exact guard; it is labelled post-hoc in the paper and changes no
frozen conclusion.

Verdict up front: **Draft 4 is correct, provenance-perfect and already unusually honest, but it still reads
every absolute CLAP number against an implicit zero, attributes the whole "duration response" to the
systems although half of it could be the scorer, states its headline at severity 2 on a cross-set dense
reference, and buries its most human-readable quantity (the fraction of the gap that fine-tuning closes) in
one sentence.** All four are fixed at zero credit with data already on disk, and the paper is rewritten
around them. One structural weakness remains and is the only item worth paying for: a paired dense control
on the severity-2 prompt set (protocol frozen and launch script ready; ≈1.2 cr on a T4 or ≈18 h of CPU;
**not launched** — Gabriel's call).

---

## A. Scientific / methodological findings (new relative to the Draft-1/2/3 reviews)

### A1. Absolute CLAP levels had no chance floor [FIXED, new analysis, 0 cr]
The paper's most striking numbers — P at 83 % pruning scores 0.015 at 3.84 s and 0.055 at 10.24 s; music
cells score 0.005–0.094 — are read against zero, but the CLAP cosine of an *unrelated* audio–caption pair is
not zero and depends on the battery and on the audio. Without a floor, "barely responds", "not a floor" and
"absent" are all under-defined. Fix: a **shuffled-caption chance floor** per cell, computed from the *same*
embeddings the frozen scores came from. Text embeddings are batch-invariant and the frozen convention (one
seed-once fixed-order call per cell) reproduces the audio embeddings exactly, so for each frozen cell we
rebuild the text × audio cosine matrix, require its diagonal to equal the frozen per-item cosines
(**guard: max |diff| 2.4·10⁻⁷ across 28 groups, 3 296 clips**) and take the off-diagonal mean as the floor
(replicates of the same prompt excluded). `scripts/research/draft5_floor_ceiling.py` →
`configs/research/draft5_floor_ceiling_result.json` (seed namespace `DRAFT5-FLOOR-CEILING|BOOTSTRAP|2026-09-02`).

**Result (28 groups, 3 296 clips; percentile bootstrap B = 10 000, unit = prompt).** The chance floor is small
and battery-dependent, not zero: from −0.048 (dense, severity-1 prompts, 3.84 s) to +0.020 (P at 83 %,
10.24 s) across the AudioCaps cells, and as high as **+0.07 for the hip-hop captions at 10.24 s** (long,
generic captions are similar to any clip). Consequences:

| Cell (system, op-point) | raw | floor | above chance [95 % CI] |
|---|---:|---:|---|
| P 83 %, AudioCaps 3.84 s | 0.015 | −0.005 | +0.020 [+0.004, +0.036] |
| P 83 %, AudioCaps 10.24 s | 0.055 | +0.020 | +0.035 [+0.018, +0.052] |
| P+FT 83 %, AudioCaps 3.84 s | 0.100 | −0.015 | +0.115 [+0.098, +0.132] |
| P+FT 83 %, AudioCaps 10.24 s | 0.299 | −0.022 | +0.321 [+0.295, +0.346] |
| P 83 %, music 3.84 s / 10.24 s | 0.005 / 0.089 | −0.013 / +0.070 | +0.018 [−0.002, +0.039] / +0.019 [−0.006, +0.045] |
| P+FT 83 %, music 3.84 s / 10.24 s | 0.014 / 0.094 | −0.004 / +0.061 | +0.018 [+0.001, +0.036] / +0.033 [+0.009, +0.057] |
| P 65 %, music 3.84 s | 0.117 | +0.055 | +0.061 [+0.039, +0.084] |
| P+FT 65 %, music 3.84 s | 0.023 | +0.001 | +0.022 [+0.001, +0.042] |

* **The 83 %-pruned checkpoint is barely above chance in-domain at either duration** (0.020 / 0.035) — the
  right wording, instead of Draft 4's "has lost most of the dense duration response".
* **Draft 4's "not a floor" argument was itself a floor artefact.** "At 10.24 s the pruned checkpoint scores
  higher on music (0.089) than in-domain (0.055)" holds only because the music floor is 0.070 vs 0.020;
  above chance the pruned checkpoint is *lower* on music (0.019 vs 0.035). Draft 5 retracts the sentence
  and states the null as "a null near chance for both checkpoints, as CLAP measures it".
* **After fine-tuning, hip-hop alignment sits 0.02–0.03 above chance at both severities and durations**
  (0.022 / 0.018 / 0.033), against 0.115 / 0.321 in-domain at 83 %. The 65 % "penalty" is real but smaller
  than it looked: chance-corrected R_music = −0.040 [−0.060, −0.020] (raw −0.094), because the 65 %-pruned
  checkpoint's music output was partly *generic* (floor +0.055).
* **Floor-corrected interaction:** J_c = +0.191 [+0.162, +0.220] at 83 % (raw +0.159) and **+0.066
  [+0.020, +0.111] at 65 % (raw +0.044 [−0.001, +0.087])** — the severity-1 interaction is resolved once the
  pruned checkpoint's floor rise with duration (+0.019; the fine-tuned checkpoint's floor does not move,
  −0.003) is removed. Reported as post-hoc sensitivity; the pre-specified raw-scale verdict stands.
* Floor shift between durations ≤ 0.025 for every system: the scale objection is closed.

### A2. The "duration response" s(·) silently bundled a scorer effect [FIXED, new analysis, 0 cr]
Fused CLAP repeat-pads 3.84 s audio to 10 s and centre-crops 10.24 s audio; Draft 4 credits every system's
score change with duration to the *system*. Two anchors separate the parts: (i) the chance floor at each
duration (a floor that moves with duration is a level/scale effect); (ii) the **real AudioCaps audio of the
same prompts**, band-limited to the generators' 16 kHz and scored at its full length and as its first
3.84 s under the identical convention — s(real) is the scorer's plus content-window response with no
generation involved. Both are in the same result file.

**Result.** Real audio of the same prompts scores 0.274 (first 3.84 s) and 0.440 (full clip) on the 192
severity-2 prompts (0.264 / 0.442 on the 80 severity-1 prompts): s(real) = +0.167 [+0.150, +0.184] and
+0.178 [+0.155, +0.203]. So a duration response of ≈ +0.15–0.18 is what the scorer returns for *any*
well-aligned audio when it sees the whole clip instead of the first 3.84 s; the dense model (+0.150), the
65 %-pruned checkpoint (+0.149) and the fine-tuned checkpoints (+0.193, +0.200) are all in that band, and
only the 83 %-pruned checkpoint (+0.040; chance-corrected +0.015) is not. The "lost duration response"
sentence is therefore replaced by "barely above chance at either duration", and the fine-tuned
checkpoint's response is "normal", not "restored".

**Recovery ratios against real audio (paired, same prompts):** at 83 % pruning fine-tuning closes **63 %
[56 %, 71 %]** of the pruned checkpoint's gap to real audio at 10.24 s but **33 % [26 %, 39 %]** at 3.84 s;
at 65 %: 27 % [6 %, 46 %] vs 5 % [−16 %, 24 %] (against dense: 52 % vs 8 %, Draft 4). These four numbers
are now the headline of the abstract, the intro bullets and the conclusion. The dense model itself sits at
53–62 % of the real-audio alignment above P; the fine-tuned 83 % checkpoint at 10.24 s is 0.141
[0.110, 0.172] below real audio.

**Crop decomposition, chance-corrected (83 %, P+FT):** generation-length part +0.115 [+0.092, +0.138],
scoring-window part +0.091 [+0.077, +0.106] — the Draft-4 reading (mostly generation length) survives.

### A3. The headline at the primary severity rested on a cross-set dense reference [PARTLY FIXED]
"At 83 % the pruned checkpoint has lost most of the dense model's duration response and fine-tuning restores
one of dense magnitude" compares s(P)=+0.040 and s(P+FT)=+0.200 (192-prompt set) with s(dense)=+0.150
(80-prompt set). Draft 5 (a) keeps the sentence explicitly labelled cross-set, (b) replaces the severity-2
recovery ratio by a **paired ratio against real audio** (same 192 prompts), and (c) freezes the design that
makes it paired against dense: `docs/xsev_dense_192_control.md` + `scripts/research/run_xsev_dense_192_gen.sh`
(dense at both durations on the 192 prompts, CRN-matched to the frozen P/P+FT clips). Cost measured on this
Studio: dense at latent 96 ≈ 1.7 s/DDIM step (≈95 s/clip), latent 256 ≈ 4× → ≈18 h CPU for 384 WAVs, or
≈1.1–1.3 cr on a T4 (hard cap 1.5). **Not launched** — GPU spend needs Gabriel; CPU would block the Studio
for a day. The paper as written does not need it; a reviewer who insists gets it in one job.

### A4. The natural headline quantity was used once [FIXED, writing + numbers]
Readers of a pruning paper want to know *how much was recovered*. Draft 4 computes the fraction of the
pruned checkpoint's gap to dense closed by fine-tuning (8 % vs 52 % at severity 1) and uses it in one
sentence. Draft 5 defines the **recovery ratio** ρ = R/(ref − P) in Sec. 3.4, reports it against real audio
at both severities and both durations and against dense at severity 1 (new Table 2; the analysis-plan table was folded into two sentences in Sec. 3.4 for space), and builds the
abstract, intro bullets, discussion and conclusion on it.

### A5. The music reading was severity-fragmented [FIXED, writing, uses A1]
Draft 4: "the music penalty is severity-specific; the absence of a gain replicates." The floor reads the two
severities together: after fine-tuning, alignment on the hip-hop captions sits near the chance floor at
**both** severities (0.023 at 65 %, 0.014 at 83 %, at 3.84 s); whether that registers as a "penalty" depends
only on whether the pruned checkpoint still had music alignment to lose (0.117 at 65 % vs 0.005 at 83 %).
That is one statement instead of two, and it is what the data say.

### A6. "Narrowly includes zero" hid an underpowered test [FIXED, wording]
The Arm-D protocol was powered (MDE ≈ 0.065 at n = 80) for a larger J than the +0.044 observed. Draft 5
says so: an effect of the observed size could not be resolved on the raw scale at severity 1; the effect is
directionally consistent on every scale and established by the severity-2 replication. Reviewers accept an
explicit power statement; they punish "narrowly misses".

### A7. Caption-length/truncation claim checked in code [VERIFIED, wording]
`audioldm_train/conditional_models.py: CLAPAudioEmbeddingClassifierFreev2.tokenizer` uses `max_length=512`,
so **no caption is truncated at generation** (the 77-token limit in `modules/clap/training/data.py` is the
CLAP pre-training setting). 47 % of the music captions exceed 77 RoBERTa tokens (median 75; AudioCaps
median 12, max 38): they are near/above the caption length CLAP's text tower was pre-trained with, for both
the conditioner and the scorer. Draft 5 states this precisely as a caption-style covariate, not a truncation
artefact.

### A8. "Held-out music" over-generalised one sub-genre [FIXED, wording]
The battery is hip-hop/rap captions from MusicCaps under one eligibility rule. Abstract, bullets and
Sec. 3.2 now say "hip-hop/rap captions"; "music" is kept as the short label after being defined.

### A9. Prose density [FIXED, writing]
Draft 4 carried a 95 % CI in almost every sentence of Sec. 4 (>60 intervals in prose). Draft 5 keeps every
interval in Tables 2–3 and states point estimates in prose, keeping intervals only where the interval *is*
the point (J at both severities, the sev-1 dense differences, the negatives).

### A10. Related work gaps for an ICASSP audience [FIXED]
Added: pruning + fine-tuning of self-supervised speech encoders (PARP, DPHuBERT) as the audio precedent for
"recovery"; TTA systems that report CLAP score (Make-An-Audio, Tango). 21 references; ICASSP 2027 paper kit:
**4 pages of technical content + an optional 5th page containing only references** (confirmed on the
paper-kit page); full-paper deadline **16 September 2026** (CFP).

### A11. Things verified correct and kept
Bit-exact reproduction of every frozen number by the new script (11 consistency guards to 1e-9 on frozen
points; scorer guard 2.4e-7 on 3 296 per-item cosines); pre-specified gates; disjoint replication set;
seam sensitivity; Holm; rank-scale checks; FineLAP as a non-CLAP corroborator; crop analysis; every negative.

---

## B. Cheap improvements — what was run (all CPU, 0 cr, this pass)

| # | Item | Artifact | Label in paper |
|---|---|---|---|
| B1 | Chance floor per cell (28 groups, 3 296 clips re-embedded under the frozen convention; guard 2.4e-7) | `configs/research/draft5_floor_ceiling_result.json` | post-hoc anchor |
| B2 | Real-audio ceiling (272 real clips at full length and first 3.84 s, 16 kHz) + s(real) | same | post-hoc anchor |
| B3 | Recovery ratios ρ vs real audio (paired, both severities, both durations) and floor-corrected s, R, J | same | post-hoc sensitivity |
| B4 | Caption-token check vs the conditioner (no truncation; 47 % of music captions > 77 tokens) | same | descriptive |
| B5 | Draft-5 rewrite, anchors Table 2, Fig. 1 with floor ticks + real-audio ceiling, fill/verify scripts | `icassp/` | — |
| B6 | Paired dense control on the 192 set: protocol frozen + launch script + 1-line generator change | `docs/xsev_dense_192_control.md`, `run_xsev_dense_192_gen.sh` | **not launched** |

Not run (unchanged verdict): DDIM-200, third severity, more prompts, dense-FT reconstruction, listening study.

---

## C. The rewrite (Draft 5): what changed and why it sells better

* **Title** unchanged (declarative, keyword-complete). Fallback: the Draft-3 question title.
* **Abstract** leads with the recovery ratios against real audio and dense, names the anchors, keeps the
  robustness list and the evaluation-not-mechanism boundary (≈170 words).
* **One concept carries the paper — the recovery ratio ρ at an operating point**, defined next to R, s and
  J; Table 2 tabulates floors, real-audio levels and ρ; Fig. 1 shows floor ticks and the real-audio ceiling
  so a reader sees at a glance where P, P+FT and dense sit between chance and real audio.
* **Findings-first bullets** with ρ instead of raw cosines; the music bullet reads the two severities as one.
* **Honesty made structural:** analysis-plan table folded into Sec. 3.4 (space); power statement;
  Holm; cross-set caveat and the real-audio substitute; caption-style covariate; one sub-genre.
* **Terminology fixed:** recovery fine-tuning · P / P+FT · recovery gain R · duration response s · recovery
  ratio ρ · chance floor · real-audio ceiling · native/short · in-domain / held-out hip-hop captions ·
  pre-specified (defined) · post-hoc.
* **Layout:** 4 content pages + references on page 5 (paper-kit-compliant); verifier 97/97; forbidden-claim scan 0 hits.

## D. Remaining reviewer risks (disclosed in the paper)
1. No matched dense fine-tuned control → evaluation, not mechanism.
2. CLAP-family primary; no human ratings.
3. Off-recipe sampler (DDIM 50 / guidance 2.5 / single generation).
4. Two durations; one held-out sub-genre; dense control paired at severity 1 only (B6 fixes it if funded).
5. Severity-1 effects underpowered and not Holm-robust — stated.
