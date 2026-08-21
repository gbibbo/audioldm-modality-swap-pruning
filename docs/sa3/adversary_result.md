# Single-block E adversary — result (PILOT-directional, CLAP+KL only)

**Job `sa3-adversary-1` (Tesla T4, 0.195 cr, commit 9133829), N=16 `panel_pilot` prompts, all 20
blocks removed one at a time @ 8 steps, R=5 dense-8 seed streams, dense ladder 8/7/6/5/4.** Scored
CLAP + KL_passt (`--no-fd`; OpenL3/FD is ~2 h on CPU for 464 clips — deferred to a GPU pass). Raw:
`artifacts/sa3/adversary_analysis.json`. **PILOT-directional, point estimates, NOT the main-panel
CI CASE-E decision.**

## Margins (8→7, with the R=5 resolution floor)

* `m_CLAP = 0.164`, but the 8→7 **deterioration is 0.000** (dense-7 CLAP 0.292 ≈ dense-8 0.289) —
  the margin is set entirely by the seed floor `r_CLAP = 0.164` (CLAP varies this much across the 5
  dense seed streams). So the CASE-E test has **low resolution at N=16**.
* `m_KL = 3.50`, again floored by the seed spread (PaSST-KL is very high-variance). KL is not
  discriminating at this scale.
* **Dense CLAP is flat across 8/7/6/5/4 steps (0.28–0.30)** — the post model barely loses prompt
  adherence when sampling steps drop; KL grows slowly (0 → 0.73).

## Single-block verdict (CLAP deficit vs the dense-7 comparator)

* **Inferior (removing the block is catastrophic): only blocks 0, 1, 19** — CLAP collapses to
  ≈ 0 / negative (garbage), deficits 0.38 / 0.27 / 0.25 ≫ margin. These are the input/output
  boundary blocks (matching the field pilot's D_P > 1 there).
* **All 12–17 interior blocks (2–18): NOT inferior on CLAP** — every single-middle-block removal
  keeps CLAP within the seed-noise margin of dense-7 (deficits 0.01–0.10 < 0.164). The largest are
  block 6 (0.104), 18 (0.088), 17 (0.082), 5 (0.076).
* **3/20 inferior → MIXED, not a clean CASE E.**

## Interpretation (honest)

1. **The smoke over-read.** In the n=4 CPU smoke, skip-block-5 looked badly degraded (CLAP 0.246,
   deficit ~0.11). With the proper R=5 margin at N=16, block 5's CLAP deficit (0.076) is **inside
   the seed-noise floor (0.164)** — the apparent degradation was largely seed variance. This is
   exactly the failure mode the protocol's R=5 margin + CIs exist to prevent (post-mortem rule S7:
   detectable ≠ decision-relevant).
2. **CASE E is NOT established.** Depth pruning of a *single* middle block appears CLAP-tolerable
   here (vs the latency-matched dense-7). So the line does not die at §6.1; the interesting RQ1/RQ2
   middle ground stands — which fits the field pilot (middle blocks: small D_P, but the post is
   ~10× more sensitive than the base → a real but sub-catastrophic reorganization).
3. **This is CLAP-only.** The smoke's FD_openl3 showed large middle-block drift (skip5 FD 99.9 ≫
   dense7 20.7). The full multivariate (CLAP, KL, FD) verdict could be stronger against the middle
   blocks. **FD must be added (GPU scoring pass) before any CASE-E call**, and the decision is
   main-panel with CIs, not this pilot.

## Next single step

A GPU FD_openl3 scoring pass over the 464 adversary wavs (OpenL3 is fast on GPU), then re-run the
multivariate verdict; and, if the middle blocks remain non-inferior, proceed to the k>1 greedy
sets and A_tan on the main panel. If FD flips the middle blocks to inferior, that is the CASE-E
signal — to be confirmed on the main panel.
