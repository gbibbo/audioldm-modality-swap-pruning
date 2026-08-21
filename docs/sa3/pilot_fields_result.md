# RQ1 field pilot — result (PILOT / sizing, NOT a section-8 decision)

> **⚠ SEE `docs/sa3/review_erratum_2026-08-21.md` (E6/E8).** The `D_P/D_B≈10×` claim needs the field-norm ratio to rule out a normalization artifact; the phrasing is corrected below.


**Job `sa3-pilot-fields-2` (Tesla T4, 0.2888 cr, commit 31d9ad6), N=32 prompts of `panel_pilot`,
all 20 blocks, forward-only, fp16.** Raw: `artifacts/sa3/pilot_fields.json`. This is pilot data —
it SIZES the main experiment and shows direction; no case-A–E decision is read from it (that is
main-panel, after freezing N_main / n_u / margins, per §6.0).

## What was measured (protocol §3.1–3.3, on the dense-post S_traj states)

* `D_P(g)` post-field damage from removing block g; `D_B^common(g)` base-field damage on the SAME
  states; `I_PT^raw(g)` normalized post-training-delta distortion (pooled); `W(g)` parameter-delta
  covariate (never an effect estimate).

## Findings (directional)

1. **Block importance is strongly structured, not uniform.** `D_P` spans 0.016 (block 12) to
   **2.31 (block 0)** and **1.67 (block 19)**; the input/output boundary blocks (0, 19, 17, 18, 1)
   dominate, interior blocks 5–13 are ~0.02–0.05. Removing a boundary block distorts the post
   field by **more than the field's own norm** (D_P > 1) — those blocks are near-critical.
2. **Few-step post-training concentrates functional reliance on the INTERIOR blocks.** The
   post-vs-base sensitivity ratio `D_P/D_B` is **6.8–11.5 for blocks 3–11** but ~1.0–1.5 for the
   boundary blocks (0, 17, 18, 19). i.e. the post model is ~an order of magnitude more sensitive
   to removing a middle block than the base is, while both are equally (very) sensitive at the
   boundaries. This is the RQ1 signal (phrasing per erratum E8): **few-step post-training strongly amplifies relative interior-block sensitivity while much of the global block-importance ranking remains shared with the base** (ρ(D_P,D_B)=0.82). The amplification is **confirmed real, not a normalization artifact**: ‖F_P‖²/‖F_B‖²=0.889 (fields nearly equal magnitude), so the *non-normalized* damage ratio is 9–10× for interior blocks 5/6/7/9 (median 5.3×) and ~1× at the boundaries (`rq1_field_norms.py`; erratum E6).
3. **Parameter-space change does NOT predict functional importance.** `W(g)` is tiny (~1e-5) and
   nearly uniform across all 20 blocks; **Spearman(D_P, W) = −0.21** (≈ 0). The weights barely move
   and move uniformly, yet the functional impact varies ~100× — a `‖ΔW‖`-based proxy would miss the
   structure entirely (rule S1: parameter proxies are covariates, never the effect estimate).
4. `I_PT^raw` tracks `D_P` closely (**Spearman 0.93**); `D_P` vs `D_B` rank-correlate 0.82
   (boundary blocks matter for both, interiors diverge).

## Caveats / what this is NOT

* PILOT (N=32), point estimates, no bootstrap CIs yet → cannot freeze N_main or read §8. Boundary
  dominance is partly expected (input/output interface). The interior-reliance story needs the main
  panel + CIs to confirm.
* `D_P` is a *field* quantity on the 8 ping-pong states; a small `D_P` need NOT mean small
  end-to-end damage (a small per-step field change compounds over 8 steps). The deployment-relevant
  test is `E` on generated audio (the single-block adversary), measured separately.
* `I_PT^dep` (base deploy/CFG field), `A_tan` (adaptability), the R=5 seed streams / 8→7 margins,
  and the N_main/n_u bootstrap are NOT in this run — next steps.

## Next single step

The E single-block adversary on generated audio (`scripts/sa3/e_adversary.py`, CPU-validated) to
see whether the interior blocks whose field barely moves nonetheless degrade *output* beyond the
latency-matched dense comparator — the CASE-E question. The smoke already hinted yes for block 5.
