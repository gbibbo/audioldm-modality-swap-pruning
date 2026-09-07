# REVIEW3-DENSERAW — results: the raw dense baseline (third review, item 1b)

**Date:** 2026-09-06 (MVD 20:5x). **Compute:** one T4 job `r3-denseraw` (Completed, **1.3248 cr**, 77 min running, cap 2.6 /
180 min, not killed) + CPU scoring (0 cr). **Provenance:** protocol `docs/review3_denseraw.md` (frozen + sha256 before
launch, `807309e3…`); result `configs/research/r3_denseraw_result.json` (sha256 `ba3631e4…`); scorer
`scripts/research/r3_denseraw_score.py`; fused CLAP `laion/clap-htsat-fused` rev `365dea6e`, prompt bootstrap B = 10⁴, seed
namespace `REVIEW3|BOOTSTRAP|2026-09-06`; intervals 95 %. Generation: 192 + 192 `dense_raw` WAVs (61 472 / 163 872 samples,
Tesla T4, checkpoint `dense_raw:936914a388905e1f`, weight label `raw`, job tree `c186572`), CRN-paired by prompt with the
frozen dense-EMA (`xsev_dense192`), `denseft_short`, `denseft_native`, P and P+FT clips. Device check: 4 dense-EMA clips
regenerated on the job's T4 vs the frozen `xsev_dense192` clips, max |ΔCLAP| = 1.9e-5 (descriptive; the bit-identical rule
has failed at ≈ 4e-6 on every job and is reported as written).

Notation: `O(d) = CLAP(dense_raw) − CLAP(dense_EMA)`; `G′(d) = CLAP(denseft) − CLAP(dense_raw)`; `G(d)` = the reported
contrast against the EMA baseline; `J′ = G′(10.24) − G′(3.84)`; `ΔJ′_dense = J′_native − J′_short`.

## The raw dense weights score the same as the EMA weights

| Duration | CLAP dense_raw | CLAP dense_EMA | **O(d)** | reading |
|---|---:|---:|---:|---|
| 3.84 s | 0.208 (floor −0.022) | 0.207 | **+0.001 [−0.017, +0.020]** (n = 192) | within ±SESOI |
| 10.24 s | 0.337 (floor −0.017) | 0.354 | **−0.017 [−0.044, +0.010]** (n = 192) | unresolved (|point| < SESOI) |

`Δ_O = O(10.24) − O(3.84) = −0.019 [−0.050, +0.013]` — unresolved. The raw-vs-EMA convention moves CLAP by at most ≈ 0.02
at step 0; it cannot account for a drop of 0.2.

## The 20 000-step fine-tune genuinely degraded the dense model

| Dense arm | G′(3.84 s) vs raw | G′(10.24 s) vs raw | G(3.84 s) vs EMA | G(10.24 s) vs EMA | J′ |
|---|---:|---:|---:|---:|---:|
| `denseft_short` (train @3.84) | **−0.183 [−0.206, −0.160]** (192) | **−0.220 [−0.258, −0.181]** (170) | −0.182 | −0.235 | −0.034 [−0.071, +0.002] |
| `denseft_native` (train @10.24) | **−0.168 [−0.190, −0.144]** (192) | **−0.206 [−0.235, −0.177]** (192) | −0.166 | −0.223 | −0.038 [−0.069, −0.008] |

`ΔJ′_dense = −0.005 [−0.033, +0.025]` (n = 170); guard vs the reported `ΔJ_dense` (`r2_EXT2x2_result.json`): |diff| = 0.0
(the baseline cancels algebraically, as declared).

## Pre-specified reading: **DEGRADATION**

Rule (`docs/review3_denseraw.md` §4): all four `hi95(G′(d)) < 0` **and** `|O(d)| < |G(d)|/2` at both durations. Both hold
(`hi95(G′)` ≤ −0.144; `|O|` ≤ 0.017 against `|G|/2` ≈ 0.08–0.12). Consequences, as pre-specified:

1. The −0.2 drop of the dense 2×2 is **not** a weight-convention artifact. The recipe itself — full-U-Net AdamW, constant
   lr 1e-4, effective batch 2, no EMA, 20 000 steps, the recipe that *improves* the pruned checkpoint (+0.05 to +0.08 at
   10.24 s) — degrades a dense model that had already converged on AudioCaps (0.25 M steps by its authors). The
   fine-tuned dense checkpoints land at ≈ 0.03 / 0.12–0.13 CLAP, the same profile as the fine-tuned pruned checkpoints
   (0.02–0.03 / 0.10–0.13): the recipe pulls both starts to a common operating-point-dependent attractor.
2. **The raw-vs-EMA explanation must be deleted from the manuscript** (Sec. 3.2, Sec. 4.2, Sec. 5), not softened, and
   replaced by the recipe statement with the measured `O(d)`. The dense 2×2 stays a negative / limitation; it still shows
   no training-duration specialization (`ΔJ_dense ≈ 0`).
3. Item 1(c) of the third review (P is EMA-derived, the pruned fine-tunes are raw): the only measurement of the
   convention offset is on the dense model at step 0, `O(d)` within ±0.02 and `Δ_O = −0.019 [−0.050, +0.013]`. It is a
   caveat magnitude, not a correction: small relative to `J_3.84 = +0.065`, of the order of the lower bound of
   `J_10.24 = +0.031`. `ΔJ_pruned = −0.035 [−0.064, −0.004]` is raw-vs-raw and unaffected.

The earlier audit (`docs/reviewer2_followup_ext_results.md` §Audit, 2026-09-06 13:5x) offered "raw vs EMA + no headroom"
as the two explanations of the dense drop. The first is now **falsified by measurement**; the second stands and is the
whole explanation. `docs/reviewer2_scientific_report.md` Point 2 carries the same sentence; both are annotated.

## Exact manuscript edits under DEGRADATION (Draft 14; every number from `r3_denseraw_result.json`; Gabriel's prose to sign off)

**E1 — Sec. 3.2 (`draft14_2.tex`), dense 2×2 paragraph.** Replace

> "The dense baseline is EMA-averaged, whereas the short experimental exports are raw weights, and the dense model had
> already received substantial AudioCaps training."

with

> "The dense model had already received substantial AudioCaps training. Because the released baseline is EMA-averaged
> while the experimental exports are raw weights, we also generated the released checkpoint's raw weights on the same
> prompts: they score within $\pm0.02$ CLAP of the EMA weights at both durations, so the weight convention does not enter
> the comparison."

**E2 — Sec. 4.2 (`draft14_3.tex`), dense paragraph.** Replace

> "The training loss remains stable, which is consistent with the anticipated raw-weight-versus-EMA mismatch and limited
> AudioCaps headroom rather than optimizer divergence."

with

> "The training loss remains stable, and the raw weights of the released checkpoint score the same as its EMA weights
> ($+0.001$ \ci{-0.017}{+0.020} at $3.84$\,s, $-0.017$ \ci{-0.044}{+0.010} at $10.24$\,s), so the drop is not a
> weight-convention artifact: the same $20{,}000$-step recipe that improves the pruned checkpoint degrades a dense model
> that had already converged on AudioCaps, and both fine-tuned dense checkpoints land at the low CLAP profile of the
> fine-tuned pruned ones."

**E3 — Sec. 5 (`draft14_4.tex`), causal-limitation paragraph.** Replace

> "The experimental dense $2\times2$ uses a short raw-weight fine-tune against an EMA baseline and substantially degrades
> the dense model, so it cannot replace …"

with

> "The experimental dense $2\times2$ substantially degrades the dense model under the same $20{,}000$-step recipe that
> improves the pruned checkpoint, so it cannot replace …"

**E4 — Sec. 3.2 or 4.2, one sentence for item 1(c) (0 cr, independent of E1–E3).**

> "P is derived from the dense EMA weights while both $20{,}000$-step checkpoints are raw exports; this mismatch cancels in
> $\Delta J$ and, measured on the dense model at step zero, shifts CLAP by at most $\pm0.02$ with an unresolved duration
> dependence ($-0.019$ \ci{-0.050}{+0.013}), a caveat on the individual $J$ values rather than a correction."

After applying: extend `verify_draft14_numbers.py` with the four new numbers (O(3.84), O(10.24), Δ_O, and the ±0.02
bound) and re-run `pagecheck_times.py` (E1/E2 add ≈ 3 lines; the current build has ≈ 2 characters of slack on page 5, so
a compensating trim of similar length is needed — candidates: the "Additional checks extend…" sentence of Sec. 3.2 or the
FAD sentence of Sec. 2.2).

## Cost lesson

Dense **inference-only** per-WAV cost on the T4 = (1.3248 − 0.145) / 388 ≈ **0.0030 cr/WAV** at mixed latent 96/256 —
below the 0.0041–0.0052 used for the estimate, which was fitted on jobs whose eval share was entangled with training.
`total_spent` 136.9389 cr at scoring (135.5623 at launch → +1.38 cr incl. Studio); of Gabriel's 4-cr pool ≈ 2.6 cr remain.
