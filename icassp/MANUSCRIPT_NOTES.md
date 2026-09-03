# MANUSCRIPT_NOTES - ICASSP Draft 5

**Everything lives in `icassp/`:** `icassp_operating_point.tex`, official style files
`spconf.sty` + `IEEEbib.bst`, `figs/`, the built `icassp_operating_point.pdf`, this file,
`README_OVERLEAF.md`, and the ready-to-upload `icassp_operating_point_overleaf.zip`
(gitignored; regenerable). **4 content pages + references on page 5** at 9 pt (`\ninept`); the ICASSP
2027 paper kit allows "4 pages of technical content" plus "one additional optional 5th page containing
only references" (confirmed 2026-09-02). Draft 5 = rewrite of Draft 4 per
`docs/review/2026-09-02_manuscript_draft4_review.md` (Gabriel's request 2026-09-02 22:18: find the
scientific gaps, run the cheap improvements, rewrite to the level of the best ICASSP papers). Gabriel
compiles on Overleaf; the local `tectonic` build is only the page-limit check.

## Build

```bash
# 1) anchors (chance floor + real-audio ceiling; CPU ~35 min; needs the frozen WAVs on disk)
OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/draft5_floor_ceiling.py --emit
OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/draft5_floor_ceiling.py --score
OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/draft5_floor_ceiling.py --verdict
# 2) figures (reads durable artifacts, writes icassp/figs/*.pdf)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_draft5_figs.py
# 3) placeholders (only needed after editing a `@@` placeholder back into the .tex)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/fill_draft5.py
# 4) manuscript (pdfLaTeX on Overleaf; locally tectonic -- spconf.sty must sit beside the .tex)
cd icassp && mkdir -p build && ~/.local/bin/tectonic -X compile icassp_operating_point.tex --outdir build --keep-logs
# 4b) dense-192 integration (idempotent; already applied in the committed .tex)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_dense192.py
# 5) number provenance (must print all OK, no placeholders)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/verify_draft5_numbers.py
# 6) A8 secondary metrics at BOTH durations (CPU ~20 min; frozen WAVs, no generation)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_secondary_metrics_short.py
# 7) A10 credit estimate for the reviewer's GPU asks (CPU, 0 cr, launches nothing)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/a10_gpu_cost_estimate.py
```

## Draft 4 -> Draft 5: what changed and why

* **New anchors (post-hoc, 0 cr, no generation; `scripts/research/draft5_floor_ceiling.py` ->
  `configs/research/draft5_floor_ceiling_result.json`, seed ns `DRAFT5-FLOOR-CEILING|BOOTSTRAP|2026-09-02`):**
  * **Chance floor** per (system, operating point) cell: the mean cosine between each clip and the captions
    of the other prompts in its battery, computed from the same embeddings as the frozen scores (frozen
    convention re-run group by group; guard: the diagonal reproduces every frozen per-item cosine, max
    |diff| 2.4e-7 over 28 groups / 3 296 clips; 11 frozen point estimates reproduced to 1e-9).
  * **Real-audio ceiling**: the real AudioCaps clip of each prompt (16 kHz band-limited), scored at its
    full length and as its first 3.84 s under the identical convention; s(real) = scorer + content-window
    duration response with no generation.
  * **Recovery ratio** rho = R/(ref - P) against real audio (paired, both severities, both durations) and
    against dense (severity 1, from Draft 4); floor-corrected s, R, J; crop decomposition floor-corrected;
    caption-token check (no truncation at generation: conditioner max_length=512; 47 % of music captions
    exceed the 77-token CLAP pre-training length).
* **Rewrite:** recovery ratio is the headline quantity (abstract, bullets, discussion, conclusion);
  new Table 2 (floors P / P+FT, real audio, rho_real; rho_dense in its caption); the analysis-plan table folded into Sec. 3.4; Fig. 1 shows floor ticks and the
  real-audio ceiling; CIs moved from prose to Tables 1-2 (prose keeps point estimates and the few
  intervals that are the point); severity-1 stated as underpowered (MDE 0.065 at n=80) instead of
  "narrowly misses"; music battery named precisely (hip-hop/rap captions); the two severities' music
  results read as one statement via the floor; two speech-pruning refs + two TTA refs added (22 refs).
* **XSEV-DENSE-192-CONTROL (run 2026-09-03, Gabriel GO; T4 job 1.2633 cr):** dense at both durations on the
  severity-2 192 prompts, CRN-matched. Integrated by `scripts/research/paper_figs/integrate_dense192.py`
  (idempotent; marker `%% dense192-integrated`): the cross-set sentence is now paired (s(dense) +0.147;
  s(P+FT) − s(dense) +0.053 [+0.017,+0.089]); rho_dense 44 % / 82 % in abstract, bullets, Sec. 4.1, Table 2
  and conclusion; dense − P+FT +0.055 [+0.021,+0.088] at 10.24 s in Sec. 4.4 (not restored); Fig. 1(b) dense
  line; cross-set caveat removed from Limitations.

## Draft 5 -> Draft 5r: the external reviewer's action list A2-A9 (2026-09-03, CPU only, 0 cr)

Implemented from `docs/review/2026-09-03_manuscript_draft5_icassp_reviewer_simulation.md` on Gabriel's
instruction; A10 (every GPU run) was **costed, not launched** (`docs/compute_budget.md` §A10).

* **A8 is the one new measurement.** `scripts/research/xsev_secondary_metrics_short.py` ->
  `configs/research/xsev_secondary_metrics_short.json`. KL and PANNs top-10 capture, previously run at
  the native point only, are now measured at **both** durations on the frozen severity-2 WAVs against
  the real clip of the same prompt (full <=10 s / first 3.84 s, the files
  `draft5_floor_ceiling.py --emit` already wrote). J_KL = +1.56 [+1.19,+1.92], J_PANN = +0.67
  [+0.49,+0.86], both seam-robust; guard reproduces the frozen native means exactly (max |diff| 0.0).
  New text in Sec. 4.4; the abstract's corroboration sentence is now true as written.
* **A2/A3/A4/A5.** Abstract corroboration reworded; the Domain bullet states what was measured ("at
  both severities at 3.84 s and at both durations at severity 2" - there is no severity-1 music cell at
  10.24 s); the three analysis statuses are defined **once** in Sec. 3.4 (pre-specified / registered
  after the primary result / post-hoc) and Table 2 and the Limitations point at that definition; a
  one-clause co-author/checkpoint disclosure sits in Sec. 3.1.
* **A6 figures.** Both are now included at `\columnwidth` and DRAWN at 3.35 in (one ICASSP column), so
  the lettering renders at its stated point size instead of 0.74x/0.58x. `R_short` moved off the zero
  line and the chance-floor ticks; Fig. 1 legend to one row; Fig. 2 "early/late" moved to the zero
  line, clear of the `late-early T` annotation; y-labels shortened; legends given a translucent frame.
* **A7 readability.** Secs. 4.1-4.2 lead with the claim; nine intervals that Tables 1-2 already carry
  left the prose. `verify_draft5_numbers.py` retires the three prose checks and adds three table-form
  ones, so the count stays 108/108 with the same artifacts behind every number.
* **A9 references and policy.** ICASSP 2027 is **single-anonymous** (editorial policies: "Papers
  undergo single-anonymous review"; paper kit: "ICASSP does not perform blind reviews, so be sure to
  include the author list"), so the author block, e-mails and audio URL stay. Three references were
  published since their preprint and were corrected: Tango -> ACM MM 2023 pp. 3590-3598 (title
  "instruction guided"); Human-CLAP -> APSIPA ASC 2025 pp. 131-136; FineLAP -> ACL 2026
  pp. 10393-10408.
* **Page budget.** The full-column figures plus A2-A5/A9 cost ~20 lines, paid for by A7, by tightening
  the float/caption skips in the preamble, by trimming caption and Discussion text that repeated
  Table 2, and by two figure-height reductions. Result: **4 content pages + references page**, 0
  overfull boxes, 108/108, zip compiles standalone.

## Number provenance (every number in the .tex)

| Where | Source artifact |
|---|---|
| sev-2 R_native/R_short/R_music/J, B', dense control G | `configs/research/xsev_result.json` (frozen) |
| sev-1 R_short/R_native/J, HC J, raw cosines | `configs/research/op_duration_discriminator_1_result.json` (frozen) |
| sev-1 domain test R_AC, I | `configs/research/reversal_v1_1_result.json` (frozen) |
| sev-1 music R (-0.094), means 0.117/0.023, W 0.20 | `configs/research/reversal_v1_r_music_clap.json` + persisted `artifacts/icassp_gate0/_phenom_groups_out.json` |
| HC sev-2, KL, PANN, FAD/FD | `configs/research/xsev_hc_secondary.json`, `xsev_secondary_metrics.json` |
| FineLAP D_early/D_late/T, mass/occupancy/coverage/peak, n | `configs/research/finelap_temporal_result.json` |
| win-rates, dW, sev-2 duration responses, sev-2 domain gap | `configs/research/draft3_sensitivity_result.json` (post-hoc) |
| R_crop and differences | `configs/research/native_crop_analysis_result.json` (post-hoc) |
| music @10.24 s, J_music, D_nat | `configs/research/xsev_music_native_1_result.json` (frozen protocol `docs/xsev_music_native_1.md`) |
| dense 0.202->0.352, s(dense), s(P), s(P+FT) sev-1, differences, gap closed 8 %/52 %, batch <=0.002 | `configs/research/draft4_dense_duration_control_result.json` (post-hoc control) |
| Holm p-values, median/Wilcoxon J, caption words, Spearman, pruned music 0.089 vs 0.055 | `configs/research/draft4_robustness_result.json` (post-hoc sensitivity) |
| **chance floors, real-audio levels, s(real), rho_real, J_c, floor-shift max, 47 % tokens** | **`configs/research/draft5_floor_ceiling_result.json` (post-hoc anchors)** |
| **sev-2 dense 0.207->0.354, s(dense) +0.147, s(P)/s(P+FT) − s(dense), rho_dense 44 %/82 %, dense−P+FT gaps** | **`configs/research/xsev_dense_192_control_result.json` (XSEV-DENSE-192-CONTROL; prospective design completion, T4 job 1.2633 cr)** |
| MDE 0.065 at n=80 (sev-1 power) | `docs/op_duration_discriminator_1.md` (frozen protocol) |
| parameter counts | Draft-2 CPU count (415.96 / 145.67 / 71.08 M), validated bit-exact |

## Known compile notes

`tectonic` (XeTeX engine) prints harmless `TU/ptm` Times-shape substitution warnings; Overleaf
pdfLaTeX uses real Times. `verify_draft5_numbers.py` re-uses the Draft-3/4 check lists (minus the strings
Draft 5 no longer prints) and adds the Draft-5 anchors (97/97). **Verify venue/pages of every reference before
submission** (bibliographic details were written from memory of the records; the four references added in
Draft 5 — PARP (NeurIPS 2021), DPHuBERT (Interspeech 2023), Make-An-Audio (ICML 2023), Tango (arXiv
2304.13731) — included).

## 2026-09-03 post-integration fixes (external-reviewer simulation, 0 cr)

Review artifact: `docs/review/2026-09-03_manuscript_draft5_icassp_reviewer_simulation.md` (ICASSP-form
scores, overall 3/5 borderline; strengths/weaknesses; review text; action list A1-A10). Two defects in the
committed package fixed, no number changed: (D1) Sec. 4.1 printed a literal "eftab:anchors" (lost `\r`
after a manual edit; now `(Table~\ref{tab:anchors})`); (D2) Fig. 1 caption said only panel (a) carries the
dense control although both panels do since the dense-192 integration. Rebuilt PDF (5 pages: 4 content +
references) and Overleaf zip (compiles standalone from a fresh extraction); `verify_draft5_numbers.py`
108/108. Open editorial actions from the review (A2-A9: abstract corroboration wording, intro "both
severities and durations", one status label for the severity-2 dense control, co-author disclosure,
figure widths, readability pass, KL/PANNs at 3.84 s on the frozen WAVs, anonymity/reference check) are
Gabriel's call.
