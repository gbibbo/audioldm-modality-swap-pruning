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
