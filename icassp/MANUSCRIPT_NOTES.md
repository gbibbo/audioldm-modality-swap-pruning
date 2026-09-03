# MANUSCRIPT_NOTES - ICASSP Draft 4

**Everything lives in `icassp/`:** `icassp_operating_point.tex`, official style files
`spconf.sty` + `IEEEbib.bst`, `figs/`, the built `icassp_operating_point.pdf`, this file,
`README_OVERLEAF.md`, and the ready-to-upload `icassp_operating_point_overleaf.zip`
(gitignored; regenerable). **4 content pages + references on page 5** at 9 pt (`\ninept`); ICASSP
permits a 5th page containing only references (confirm against the ICASSP 2027 CFP — if the rule is
4 pages total, cut Sec. 2 by ~8 lines or compact the bibliography). Draft 4 = rewrite of Draft 3 per
`docs/review/2026-09-02_manuscript_draft3_review.md` (Gabriel's request 2026-09-02 21:38: find the
scientific gaps, run the cheap improvements, rewrite to ICASSP standard). Gabriel compiles on Overleaf;
the local `tectonic` build is only the page-limit check.

## Build

```bash
# figures (reads durable artifacts, writes icassp/figs/*.pdf)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_draft3_figs.py
# manuscript (pdfLaTeX on Overleaf; locally tectonic -- spconf.sty must sit beside the .tex)
cd icassp && mkdir -p build && ~/.local/bin/tectonic -X compile icassp_operating_point.tex --outdir build --keep-logs
# number provenance (must print 82/82)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/verify_draft4_numbers.py
```

## Draft 3 -> Draft 4: what changed and why

* **Title:** "Recovery Fine-Tuning Recovers Where It Was Trained: Duration- and Domain-Dependent Gains
  in Pruned Text-to-Audio Diffusion" (declarative; states the finding). Alternatives if Gabriel
  prefers: the Draft-3 question title; "Post-Pruning Recovery in AudioLDM Restores Native-Duration
  Alignment In-Domain Only".
* **New concept: the duration response** $s(\cdot)$ = a system's mean per-prompt score change from
  3.84 s to 10.24 s, so $J = s(\mathrm{P{+}FT}) - s(\mathrm{P})$ is an identity and the dense model gets
  its own $s$. Table 2 has a "duration $s$; $J$" row per severity.
* **New analyses (all CPU, 0 cr; labelled post-hoc; no frozen verdict changed):**
  * **Matched dense duration control** (`scripts/research/draft4_dense_duration_control.py` ->
    `configs/research/draft4_dense_duration_control_result.json`; seed ns
    `DRAFT4-DENSE-DURATION-CONTROL|BOOTSTRAP|2026-09-02`): the 80 existing V1.1 dense r0 3.84 s WAVs
    re-scored as ONE 80-item frozen-convention call (identical to how the Arm-D short groups were made),
    paired with the frozen dense 10.24 s scores on the same 80 prompts. s(dense) +0.150 [+0.100,+0.198]
    (0.202 -> 0.352); s(P) +0.149 [+0.119,+0.178]; s(P+FT) +0.193 [+0.152,+0.232]; s(P)-s(dense) -0.001
    [-0.056,+0.055]; s(P+FT)-s(dense) +0.043 [-0.020,+0.109]; dense gap closed 8% [-30%,+36%] at 3.84 s
    vs 52% [+11%,+103%] at 10.24 s. Consistency guards reproduce the frozen Arm-D and DENSE_CONTROL points
    to 0.0. Batch-composition diagnostic: re-scoring the same clips in another batch shifts means by
    <=0.002 (stated in Sec. 3.3).
  * **Multiplicity** (`scripts/research/draft4_robustness.py` R1 ->
    `configs/research/draft4_robustness_result.json`; ns `DRAFT4-ROBUSTNESS|BOOTSTRAP|2026-09-02`): Holm
    over all 13 reported contrasts. Every severity-2 dagger survives (p < 1e-4); at severity 1 R_nat
    (p = 0.016) and J (p = 0.052) do not; the music and domain contrasts do. Stated in Sec. 4.1, Table 2
    caption, Limitations.
  * **Rank-scale J** (R2): median per-prompt interaction +0.051 [+0.012,+0.077] / +0.172 [+0.153,+0.214];
    Wilcoxon p = 0.020 / < 1e-17; pooled-rank J +0.063 [-0.015,+0.143] / +0.199 [+0.153,+0.244].
  * **Caption-length check** (R3): music captions median 56.5 words vs AudioCaps 8; within AudioCaps the
    native gain is uncorrelated with caption length (Spearman rho +0.04 [-0.12,+0.18], sev-2).
  * **Floor rebuttal** (R3, descriptive): at 10.24 s the pruned checkpoint scores higher on music (0.089)
    than in-domain (0.055).
* **Wording / reporting fixes:** "pre-registered" -> "pre-specified" (defined once: committed to the
  version-controlled repository before any score was seen); "absent on music" -> "absent or negative";
  FineLAP n stated (110 / 49 eligible prompts); severity-2 music at 10.24 s = one replicate; the two
  severity-1 short-duration numbers (n=80 subset vs n=96 pre-specified set) are bridged in the Table 2
  caption; process narrative ("reframed the work") removed from Results; Discussion reads the dense
  control (65 %: P responds like dense; 83 %: P lost the response, P+FT restores a dense-magnitude one,
  in-domain only); Limitations add the cross-set caveat, the domain/caption-style bundling and the
  severity-1 Holm failure.
* **Figures:** Fig. 1(a) grey line now = the MATCHED dense control (0.202 -> 0.352, same 80 prompts,
  same convention; was the unmatched n=96 anchor) with the three duration responses annotated;
  R_short labels moved off the y-axis. Fig. 2 unchanged.
* **Layout:** URL inline in the Conclusion (footnote removed); Table 2 `\tabcolsep` 2.3 pt; ~10 lines
  trimmed (Sec. 3.4, 4.1, 4.4, 5, 6). Remaining overfull boxes < 2.5 pt.

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
| **dense 0.202->0.352, s(dense), s(P), s(P+FT) sev-1, differences, gap closed, batch <=0.002** | **`configs/research/draft4_dense_duration_control_result.json` (post-hoc control)** |
| **Holm p-values, median/Wilcoxon/rank J, caption words, Spearman, pruned music 0.089 vs 0.055** | **`configs/research/draft4_robustness_result.json` (post-hoc sensitivity)** |
| parameter counts | Draft-2 CPU count (415.96 / 145.67 / 71.08 M), validated bit-exact |

## Known compile notes

`tectonic` (XeTeX engine) prints harmless `TU/ptm` Times-shape substitution warnings; Overleaf
pdfLaTeX uses real Times. `verify_draft4_numbers.py`: 82/82 numbers reproduced from artifacts (it
re-uses the Draft-3 check list and adds the Draft-4 numbers). **Verify venue/pages of every reference
before submission** (bibliographic details were written from memory of the records).
