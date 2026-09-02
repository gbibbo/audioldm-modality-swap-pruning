# MANUSCRIPT_NOTES - ICASSP Draft 3

**Everything lives in `icassp/`:** `icassp_operating_point.tex`, official style files
`spconf.sty` + `IEEEbib.bst`, `figs/`, the built `icassp_operating_point.pdf`, this file,
`README_OVERLEAF.md`, and the ready-to-upload `icassp_operating_point_overleaf.zip`
(gitignored; regenerable). **4 pages total** (content + references) at 9 pt (`\ninept`).
Draft 3 = rewrite of Draft 2 per the scientific review
`docs/review/2026-09-02_manuscript_draft2_scientific_review.md` (Gabriel GO 2026-09-02, 18:16,
condition: minimal credit use). Gabriel compiles on Overleaf; the local `tectonic` build is only
the page-limit check.

## Build

```bash
# figures (reads durable artifacts, writes icassp/figs/*.pdf)
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_draft3_figs.py
# manuscript (pdfLaTeX on Overleaf; locally tectonic -- spconf.sty must sit beside the .tex)
cd icassp && mkdir -p build && ~/.local/bin/tectonic -X compile icassp_operating_point.tex --outdir build --keep-logs
```

## Draft 2 -> Draft 3: what changed and why

* **Title:** "How Much Does Recovery Fine-Tuning Recover? Operating-Point-Dependent Gains in Pruned
  Text-to-Audio Diffusion" (review D1, option 1).
* **Terminology (review D2):** *recovery fine-tuning* (the stage; Singh et al.'s term), *pruned
  checkpoint* P and *fine-tuned checkpoint* P+FT, *recovery gain* R, native/short duration,
  in-domain / held-out music; "pre-registered" once; "post-hoc sensitivity" / "post-result
  diagnostic" as labels. The frozen-story ban on "recovered model" is honoured (neutral labels).
* **Contributions are findings** (duration; domain at matched duration; where the gain lives), not
  apparatus. Abstract (~150 w) opens with the vivid absolute numbers.
* **Evidence hierarchy** is a 7-row Table 1 (analysis / registered / role) instead of ~30 % of the
  prose. Parameter-count table removed (numbers in text + Singh's 39 % MAC figure cited).
* **Pruned-baseline provenance stated correctly** (review A5): released L1 channel selection applied
  to the dense EMA weights; bit-exact vs the released raw weights at sev-1; 3-tensor seam ambiguity
  at sev-2 -> A'/B'.
* **New analyses added (all 0 cr; authorised 2026-09-02):**
  * **B1 scale-free interaction** (`configs/research/draft3_sensitivity_result.json`, post-hoc
    sensitivity): win-rate W(P+FT>P) 0.44->0.64 (sev-1, dW +0.20 [+0.06,+0.34]) and 0.72->0.87
    (sev-2, +0.15 [+0.07,+0.22]); duration slopes P/P+FT; paired d. Column W in Table 2.
  * **B2 matched-duration domain contrast** replaces K: sev-1 frozen I = +0.092 [+0.054,+0.131]
    (reversal_v1_1_result.json), sev-2 R_short - R_music = +0.076 [+0.047,+0.105].
  * **B3 crop analysis** (`configs/research/native_crop_analysis_result.json`, post-hoc diagnostic;
    first 3.84 s of the frozen native WAVs scored with the frozen CLAP convention): sev-1 R_crop
    +0.051 [+0.016,+0.085] = R_native (R_crop - R_short +0.043 [+0.002,+0.085]); sev-2 R_crop +0.172
    [+0.150,+0.194], R_crop - R_short +0.087 [+0.065,+0.110], R_nat - R_crop +0.072 [+0.055,+0.089]
    (B' identical). Fig. 2(b). Reading: the short-duration deficit is a generation-length effect.
  * **Frozen Human-CLAP sev-1 J** (+0.075 [+0.012,+0.137]) and Singh et al.'s in-domain
    "fine-tuned pruned > unpruned" (FAD 1.57 vs 3.95) now used (review A6/A7).
  * **XSEV-MUSIC-NATIVE-1** (protocol `docs/xsev_music_native_1.md`, frozen before generation;
    music-64 x r0 @10.24 s, P and P+FT, generated on CPU, 0 cr): fills the `[[MN-*]]` placeholders
    (Table 2 rows "music 10.24 s" and "domain 10.24 s", Sec. 4.2 sentence, Sec. 3.2 cell list,
    Intro bullet). **Until filled, the .tex still contains `[[MN-...]]` markers** — a dummy-filled
    scratch compile (`build/scratch_fill.tex`) is used to verify the 4-page fit with realistic
    widths.
* **Figures:** Fig. 1 unchanged in content (interaction, CI whiskers, dense reference). Fig. 2 =
  (a) FineLAP grounding-gain-vs-time (label collisions fixed) + (b) generation length vs scoring
  window (R at generated 3.84 s / first 3.84 s of 10.24 s / full 10.24 s). The Draft-2 forest panel
  is redundant with Table 2 and dropped (`fig_summary.pdf`, `fig2_forest.pdf`, `fig3_finelap.pdf`
  kept on disk as spares).
* **Citations fixed against arXiv (2026-09-02):** Singh et al. cited as arXiv:2607.13330 ("Submitted
  to DCASE 2026 Workshop" — not a proceedings entry); Human-CLAP = Takano, Okamoto, Kanamori et al.,
  arXiv:2506.23553 (the Draft-2 *PDF* still carried a stale "Yamamoto" byline — PDF was out of
  date vs the .tex); FineLAP cited as arXiv:2604.01155 (ACL venue not confirmed by the record).
  Added: TinyFusion (CVPR 2025), AudioLDM 2 (TASLP 2024), Stable Audio Open (ICASSP 2025), Gui et
  al. FAD (ICASSP 2024), KAD (arXiv 2502.15602), Kumar et al. (ICLR 2022). 18 references.
  **Verify venue/pages of every added reference before submission** (bibliographic details were
  written from memory of the records, not fetched one by one).
* **Documented protocol details** (review A8-A10): cross-duration pairing is prompt-paired, not
  noise-paired; CLAP repeat-pads 3.84 s to 10 s and centre-crops 10.24 s to 10 s; sampler deviations
  from the published recipe listed (DDIM 50 vs 200, guidance 2.5 vs 3.5, single vs best-of-3).
* **Discussion:** recovery gain tracks the fine-tuning operating point; four-item reporting
  recommendation (was seven); limitations folded in.

## Number provenance (every number in the .tex)

| Where | Source artifact |
|---|---|
| sev-2 R_native/R_short/R_music/J/K, B', dense control | `configs/research/xsev_result.json` (frozen) |
| sev-1 R_short/R_native/J, HC J, raw cosines | `configs/research/op_duration_discriminator_1_result.json` (frozen) |
| sev-1 domain test R_AC, I, dense 3.84 s | `configs/research/reversal_v1_1_result.json` (frozen) |
| sev-1 music R (-0.094), means 0.117/0.023, W 0.20 | `configs/research/reversal_v1_r_music_clap.json` + persisted `artifacts/icassp_gate0/_phenom_groups_out.json` |
| HC sev-2, KL, PANN, FAD/FD | `configs/research/xsev_hc_secondary.json`, `xsev_secondary_metrics.json` |
| FineLAP D_early/D_late/T, mass/occupancy/coverage/peak | `configs/research/finelap_temporal_result.json` |
| win-rates, dW, slopes, sev-2 domain gap | `configs/research/draft3_sensitivity_result.json` (post-hoc) |
| R_crop and differences | `configs/research/native_crop_analysis_result.json` (post-hoc) |
| music @10.24 s (`[[MN-*]]`) | `configs/research/xsev_music_native_1_result.json` (pending generation) |
| parameter counts | Draft-2 CPU count (415.96 / 145.67 / 71.08 M), validated bit-exact |

## Known compile notes

`tectonic` (XeTeX engine) prints harmless `TU/ptm` Times-shape substitution warnings; Overleaf
pdfLaTeX uses real Times. Remaining overfull boxes: <4 pt (equation line and Table 1), negligible.
