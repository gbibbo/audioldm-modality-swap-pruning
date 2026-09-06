# Recovery Gain Is Operating-Point Dependent in Pruned Text-to-Audio Diffusion

Research code, protocols, provenance and results for an evaluation study of recovery fine-tuning in pruned text-to-audio diffusion. The paper uses released AudioLDM-M pruned and recovered checkpoints as its main case study and adds targeted short-budget interventions to distinguish operating-point dependence from training-duration specialization.

## Current manuscript

The current working manuscript is **Draft 14**.

* Source: [`icassp/icassp_operating_point.tex`](icassp/icassp_operating_point.tex)
* Modular sections: [`icassp/sections/draft14_1.tex`](icassp/sections/draft14_1.tex) through [`icassp/sections/draft14_4.tex`](icassp/sections/draft14_4.tex)
* Expanded numerical layer: [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md)
* Paper-to-repository map: [`PAPER_COMPANION.md`](PAPER_COMPANION.md)

The manuscript is self-contained. The repository exists for audit, exact intervals, provenance and reproduction.

## Scientific question

Recovery fine-tuning is commonly summarized by the final score of a compressed checkpoint at one benchmark setting. This project instead measures the paired gain from the pruned checkpoint P to the recovered checkpoint P+FT and asks whether that gain changes with the inference operating point.

The main axes are requested duration, prompt domain and sampler recipe.

## Main findings

1. **Recovery gain depends strongly on requested duration.** At 82.9% pruning, CLAP recovery rises from +0.085 at 3.84 s to +0.244 at 10.24 s. Human-CLAP, KL to matched references and PANNs top-10 capture show the same larger gain at longer durations.
2. **Training duration does not determine the favorable evaluation duration at the matched 20k-step budget.** The same pruned checkpoint was fine-tuned for 20,000 steps at 3.84 s and at 10.24 s. Both checkpoints gain more when evaluated at 10.24 s. The specialization contrast is Delta J = -0.035 [-0.064,-0.004], opposite to the positive shift predicted by training-duration specialization.
3. **The response beyond 10.24 s remains uncertain.** The matched 10.24 to 15.36 s step is +0.021 [-0.023,+0.067]. The paper therefore says that recovery does not clearly increase beyond 10.24 s, not that it plateaus.
4. **Domain transfer is selective.** Recovery transfers strongly to Clotho but is about an order of magnitude smaller on the hip-hop battery. With dense anchors pooled over all 127 hip-hop prompts, the battery remains discriminative and recovery closes about 11 to 12% of the dense gap.
5. **Severity 1 shows prompt-set heterogeneity.** The original 80-prompt set gives J = +0.044 while a disjoint 96-prompt set gives +0.169. The pooled value resolves, but the manuscript reports the two subsets separately rather than treating the pooled magnitude as stable.
6. **A reduced dense 2x2 is a negative diagnostic, not the missing control.** The two 20k-step dense fine-tunes degrade the EMA dense baseline, consistent with the pre-declared raw-weight-versus-EMA issue and limited AudioCaps headroom. The unavailable dense checkpoint after the released million-step recovery remains the principal causal limitation.

## Figure status

Draft 14 intentionally contains **figure placeholders rather than generated graphs**. The final intended footprint is already reserved in the LaTeX source. Detailed comments immediately above Fig. 1 specify the exact data, uncertainty, axes and annotations to render once a data-connected plotting agent produces the final artwork.

The planned figure has two panels.

* Panel (a) shows the released recovery gain over 3.84, 5.12, 7.68 and 10.24 s, plus the matched 96-prompt 10.24 to 15.36 s extension.
* Panel (b) shows the symmetric 20k-step train@3.84 versus train@10.24 intervention and the Delta J contrast that tests specialization directly.

No synthetic values or illustrative curves are used in the placeholder.

## Reproducibility

The scientific report for the second reviewer round is in [`docs/reviewer2_scientific_report.md`](docs/reviewer2_scientific_report.md). The new campaign is backed by the frozen protocol `docs/reviewer2_followup_ext.md` and committed result artifacts including `configs/research/r2_EXT2x2_result.json` and `configs/research/r2_posthoc_pooled_anchors.json`.

The original confirmatory and follow-up artifacts remain under `configs/research/`. Prompt is the statistical unit throughout, with common generation noise for paired system comparisons and prompt-level percentile bootstrap intervals.
