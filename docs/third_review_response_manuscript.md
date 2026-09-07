# Third ICASSP review: manuscript response map

This file maps the third review to Draft 15. It is not manuscript prose.

1. **Dense 2x2 and raw-vs-EMA.** Evaluated the dense raw baseline on the matched 192 prompts at both durations. Raw and EMA differ by at most about 0.02 CLAP, so the approximately 0.2 dense degradation is genuine under the 20k recipe rather than a convention artifact. Draft 15 reports this in Secs. 3.2, 4.2 and 5. The missing dense 10^6-step control remains a limitation.
2. **Public dense text-FT reference.** Restored its numerical result to Sec. 4.2 so it no longer appears only in Discussion.
3. **Severity-2 prompt sensitivity.** Added the post-hoc outcome-blind split-half analysis. The two n=96 halves give J=0.159 and 0.160 with difference 0.001 [-0.056,+0.057]. Draft 15 distinguishes independent prompt evidence from sampler/scorer robustness on reused prompts.
4. **Hip-hop duration interaction.** Added J_music=0.001 [-0.026,+0.028] and connected the duration and domain axes. Clotho is intermediate, while hip-hop has weak recovery and no resolved duration interaction.
5. **Short-generation wording.** Replaced any broad claim that short generation is healthy with the narrower statement that, under CLAP, base-model degradation does not account for the interaction. The paper explicitly notes that this is not perceptual evidence.
6. **Minor points.** Added an explanation of uncertainty in rho_dense, retained the two-column figure footprint, corrected the primary J CI upper bound from 0.188 to 0.187, and verified a five-page build with page 5 containing references only.

Figure policy remains unchanged from Draft 14: no synthetic graphs are generated. The final plotting specification is embedded in `icassp/sections/draft15_2.tex`.
