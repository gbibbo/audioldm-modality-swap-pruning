# Fourth ICASSP review: camera-ready response map

1. The 20k recipe is now specified in the manuscript as full-U-Net AdamW with constant LR 1e-4, default betas, weight decay 0.01, no scheduler, effective batch 2. No smaller-LR dense run exists. The text therefore treats excessive LR for the already converged dense model as an untested hypothesis and describes Delta J_dense as measurable but not a clean recovery analogue in the degraded regime.
2. Results explicitly state that Human-CLAP is essentially saturated by 7.68 s, increasing only from 0.371 to 0.375 by 10.24 s.
3. The severity-1 heterogeneity paragraph restores the selection difference. The second outcome-blind draw used a different seeded-hash salt and a five-caption-row rule.
4. The dense Results and Discussion paragraphs were compressed while retaining the raw-vs-EMA control and the recipe-specific conclusion.
5. The paper citation now points to the coherent `paper-operating-point-recovery/` landing subdirectory. For regular camera-ready submission author identification is intentional.
