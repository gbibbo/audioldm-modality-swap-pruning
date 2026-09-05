# ICASSP reviewer response map

This document records how the manuscript was revised after the ICASSP review and the pre-specified follow-up experiments in `docs/reviewer2_followup.md`.

## 1. Missing matched dense fine-tune and over-strong specialization framing

**Reviewer concern.** Without the dense model receiving the same recovery fine-tune, the paper cannot attribute the effect to pruning. The old title implied specialization to the recovery training point.

**Action.** The title and abstract were reframed around the directly supported claim, operating-point dependence of recovery gain. The manuscript now treats pruning as the context in which the released checkpoints arise, not as the identified cause of the duration interaction.

**New evidence.** A direct intervention fine-tuned the pruned checkpoint at 3.84 s for 20,000 steps. Its gain remained larger at 10.24 s, with J = +0.065 [+0.043, +0.087]. A public dense text-fine-tuned reference shows the same direction, J = +0.113 [+0.051, +0.173]. The 15.36 s follow-up shows a plateau rather than a peak at 10.24 s.

**Manuscript change.** Secs. 1, 3.2, 4.2, 5, and the title and abstract.

## 2. Hip-hop may be floor-limited

**Reviewer concern.** Without dense and real anchors, a near-floor P+FT value could indicate that the battery cannot resolve alignment for any model.

**Action.** Dense anchors are now reported for the hip-hop battery. The dense model is above the shuffled-caption floor at both severity-2 durations, including +0.106 [+0.079, +0.135] at 10.24 s.

**Interpretation.** The battery has measurable headroom. Recovery closes only about 2% to 4% of the dense gap in the original anchored cells, so the weak hip-hop transfer is not a simple floor artifact.

**Manuscript change.** Table 1 and Sec. 4.3.

## 3. Only one extreme domain shift

**Reviewer concern.** Hip-hop changes both sound content and caption style. Clotho is a more informative held-out control.

**Action.** Added a 96-prompt Clotho battery with P, P+FT, dense, real-audio, and chance anchors.

**New evidence.** Recovery transfers strongly to Clotho. At 10.24 s, R = +0.210 [+0.176, +0.243], rho_real = 0.59, and rho_dense = 0.74. The AudioCaps-minus-Clotho contrast is unresolved at +0.032 [−0.023, +0.088].

**Interpretation.** The paper no longer claims general failure of out-of-domain recovery. It reports strong Clotho transfer and much weaker hip-hop transfer.

**Manuscript change.** Secs. 3.1 and 4.3 and Table 1.

## 4. Sweep stops at the native duration and short clips may be globally broken

**Reviewer concern.** The original sweep could not distinguish a peak at the fine-tuning duration from a general longer-is-easier effect.

**Action.** Added a 15.36 s point and a direct 3.84 s fine-tuning intervention.

**New evidence.** R is +0.264 [+0.216, +0.310] at 15.36 s, while the matched 10.24 s value is +0.242 [+0.198, +0.285]. The step is unresolved at +0.021 [−0.023, +0.067], consistent with a plateau. Fine-tuning at 3.84 s does not move the maximum gain to 3.84 s.

**Interpretation.** The manuscript explicitly rejects training-duration specialization as the explanation. The existing crop analysis remains important because it shows that the short-duration difference arises during generation rather than from the CLAP scoring window.

**Manuscript change.** Fig. 1 and Secs. 4.1, 4.2, and 5.

## 5. Too much robustness evidence lives only in GitHub

**Reviewer concern.** The four duration values and alternative scorers should be visible in the paper.

**Action.** Added a compact full-width table with all four durations for CLAP, Human-CLAP, KL recovery, and PANNs top-10 capture, including 95% intervals and the native-minus-short contrast.

**Manuscript change.** Table 1.

## 6. Modest sample sizes and informal listening

**Reviewer concern.** Severity 1 was underpowered and the hip-hop battery had 64 prompts. Author listening should remain a sanity check.

**Action.** Added 96 severity-1 prompts and 63 hip-hop prompts. The pooled severity-1 interaction is +0.112 [+0.076, +0.149] at n = 176. The hip-hop battery at n = 127 resolves a small positive gain at both durations. The manuscript describes the author listening only as descriptive and does not use it to support an inferential claim.

**Manuscript change.** Secs. 4.1, 4.3, and 5.

## 7. Wording, figure density, and anonymity

**Reviewer concern.** The abstract should not say "no gain," the original figure was dense, and a repository URL could be problematic under double-blind review.

**Action.** The abstract now reports the quantitative contrast without claiming zero hip-hop gain. The main visualization is split into two simpler panels, and the robustness evidence is moved into a readable table. The expanded hip-hop battery shows a small positive gain, so the paper uses "much smaller" rather than "none."

ICASSP 2027's official Paper Kit states that the regular review process is not double blind and instructs authors to include the author list in submitted manuscripts. The repository citation therefore does not violate the regular ICASSP 2027 anonymity policy. The repository URL is nevertheless kept concise and nonessential to understanding the paper.
