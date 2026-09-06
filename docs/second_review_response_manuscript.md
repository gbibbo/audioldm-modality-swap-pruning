# Second-review manuscript response map

This document records how Draft 14 addresses the second ICASSP review. It is a response aid, not paper prose.

## 1. Symmetric training-duration control

The manuscript now uses the matched 20,000-step train@3.84 versus train@10.24 intervention as the direct specialization test. The key contrast is Delta J = -0.035 [-0.064,-0.004]. The abstract no longer treats the earlier one-arm test as conclusive. The current claim is limited to the matched short-budget intervention.

## 2. Dense 2x2

The dense 2x2 is reported as a negative diagnostic. Both dense fine-tunes degrade the EMA dense baseline, so this experiment does not replace the missing million-step dense recovery checkpoint. The paper retains that limitation explicitly.

## 3. Revision chronology removed

Draft 14 contains no references to a previous title, revised experiment, earlier reviewer or rebuttal chronology. Analysis status is described only as experimental design.

## 4. 15.36 s wording

The paper replaces plateau language with the supported statement that recovery does not clearly increase beyond 10.24 s. The matched step remains +0.021 [-0.023,+0.067].

## 5. Severity-1 heterogeneity

The original n=80 and new disjoint n=96 subsets are both reported, together with their resolved difference. The pooled n=176 value is interpreted only as evidence that the interaction can resolve at severity 1.

## 6. PANNs, references and full hip-hop anchors

PANNs top-10 capture is defined in Sec. 3.3. Unused bibliography entries were removed. Hip-hop dense anchors and rho_dense now use all 127 prompts, replacing the earlier 64-prompt footnote.

## 7. Short operating-point hypothesis

Sec. 5 formulates a weaker adaptation-gain expression hypothesis and explicitly rejects the stronger statement that the dense base model is simply broken at 3.84 s under CLAP. The dense floor-corrected duration response is compared with the real-audio response. The underlying mechanism remains open.

## Presentation

The long replication-instructions paragraphs from Draft 13 are removed. The main figure is now a two-panel placeholder with a fixed footprint and detailed generation instructions embedded as TeX comments. The robustness table remains in the paper.
