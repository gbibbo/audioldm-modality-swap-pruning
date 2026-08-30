# RECOVERY-METRIC-AUDIT-1 — literature note (primary-source facts)

Compact note for eventual paper writing. Only facts that materially affect the metric-concordance
interpretation. Facts marked **[repo-verified]** were established by the prior zero-GPU audit
(`docs/review/2026-08-27_recovery_reversal_audit.md`, ledger RECOVERY-REVERSAL-AUDIT-1) against the
primary sources; facts marked **[to-confirm]** still need a direct primary-source check before use in
a submission. No novelty claim is made here.

## Singh et al. 2026 — "Efficient Text-to-Audio Generation via Pruning" (arXiv 2607.13330, as cited in-repo)

* **[repo-verified] Recovery is reported with FAD and KL.** Unpruned M-Full **FAD 3.95 / KL 2.16**;
  the `(1,2,3,1)` budget **after fine-tuning** reaches **FAD 1.57 / KL 1.678**. These appear in the
  **Section-5 running text, NOT Table 3** (a prior in-repo mis-attribution of "FAD 1.57" to Table 3 was
  corrected).
* **[repo-verified] Table 3 is the per-event PANNs loss/recovery analysis** (event families incl.
  "safety-critical events"), i.e. the semantic-event-capture axis. The exact AudioSet-mid → family
  mapping is **not reproduced in this repo**, so the family-level analysis is **omitted** from this
  audit (no invented families).
* **[repo-verified] PANN top-10 event-capture methodology** underlies the event analysis (the machinery
  reproduced in `scripts/research/panns_topk.py`).
* **[repo-verified] Evaluation protocol:** AudioCaps test (**964 pairs**), **10 s** clips,
  **200 inference steps**. Pre-finetuning pruned absolute FAD/KL are **not** in the paper (only Fig. 3
  deltas). → This audit's absolute FAD/KL (3.84 s / 50 steps / n=96) are therefore **not** comparable to
  1.57 / 1.678; only within-audit ordering is read.
* **[repo-verified] Materially important confound:** the fine-tuned model (**FAD 1.57**) beats the
  authors' **own unpruned** model (**FAD 3.95**). So the 1M-step fine-tune is **not pure post-pruning
  recovery — it is also AudioCaps domain adaptation.** Any "recovers most of the loss" reading is
  entangled with in-domain adaptation. This is central to interpretation branch A.
* **[repo-verified]** The recovered full-FT `(1,2,3,1)` checkpoint is **not public**; our `recovered`
  system is the released fine-tuned AudioLDM-M-Full checkpoint used as the published reference.

## AudioLDM 2023 (Liu et al.)

* **[to-confirm] Authors' own caution:** AudioCaps fine-tuning improves several AudioCaps metrics, but
  performance on that limited/similar distribution does not necessarily imply better overall
  generalization. Direction is consistent with the Singh 3.95→1.57 evidence above (in-domain adaptation
  gains). Confirm the exact wording/section against the AudioLDM paper before quoting in a submission.

## Human-CLAP

* **[to-confirm, limitation only]** Conventional CLAPScore has imperfect correspondence with subjective
  alignment, which motivates human-aligned CLAP variants (e.g. sarulab-speech human-clap-wsce-mae, the
  V1.1 secondary scorer). **Human-CLAP is a CLAP-family model, NOT actual human evaluation** — it must
  never be reported as human listening results.

## Why this matters for the audit

If the authors' metrics (FAD↓, KL↓, capture↑) favor `recovered` while CLAP does not (branch A), the
Singh 3.95→1.57 confound sharpens the thesis: post-pruning "recovery" is **evaluation-axis-dependent**,
and the distributional/event gains are consistent with **in-domain AudioCaps adaptation** rather than a
restoration of general text–audio alignment — a multidimensional-recovery framing, **never** "the paper
is wrong". This connects to the broader TTA-evaluation problem (FAD/CLAP as non-substitutable axes; see
the reliability literature catalogued in the 2026-08-27 audit: Gui et al. ICASSP'24, KAD ICML'25, FAD
encoder-bias work) — cited only to support axis non-substitutability, not to claim "FAD is bad".
