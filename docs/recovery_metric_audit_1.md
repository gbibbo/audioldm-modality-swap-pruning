# RECOVERY-METRIC-AUDIT-1 — Post-hoc metric-concordance diagnostic

**STATUS: POST-HOC DIAGNOSTIC / OUTCOME-MOTIVATED. NOT preregistered confirmation.**

## 0. Provenance and scientific status (binding preamble)

* RECOVERY-REVERSAL-V1.1 has **already been observed** and was a **pre-registered NEGATIVE**
  (PRIMARY CLAP `PASS = FALSE`, `R_AC = -0.0024 [-0.0267, 0.0214]`, commit `cc1b0cf`).
* This diagnostic was **motivated by that result** (recovered ≈ pruned in-domain under CLAP, yet
  recovered collapses on the music battery). It is therefore **outcome-motivated / post-hoc** and is
  **NOT** a preregistered or confirmatory test.
* **Purpose:** mechanism / evaluation-axis diagnosis and hypothesis generation — specifically, whether
  the metrics used by Singh et al. / AudioLDM to *call the checkpoint "recovered"* (FAD, KL, PANN event
  capture) tell a different story than text–audio alignment (CLAP) on the **same** V1.1 outputs.
* **V1.1 `PASS = FALSE` cannot be changed by anything in this diagnostic.** Nothing here rewrites any
  V1 / V1.1 artifact, manifest, claim, or verdict. This is a *separate* diagnostic record. Terminology:
  **POST-HOC METRIC-CONCORDANCE DIAGNOSTIC** — never "V1.1 extension / rescue / confirmatory".
* There is **no gate, no PASS/FAIL, no SESOI, no composite score, no majority vote** in this audit.
  The deliverable is a faithful metric-concordance table plus interpretation.

## 1. Inputs (frozen; NO new audio, NO new selection)

* **Generated audio:** the existing V1.1 set only — 3 systems × 96 ytids × 2 replicates = **576 WAVs**,
  16 kHz / 3.84 s, at
  `/teamspace/jobs/reversal-v11-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_v1_1_gen`
  (names `{dense_noadapter | p1_pruned_ema_reconstructed_noadapter | p1_recovered_noadapter}_p{0..95}_r{0,1}.wav`).
  Systems: `dense`, `pruned`, `recovered`. Job `reversal-v11-gen-1` @git `5f2fa55`.
* **Reference audio:** the real AudioCaps-test clip for each of the 96 V1.1 ytids, `Y<ytid>.wav` under
  `data/dataset/audioset/zip_audios/**` (96/96 located).
* **GT event labels:** `data/dataset/metadata/audiocaps/datafiles/audiocaps_test_label.json`
  (sha256 `6b09d3fd…`), field `labels` = comma-separated AudioSet mids. Verified: all 96 ytids present,
  and the label-string is **identical across the ~5 caption rows per ytid** (0 ambiguous) → one
  unambiguous GT event set per ytid.
* **Manifest:** `configs/research/reversal_v1_1_audiocaps_manifest.json` (file sha256 `83f50bef…`) —
  the frozen 96-ytid order and chosen captions. Unchanged.

## 2. Evaluator provenance (exact, reproducible match)

All PANN-based metrics use the **same** classifier and preprocessing for every system and the reference,
per the `audioldm_eval` warning that FD/KL are resampling-sensitive.

| Component | Exact spec |
| --- | --- |
| Classifier | PANNs **Cnn14 16 kHz**, `ckpt/Cnn14_16k_mAP=0.438.pth` (sha256 `ba238e60…`) |
| Cnn14 params | sr 16000, window 512, hop 160, mel 64, fmin 50, fmax 8000, classes_num 527 |
| Preprocessing (all clips) | `audioldm_eval` WaveDataset: torchaudio.load → channel 0 → subtract mean (DC) → resample→16 kHz if needed → pad to ≥ 32000 samples; **full clip length, no truncation** |
| Cnn14 outputs used | `clipwise_output = sigmoid(logits)` (capture), `logits` (KL), `2048` embedding (FD) |
| Label mapping | `class_labels_indices.csv` (sha256 `cdd10498…`): AudioSet mid ↔ 527-class index |
| Library | `audioldm_eval` (pinned in `.venv`; import banner "2023-06-22"), torch 1.13.1+cu117 (`.venv`) |
| Device | CPU (Studio has no CUDA); PANN + VGGish inference is CPU-tractable |

**Deviations from Singh et al. — labelled precisely, NOT hidden.** Singh reports FAD 1.57 / KL 1.678 at
**10 s / 200 DDIM steps** over the **full** AudioCaps-test set. This audit runs on the frozen V1.1
outputs: **3.84 s / DDIM 50 / guidance 2.5 / single generation / n = 96 ytids**, the *same* controlled
operating point as the music arm. Therefore **absolute** FAD/KL here are NOT comparable to Singh's
published values; only the **within-audit ordering** (dense vs pruned vs recovered) and **cross-metric
concordance** are interpreted. Gen clips (3.84 s) and reference clips (up to 10 s) differ in length —
an inherent consequence of the controlled operating point, applied identically to all systems.

## 3. Metrics and exact computation

Replicate/cluster convention (all per-ytid metrics): compute the metric separately for replicate 0 and
replicate 1, **average the two replicate-level values within ytid**, treat the **96 ytids as the
independent bootstrap clusters**. Prompt-clustered percentile bootstrap, B = 10000, seed **20260828**
(new namespace; distinct from V1 20260826/20260827). **No significance gate.**

### 3a. PANN event capture (highest priority)
For generated clip *i*: `P_i` = Cnn14 **top-10** predicted AudioSet classes (by `clipwise_output`);
`G_i` = GT AudioSet event classes for that ytid (from `labels` mids → indices). Singh recall:

```
Recall = sum_i |G_i ∩ P_i| / sum_i |G_i|
```

Report `capture_dense/pruned/recovered`, paired contrasts `Δ_rec−pruned`, `Δ_rec−dense`,
`Δ_pruned−dense` with clustered-bootstrap CIs, and the **fraction of ytids where recovered captures
more GT events than pruned**. Event-family analysis is **descriptive only** and included **only if** the
exact family mapping is recoverable from the paper/code; otherwise omitted (no invented families). At
n = 96 report counts/denominators per family; do not overinterpret sparse groups.

### 3b. KL divergence (lower = better)
Exact `audioldm_eval.metrics.kl.calculate_kl` formula on Cnn14 `logits`:
`KL(gen‖gt) = mean_over_classes( KL_div( log(softmax(gen_logits)+1e-6), softmax(gt_logits) ) )`, per
(gen clip, its ytid reference) pair. Also report the sigmoid variant descriptively. Per-ytid: average
the two replicates' KL against the **same** reference, bootstrap 96 ytids. Report `KL_dense/pruned/
recovered` and `Δ_KL_rec−pruned` with CI. My per-example implementation is **validated against
`audioldm_eval`'s aggregate KL** on a matched set before use.

### 3c. FAD (VGGish; lower = better) — statistically cautious
VGGish Fréchet Audio Distance (`audioldm_eval.metrics.fad`, torch.hub `harritaylor/torchvggish`),
distributional over a system's clip set vs the 96 real references. Compute **replicate-specific**
`FAD_r0` (96 gen-r0 vs 96 ref) and `FAD_r1` (96 gen-r1 vs 96 ref) per system; report both and their
mean **descriptively**. A ytid-block bootstrap contrast is reported **only if** VGGish per-clip
embeddings support a defensible CI at n = 96; **if FAD is too unstable at n = 96, say so plainly and
keep FAD descriptive.** Never fabricate a 192-item reference by duplicating clips. FD (PANN-2048
Fréchet) reported alongside as a secondary distributional read.

## 4. Concordance table (central output)

One table: rows = {primary CLAP, Human-CLAP, PANN capture, KL, FAD (+ FD)}; columns = direction,
dense, pruned, recovered, rec−pruned contrast, CI/status. CLAP and Human-CLAP rows reuse the existing
frozen V1.1 numbers. **No composite, no majority vote** — the object is concordance/discordance.

## 5. Interpretation branches (guides, NOT gates)

* **A — authors' metrics favor recovered, CLAP does not** (FAD↓, KL↓, capture↑ for recovered, but
  CLAP rec≈pruned): candidate thesis — *post-pruning recovery is evaluation-axis-dependent; finetuning
  restores distributional/event metrics without restoring text–audio alignment, with a much larger
  semantic penalty under music-domain shift.* Phrase as multidimensional "recovery", **never** "the
  paper is wrong".
* **B — authors' metrics also show no recovered advantage:** metric mismatch is insufficient; next
  candidate is **operating-point / inference-recipe dependence** (Singh 200 steps / 10 s vs V1.1
  50 steps / 3.84 s). **Do NOT test the alternate operating point in this audit — return first.**
* **C — metrics disagree among themselves:** report the discordance faithfully; do not cherry-pick a
  "recovery"-supporting metric.

## 6. Budget

CPU first. `MAX NEW GPU SPEND FOR METRIC AUDIT = 0.20 cr` (hard). No generation. If any step would need
> 0.20 cr GPU, STOP and report. Record starting `total_spent` before any GPU launch.

## 7. Literature note

A compact primary-source note (`docs/recovery_metric_audit_1_literature.md`) captures only facts that
materially affect interpretation: Singh et al. 2026 (FAD/KL as recovery metrics, 200 steps, (1,2,3,1)
FAD 1.57/KL 1.678, PANN top-10 capture methodology, finetuning-recovers claim); AudioLDM 2023 (authors'
own caution that AudioCaps-finetuning gains on that similar distribution need not imply better
generalization); Human-CLAP (conventional CLAPScore imperfectly matches subjective alignment — a
CLAP-family limitation, **not** actual human evaluation). No novelty claim.

## 8. Freeze

This protocol is frozen and committed **before** any metric is computed. Downstream results land in
`configs/research/recovery_metric_audit_1_result.json` and ledger `RECOVERY-METRIC-AUDIT-1-RESULT`;
they do not modify this protocol or any V1/V1.1 artifact.
