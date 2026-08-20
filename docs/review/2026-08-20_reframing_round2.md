# Hostile review — round 2 (2026-08-20, Reviewer B reply)

Companion to `2026-08-20_reframing_round1.md`. Output of this round:
`docs/master_plan_v4_draft.md` (draft, pending Gabriel's decisions).

## 1. Reviewer A's round-2 claims — verification status

| Claim | Status | Evidence |
|---|---|---|
| Importance-Aware OBS = arXiv 2607.20048 (22 Jul 2026); CFG-response importance maps `M_t = |ε(x_t,c) − ε(x_t)|`; structured + unstructured; SD3-Medium, PixArt-Σ; §4.4 category-targeted calibration (Woman/Cat/Airplane/Motorcycle, 60 % sparsity); plain OBS-Diff with targeted calibration also improves the target (MUSIQ 64.35→69.81; theirs 71.64); **T2I only**, limitations say other conditional tasks are future work | ✔ verified | WebFetch of the HTML |
| 27 Jul paper = arXiv 2607.24731 (Li et al.), on-policy distillation for dense→sparse video control; Positive-Direction Matching; no pruning, no per-concept analysis | ✔ verified | abstract |
| ELSA = arXiv 2606.17404, Interspeech 2026; **GPT-5.2** decomposition + **SAM Audio** + **Human-CLAP**; +13.1 Kendall τ over CLAPScore on AudioCaps | ✔ verified; **code release unconfirmed**, API-dependent | HTML |
| AudioCaps exposure gradient (Speech 41.5 %, Siren 1.8 %, Gunshot 1.7 %, Drill 1.5 %, Explosion 0.47 %) | ✔ **reproduced in-repo** from `data/dataset/metadata/audiocaps/datafiles/audiocaps_train_label.json` (49 502 labelled clips; all values within 0.05 pp of A's) | this session |
| Arshdeep Table 3 monotonic (unpruned count vs % loss, 6 families, ρ = −1) | ✔ verified against the HTML table | WebFetch |

Two facts A did not use:

* **"Music" has 0 clips in AudioCaps train** (curated out), yet in the M4 screening PANNs
  top-1 = "Music" on 9 base clips and **24 P3 clips**. Pruned models drift toward a class
  the fine-tuning distribution does not contain — a cheap, already-available signature of
  "forgetting toward a generic prior" worth a pre-registered look.
* Arshdeep's **fine-tuned pruned model has FAD 1.57 vs 3.95 for the unpruned model**.
  1M AudioCaps steps is domain adaptation as much as recovery. RQ3′ must not call the
  full-FT number "recovery" without that caveat, and ideally includes an unpruned model
  fine-tuned with the same PEFT budget as a control (costed as optional).

## 2. Response to A's five mandatory changes

1. **Event-level unit, mixed-effects model — ACCEPTED.** Unit = (clip, requested event)
   occurrence; requested events = AudioSet labels of the source clip whose display name
   (or a pre-registered synonym list) appears in the caption — this also answers A's
   construct-validity objection (AudioSet labels ≠ caption content). Family stays as a
   grouping level, never the regression unit. Minimum support per event pre-registered.
2. **"Rarity" = AudioCaps exposure — ACCEPTED, with a second covariate.** Primary:
   log AudioCaps-train clip count (reproducible in-repo, and it is the distribution of
   the calibration set and of Arshdeep's fine-tuning). Secondary: log AudioSet-unbalanced
   count (needs the 2M-row CSV; closer to pretraining exposure). Both pre-registered;
   if they disagree, that is itself informative.
3. **Gate B′ needs a null distribution — ACCEPTED, and made free.** Taylor saliency is a
   sum over slots of `|g·∂L/∂g|`; storing the **per-slot contributions** once (28 layers ×
   ≤960 channels × slots, ~0.3 GB) turns every subset/reweighting — natural halves,
   tail-enriched, guidance-weighted — into a CPU sum. One GPU run on an enriched pool
   (natural 256 + tail-enriched 256) yields thousands of null splits at zero extra GPU.
4. **RQ2′ is a mechanism-informed intervention, not a method — ACCEPTED.** Explicit
   positioning: targeted calibration is Importance-Aware OBS §4.4's principle; our
   question is whether the *mechanism identified in audio* (exposure vs guidance vs
   acoustics) tells you **which** calibration to target, and whether it works for
   structured channel pruning of a TTA U-Net at fixed compute. If the intervention merely
   replicates §4.4 in audio, we say so.
5. **P0-standard as primary scientific baseline — AGREED, but it is Gabriel's decision.**
   DECISION-M3B-002/003 currently say the opposite. Listed as DECISION-V4-01 in the draft.
   Note: the M3A/M3B runs used the published (inverted) model as "the" L1 model; under
   v4 the event-level runs must use P0-standard as the primary pruned system.

## 3. Where I still push back

* **"Two convergent negatives are real evidence" — partly.** Gate B's negative is
  genuine (the instrument separates Taylor from L1 at ρ=0.57). Gate A's negative is
  genuine *for the statistic as defined*, but `R_mod` is unsigned and the regime is
  saturated (D_mod ≳ the full model's whole conditional response). D1 (signed asymmetry
  + per-stratum) stays mandatory before the paper words RQ1-swap as "not supported"
  rather than "not detectable".
* **ELSA as a headline metric — not yet.** GPT-5.2 + SAM Audio + Human-CLAP is an
  API-bound, three-model stack with unconfirmed code; for a reproducibility-first project
  it is a secondary metric at most, budgeted separately. Primary event metric: PANNs
  top-10 recall per requested event (Arshdeep-compatible); secondary local metric:
  CLAP score of the event phrase vs the generated clip.
* **The CFG grid (H-guidance test) is the most expensive item.** Generation costs 8.44
  s/clip at S=50; a 3-CFG × 2-system × 300-prompt grid is ~4 GPU-h at one seed. It goes
  in Tier 1, not Tier 0. Tier 0 tests H-guidance only through the forward-only
  guidance-gap damage per event (D2), which is ~15 min.
* **D3 is demoted** to "exploratory, CPU-only, unrecorded in the claims matrix", as A
  asked. The Music-drift observation comes from it and is labelled the same way.

## 4. The binding constraint A has not seen: credits

Total job spend to date **4.205 credits** (SDK, 14 jobs); CG-001 balance was ~9.6 with a
2.0 reserve. Remaining spendable is of the order of **3–5 credits ≈ 3–5 T4 GPU-hours**,
and the v3 submission target is **2026-09-16**. The v4 draft is therefore written in
**credit tiers**: Tier 0 (≤ 3 credits) closes the v3 negatives and runs the event-level
screening + Gate B′; Tier 1 (~15–20 credits) is the confirmatory RQ1′/RQ2′ paper;
Tier 2 (~30–50 credits) adds PEFT recovery. Gabriel decides the tier; the plan must not
pretend Tier 2 is funded.

## 5. Questions for Reviewer A, round 3 (only if they change a design choice)

1. Any TTA or audio work using **framewise PANNs / SED outputs** to define per-event
   temporal occupancy in generated audio (needed for H-acoustic covariates; the
   `seg_label` .npy files referenced by the upstream manifest are **not on disk**)?
2. Is there a published **synonym map from AudioSet display names to caption vocabulary**
   (for the requested-event filter), or must we pre-register our own?
3. Does any 2025–2026 work report **class drift toward a generic class** ("Music",
   "Speech") under generative-model compression — i.e., a prior-collapse signature?
