# ICASSP-2027 — Scenario B: Deterministic Mask-Induced Adapter Transfer under Structured Pruning of Text-to-Audio Diffusion

> **Skeleton only (started 2026-08-26, DECISION-V4-11).** Result-independent sections are drafted;
> Results/Conclusion are placeholders and MUST NOT be written before data. Target: ICASSP 2027,
> 4 pages + references. Author/affiliation: Gabriel Bibbó (fill final). This file is a working
> outline in Markdown; port to the ICASSP LaTeX template once the central result exists.

## One-sentence contribution

We show that a dense-trained LoRA adapter can be transferred **without any retraining or adapter
data** onto a structurally pruned AudioLDM backbone by deterministic mask-induced weight slicing, and
we measure whether — and at what pruning/recovery severity — that legacy adapter's benefit survives,
isolating an *excess, adapter-specific* fragility beyond generic capacity loss.

## 1. Introduction (result-independent — DRAFT)

- Text-to-audio latent diffusion (AudioLDM family) is increasingly deployed with **third-party
  parameter-efficient adapters** (LoRA) trained against a specific released backbone.
- Structured pruning changes the backbone's channel geometry, so a legacy adapter trained on the
  dense model is not directly loadable on the pruned one. Practitioners either retrain the adapter
  (needs the adapter's data, often unavailable) or discard it.
- **Question (Scenario B):** can a legacy dense-trained adapter be transferred onto the pruned
  backbone *mechanically* (no retraining, no adapter data), and does its benefit survive?
- We separate two failure modes that are usually conflated: (i) **generic** degradation the pruned
  backbone suffers standalone, vs (ii) **excess** degradation specific to the transferred adapter.
- Contribution list (bullets): deterministic mask-induced transfer operator; a pre-registered,
  prompt-clustered CLAP protocol with a dual non-inferiority + differential-fragility gate; a
  measurement across dense → pruned-only → **published-recovery** backbones; [PLACEHOLDER: headline
  finding — DO NOT WRITE UNTIL DATA].

## 2. Related work (from audited memo §4 — DRAFT)

- **Cross-architecture / cross-model adapter reuse:** X-Adapter (adapters across base models),
  LoRA-X (training-free cross-model LoRA transfer via subspace projection). Contrast: they transfer
  across *different* models; we transfer across a *pruning mask* of the same model, by kept-index.
- **Compression-aware / quantization-aware adapters:** CA-LoRA, SVDQuant, QLoRA-style — adapters
  designed *for* a compressed backbone or absorbing quantization error; quantization preserves tensor
  shapes (no slicing). We address *structured channel pruning*, which changes shapes.
- **Adapter-preserving pruning (ASSP and related):** [PLACEHOLDER: precise positioning].
- **Framing (locked wording):** *"we found no prior work that measures the differential fragility of
  a legacy adapter's benefit under structured pruning of a text-to-audio backbone"* — never "no study
  in any domain". The AudioLDM third-party-adapter ecosystem is small today (n≈1 released family), so
  the ecosystem motivation is framed **prospectively** (cf. the ~196 MusicGen LoRAs; documented
  FLUX-lite / SSD-1B adapter-breakage pain).

## 3. Scenario B — definition (result-independent — DRAFT)

- **Dense backbone** `B0` (AudioLDM-M-Full, channel_mult (1,2,3,5), U-Net 415.955 M params).
- **Legacy adapter** `A`: a LoRA on the dense backbone's attention `to_q`/`to_v` (Kim et al. recipe).
- **Pruned backbone** `B_s` at severity `s`: L1 structured channel pruning (Arshdeep's published
  artifact); (1,2,3,1) = −65 % U-Net params (145.674 M). Pure selection of dense weights (bit-exact,
  never finetuned — verified).
- **Recovered backbone** `B_s^rec`: `B_s` + published 1M-step recovery (Zenodo 21977996).
- **Scenario B transfer** `A ↦ A_s`: slice each LoRA factor to the kept-channel set of `B_s`.

## 4. Deterministic mask-induced adapter transfer (result-independent — DRAFT)

- Kept-set `K_ℓ` per pruned layer ℓ from the published L1 ranking (kept-index vector).
- For a LoRA on a linear/conv weight, the input/output channel dimensions are sliced by the adjacent
  layers' kept-sets; the low-rank factors are indexed accordingly. **Zero learned parameters**, fully
  deterministic, exact at every seam.
- **Verified (CPU, this work):** `test_lora_mask_transfer` 6/6 + `test_scenario_b_seam_transfer`
  S1–S6 (238/238 changed tensors mapped, zero learned parameters; nested-ladder composition exact).
- Figure 1 [TEMPLATE]: schematic of a dense LoRA sliced by the kept-set onto the pruned attention
  block. [placeholder diagram]

## 5. Experimental design (pre-registered — DRAFT; numbers frozen, results blank)

- **Gate 0 (adapter is real on M-Full):** replicate the published LoRA uplift on the DENSE backbone.
  LoRA `to_q`/`to_v`, r8/α16 (paper op-point), **gaussian init**; AdamW lr 1e-5, betas (0.9,0.999),
  weight_decay 1e-5, eps 1e-8, grad-clip 1.0, **FP32**; polynomial LR over the 97 000-step horizon;
  **200 epochs = 19 400 updates** (drop_last=False); 193 four-second clips, **cropped to 3.84 s**
  (near-4-s M-Full transposition; latent_t=96, U-Net divisibility-forced). Primary endpoint **ΔCLAP**
  (LAION-CLAP fused, Kim's scorer) on a **held-out 64-prompt MusicCaps battery**, prompt-clustered
  bootstrap (B=10 000, seed 20260826). SESOI +0.025; PASS = point ≥ SESOI AND lower-CI95 > 0.
- **Phenomenon falsifier (generation only):** severities {dense, p1 pruned-only, p1 recovered} ×
  {backbone, backbone + sliced legacy LoRA}. Dual gate per severity: (i) standalone non-inferiority
  `upper-CI95[E(s)] ≤ 0.025`; (ii) differential fragility `F(s)=D(s)−E(s)`, point ≥ 0.025 AND
  lower-CI95 > 0.
- Guidance `2.5` (Diffusers 0.32.2 AudioLDM default), 50 DDIM steps, 3.84 s generations, 3 seeds.
- Table 1 [TEMPLATE]: per-severity C, ΔCLAP, E, D, F with cluster-bootstrap CIs. [values blank]
- Figure 2 [TEMPLATE — CENTRAL FIGURE]: E(s) vs F(s) across severities, with the non-inferiority and
  fragility thresholds drawn. [blank until data]

## 6. Results — [PLACEHOLDER — DO NOT WRITE BEFORE DATA]

## 7. Discussion / Conclusion — [PLACEHOLDER — DO NOT WRITE BEFORE DATA]

## 8. Limitations (known now — DRAFT)

- **One adapter family.** The legacy adapter is a single published recipe (Kim et al.) on one
  backbone family (AudioLDM-M-Full). Ecosystem generality is argued prospectively, not shown.
- **Adaptation data provenance.** Kim's "193 training clips" are **193 overlapping 4-s windows drawn
  from only 44 source recordings** (MusicCaps hip-hop; captions = MusicCaps caption + subgenre
  suffix). The effective adaptation diversity is ~44 songs. Our held-out battery is disjoint at the
  **source-ytid** level (44 recovered ytids excluded), not merely by caption similarity.
- **Backbone transposition.** Kim's recipe is on S-Full-v2 (diffusers); we transpose it to M-Full
  (audioldm_train) — a near-4-s (3.84 s) reproduction of a published operating point, **not an exact
  replication** of the S-Full headline.
- **Recovered-backbone caveat.** The sliced LoRA lands on the recovered backbone's kept-set but
  **recovered (non-selection) weights**; reported honestly.
- **Statistical scope.** Claim is standalone **text-audio semantic alignment (CLAP)**; FAD/KAD are
  corroborative only and cannot rescue a failed CLAP gate.

## 9. Reproducibility (DRAFT)

- All numbers traceable to Git commit, resolved config (`configs/research/icassp_gate0_prereg.yaml`
  v3), checkpoint hash, dataset manifest sha256, seeds, raw output, and GPU metadata.
- Frozen artifacts: battery `icassp_gate0_battery.json` (sha `ba4ebb50…`), source-ytid exclusion set,
  Kim-clip 3.84-s manifest, adapter checkpoint + meta.
