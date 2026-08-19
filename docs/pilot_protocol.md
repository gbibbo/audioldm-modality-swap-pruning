# M3 Pilot Protocol

**Status: BORRADOR — PENDIENTE DE REVISIÓN. NOT FROZEN.** M3 is blocked until this
file is reviewed against `docs/master_plan_v3.md`, completed, and committed
**before any saliency or D_gen/D_mod/R_mod result on the real pruned checkpoint is
inspected**. Every numeric value below is a **reasoned proposal**, provisional until
(a) the first GPU benchmark populates `docs/compute_budget.md` (so `T_sal`/`T_fwd`
are known) and (b) Compute Gate CG is resolved. Do not treat this draft as a
pre-registration. `Freeze commit`/`Freeze timestamp` are intentionally left blank.

## Machinery status (this unit, M3-000)

The M3A machinery exists and is tested on CONTROL models only (no result was
computed on `l1_audioldm-m-full_p1.ckpt`):

* Real noised latent `z_t = sqrt(a_t) z_0 + sqrt(1-a_t) eps`, `z_0 =
  scale_factor * VAE.encode(mel).mode()`, `scale_factor = 0.9138255715370178`
  read from the checkpoint. Code: `research_pruning/diagnostics/conditioning.py`.
* Diagnostics `D_gen`, `D_mod`, `R_mod`: `research_pruning/diagnostics/modality_diagnostics.py`.
* Matched random null (20 masks) + Gate A statistic:
  `research_pruning/diagnostics/random_masks.py`, `matched_null.py`.
* Tests PASS (control models / synthetic data only, L1 ckpt never opened):
  `test_diagnostics.py` D1..D5, `test_random_masks.py` R1..R4,
  `test_matched_null.py` N1..N4.

## Frozen diagnostic conventions (proposed)

* **Norm:** L2 (Euclidean) over the flattened per-example epsilon-error latent
  (one scalar per example over C·H·W). Rationale: the diffusion loss is L2 on
  epsilon; makes `R_mod` scale-free and bounded in [0,1] by the triangle
  inequality.
* **epsilon in R_mod denominator:** `1e-12`. Rationale: only prevents division by
  zero when both errors vanish (identity → exactly 0); ~12 orders of magnitude
  below any real error norm (O(1e-1..1e2)).
* **Timestep aggregation (master plan §6):** equal-weight mean across
  pre-registered timestep strata, then mean across examples. Timestep-specific
  curves are secondary only.

## Calibration budget (master plan §5) — B1 resolved

One gradient-evaluation unit = one forward+backward saliency evaluation for one
modality at one pre-specified `(example, noise, timestep)` **slot**.

**`B` counts SLOTS, not examples** (the §5 unit is the slot). A slot is one
`(example, noise, timestep)` triple. The number of base examples is `E` and the
number of timestep strata is `K`, with one timestep drawn per stratum per example,
so:

    B (base slots) = E * K

**PROPOSAL: `E = 256` base examples, `K = 5` strata ⇒ B = 1280 base slots.**
Rationale: 256 examples stabilise per-layer saliency across the 28 ranked conv
layers; 5 strata cover the noise range (see timestep section). **Provisional**
until `T_sal` is measured on GPU; revise `E` so total calibration GPU-hours fit the
~US$50 budget (keep `E` a multiple of nothing in particular, but keep `E·K` the
same across P0–P3).

Gradient-evaluation arithmetic (a reviewer can sum the last column):

| Criterion | what is evaluated | per stratum (×K=5) | total gradient evals |
|---|---|---|---|
| P1 text-only Taylor | 2 text draws per (example, stratum) | 2·E = 512 text | **2B = 2560** |
| P2 paired mean | 1 text + 1 audio per (example, stratum) | E text + E audio = 256+256 | **2B = 2560** |
| P3 paired max | (shares P2's `S_a`, `S_t`) | — reuses P2 — | **2B = 2560** |

* P1 == P2 == P3 == **2560** total gradient evaluations. ✔ matched budget.
* P0 (L1) is data-free: **0** gradient evaluations (its cost advantage stays visible).
* P2 and P3 share the same computed `S_a`, `S_t`; no duplicate calibration compute.
* Report per criterion: calibration GPU-hours, peak VRAM, #forward/backward evals,
  wall time, examples used.

### P1 fairness under stratification — B2 resolved

P1's `2B` text evaluations must cover the **same strata with the same weights** as
P2/P3, or P1's timestep distribution could handicap the mandatory baseline and any
cross-modal advantage would be a design artefact (RQ2 would collapse). Rule:

* P2/P3 spend `2E` units per stratum (`E` text + `E` audio). P1 spends `2E` **text**
  units per stratum — **two pre-registered `(noise, timestep)` draws per
  (example, stratum)**, the timesteps drawn from that stratum.
* Therefore every criterion spends exactly `2E = 512` units per stratum across all
  `K = 5` strata, and P1's text-timestep distribution is proportional to P2/P3's.
  The two P1 draws per (example, stratum) are the "two pre-registered draws per base
  example" of §5, refined to fall inside the stratum.

## Slot construction (PROPOSAL)

* **Example pool:** AudioCaps **train** manifest
  (`datafiles/audiocaps_train_label.json`, 49 502 items), which is disjoint from
  the test set by wav id (verified: `train ∩ test = 0`). Calibration never draws
  from the test set or from the reserved validation split.
* Draw `E` base examples deterministically (sort wav ids, seeded permutation,
  master seed `20260818`).
* **Caption-selection rule (B3):** where a wav has multiple captions, use the
  **first caption in source-file order** (`dict.setdefault`, as in
  `scripts/research/build_val_split.py:62`). Deterministic; text conditioning
  depends on it, so it is stated here. Apply the SAME rule to the calibration pool
  drawn from train. (The val manifest has 5 captions per wav = 2475 entries / 495
  wavs; the calibration pool is de-duplicated to one caption per wav the same way.)
* **`z_0`:** `scale_factor * VAE.encode(real_mel).mode()` (deterministic mode).
* **Noise policy:** `eps ~ N(0, I)` per slot from the master seed; `z_t =
  q_sample(z_0, t, eps)`. Same `z_t`, `t`, `eps` shared across audio and text for
  each slot (pairing invariant, tested in M2 T3/T4).
* **CLAP `unconditional_prob = 0.0`** in all calibration/diagnostic paths
  (**M2 trap #1**: the upstream default 0.1 is not overridden by the config and
  would inject stochastic unconditional dropout).
* **Audio-branch 0.24 s truncation** (**M2 trap #2**): the vendored
  `get_audio_features` deterministically keeps the first 10.0 s @ 48 kHz, dropping
  ~0.24 s of every 10.24 s clip. Fixed and identical across models; recorded so it
  is not mistaken for a bug or for `laion_clap`'s stochastic fusion.
* Freeze the resulting slot list (example wav ids, seeds, timesteps) as a manifest
  with a sha256 before inspecting results.

## Primary timestep aggregation

Equal-weight mean across preregistered timestep strata, followed by averaging
across examples, unless the master plan is explicitly amended before results are
inspected.

* **Strata (PROPOSAL): `K = 5` equal-width strata** over `[0, 1000)`:
  `[0,200), [200,400), [400,600), [600,800), [800,1000)`. Rationale: covers
  low/mid/high-noise regimes evenly and prevents post-hoc selection of a
  favourable timestep. For P2/P3, draw **1 timestep per stratum per example**
  (so slots per example = K = 5); for P1, **2 timesteps per stratum per example**
  (so P1's `2B` text units match P2/P3's per-stratum weights — see B2 above).
  All draws seeded from the master seed.

## M3A matched random null — machinery implemented (M3-001)

* `Krand = 20` structured random masks at the `(1,2,3,1)` budget, with EXACTLY the
  same per-layer `k` as L1, over the 28 ranked layers. Generator:
  `research_pruning/diagnostics/random_masks.py` (mechanic ported verbatim from the
  frozen reference `pruned_unet_dict_creation.py`). **The per-layer `k` come from
  the pruned U-Net's target SHAPES (`channel_mult=[1,2,3,1]`), NOT from the pkl and
  NOT from the L1 checkpoint** — `sorted_indexes_dict.pkl` holds full channel
  *permutations*, not counts. Weights are materialised from the BASE checkpoint;
  the L1 checkpoint is never opened.
  * Per-layer `k`: 15 layers keep 192 of 960, 12 keep 576, 1 keeps 384;
    total 10 176 kept channels/mask.
  * Pre-registered seeds `20260818..20260837`; random-mask-set sha256
    `3e6666bcdf0bab77568650732aaf9aab37241527903c6023031d01aac84e8f7e`;
    L1 reference-mask sha256
    `9a2593c20555d510d0edef76deb2075121f5e865f5f931d0c28584fa83524360`
    (full-ranking fingerprints; regenerated in M3-002 after the materializer fix).
    Record: `artifacts/m3_pilot/random_null_masks.json`.
  * **Materializer is bit-exact to the published artifact (M3-002):**
    `materialize(base, L1 ranking)` == `l1_audioldm-m-full_p1.ckpt`, **690/690**
    tensors (`artifacts/m3_pilot/l1_bitexact_check.json`, test R5). Random masks use
    the SAME materializer, so they coincide with L1 at every positional/identity
    seam and differ only in the **12 ranking-driven layers** (see below).
* Diagnostics per mask and for L1: aggregated `D_gen`, `D_mod`, `R_mod` (forward-
  only; real pruned/full epsilons feed `modality_diagnostics`).
* Primary statistic: `Delta_swap = R_mod^L1 - E[R_mod^random | D_gen^L1]`.
  Implementation: `research_pruning/diagnostics/matched_null.py`.
  * **Fit functional form: LINEAR** (`R_mod = a + b·D_gen`) over the per-mask
    control points, by OLS. Rationale: within the narrow `D_gen` band around L1 the
    relationship is expected smooth/monotone; a line is the simplest defensible
    model and its residual scatter is the control SD for the standardized-residual
    gate. `R²` and residual SD are stored as fit diagnostics. Revisit the form
    (isotonic/log) only if curvature appears, recording the change in the ledger.
  * **Standardized residual** = `Delta_swap / (mask-level residual SD)`, in
    random-control SD units.
* **Bootstrap unit = WAV (never the caption-wav entry), seed `20260818`,
  `10 000` resamples.** Resample wavs with replacement AND resample the 20 masks.
  `bootstrap_delta_swap` **raises** if the pool has repeated wav ids
  (pseudo-replication would narrow the CI and let Gate A pass by construction).
  Report the 95% CI of `Delta_swap`.

## Seam conventions of the published artifact (pre-registration for M3A/M3B)

The published `l1_audioldm-m-full_p1.ckpt` is **internally inconsistent at its
seams**, and the public reference script does not reproduce it. The materializer
`research_pruning/diagnostics/random_masks.py` reproduces the artifact **690/690**
bit-exact (test R5); the per-layer convention below IS that reproduction and is part
of the pre-registration, because the random null inherits it and because **P1/P2/P3
compete only on the ranking-driven layers**.

Of the 15 tensors that reduce a 960-channel output to 192, **12 are ranking-driven
in their output** (out = `perm[:192]`) and **3 are positional** (out = first-192,
ranking ignored):

* **Ranking-driven output (12) — where the null differs from L1 and where M3B
  overlap competes:** `input_blocks.10.0.in_layers.2.weight`,
  `input_blocks.10.0.out_layers.3.weight`, `input_blocks.11.0.in_layers.2.weight`,
  `input_blocks.11.0.out_layers.3.weight`, `middle_block.0.in_layers.2.weight`,
  `middle_block.0.out_layers.3.weight`, `middle_block.2.in_layers.2.weight`,
  `middle_block.2.out_layers.3.weight`, `output_blocks.0.0.out_layers.3.weight`,
  `output_blocks.1.0.out_layers.3.weight`, `output_blocks.2.0.out_layers.3.weight`,
  `output_blocks.2.2.conv.weight`.
* **Positional-output seams (3) — random == L1 by construction:**
  `output_blocks.0.0.in_layers.2.weight`, `output_blocks.1.0.in_layers.2.weight`,
  `output_blocks.2.0.in_layers.2.weight`.
* **Identity-input seam (1):** `input_blocks.10.0.in_layers.2.weight` keeps its input
  columns in identity order (the reference reorders them by the `input_blocks.9.0.op`
  ranking).
* **Bias anomaly:** `output_blocks.2.0.in_layers.2.bias` is kept **ranked**
  (`base[perm[:192]]`) while its own **weight** is kept **positional** — the bias
  values are attached to different channels than the weight rows. Reproduced exactly
  because M3A diagnoses this exact checkpoint; flagged as a question for Arshdeep.

Proof for each seam: `scripts/research/verify_l1_bitexact.py` →
`artifacts/m3_pilot/l1_bitexact_check.json`. Full per-tensor derivation:
`research_pruning/diagnostics/random_masks.py` docstring.
* **Evaluation examples (PROPOSAL): `N_eval = 200`** drawn from the disjoint
  validation split defined below (never the test set). Provisional; CPU plan-B
  cost at this size is ~3 CPU-hours per full pass (M2 timing), GPU cost TBD.
* **Data split constraint (RESOLVED, from M0 9.1):** upstream `dataset_root.json`
  aliases `val` to `test`. A disjoint validation split is now defined:
  `configs/research/val_split_disjoint.json` — the upstream AudioCaps val set
  (`datafiles/audiocaps_val_label.json`, **495 items**), proven disjoint by wav id
  from BOTH test (`val ∩ test = 0`) and train (`val ∩ train = 0`).
  **sha256 `e540146d62d01ca70ed92e8b1adc1991da8c967e3e5229241c13f78edc8ff45e`.**
  Upstream `dataset_root.json` is NOT modified; the split lives in its own manifest.
* Gate A PASS requires 95% bootstrap CI of `Delta_swap` above zero and the
  standardized residual at least 0.5 random-control SD.

## M3B saliency disagreement

* **P0 / L1 baseline convention (DECIDED and pre-registered — DECISION-M3B-002,
  2026-08-19).** RQ2's L1 baseline **is** Arshdeep's official published pruning
  artifact (Zenodo 21376822), and that artifact keeps, per pruned layer, the
  **LOWEST**-L1 conv filters — inverted from standard L1 magnitude pruning
  (verified 4 ways; write-up
  `docs/m0_baseline_reproduction/l1_pruning_direction_finding.md`). The project
  therefore adopts the published inverted convention: **P0 keeps the lowest-L1
  channels**, via `research_pruning.taylor.p0_importance(convs,
  convention="published")` (= `-L1`, so `keep_topk` keeps the low-L1 set); module
  default `P0_CONVENTION = "published"`. Verified on the real base `(1,2,3,5)`
  U-Net to reproduce the published kept-set **exactly on 12/12 ranking-driven
  layers** (`scripts/research/verify_p0_convention.py`; control test
  `test_taylor_saliency.py::C8`). `"standard"` (keep-highest-L1, Li et al. 2017)
  is retained in code for non-Arshdeep baselines and is **not** the RQ2 baseline.
* **P0-standard as a SECONDARY reference (DECIDED, Gabriel 2026-08-19).** In addition
  to the primary P0-published baseline, the run also computes and reports
  `p0_importance(convs, convention="standard")` (keep-highest-L1) as a **secondary
  reference point**, so RQ2 can separate "beats the published artifact" from "beats a
  competently-directed L1 criterion". This does not change the pre-registered primary
  baseline and costs no extra compute (same per-channel L1, opposite sign). Both
  numbers are reported side by side; the gate decision uses the primary.
* **Direction asymmetry — mandatory reporting constraint.** P1/P2/P3 keep the
  channels of **highest** Taylor saliency (most important by their own criterion),
  while P0 as pre-registered above keeps the channels of **lowest** L1 magnitude.
  A P1/P2/P3-vs-P0 margin therefore confounds criterion *direction* with criterion
  *quality*. Every reported comparison must be worded "vs the published L1 pruning
  artifact", never "vs standard L1 magnitude pruning", and must carry this
  asymmetry in the caption/caveat. This constrains RQ2 wording in
  `docs/claims_matrix.md`; it does not change the pre-registered baseline.

* **Target prune tail (PROPOSAL):** the channels actually removed at the
  `(1,2,3,1)` budget, per layer. The per-layer kept count `k_l` comes from the
  pruned U-Net's target **shapes** (built from the frozen config with
  `channel_mult=[1,2,3,1]`), **not** from `sorted_indexes_dict.pkl` — the pkl holds
  full channel permutations, not counts. **P1/P2/P3 compete only on the 12
  ranking-driven layers** enumerated in "Seam conventions" above: the 3
  positional-output seams have no ranking to disagree on (L1 itself is positional
  there), so overlap is defined only where channel selection is ranking-driven.
  `overlap@k` uses the kept sets of size `k_l = 192` from the audio and text
  saliency rankings on those 12 layers.
* Targeted layers: **constrained by M0.** The public L1 baseline
  (`sorted_indexes_dict.pkl`, Zenodo 10.5281/zenodo.21376822) ranks exactly 28
  conv layers -- `input_blocks.7..11` (9), `middle_block` (4),
  `output_blocks.0..6` (15) -- widths 384/576/960. P1, P2 and P3 must be computed
  over this same layer set with the same per-layer channel counts, or the
  comparison against L1 is not structure-matched. Evidence:
  `docs/m0_baseline_reproduction/public_artifact_inventory.md`.
* **Weighted overlap calculation (AMENDED and pre-registered — DECISION-M3B-003,
  2026-08-19; resolves audit findings G1 and G2).** There is now exactly ONE overlap
  definition: the **KEPT set**. For each ranking-driven layer `l` with `k_l` kept
  channels out of `N_l`, let `K_a^l`, `K_t^l` be the top-`k_l` channels by audio and
  text saliency. Then

  ```text
  overlap_l        = |K_a^l ∩ K_t^l| / k_l                    in [0, 1]
  weighted overlap = sum_l k_l * overlap_l / sum_l k_l
  chance_l         = k_l / N_l                                (= 0.20 at 192/960)
  adjusted_l       = (overlap_l - chance_l) / (1 - chance_l)  (0 at chance, 1 at identity)
  ```

  Report per-layer overlaps, the weighted aggregate, and the chance-adjusted values;
  global Spearman is secondary. The equivalent prune-set number may be reported for
  transparency via the exact identity `prune_overlap_l = (N_l - 2*k_l + |K_a ∩ K_t|) / p_l`
  (at 960/192/768: `(576 + I)/768`), but it is **never** the gate.
* **Gate B PASS (AMENDED — DECISION-M3B-003) requires weighted KEPT-set overlap
  `<= 0.80` AND at least two ranking-driven layers with kept-set overlap `<= 0.70`.**

  *Why amended.* The master plan states these two numerals against the **prune-set**
  overlap. At the `(1,2,3,1)` budget each ranking-driven layer prunes `p = 768` of
  `N = 960` channels, so prune-set overlap is confined to `[0.75, 1.0]` by pigeonhole
  (`(2p - N)/p = 0.75`) with chance at `p/N = 0.80`. Under that reading condition 1
  demanded "no more agreement than pure chance" and condition 2 (`<= 0.70`) was
  **mathematically impossible** — Gate B could never PASS. Full derivation: ledger
  `AUDIT-M3-002`, finding G1. Gabriel's amendment (option (a), 2026-08-19) transfers the
  plan's numerals verbatim onto the kept-set definition, where the range is the full
  `[0, 1]` and chance is `0.20`; the gate then fails only if the two modalities agree on
  `>= 80%` of the kept channels. The draft previously carried BOTH definitions in
  adjacent bullets (finding G2); the prune-set one is now deleted as a gate object.

## Frozen identifiers

* Reference architecture: `channel_mult = [1,2,3,1]`, `model_channels = 192`,
  verified from `l1_audioldm-m-full_p1.ckpt` (md5 `2666e6fc108a9c4fc0d19bbf26832905`).
* L1 index manifest md5: `a4cd11ff83438ee0f9aa5fe0917f39e3`
* `scale_factor`: `0.9138255715370178` (read from `audioldm-m-full.ckpt`).
* Validation split manifest sha256: `e540146d62d01ca70ed92e8b1adc1991da8c967e3e5229241c13f78edc8ff45e`
* Diagnostic norm: L2 (flattened per-example); `R_mod` epsilon: `1e-12`.
* Master seed (proposed): `20260818`.
* Random-null seeds: `20260818..20260837`; random-mask-set sha256
  `3e6666bcdf0bab77568650732aaf9aab37241527903c6023031d01aac84e8f7e`;
  L1 reference-mask sha256
  `9a2593c20555d510d0edef76deb2075121f5e865f5f931d0c28584fa83524360`
  (full-ranking fingerprints, M3-002). Materializer bit-exact to published ckpt (690/690).
* Matched-null fit form: LINEAR (OLS); bootstrap unit: WAV; `n_boot` 10 000.
* Calibration manifest SHA256: _pending (built at freeze time; E, K, caption rule fixed)_
* Timestep list/hash: _pending (built at freeze time)_
* Code commit: _pending_
* Resolved config: `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`
* Freeze commit: _left blank — reviewed before freezing_
* Freeze timestamp: _left blank — reviewed before freezing_
