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
* Tests `tests/research/test_diagnostics.py` D1..D5 PASS (identity, bounds,
  monotonicity, symmetry, isolation).

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

## Calibration budget (master plan §5)

One gradient-evaluation unit = one forward+backward saliency evaluation for one
modality at one pre-specified `(example, noise, timestep)` slot.

* **Base slots `B` (PROPOSAL): `B = 256`.** Rationale: enough examples to stabilise
  per-layer saliency across the 28 ranked conv layers while keeping total gradient
  evaluations affordable within the ~US$50 cloud budget. **Provisional** until
  `T_sal` is measured on GPU; revise so total calibration GPU-hours fit the budget.
* **P1 (text-only Taylor): `2B` text evaluations** — two pre-registered
  `(noise, timestep)` draws per base example, to avoid giving P1 redundant
  duplicate gradients (§5).
* **P2/P3 (paired): `B` text + `B` audio** on the same `B` base slots; P2 and P3
  share the same computed `S_a`, `S_t` (no duplicate calibration compute).
* All criteria therefore consume `2B = 512` gradient evaluations. Report
  calibration GPU-hours, peak VRAM, #forward/backward evals, wall time, examples.

## Slot construction (PROPOSAL)

* **Example pool:** AudioCaps **train** manifest
  (`datafiles/audiocaps_train_label.json`, 49 502 items), which is disjoint from
  the test set by wav id (verified: `train ∩ test = 0`). Calibration never draws
  from the test set or from the reserved validation split.
* Draw `B` base examples deterministically (sort wav ids, seeded permutation,
  master seed `20260818`).
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
  favourable timestep. Sample the same number of timesteps per stratum per base
  example (proposal: 1 per stratum for the pilot), seeded.

## M3A matched random null

* `Krand = 20` structured random masks at the `(1,2,3,1)` budget (same per-layer
  channel counts as L1, over the 28 ranked layers).
* Diagnostics per mask and for L1: aggregated `D_gen`, `D_mod`, `R_mod` using the
  conventions above (forward-only; the real pruned/full epsilons feed
  `modality_diagnostics`).
* Primary statistic: `Delta_swap = R_mod^L1 - E[R_mod^random | D_gen^L1]`, from a
  fit of `R_mod` vs `D_gen` across the random controls evaluated at L1's `D_gen`.
* **Bootstrap unit and seed (PROPOSAL):** resample **evaluation examples** with
  replacement AND resample the 20 random masks; `10 000` bootstrap resamples;
  master seed `20260818`. Report the 95% CI of `Delta_swap`.
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

* **Target prune tail (PROPOSAL):** the channels actually removed at the
  `(1,2,3,1)` budget, per layer, taken from `sorted_indexes_dict.pkl`'s per-layer
  counts. `overlap@k` uses `k = p_l` = number of channels pruned in layer `l`.
* Targeted layers: **constrained by M0.** The public L1 baseline
  (`sorted_indexes_dict.pkl`, Zenodo 10.5281/zenodo.21376822) ranks exactly 28
  conv layers -- `input_blocks.7..11` (9), `middle_block` (4),
  `output_blocks.0..6` (15) -- widths 384/576/960. P1, P2 and P3 must be computed
  over this same layer set with the same per-layer channel counts, or the
  comparison against L1 is not structure-matched. Evidence:
  `docs/m0_baseline_reproduction/public_artifact_inventory.md`.
* **Weighted overlap calculation (PROPOSAL):** for each targeted layer `l` with
  `p_l` pruned channels, let `S_a^l`, `S_t^l` be the bottom-`p_l` channels by
  audio/text saliency. `overlap_l = |S_a^l ∩ S_t^l| / p_l`. Weighted overlap =
  `sum_l p_l * overlap_l / sum_l p_l`. Report per-layer overlaps and the weighted
  aggregate; global Spearman is secondary.
* Gate B PASS requires weighted prune-set overlap `<= 0.80` and at least two key
  prunable layers with overlap `<= 0.70`.

## Frozen identifiers

* Reference architecture: `channel_mult = [1,2,3,1]`, `model_channels = 192`,
  verified from `l1_audioldm-m-full_p1.ckpt` (md5 `2666e6fc108a9c4fc0d19bbf26832905`).
* L1 index manifest md5: `a4cd11ff83438ee0f9aa5fe0917f39e3`
* `scale_factor`: `0.9138255715370178` (read from `audioldm-m-full.ckpt`).
* Validation split manifest sha256: `e540146d62d01ca70ed92e8b1adc1991da8c967e3e5229241c13f78edc8ff45e`
* Diagnostic norm: L2 (flattened per-example); `R_mod` epsilon: `1e-12`.
* Master seed (proposed): `20260818`.
* Calibration manifest SHA256: _pending (built at freeze time)_
* Timestep list/hash: _pending (built at freeze time)_
* Random mask seed list/hash: _pending (built at freeze time)_
* Code commit: _pending_
* Resolved config: `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`
* Freeze commit: _left blank — reviewed before freezing_
* Freeze timestamp: _left blank — reviewed before freezing_
