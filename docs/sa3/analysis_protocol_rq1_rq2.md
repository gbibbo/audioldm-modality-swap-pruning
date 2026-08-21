# SA3 analysis protocol — RQ1 / RQ2 (analysis-first) — **rc3**

**Status: ANALYSIS PROTOCOL rc3, 2026-08-20 23:44 (Montevideo). rc1 (`76fdcda`) → rc2
(`a57176d`) → rc3 after the second external review. Reviewer decision on rc2: GO for Step 0
(CPU): `.venv-sa3`, checkpoint/config verification, `research_sa3/` skeleton, tests, CPU
dry-run; NO GO for any scientific GPU smoke until rc3. No GPU has been run. No pruning
method, no LoRA recovery, no RQ3/RQ4 design is part of this document (§11).**
This is not a master plan. It instantiates the adoption rules S1–S7 of
`docs/review/2026-08-20_postmortem_v3_v4.md` §3 (DECISION-RULES-001) for two analysis
questions on Stable Audio 3 `small-sfx-base` / `small-sfx`.

Every number is either (a) read from a cited source (file:line at `Stability-AI/stable-audio-3`
commit `a0b57f5483c4588f827f3552b7d5c6ca2a9687be`, HF API, paper) or (b) marked **TBM** (*to be
measured*), or (c) flagged **[pre-registered constant]** — a stopping/tolerance constant chosen
now, before data, and never tuned afterwards. No estimate is presented as a measurement.

**rc1 → rc2 changelog (review of 2026-08-20 23:35):**
1. `E(M)` is now an operational **tuple with a pre-registered non-inferiority rule** (§9.2);
   scalar differences use the primary metric only (§4).
2. The localized-probe control is demoted to a **unit test**; a **trained LoRA restricted to
   known blocks** is the mandatory scientific positive control for RQ2 (§5.1).
3. `A` split into **`A_tan`** (infinitesimal regime, κ-convergence required) and
   **`A_eco`** (held-out real adapters at their normal strength); κ is **not** calibrated to a
   real LoRA; the RQ2 claim is "`A_tan` predicts `A_eco`" (§3.4, §8).
4. The ecological LoRA is trained with **`--include transformer.layers`** (backbone only);
   full model+conditioner LoRA is a secondary sensitivity (§5.2).
5. Post-only and adaptability-aware selections are **sequential greedy, re-evaluated at each
   step** (105 candidate evaluations to k = 6), not top-k of single-block scores (§6.2, §7).
6. "Iso-compute" removed: **block-pass bracket** now, **latency-matched** comparison after the
   smoke (§4.2).
7. `I_PT` renamed **normalized post-training-delta distortion**, unbounded above, pooled
   ratio with a measured denominator guard, reported per noise level (§3.2).
8. **Pre-registered sample-size rules** for `N_p` (now `N_main`, rc3) and `n_u` from precision/stability of each
   criterion's own quantities, never from cross-criterion divergence (§2.3).
Minor: `D_B^common` vs `D_B^deploy` (§3.1); the phase is *analysis-first* (forward-only
structural analysis + one ecological adapter validation that requires LoRA training), not
strictly forward-only.

**rc2 → rc3 changelog (second review, 2026-08-20 23:44):**
1. Set-divergence floor now in the **same units** as `δ` (bootstrap disagreement count, 95th
   percentile, exceed the floor of **both** criteria); Jaccard is descriptive only (§4.1).
2. **Measurement uncertainty** (≥ 5 independent seed streams → distributions, CIs) separated
   from the **non-inferiority margin** (operational: one sampling step's worth of dense
   degradation, 8 → 7, measured on the pilot before any mask) (§9.2).
3. Positive control = **trained single-block LoRAs** `L_6`, `L_13` with identifiable
   localisation; the `layers[4-7]` version dropped (§5.1).
4. Each ecological adapter gets a **frozen train/eval split** and a **task-level held-out
   metric** `T(L; system)`; field preservation alone never demonstrates function preservation
   (§5.2, §3.4).
5. `L_ceil` replaced by the **dense-transfer compatibility band**; the claim is "stays within
   the band or recovers a pre-registered fraction of the pruning-induced extra loss" (§4.3).
6. **Disjoint pilot panel** fixes `N_main` and `n_u` from precision tied to the set-level
   SESOI; the main panel never informs its own size; the "largest the smoke allows, then
   double" wording removed (§2.3). S1 is satisfied by the pilot, **not** by a theoretical
   `I_PT` range (§11).
7. `E`-greedy is **mandatory for cases C/D to authorise RQ3**; without it RQ1/RQ2 may
   conclude but no method is authorised; `E`-greedy uses the multivariate rule, not CLAP-only
   (§3.5, §6.2, §8, §11).
8. **Probe → adapter correspondence frozen**: `U_gen` → standard `lora` r = 16 (primary);
   `U_xs` → real `lora-xs` adapters (secondary, matched); `dora-rows` external validity; no
   best-of-two (§3.4).
Also: `η_i` per noise level (§3.2); "≥ 4 adapters" is a feasibility minimum, the unit of
generalisation is the **domain** (§5.2).

---

## 0. Scientific question and decision

Stable Audio 3 ships, for the same 433 M DiT, a **base** checkpoint (rectified flow, ~50 Euler
steps with CFG) and a **post-trained** checkpoint (distillation warm-up + adversarial
post-training, 8 ping-pong steps, no CFG), and documents that **LoRAs are trained on the base
and applied unchanged to the post-trained model** (`docs/guides/model-overview.md`). The pair
is a controlled perturbation of one architecture. We ask two *analysis* questions, block by
block, before deciding whether any pruning method is worth building.

* **RQ1 — Does few-step post-training materially change structural importance?** For each of
  the 20 transformer blocks: how much does removing it distort the base field, the
  post-trained field, and — separately — the *post-training transformation* (difference of the
  two fields)? Is the transformation's distortion predictable from the plain post-trained
  damage, or from the parameter delta?
* **RQ2 — Does sensitivity to low-rank adaptation occupy structure distinct from what the
  standalone 8-step generator needs?** For each block: how much of the functional effect of
  low-rank perturbations in the LoRA-eligible subspace is carried by the block or disturbed by
  its removal; does a tangent-regime measurement made without seeing any real adapter predict
  the degradation of *held-out real adapters*; and do the blocks a deployment-only selection
  would remove coincide with the blocks an adaptability-aware selection would keep?

**What would justify building a method (RQ3 design):** the greedy removal sets at k ∈ {2, 4, 6}
under the deployment criterion and under the adaptability (and/or PT) criterion **differ**
above the bootstrap stability floor; `A_tan` predicts `A_eco` on held-out adapters; and the
adaptability-aware set is **non-inferior** in deployment (§9.2) to the post-only set, or loses
less than the latency-matched dense alternative would (§4.2). Cases C/D in §8.

**What kills the line:** the sets coincide at every k within the floor and `I_PT` is flat
(case A); or no single block can be removed from the post-trained 8-step model without being
dominated by the dense model at the matched step count (case E). Either is reported as the
result; no repair method is invented after seeing it.

---

## 1. Objects (exact)

### 1.1 Checkpoints and configs

| object | HF repo | access | file | size (bytes) | git blob oid (HF tree API, 2026-08-20) | sha256 |
|---|---|---|---|---|---|---|
| base | `stabilityai/stable-audio-3-small-sfx-base` | public, **not gated** | `model.safetensors` | 2 270 384 940 | `da83db9b43a3c2f028d45aa172fb938fd0958d81` | **TBM at download** (ledger) |
| base config | same | public | `model_config.json` | 8 476 | `bcc43173b590e4ff34e995dff4201d1a0d164d43` | TBM |
| base SVD bases | same | public | `svd_bases.pt` | 1 270 378 123 | `ce26e2ca33f43da49c86f5507aedc9d0cb6a23f1` | TBM |
| post | `stabilityai/stable-audio-3-small-sfx` | **gated = auto** (accept license) | `model.safetensors` | 2 270 384 940 | `eff880f5df27fa7a6289c737595f169028bea1f8` | TBM |
| post config | same | gated | `model_config.json` | 10 454 | `0321320a34e3e0ceda48e47c2c36b70234df746f` | TBM |

Both repos bundle the text encoder `t5gemma-b-b-ul2/`. **No `svd_bases.pt` in the post repo.**
License: Stability AI Community License (+ Gemma license for the text encoder); research use.

The two `model.safetensors` have identical byte size — *consistent with*, not proof of,
identical tensor shapes. **Step 0 (mandatory, CPU, before anything else):** load both state
dicts, assert identical key sets and shapes (report any difference), diff the two
`model_config.json` (the post one is 1 978 bytes longer and could not be read without the
gate: HTTP 401). Expected from code: `diffusion_objective` `rectified_flow` (base, read) vs
`rf_denoiser` (post; `diffusion_cond.py:277` "includes ARC models") plus sampling-shift
options. **Any difference inside `model.diffusion.config` invalidates the 1:1 block mapping
and stops the protocol until resolved.**

### 1.2 Architecture (read from the base `model_config.json`)

`model.diffusion.type = dit`; `config`: `io_channels 256`, `embed_dim 1024`, `depth 20`,
`num_heads 16`, `cond_token_dim 768`, `global_cond_dim 768`, `local_add_cond_dim 257`,
`global_cond_type adaLN`, `timestep_features_type expo`, `timestep_features_logsnr False`,
`attn_kwargs.qk_norm rms`, `norm_type rms_norm` (`force_fp32 True`), `ff_kwargs.mult 4.0`,
`num_memory_tokens 64`. Conditioning: `prompt` (`t5gemma`, cross-attention tokens) and
`seconds_total` (`number`, global). Pretransform: SAME autoencoder, `latent_dim 256`,
`downsampling_ratio 4096`, 44.1 kHz stereo; `sample_size 5 324 800` (≈ 120.7 s). Latent rate
44 100 / 4 096 ≈ 10.77 frames/s (derived).

### 1.3 The 20 blocks and the 1:1 mapping

`DiffusionTransformer.transformer` is a `ContinuousTransformer` whose `self.layers` is an
`nn.ModuleList` of 20 `TransformerBlock`s (`transformer.py:1103,1142-1144`), iterated in order
(`transformer.py:1235`); `return_info=True` exposes every block's hidden state and
`exit_layer_ix` supports early exit (`transformer.py:1169-1171, 1254-1262`). Parameter names
are `transformer.layers.{i}.…` (e.g. `transformer.layers.0.self_attn.to_qkv`, `lora.md`).
Block `g ∈ {0,…,19}` of the base maps to block `g` of the post by name; step 0 asserts the
per-block shapes agree.

Each block is residual (`x = x + residual` around self-attention, cross-attention and
feed-forward, `transformer.py:1023-1044`; branches `zero_init_branch_outputs=True`,
`transformer.py:870`), so **removing block g = identity on its input** — no re-wiring, no
dimension change; a LoRA slice on a surviving block keeps exactly its geometry.

### 1.4 Fields, samplers, CFG (read from code)

Both checkpoints are called as `v = model(x, t, **cond)`; both samplers use
`x_t = (1 − t)·x_0 + t·ε`, `t ∈ [0, 1]`, `denoised = x − t·v` (`sampling.py:147-186` Euler;
`:308-353` ping-pong: `denoised = x − t_i·v`, `x = (1 − t_{i+1})·denoised + t_{i+1}·randn`).
**`F_B` and `F_P` are functions of the same `(x_t, t, c)` and are evaluated on identical
states.**

* **Base sampler (frozen):** Euler, 50 steps, `cfg_scale = 7.0` (`diffusion_cond.py:344-351,
  380-385`; `inference.md:52-57`).
* **Post sampler (frozen):** ping-pong, 8 steps, `cfg_scale = 1.0` (`sampling.py:434`;
  `diffusion_cond.py:346,350`; `inference.md:56`: CFG "no effect on post-trained checkpoints").
* **CFG (base only):** combined in denoised space and mapped back to velocity (`dit.py:579-611`):
  `d_c = x − t·v_c`, `d_u = x − t·v_u`, `d_cfg = d_c + (s − 1)·cfg_diff`, `v_cfg = (x − d_cfg)/t`.
  **Production default is full APG** (`apg_scale = 1.0`: `cfg_diff` = component of `d_c − d_u`
  orthogonal to `d_c`, `dit.py:370,599-602`; Gradio default `diffusion_cond.py:76,377`;
  `cfg_rescale` 0.0 `:74`). **Deploy field frozen to the production default** (`s = 7`,
  `apg = 1.0`, `scale_phi = 0`, `cfg_norm_threshold = 0`, interval (0, 1)); vanilla CFG
  (`apg = 0.0`) secondary. Both recorded in every output JSON.
* **Schedules:** `build_schedule(steps, sigma_max = 1.0, dist_shift, effective_seq_len)`
  (`sampling.py:9-64`) = `linspace(1, 0, steps + 1)` warped by the checkpoint's own shift (base:
  `distribution_shift_options.type = full`, `min_length 256`, `max_length 4096`,
  `use_effective_length_for_schedule True`). The 8 post noise levels `τ_1 > … > τ_8` are
  computed from the **post** checkpoint's `sampling_dist_shift` for the panel duration, stored
  in `configs/sa3/schedule_post_10s.json`, never typed by hand.
* **Duration:** one fixed `seconds_total` for the panel (§2) ⇒ identical effective sequence
  length and schedule across systems.

### 1.5 Common states

* **`S_traj` (primary):** states visited by the **dense post-trained sampler** on the panel:
  for prompt `p` with seed-paired initial noise `ε_p`, the 8 pairs `(x_i, τ_i)` captured through
  the sampler callback (`sampling.py:347` ping-pong, `:182` Euler; fields `x, t, i, denoised`).
* **`S_noised` (secondary):** `x_τ = (1 − τ)·x̂_0 + τ·ε'` at the same 8 levels, `x̂_0` = the
  dense post model's final latent for the prompt, independent `ε'`.
* **`S_traj^B`:** states of the dense **base** Euler-50/CFG-7 trajectory — used only for
  `D_B^deploy` (§3.1), never for `F_B` vs `F_P` comparisons.
* **`S_real` (optional, not available):** noised latents of real 44.1 kHz audio; our AudioCaps
  copy is 16 kHz mono. Not required for RQ1/RQ2.

---

## 2. Panel, seeds, sample-size rules

### 2.1 Prompts and duration

Fixed, seeded subset of AudioCaps **test** captions (ARC's evaluation set, arXiv 2505.08175
§3.3: 881 audios / 4 875 captions; captions held in `data/dataset/metadata/audiocaps/`).
Construction: master seed 20260818, one caption per source wav, stratified by caption-length
tercile; sha256 of `configs/sa3/panel_prompts.json` in the ledger. `seconds_total = 10` for the
whole panel (AudioCaps clip length; ≈ 108 latent frames, derived). Text only; no 16 kHz audio
enters any SA3 model.

### 2.2 Seeds and conditioning cache

Per prompt `p`: one initial noise `ε_p` shared by **every** system (dense base, dense post, all
block-removed variants, all probes); a second per-prompt, per-step seed stream for ping-pong
re-noising, also shared. Seed table hashed. T5Gemma token embeddings and the `seconds_total`
embedding are computed once per prompt and cached; only the DiT is evaluated repeatedly.
`local_add_cond` is the no-inpainting default, identical across systems.

### 2.3 Pilot panel and sample-size rules (pre-registered)

Two **disjoint** prompt panels are drawn from the same construction (§2.1) with different
seeds: a **pilot panel** `P_pilot` and a **main panel** `P_main` (no shared source wav).

* **Role of the pilot (S1 + precision):** the pilot is the *only* data used to (i) bound the
  magnitude and variance of `D_P`, `I_PT`, `A_tan` (this is how S1 is satisfied — no
  theoretical `I_PT` range is written), (ii) measure the dense seed-stream distributions and
  the 8 → 7 step margin of §9.2, (iii) fix `N_main` and `n_u`. Pilot results are reported as
  pilot; no §8 decision is read from them.
* **`N_main` rule (tied to the set-level SESOI).** The decision unit is the greedy set at
  k ∈ {2, 4, 6}. On the pilot, for each criterion `X` separately (`D_P`, `I_PT^raw`,
  `A_tan`), compute by prompt bootstrap (B = 1 000 **[pre-registered constant]**) the
  disagreement count between pairs of replicates, `d_X(k) = |R_X^a(k) △ R_X^b(k)| / 2`, as a
  function of the subsampled prompt count `N`, and fit its decay. `N_main` is the smallest `N`
  at which the 95th-percentile **[pre-registered constant]** of `d_X(k)` is **0 blocks** for
  every `X` and every `k` (i.e. the floor `f_X(k)` of §4.1 is below the one-block SESOI unit),
  extrapolated from the pilot curve and then **verified on the main panel only after it is
  frozen**. If the pilot predicts `N_main` beyond the authorised budget, the experiment is
  declared **underpowered before it runs** and §8 is not applied. `N_main` never uses the
  divergence *between* criteria and is never revised after main-panel data exist.
* **`n_u` rule.** On the pilot, for each probe family, increase `n_u` (doubling from 8
  **[pre-registered constant]**) until the probe-bootstrap 95th percentile of `d_{A_tan}(k)` is
  0 blocks at every k; that `n_u` is frozen for the main panel.
* Jaccard of bootstrap sets is **reported as a descriptive stability diagnostic** (not a
  criterion).
* The cross-criterion statistics of §4 are computed once, on the main panel, after `N_main`
  and `n_u` are frozen and written to the ledger.

---

## 3. Executable definitions

Notation: `F_B`, `F_P` are the **raw conditional velocity fields** (`DiffusionTransformer._forward`,
`dit.py:179`); `F_B^dep` is the production CFG/APG field of §1.4. `F^{−g}` = same model with
block `g` replaced by identity; `F^{−M}` for a set `M`. `‖·‖²_S` = mean over a state set `S` of
the per-state squared L2 norm over the latent `(256 × T)`, padding-masked and divided by `T`.
Quantities are computed on `S_traj` (primary) and `S_noised` (secondary) unless stated.

### 3.1 Block damage (RQ1)

* `D_B^common(g) = ‖F_B − F_B^{−g}‖²_{S_traj} / ‖F_B‖²_{S_traj}` — base damage on the **common**
  states (comparable with `D_P`).
* `D_B^deploy(g) = ‖F_B^dep − F_B^{dep,−g}‖²_{S_traj^B} / ‖F_B^dep‖²_{S_traj^B}` — base damage on
  its **own** deployment trajectory (a different question; never tabulated against `D_P` as
  if equivalent).
* `D_P(g) = ‖F_P − F_P^{−g}‖²_{S_traj} / ‖F_P‖²_{S_traj}`.

### 3.2 Normalized post-training-delta distortion (RQ1)

`Δ^raw = F_P − F_B`, `Δ^dep = F_P − F_B^dep` (both on `S_traj`).

Per noise level `τ_i` and block `g`:
`num_i(g) = ‖Δ_i − Δ_i^{−g}‖²`, `den_i = ‖Δ_i‖²`, with `Δ^{−g} = F_P^{−g} − F_B^{−g}` (resp.
`F_B^{dep,−g}`, CFG recomputed on the block-removed base, `s = 7`, `apg = 1.0`).

* **Pooled ratio:** `I_PT(g) = Σ_i num_i(g) / Σ_i den_i` (ratio of sums, not mean of ratios).
* **Per-level:** `I_PT,i(g) = num_i(g) / den_i`, reported as a 20 × 8 table.
* **Denominator guard (measured, not chosen):** `η_i` = the fp16-vs-fp32 discrepancy of the post
  field **per noise level**, `η_i = ‖F_{P,i}^{fp16} − F_{P,i}^{fp32}‖² / ‖F_{P,i}‖²` (smoke, on
  CPU-reference states). Levels with `den_i / ‖F_{P,i}‖² < η_i` are reported but excluded from
  the pooled ratio and flagged. `η = max_i η_i` is the global precision reference of §3.4.
* **Interpretation:** `I_PT` is a **relative distortion of the post-training delta**; it is
  ≥ 0 and **unbounded above** (removing a block can alter the delta by more than the delta's
  own norm). It is *not* a fraction destroyed. Both `I_PT^raw` and `I_PT^dep` are reported.

### 3.3 Parameter-delta covariate (S1: never an effect estimate)

`W(g) = Σ_{θ∈g} ‖θ_P − θ_B‖²_F / Σ_{θ∈g} ‖θ_B‖²_F`, used only for `ρ(W, I_PT)`, `ρ(W, D_P)`.

### 3.4 Adaptability (RQ2): tangent regime and ecological regime

**Probe space.** LoRA-eligible layers = `nn.Linear` / `nn.Conv1d` under `transformer.layers.*`
(the repo's `--include transformer.layers` target set, `lora.md` "Layer Filtering"), applied
through the official `LoRAParametrization` (`stable_audio_3/models/lora/model.py`:
`lora_forward` `W + scaling·strength·BA`; `lora_xs_forward` `W + scaling·strength·U M Vᵀ`). Two
families, `n_u` probes each (§2.3):

* **`U_gen`:** rank `r = 16` (`train_lora.py` default); `A` Kaiming-uniform as in the repo's
  init; `B ~ N(0, 1)` (the repo zero-inits `B`; a zero probe is inert); rescaled per layer to
  `‖ΔW_ℓ‖_F / ‖W_ℓ‖_F = κ`.
* **`U_xs`:** `ΔW_ℓ = U_ℓ M_ℓ V_ℓᵀ`, `U_ℓ, V_ℓ` from the **base checkpoint's `svd_bases.pt`**
  (top-`r` singular vectors of the base `W_ℓ`, `model.py:97-111`), `M_ℓ ~ N(0, 1)`, rescaled to
  the same `κ`. **Never the SVD of post weights** (no bases ship with the post; an XS adapter
  transfers base→post only in the base's bases).

**Tangent regime `A_tan` (no real adapter is seen).** For a probe `u` on the post model:
`δF(u) = F_{P+u} − F_P`; after removing `g`, with `u_{−g}` the probe restricted to surviving
blocks: `δF^{−g}(u) = F^{−g}_{P+u_{−g}} − F^{−g}_P`. Define

* `A_carry(g) = E_u ‖δF(u_g)‖² / E_u ‖δF(u)‖²` (effect living in `g`'s own slots),
* `A_int(g) = E_u ‖δF(u_{−g}) − δF^{−g}(u_{−g})‖² / E_u ‖δF(u_{−g})‖²` (disturbance of the
  others' effect),
* `A_tan(g) = E_u ‖δF(u) − δF^{−g}(u)‖² / E_u ‖δF(u)‖²`,

each for `U_gen` and `U_xs`. **Infinitesimal-regime requirement:** all three are computed at
`κ ∈ {κ_0, κ_0/2, κ_0/4}` (`κ_0` **TBM**: the largest value at which linearity holds, see next
item) and reported only if, for every block, the value at `κ_0/2` and `κ_0/4` agrees with the
value at `κ_0` within the bootstrap CI; **linearity check** `‖δF(2u)‖ / ‖δF(u)‖ ∈ [1.9, 2.1]`
**[pre-registered constant]** on the dense post model; **precision floor:** `‖δF(u)‖² / ‖F_P‖²`
must exceed `η` (§3.2) by a factor ≥ 10 **[pre-registered constant]** — if fp16 cannot
resolve the tangent regime on T4, `A_tan` is computed in fp32 (cost TBM) or declared not
measurable. `κ` is **never** set to match a real adapter's magnitude.

**Ecological regime `A_eco` (held-out real adapters, normal strength).** For each real LoRA
`L` of §5.2 (trained on the base, applied to the post at `strength = 1.0`):
`A_eco(g; L) = ‖δF(L) − δF^{−g}(L)‖² / ‖δF(L)‖²` on `S_traj`, plus its end-to-end analogue
(§9: adapter effect on generated audio with/without `g`). `L` is **never used** to choose
probes, `κ`, `N_main`, `n_u` or any set.

**Frozen probe → adapter correspondence (no best-of-two).** *Primary pair:* `U_gen` predicts
standard **`lora`, rank 16** held-out adapters. *Secondary, matched pair (pre-specified):*
`U_xs` predicts real **`lora-xs`** adapters, run only if such adapters are trained. `dora-rows`
adapters are **external validity** (reported, never used to choose between probe families).
The probe family that "predicts better" is never selected after the fact.

**RQ2 claim structure.** (i) `A_tan` from the primary pair (blind) **predicts** `A_eco` across
blocks for held-out adapters (pre-registered: rank correlation and, decision-relevant, the
set-disagreement statistic of §4.1); (ii) the adaptability-aware greedy set differs from the
post-only greedy set (§4.1); (iii) the adapters' **task-level function** `T(L; ·)` (§5.2) is
preserved per §4.3. All three are required for case C/D.

### 3.5 Selections (sets) — non-additive

For a per-block or per-set score `X`, the **sequential greedy** selection `R_X^greedy(k)`:
start from the dense post model; evaluate all 20 single removals; remove the best (smallest
`X`); re-evaluate all 19 remaining removals *from the new architecture*; repeat to k = 6. That
is 20 + 19 + 18 + 17 + 16 + 15 = **105 candidate evaluations** per criterion; sets at k = 2, 4,
6 are read off the path. Criteria: `D_P` (field proxy), `E` (end-to-end, §9 — **the strongest
post-only adversary**), `A_tan` (adaptability-aware, field), `I_PT` (PT-aware, field).
**`E`-greedy selection rule (same multivariate rule as deployment, §9.2, never CLAP-only):**
at each step, among the candidate removals, keep those whose KL and FD drift from the dense
post model are within the margins `m_KL`, `m_FD`; among them choose the highest CLAP; if none
satisfies both caps, choose the candidate with the smallest maximum normalised violation
`max(ΔKL/m_KL, ΔFD/m_FD)`. **If `E`-greedy (105 × `N_main` generations) cannot be afforded,
RQ1/RQ2 may still conclude, but cases C/D cannot authorise RQ3** (§8, §11); `D_P`-greedy is
then reported as a *field* adversary only. Single-block leave-one-out tables (§7) are **analysis**, not
selection; the additivity gap `X(R(k)) − Σ_{g∈R(k)} X({g})` is reported as a diagnostic.

### 3.6 End-to-end `E(M)`

Defined operationally in §9.2 as a **tuple** `(CLAP, KL, FD)` with a non-inferiority relation
and a primary scalar; never used as an abstract scalar elsewhere.

---

## 4. SESOI in decision terms (S7) with measured anchors

### 4.1 Set divergence (blocks, with a floor in blocks)

`δ_XY(k) = |R_X^greedy(k) △ R_Y^greedy(k)| / 2` ∈ {0, …, k} for the pairs (`D_P`, `A_tan`),
(`D_P`, `I_PT`), (`D_B^common`, `D_P`), and (`A_tan`, `A_eco`) (prediction check).
**Floor in the same units:** `f_X(k)` = 95th percentile **[pre-registered constant]** over
bootstrap replicate pairs of `|R_X^a(k) △ R_X^b(k)| / 2` (prompt bootstrap on the main panel,
B = 1 000; for `A_tan` also probe bootstrap). A divergence between criteria is **real** iff
`δ_XY(k) > max(f_X(k), f_Y(k))`. By §2.3 the main panel is sized so that `f_X(k) = 0` for the
field criteria; if the verification on the frozen main panel finds `f_X(k) ≥ 1` for some
`(X, k)`, that `(X, k)` is reported as unresolved. Jaccard is reported descriptively.

### 4.2 Cost of respecting adaptability — two yardsticks, neither called "iso-compute"

* **Block-pass bracket (analytic, now).** A k-block-removed model at 8 steps performs
  `8·(20 − k)` block-passes: k = 2 → 144, k = 4 → 128, k = 6 → 112. The dense post model at `n`
  steps performs `20·n`: 5 → 100, 6 → 120, 7 → 140, 8 → 160. Each pruned system is bracketed
  by the two dense step counts around it: k = 2 → {7, 8}, k = 4 → {6, 7}, k = 6 → {5, 6}. Both
  bracket members are generated and scored (§9). This counts DiT block-passes only; it ignores
  projections, conditioning, the autoencoder and sampler overhead, and says nothing about GPU
  utilisation.
* **Latency-matched (measured, after the smoke).** Wall-clock per generation on the job's GPU
  (DiT + decode, batch 1 and the smoke's batch), for dense at 4–8 steps and for each pruned
  system at 8 steps. The deployment comparator of a pruned system is the dense configuration
  with the **nearest measured latency** (linear interpolation between step counts if needed).
  **Strong conclusions about deployment use this comparator**; the bracket is a pre-smoke
  stand-in.
* `C_dep(k)` = difference in the **primary scalar** (§9.2) between `R_{A_tan}^greedy(k)` and
  `R_{D_P}^greedy(k)` (or `R_E^greedy(k)` if run); reported with the tuple and the
  non-inferiority verdict.

### 4.3 Adapter function: the dense-transfer compatibility band

For each held-out adapter `L` with its frozen task metric `T(L; system)` (§5.2, higher =
function better preserved):

* **Compatibility band** `[T_lo(L), T_hi(L)]` = the interval spanned by `T(L; dense base)` and
  `T(L; dense post)` (the two systems the official pipeline declares compatible), each with
  its seed/prompt CI. This is a **baseline/band**, not an improvement threshold.
* **Pruning-induced extra loss** for a set `M`: `ℓ(M; L) = max(0, T_lo(L) − T(L; post^{−M}))`.
* **Claim structure (per adapter, summarised over adapters):** the post-only set
  `R_E^greedy(k)` (or `R_{D_P}`) pushes `T` **below the band** (`ℓ > 0` beyond its CI) while the
  adaptability-aware set `R_{A_tan}^greedy(k)` **stays within the band** (`ℓ = 0` within CI) or
  recovers at least `ρ_rec = 0.5` **[pre-registered constant]** of the post-only extra loss:
  `ℓ(R_{A_tan}) ≤ (1 − ρ_rec)·ℓ(R_{post-only})`.
* Field-level `A_eco` is reported alongside as mechanism; **only `T` carries the function
  claim.**

### 4.4 Minimal result that would justify a new pruning method

To be confirmed by the measured anchors, not signed here: at k = 4,
`δ_{D_P,A_tan}(4) ≥ 1` above both floors **and** `δ_{A_tan,A_eco}(4) ≤ max floor`
(prediction holds) **and** the band condition of §4.3 holds for the majority of held-out
adapters **and** `R_{A_tan}^greedy(4)` is non-inferior (§9.2) to `R_E^greedy(4)` or, failing
that, to its latency-matched dense comparator. If the floors, the band, the margins or the
latency-matched comparator cannot be measured, the SESOI is declared **not yet defensible**
and the main experiment is not signed.

---

## 5. Positive controls (S2) — before any null reading

### 5.1 Localised controls — unit test and identifiable scientific control

* **Unit test (code only, not evidence):** a generic probe injected only into blocks 4–7;
  `A_carry` must attribute 100 % to 4–7 and `A_tan(g)` for `g ∉ {4..7}` must equal the
  interference term. Validates restriction/attribution code; cannot fail scientifically
  because the attribution knows the support. Lives in `tests/sa3/`.
* **Scientific positive control (mandatory before RQ2 is interpreted): trained single-block
  LoRAs.** Train `L_6` with `--include "transformer.layers[6]"` and `L_13` with
  `--include "transformer.layers[13]"` (blocks 6 and 13 **[pre-registered choice]**: one in
  each half of the depth, away from the boundary blocks) on `small-sfx-base`, same recipe and
  data regime as §5.2, applied to the dense post at strength 1.0; their dense adapter effects
  are reported (no magnitude matching constant). A single-block adapter's parameters live in
  one block by construction, but whether its **function** can be localised *from outputs* is
  exactly what the instrument must show. **Pass (per control `L_b`):** `A_eco(b; L_b)` exceeds
  `A_eco(g; L_b)` for every `g ≠ b` by more than the bootstrap CI, on the field **and** on the
  end-to-end task metric `T(L_b; ·)` (§5.2); and the `L_b`-specific greedy never removes `b`
  at k ≤ 6. **Fail ⇒ the measurement chain cannot localise a known adaptation from outputs;
  RQ2 is not read.** (The earlier `layers[4-7]` control is dropped: restricting parameters to
  four blocks does not guarantee that all four are the four most damaging removals.)

### 5.2 Ecological — does the compatibility contract exist, and is the adapter's *function* measurable?

Train real adapters on `small-sfx-base` with the repo recipe (`scripts/train_lora.py`, rank 16,
1 000 steps per its quick-start; 20–50 clips per `lora.md`):

* **Per-adapter frozen data split (before training):** the domain's clips are split
  80 / 20 **[pre-registered constant]** into `train_L` and `eval_L` (minimum 5 held-out clips
  **[pre-registered constant]**), plus a held-out list of **domain prompts** `prompts_L`
  (captions not used in training). Split files are hashed in the ledger before
  `train_lora.py` runs.
* **Task-level metric `T(L; system)` (frozen before training; the *function* claim rests on
  it):** generate with `system` (+ `L`, strength 1.0) on `prompts_L` and compute (a) CLAP
  audio–audio similarity of the generations to `eval_L` embeddings, (b) CLAP text–audio score
  against `prompts_L`, (c) `FD_openl3` between the generations and `eval_L`, (d) an in-domain
  retrieval rate: fraction of generations whose nearest CLAP neighbour in
  `eval_L ∪ (generic panel generations)` is in `eval_L`. `T` is the tuple; the **primary
  scalar** for the band of §4.3 is (a); the others are reported. Generations on the same
  prompts *without* `L` give the adapter-effect baseline.
* **Primary adapters: `--include transformer.layers`** (backbone only), adapter type
  **`lora`** (the primary correspondence of §3.4). The repo's default otherwise parametrises
  **both the diffusion model and the conditioner** (`lora.md` "How LoRA Training Works" step
  3) — a conditioner adaptation would transfer through a component we never prune.
* **Secondary / sensitivity:** `lora-xs` adapters (matched to `U_xs`, if trained);
  `dora-rows` (external validity); full default model + conditioner (reported separately,
  never an anchor).
* **Compatibility check before any pruning:** `T(L; dense base)` and `T(L; dense post)` define
  the band of §4.3. If the adapter's effect on the dense post is absent or uncorrelated with
  its effect on the dense base (field and task), **the adapter axis is dropped** and RQ2 is
  reported as "no contract to preserve"; RQ1 proceeds.
* **Held-out set for RQ2:** ≥ 4 adapters from distinct SFX **domains** is the **feasibility
  minimum** **[pre-registered constant]**, not sufficient evidence for a broad "unseen
  adapters" claim; the unit of generalisation is the domain, not `lora` vs `dora`. None of
  them is used for any selection, `κ`, `N_main` or `n_u` decision.
* **Data prerequisite (the only external-data dependency):** 20–50 captioned 44.1 kHz clips
  per domain, CC0 (Freesound). Not held; a separate, small, licensed-data task. Until it
  exists, RQ1 and the `A_tan` tables can run; RQ2 cannot be interpreted.

---

## 6. The strongest adversary, first (S5)

In the **first GPU job** after the smoke, before any `I_PT` or `A` is read:

1. **Single-block tolerance of the post-trained 8-step model.** For every `g`: `E({g})` (§9) vs
   the dense post model at 7 and 8 steps (bracket for k = 1: 152 block-passes ∈ {140, 160}) and,
   once measured, vs its latency-matched dense comparator. If **every** `E({g})` is inferior
   (§9.2) to the latency-matched dense comparator, record **CASE E** and stop.
2. **Post-only greedy baselines (non-additive).** `R_E^greedy(k)` (§3.5, multivariate rule —
   the adversary that must exist for any method claim) and `R_{D_P}^greedy(k)` (field), with
   `E` of the resulting sets measured at k = 2, 4, 6 against the bracket and the
   latency-matched comparator. If `R_E^greedy` is not run, the fact is recorded and §8 C/D
   cannot authorise RQ3.

---

## 7. RQ1 / RQ2 design

* **Stage 1 (this protocol).** (a) Leave-one-block-out **analysis**: 20 variants × {base,
  post} × states; all §3 quantities; additivity diagnostics. (b) **Greedy selections** (§3.5)
  for `D_P`, `A_tan` (`U_gen`, `U_xs`), `I_PT^raw`, `I_PT^dep`, and `E` if affordable. (c)
  Cross-criterion statistics (§4) computed once, on the main panel, after §2.3 freezes `N_main`, `n_u`. (d)
  `A_eco` on held-out adapters and the prediction check `A_tan → A_eco`.
* **Stage 2 (conditional, specified, not executed).** Only if §4.1 divergence is real:
  exact enumeration of masks for k ≤ 4 (C(20,2) = 190, C(20,3) = 1 140, C(20,4) = 4 845) on the
  field proxies and the exact deployment/adaptability front; k = 6 (38 760) by stratified
  sampling. Costed from the measured s/forward. **Not authorized by this document.**
* **Replication (RQ4, excluded):** the `small-music-base` / `small-music` pair exists; untouched.

---

## 8. Result matrix (frozen before data)

"Coincide" = `δ_XY(k) ≤ max(f_X(k), f_Y(k))` at every k ∈ {2, 4, 6} (§4.1); "flat" = the
across-block spread of `I_PT` within its bootstrap spread; "predicts" =
`δ_{A_tan,A_eco}(k) ≤ max floor` for held-out adapters.

| case | pattern | decision |
|---|---|---|
| **A** | `R_{D_B^common}`, `R_{D_P}`, `R_{A_tan}` coincide; `I_PT` flat | Close. Negative analysis note. No RQ3. |
| **B** | `R_{D_P} ≠ R_{D_B^common}` or `I_PT` structured, **but** `R_{A_tan}` coincides with `R_{D_P}` (or `A_tan` does not predict `A_eco`) | PT-aware analysis is a possible smaller paper; the adapter axis adds nothing. RQ3 may be *designed* without the adaptability constraint only if `I_PT^dep` vs `I_PT^raw` shows real guidance-internalisation structure. |
| **C** | `R_{A_tan} ≠ R_E` (and `≠ R_{D_P}`) above both floors; `A_tan` predicts `A_eco`; the band condition of §4.3 holds for the majority of held-out adapters; `R_{A_tan}^greedy(k)` non-inferior (§9.2) to `R_E^greedy(k)` or to its latency-matched dense comparator | Adapter-compatible pruning justified. Proceed to **design** RQ3 — **only if `R_E^greedy` was run**; otherwise C is reported as "analysis supports, method unauthorised". |
| **D** | `I_PT` and `A_tan` both diverge from `D_P`/`R_E` and from each other, with C's conditions met | Strongest case; RQ3 design with both constraints; stage 2. Same `R_E^greedy` requirement as C. |
| **E** | Every single-block removal is inferior to its latency-matched dense comparator (§6.1) | Depth pruning without repair is not a deployment option here. The proposal dies; "mask + repair" (2607.06335 territory) is **not** pursued from this protocol. Report. |

Mixed patterns are reported as such and do not trigger RQ3. Secondary pre-registered
readings (no decisions): `ρ(W, I_PT)`, `ρ(I_PT^raw, I_PT^dep)`, `A_carry` vs `A_int`,
`U_gen` vs `U_xs`, per-level `I_PT,i`, additivity gaps, `D_B^common` vs `D_B^deploy`.

---

## 9. End-to-end validation (mechanism ≠ consequence)

### 9.1 Generation and metrics

8-step ping-pong generation on the fixed panel, seed-paired, decoded through SAME-S. Per system:

* **CLAP score** — text–audio cosine (LAION-CLAP, as ARC §3.3); higher is better.
* **KL_passt** — prompt-paired KL between the system's PaSST posteriors and the **dense post
  model's** on the same prompt and seed; lower is better (drift from dense).
* **FD_openl3** — Fréchet distance between the system's OpenL3 embeddings and the **dense post
  model's** over the panel; lower is better (drift from dense).
* **Optional, secondary:** a pre-registered T2A-bench subset (Gemini 2.5 Pro judge, paid API).
* **Not used:** PANNs top-10 recall.
* **Measurement uncertainty (not a margin):** the dense post model is generated with
  `R = 5` **[pre-registered constant]** independent seed streams on the panel. They give, per
  metric, the **distribution** of dense-vs-dense differences (CLAP: paired differences across
  streams; KL / FD: each stream against each other). Every system-vs-system comparison is
  seed-paired (stream 1) and its CI is obtained by prompt bootstrap **and** by repeating the
  comparison on the other streams; CIs, not single realised differences, enter every rule.

### 9.2 `E(M)` — operational definition

`E(M) = (CLAP(M), KL(M), FD(M))`.

* **Primary scalar** (for `C_dep` and for ordering within the multivariate rule): `CLAP(M)`.
* **Non-inferiority margins (operational, fixed on the pilot before any mask is evaluated):**
  `m_CLAP`, `m_KL`, `m_FD` = the degradation the **dense** post model incurs from dropping
  **one sampling step (8 → 7)** on the pilot panel, each as the point estimate of the paired
  difference (CLAP) / drift (KL, FD). Rationale: one step is the smallest deployment trade-off
  the model's own inference guide treats as acceptable ("reduce this number at some cost to
  quality", `inference.md:52`); it is mask-independent and measured, not chosen. Recorded in
  the ledger before the main panel is touched.
* **Non-inferiority of system X to system Y [pre-registered rule]:** the **upper 95 % CI
  bound** (§9.1 uncertainty) of the paired CLAP deficit `CLAP(Y) − CLAP(X)` is ≤ `m_CLAP`
  **and** the upper CI bounds of `KL(X) − KL(Y)` and `FD(X) − FD(Y)` are ≤ `m_KL`, `m_FD`.
* **Inferior / dominated:** X is inferior to Y if the *lower* CI bound of the CLAP deficit
  exceeds `m_CLAP`, **or** both drift deficits' lower CI bounds exceed their margins. Anything
  else is **"indeterminate"** (reported as such; never counted as non-inferior).
* Every `E` comparison in §4, §6 and §8 uses this rule; numbers are always shown as the tuple
  with CIs.
* Adapter-related end-to-end quantities use the task metric `T` of §5.2, not `E`.

---

## 10. Implementation plan (nothing here is run on GPU by this document)

**Environment.** `uv` venv `.venv-sa3` (Python 3.10; `stable-audio-3` pins `torch 2.7.1` /
`torchaudio 2.7.1`), from the pinned commit `a0b57f54…`, `--extra lora`. The frozen AudioLDM
`.venv` (torch 1.13.1) is untouched. T4 (sm_75): fp16 (no bf16, no Flash-Attention; `small`
needs none). `norm_kwargs.force_fp32 = True` is in the config; the smoke measures `η` (§3.2).

**Layout:**

```
research_sa3/
  loading.py        load base/post via the stable_audio_3 factory; sha256; key/shape diff; config diff (step 0)
  blockskip.py      BlockMask: per-layer skip flag (identity); context manager; bit-exact when empty
  states.py         callback capture of S_traj (post, 8 ping-pong) and S_traj^B (base, 50 Euler, CFG 7/APG 1);
                    S_noised; panel/seed/schedule files with sha256
  fields.py         raw field (_forward); deploy field (cond/uncond raw calls -> APG 1.0 primary, vanilla secondary);
                    padding-masked norms; per-level accumulators
  probes.py         U_gen / U_xs (base U,V) through LoRAParametrization; kappa rescaling; kappa grid; restriction u_{-g}
  metrics.py        D_B^common, D_B^deploy, D_P, I_PT (pooled + per-level + eta guard), W, A_carry, A_int, A_tan, A_eco
  greedy.py         sequential greedy selection for any criterion (105 evaluations to k=6); additivity gap
  e2e.py            8-step generation; SAME-S decode; CLAP / KL_passt / FD_openl3 vs dense; seed floors phi; E tuple + rule
  latency.py        wall-clock per generation for dense 4..8 steps and pruned systems; nearest-latency comparator
  samplesize.py     pilot-based N_main / n_u rules of section 2.3 (disagreement-count floors vs N; probe bootstrap)
  taskmetric.py     per-adapter T(L; system): CLAP audio-audio to eval_L, CLAP text, FD_openl3 to eval_L, retrieval rate
  report.py         JSON writer (schema below) + markdown tables
configs/sa3/
  panel_pilot.json, panel_main.json (disjoint), seed_table.json (R=5 dense streams), schedule_post_10s.json,
  probes.json, adapters/{domain}/{split.json, prompts_L.json} (hashed before training)
scripts/sa3/
  step0_verify_pair.py   (CPU) hashes, key/shape/config diff -- MUST pass first
  smoke_t4.py            (GPU, one short job) s/forward at batch {1,4,8}; VRAM; eta_i (fp16 vs fp32 per level); latency of
                         dense at 4..8 steps and of one block-removed variant; E of dense at 8/7/6/5 steps (pilot panel)
  rq_adversary.py        section 6: single-block E table; D_P-greedy (and E-greedy if affordable); brackets; comparators
  rq1_rq2_stage1.py      section 3 quantities; kappa grid; greedy paths; section 2.3 rules; section 4 statistics; case
  train_control_loras.py wraps scripts/train_lora.py: L_6 / L_13 (--include "transformer.layers[6]" / "[13]"),
                         primary (--include transformer.layers, lora r16) and secondary (lora-xs, dora-rows, full)
                         adapters; consumes the frozen splits; records configs + hashes
tests/sa3/
  test_blockskip.py      empty mask == original (bit-exact); skip g changes output; skip-all == project_in/out path
  test_fields.py         deploy field == DiffusionTransformer.forward(cfg_scale=7, apg_scale in {1.0, 0.0}) (bit-exact)
  test_probes.py         U_xs probe == add_lora('lora-xs') with the same M (bit-exact); kappa rescaling; linearity check
  test_metrics.py        synthetic tables -> pooled/per-level I_PT with guard; greedy paths; floors; non-inferiority rule; case
  test_localized_probe.py  the section 5.1 UNIT TEST (attribution code), on a tiny random-weights DiT
  test_samplesize.py     section 2.3 rules on synthetic data (pilot-only; stop on set-floor = 0, never on divergence)
  test_noninferiority.py section 9.2 rule on synthetic tuples with CIs (non-inferior / inferior / indeterminate)
```

**Output JSON (every run):** `{git_commit, upstream_commit, ckpt_sha256{base, post, svd_bases},
config_diff, panel_sha256, seed_table_sha256, schedule{tau_1..8}, cfg{s, apg_scale, scale_phi},
eta, kappa_grid, N_p, n_u, samplesize_trace, per_block[20]{D_B_common, D_B_deploy, D_P,
I_PT_raw{pooled, per_level[8]}, I_PT_dep{…}, W, A_carry, A_int, A_tan{gen, xs, by_kappa},
A_eco{adapter}, E_single{tuple}}, greedy_paths{criterion}{k}{set, E_tuple, additivity_gap},
floors{bootstrap_jaccard, phi}, brackets, latency{system}, comparators, case, gpu{name,
vram_peak_gb, s_per_forward{batch}}, wall_s}`. Results gitignored under `artifacts/sa3/`; hashes
to the ledger.

**CPU dry-run (free, before any GPU):** step 0; `BlockMask` identity on real weights; field
evaluation on 2 prompts × 2 states in fp32 (no decode, cached conditioning); one probe on one
block; greedy/metrics/rules on synthetic tables; localized-probe unit test.

**GPU smoke (one job; cost TBM):** `smoke_t4.py` as listed. Nothing is costed until it reports
**s/forward (batch 1/4/8), peak VRAM, `η`, s/generation at 8 steps incl. decode, and latency of
dense at 4–8 steps**. Upstream README reference points (H200, not T4, not ours): inference peak
1.69–2.40 GB for `small`; LoRA training ≈ 2.5 GB — VRAM bounds only.

**Cost model (filled only from the smoke):** field stage forwards = `N × 8 × [ 2 models ×
21 variants + CFG uncond branch × 21 + Σ_κ n_u × 21 × 2 families ]` plus greedy paths
(`105 × N × 8` per field criterion); end-to-end = `N × (R = 5 dense streams + 20 + 5 dense
step counts + greedy sets at k = 2, 4, 6 + adapters × …)` eight-step generations + decodes;
`E`-greedy adds `105 × N` generations (mandatory for any method authorisation). `N` is the
pilot size first, then `N_main` from §2.3; recorded in the ledger with the smoke.

---

## 11. Adoption gate

* Authorized by Gabriel (2026-08-20 23:14 "go"; 23:35 rc2 request): **writing this protocol**.
  Not authorized: any GPU job, any method, any LoRA recovery, RQ3/RQ4.
* **Authorised by the second review (2026-08-20 23:44): Step 0 (CPU), `.venv-sa3`,
  checkpoint/config verification, `research_sa3/` skeleton, `tests/sa3/`, CPU dry-run.**
* **Before the first GPU job** (explicit go): step 0 passes; tests and CPU dry-run pass; pilot
  and main panels, seed table (R = 5), schedule hashed in the ledger; novelty ledger (§12)
  re-read. **S1 is satisfied by the pilot panel (§2.3), not by a theoretical `I_PT` range** —
  no such number is written anywhere.
* **Before the main panel is touched:** pilot results recorded (magnitudes, variances,
  `N_main`, `n_u`, margins `m_·`, dense-stream distributions).
* **Before RQ2 is interpreted:** the single-block controls `L_6`, `L_13` of §5.1 pass; the
  primary ecological adapters of §5.2 exist with frozen splits and task metrics, and their
  dense-transfer bands are measured; `N_main`, `n_u` frozen by §2.3.
* **After RQ1/RQ2:** cases **C** / **D** permit *designing* RQ3 (constrained formulation,
  stage-2 enumeration, held-out adapter validation) as a new protocol document **only if
  `R_E^greedy` was run**; otherwise they are reported as analysis results without method
  authorisation. Cases **A**, **B without guidance structure**, **E** close the line or reduce
  it to an analysis note. No result here approves a pruning method.

---

## 12. Novelty ledger (operational; re-read at every gate)

| prior work | what they did | what we do differently | claim we will **not** make |
|---|---|---|---|
| **TinyFusion** (CVPR 2025) | Learns a depth-pruning mask for visual DiTs by optimising post-pruning *recoverability* (fine-tuning incl. LoRA) of **one** model. | Training-free measurement of which blocks carry a *transformation* (base→post) and an *adaptation subspace*; the object is a **family**, not one checkpoint's recoverability. | "first recoverability-aware depth pruning of DiTs"; "beats TinyFusion at recovering one model". |
| **2607.06335** (teacher-aligned repair) | Pruning an EDM2-XS teacher then one-step distillation (SiDA) fails without a teacher-matching repair stage (ImageNet-512). | No distillation, no repair; analysis of an already post-trained few-step model and its base. Case E stops rather than repairs. | "pruning and few-step distillation (do not) commute"; any repair method. |
| **Dynamic-in-Few-Step** (2607.06631) | Structured sparsification inside few-step distillation for video (Wan-14B), step-specific sub-models. | No distillation/training; one mask shared across steps and across the base/post pair. | "joint pruning + few-step distillation"; step-specific sparsity. |
| **CAR-LoRA** (ICLR 2026) | One LoRA trained robust to simulated compressions (quantisation/pruning) and evolved bases (LLMs) — **adapter-side**. | **Backbone-side**: which compressed backbone keeps *ordinary, unseen, untouched* adapters working; CAR-LoRA is the conceptual "pay per adapter vs once in the backbone" baseline. | "first work on LoRA robustness to compression / model evolution". |
| **TALL-Masks / NPS** (ICML 2024; task-vector compression) | Localise/compress task-specific information in fine-tuned *weights*. | Functional, normalized `I_PT` on the field; structural intervention (whole blocks); `W` is a covariate; `ρ(W, I_PT) ≈ 0` would be a result. | "task-vector-aware pruning"; weight-delta preservation as the method. |
| **Tangent-space task arithmetic** (Ortiz-Jiménez et al., NeurIPS 2023); linearised fine-tuning | Fine-tuning/editing in the tangent space; weight disentanglement via the Jacobian. | Tangent view used to *define* `A_tan` (Jacobian restricted to LoRA directions, measured functionally, κ-convergence required) and to test whether it predicts real held-out adapters. | "LoRA tangent-space preservation" as a new concept. |
| **EcoDiff** (2412.02852) | Differentiable-mask structural pruning of SDXL/FLUX, usable on timestep-distilled models. | Base-vs-post structural analysis and adaptability; depth granularity; audio DiT. | "first pruning of a distilled/few-step diffusion model". |
| **Compress-then-Serve** (PMLR v267) | Compresses collections of LoRAs for serving. | Motivates the many-adapter setting; we compress the backbone. | — |

---

## 13. Inheritance list (S3)

| inherited element | from | why necessary here | alternative |
|---|---|---|---|
| AudioCaps test **captions** as prompt panel | v3/v4 data; ARC eval set | Text-only, held, same captions ARC evaluated on; SFX domain | Freesound CC0 descriptions (would also give real 44.1 kHz latents) |
| CLAP score / KL_passt / FD_openl3 | ARC §3.3 | Published metrics for this family; dense-referenced variants avoid the 16 kHz reference problem | T2A-bench subset (secondary) |
| Seed pairing, hashed manifests, ledger discipline | plan v4 §6, AGENTS.md | Variance reduction, provenance | — |
| **Not inherited:** AudioLDM, PANNs top-10, L1 filter pruning, CLAP-swap conditioning, Arshdeep's checkpoints | v3/v4 | Not necessary for the question | — |
