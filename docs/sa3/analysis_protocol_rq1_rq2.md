# SA3 analysis protocol — RQ1 / RQ2 (analysis-first) — **rc2**

**Status: ANALYSIS PROTOCOL rc2, 2026-08-20 23:35 (Montevideo). Revision of rc1 (commit
`76fdcda`) after the external protocol review. Direction and architecture accepted by the
reviewer; rc2 fixes the eight mandatory points and the two minor ones below. No GPU has been
run. No pruning method, no LoRA recovery, no RQ3/RQ4 design is part of this document (§11).**
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
8. **Pre-registered sample-size rules** for `N_p` and `n_u` from precision/stability of each
   criterion's own quantities, never from cross-criterion divergence (§2.3).
Minor: `D_B^common` vs `D_B^deploy` (§3.1); the phase is *analysis-first* (forward-only
structural analysis + one ecological adapter validation that requires LoRA training), not
strictly forward-only.

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

### 2.3 Sample-size rules (pre-registered; fixed before any RQ1/RQ2 read-out)

`N_p` (prompts) and `n_u` (probes) are chosen by **precision of each criterion's own
quantities**, never by the divergence between criteria, never by any RQ1/RQ2 outcome.

* **`N_p` rule.** Start at `N_p^(0)` = the largest panel the smoke's measured s/forward allows
  within the authorized job (TBM). Using **only** `D_P(g)` and `I_PT(g)` of the dense/post pair
  (RQ1 quantities of one criterion at a time) and the positive-control runs, compute by prompt
  bootstrap (B = 1 000 **[pre-registered constant]**): (i) the 95 % CI half-width of each
  per-block score; (ii) the Jaccard stability of that criterion's **own** greedy sets
  `R_X(k)`, k ∈ {2, 4, 6}, across bootstrap replicates. Stop when (i) ≤ 0.10 × the inter-block
  IQR of the score **[pre-registered constant]** for every block and (ii) ≥ 0.90
  **[pre-registered constant]** for every k; otherwise double `N_p` and repeat. If the budget
  cap is reached first, the result is reported as **underpowered** and §8 is not applied.
* **`n_u` rule.** Increase `n_u` (doubling from 8 **[pre-registered constant]**) until the
  Monte-Carlo standard error of `A_tan(g)` over probes is ≤ 0.10 × its inter-block IQR
  **[pre-registered constant]** for every block, for each probe family separately. Never
  adjusted after `A_tan` is compared with anything.
* The **cross-criterion** statistics of §4 are computed once, after `N_p` and `n_u` are frozen
  and written to the ledger.

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
* **Denominator guard (measured, not chosen):** `η` = the fp16-vs-fp32 discrepancy of the post
  field on CPU-reference states, `η = ‖F_P^{fp16} − F_P^{fp32}‖² / ‖F_P‖²` (smoke). Levels with
  `den_i / ‖F_{P,i}‖² < η` are reported but excluded from the pooled ratio and flagged.
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
probes, `κ`, `N_p`, `n_u` or any set.

**RQ2 claim structure.** (i) `A_tan` (blind) **predicts** `A_eco` across blocks for held-out
adapters (pre-registered: rank correlation and, decision-relevant, agreement of the greedy
sets they induce, §4.1); (ii) the adaptability-aware greedy set differs from the post-only
greedy set (§4.1). Both are required for case C/D.

### 3.5 Selections (sets) — non-additive

For a per-block or per-set score `X`, the **sequential greedy** selection `R_X^greedy(k)`:
start from the dense post model; evaluate all 20 single removals; remove the best (smallest
`X`); re-evaluate all 19 remaining removals *from the new architecture*; repeat to k = 6. That
is 20 + 19 + 18 + 17 + 16 + 15 = **105 candidate evaluations** per criterion; sets at k = 2, 4,
6 are read off the path. Criteria: `D_P` (field proxy), `E` (end-to-end, §9 — the strongest
post-only adversary, run if the smoke shows 105 × `N_p` generations are affordable; otherwise
`D_P`-greedy is the primary adversary and `E`-greedy is run at k = 2, 4, 6 only for the
`D_P`-greedy path's alternatives, reported as such), `A_tan` (adaptability-aware, field),
`I_PT` (PT-aware, field). Single-block leave-one-out tables (§7) are **analysis**, not
selection; the additivity gap `X(R(k)) − Σ_{g∈R(k)} X({g})` is reported as a diagnostic.

### 3.6 End-to-end `E(M)`

Defined operationally in §9.2 as a **tuple** `(CLAP, KL, FD)` with a non-inferiority relation
and a primary scalar; never used as an abstract scalar elsewhere.

---

## 4. SESOI in decision terms (S7) with measured anchors

### 4.1 Set divergence

`δ_XY(k) = |R_X^greedy(k) △ R_Y^greedy(k)| / 2` for the pairs (`D_P`, `A_tan`), (`D_P`, `I_PT`),
(`D_B^common`, `D_P`), and (`A_tan`, `A_eco`) (prediction check). **Floor:** the bootstrap
instability of each criterion's *own* greedy set (§2.3 (ii)): a divergence between criteria is
real only if it exceeds `1 − Jaccard` of the within-criterion bootstrap at the same k.

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

### 4.3 Gain in adaptability and its anchor

`G_adapt(k) = A_eco(R_{D_P}^greedy(k)) − A_eco(R_{A_tan}^greedy(k))`, evaluated **on the held-out
adapters** (leave-set-out, not summed), averaged over adapters. **Anchor (measured, §5.2):**
`L_ceil` = adapter-effect loss the official dense base → dense post transfer already incurs. A
`G_adapt` below `L_ceil` is within what the official pipeline already accepts (**not
material**); above it, material.

### 4.4 Minimal result that would justify a new pruning method

To be confirmed by the measured anchors, not signed here: at k = 4,
`δ_{D_P,A_tan}(4) ≥ 1` above the floor **and** `δ_{A_tan,A_eco}(4) = 0` within the floor
(prediction holds) **and** `G_adapt(4) ≥ L_ceil` **and** `R_{A_tan}^greedy(4)` non-inferior
(§9.2) to `R_{D_P}^greedy(4)` or, failing that, non-inferior to its latency-matched dense
comparator. If the floor, `L_ceil`, or the latency-matched comparator cannot be measured, the
SESOI is declared **not yet defensible** and the main experiment is not signed.

---

## 5. Positive controls (S2) — before any null reading

### 5.1 Synthetic, localized — two levels

* **Unit test (code only, not evidence):** a generic probe injected **only** into blocks 4–7;
  `A_carry` must attribute 100 % to 4–7 and `A_tan(g)` for `g ∉ {4..7}` must equal the
  interference term. This validates the restriction/attribution code; it cannot fail
  scientifically because the attribution knows the support. Lives in `tests/sa3/`.
* **Scientific positive control (mandatory before RQ2 is interpreted):** a **trained** LoRA
  `L_4-7` on `small-sfx-base` with `--include "transformer.layers[4-7]"` (bracket syntax per
  `lora.md`), applied to the dense post. The instruments that will carry the RQ2 conclusion —
  `A_eco(g; L_4-7)` computed **functionally from outputs** (field on `S_traj` and end-to-end
  adapter effect on generated audio, §9) — must rank blocks 4–7 as the four most damaging
  removals for this adapter, with every other block at the interference floor; and the
  `A_eco`-greedy selection for this adapter must avoid 4–7 at k ≤ 6. **Fail ⇒ the measurement
  chain cannot localize a known adaptation from outputs; RQ2 is not read.** Requires the same
  data as §5.2.

### 5.2 Ecological — does the compatibility contract exist before pruning?

Train real adapters on `small-sfx-base` with the repo recipe (`scripts/train_lora.py`, rank 16,
1 000 steps per its quick-start; 20–50 clips per `lora.md`):

* **Primary adapters: `--include transformer.layers`** (backbone only), two adapter types
  (`lora` and `dora-rows`, the default), `exclude seconds_total` implied by the include. This
  confines the adaptation to exactly the structural space our analysis covers. The repo's
  default otherwise parametrizes **both the diffusion model and the conditioner**
  (`lora.md` "How LoRA Training Works" step 3) — a conditioner adaptation would transfer
  through a component we never prune and would contaminate `L_ceil`.
* **Secondary (sensitivity):** the same data with the full default (model + conditioner),
  reported separately, never used as anchor.
* Apply each to dense base and dense post; measure `δF_B(L)`, `δF_P(L)` on the common states
  and the end-to-end adapter effect with/without `L`. `L_ceil = ‖δF_P(L) − δF_B(L)‖² /
  ‖δF_B(L)‖²` (field) and its end-to-end analogue. **If the adapter effect on the post is
  uncorrelated with that on the base, the adapter axis is dropped** and RQ2 is reported as
  "no contract to preserve"; RQ1 proceeds.
* **Held-out set for RQ2:** ≥ 4 adapters **[pre-registered constant]** from distinct SFX
  domains, none used for any selection, `κ`, `N_p` or `n_u` decision.
* **Data prerequisite (the only external-data dependency):** 20–50 captioned 44.1 kHz clips per
  domain, CC0 (Freesound). Not held; a separate, small, licensed-data task. Until it exists,
  RQ1 and the `A_tan` tables can run; RQ2 cannot be interpreted.

---

## 6. The strongest adversary, first (S5)

In the **first GPU job** after the smoke, before any `I_PT` or `A` is read:

1. **Single-block tolerance of the post-trained 8-step model.** For every `g`: `E({g})` (§9) vs
   the dense post model at 7 and 8 steps (bracket for k = 1: 152 block-passes ∈ {140, 160}) and,
   once measured, vs its latency-matched dense comparator. If **every** `E({g})` is inferior
   (§9.2) to the latency-matched dense comparator, record **CASE E** and stop.
2. **Post-only greedy baseline (non-additive).** `R_{D_P}^greedy(k)` and, if affordable,
   `R_E^greedy(k)` (§3.5), with `E` of the resulting sets measured at k = 2, 4, 6 against the
   bracket and the latency-matched comparator. These are the adversaries every later set is
   compared against.

---

## 7. RQ1 / RQ2 design

* **Stage 1 (this protocol).** (a) Leave-one-block-out **analysis**: 20 variants × {base,
  post} × states; all §3 quantities; additivity diagnostics. (b) **Greedy selections** (§3.5)
  for `D_P`, `A_tan` (`U_gen`, `U_xs`), `I_PT^raw`, `I_PT^dep`, and `E` if affordable. (c)
  Cross-criterion statistics (§4) computed once, after §2.3 freezes `N_p`, `n_u`. (d)
  `A_eco` on held-out adapters and the prediction check `A_tan → A_eco`.
* **Stage 2 (conditional, specified, not executed).** Only if §4.1 divergence is real:
  exact enumeration of masks for k ≤ 4 (C(20,2) = 190, C(20,3) = 1 140, C(20,4) = 4 845) on the
  field proxies and the exact deployment/adaptability front; k = 6 (38 760) by stratified
  sampling. Costed from the measured s/forward. **Not authorized by this document.**
* **Replication (RQ4, excluded):** the `small-music-base` / `small-music` pair exists; untouched.

---

## 8. Result matrix (frozen before data)

"Coincide" = `δ_XY(k)` within the bootstrap floor at every k ∈ {2, 4, 6}; "flat" = the
across-block spread of `I_PT` within its bootstrap spread; "predicts" =
`δ_{A_tan,A_eco}(k)` within the floor for held-out adapters.

| case | pattern | decision |
|---|---|---|
| **A** | `R_{D_B^common}`, `R_{D_P}`, `R_{A_tan}` coincide; `I_PT` flat | Close. Negative analysis note. No RQ3. |
| **B** | `R_{D_P} ≠ R_{D_B^common}` or `I_PT` structured, **but** `R_{A_tan}` coincides with `R_{D_P}` (or `A_tan` does not predict `A_eco`) | PT-aware analysis is a possible smaller paper; the adapter axis adds nothing. RQ3 may be *designed* without the adaptability constraint only if `I_PT^dep` vs `I_PT^raw` shows real guidance-internalisation structure. |
| **C** | `R_{A_tan} ≠ R_{D_P}` above the floor; `A_tan` predicts `A_eco`; `G_adapt(k) ≥ L_ceil` for some k; `R_{A_tan}^greedy(k)` non-inferior (§9.2) to the post-only set or to its latency-matched dense comparator | Adapter-compatible pruning justified. Proceed to **design** RQ3. |
| **D** | `I_PT` and `A_tan` both diverge from `D_P` and from each other, with C's conditions met | Strongest case; RQ3 design with both constraints; stage 2. |
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
* **Seed floor:** the dense post model with a second, disjoint seed stream on the same panel
  gives, for each metric, `φ_CLAP` (absolute CLAP difference between the two seed streams),
  `φ_KL`, `φ_FD` (KL / FD of seed stream 2 against seed stream 1). These are the measured
  resampling floors.

### 9.2 `E(M)` — operational definition

`E(M) = (CLAP(M), KL(M), FD(M))`.

* **Primary scalar** (for `C_dep`, ordering along greedy paths, and `E`-greedy): `CLAP(M)`.
* **Non-inferiority of system X to system Y [pre-registered rule]:**
  `CLAP(X) ≥ CLAP(Y) − φ_CLAP` **and** `KL(X) ≤ KL(Y) + φ_KL` **and** `FD(X) ≤ FD(Y) + φ_FD`
  — X loses no more on any metric than seed resampling alone moves the dense model.
* **Inferior / dominated:** X is inferior to Y if it fails the CLAP condition **or** fails both
  drift conditions. (Failing exactly one drift cap with CLAP non-inferior is reported as
  "mixed"; it does not count as non-inferior.)
* Every `E` comparison in §4, §6 and §8 uses this rule; numbers are always shown as the tuple.
* Adapter-effect end-to-end analogue (§3.4, §5): the same tuple computed between "with L" and
  "without L" outputs of the same system; `A_eco` end-to-end = relative change of that tuple's
  CLAP component after removal (KL/FD reported).

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
  samplesize.py     N_p / n_u rules of section 2.3 (bootstrap CI, Jaccard stability, MC s.e.)
  report.py         JSON writer (schema below) + markdown tables
configs/sa3/
  panel_prompts.json, seed_table.json, schedule_post_10s.json, probes.json, heldout_adapters.json
scripts/sa3/
  step0_verify_pair.py   (CPU) hashes, key/shape/config diff -- MUST pass first
  smoke_t4.py            (GPU, one short job) s/forward at batch {1,4,8}; VRAM; eta (fp16 vs fp32); latency of dense
                         at 4..8 steps and of one block-removed variant; E of dense at 8/7/6/5 steps
  rq_adversary.py        section 6: single-block E table; D_P-greedy (and E-greedy if affordable); brackets; comparators
  rq1_rq2_stage1.py      section 3 quantities; kappa grid; greedy paths; section 2.3 rules; section 4 statistics; case
  train_control_loras.py wraps scripts/train_lora.py: L_4-7 (--include "transformer.layers[4-7]"), primary
                         (--include transformer.layers) and secondary (full) adapters; records configs + hashes
tests/sa3/
  test_blockskip.py      empty mask == original (bit-exact); skip g changes output; skip-all == project_in/out path
  test_fields.py         deploy field == DiffusionTransformer.forward(cfg_scale=7, apg_scale in {1.0, 0.0}) (bit-exact)
  test_probes.py         U_xs probe == add_lora('lora-xs') with the same M (bit-exact); kappa rescaling; linearity check
  test_metrics.py        synthetic tables -> pooled/per-level I_PT with guard; greedy paths; floors; non-inferiority rule; case
  test_localized_probe.py  the section 5.1 UNIT TEST (attribution code), on a tiny random-weights DiT
  test_samplesize.py     section 2.3 rules on synthetic data (stop only on precision, never on divergence)
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

**Cost model (filled only from the smoke):** field stage forwards = `N_p × 8 × [ 2 models ×
21 variants + CFG uncond branch × 21 + Σ_κ n_u × 21 × 2 families ]` plus greedy paths
(`105 × N_p × 8` per field criterion); end-to-end = `N_p × (1 + 20 + 5 dense step counts +
greedy sets at k = 2, 4, 6 + adapters × …)` eight-step generations + decodes; `E`-greedy adds
`105 × N_p` generations if run. `N_p`, `n_u` follow §2.3; recorded in the ledger with the smoke.

---

## 11. Adoption gate

* Authorized by Gabriel (2026-08-20 23:14 "go"; 23:35 rc2 request): **writing this protocol**.
  Not authorized: any GPU job, any method, any LoRA recovery, RQ3/RQ4.
* **Before the first GPU job** (explicit go each): step 0 passes on CPU; `tests/sa3/` pass; CPU
  dry-run passes; panel, seed table, schedule hashed in the ledger; S1 magnitudes for RQ1
  written down (predicted range of `I_PT` under "post-training reorganises structure" vs "it
  does not", argued from the training-pipeline description, not from `W`); novelty ledger
  (§12) re-read.
* **Before RQ2 is interpreted:** the trained localized control of §5.1 passes; the primary
  ecological adapters of §5.2 exist and `L_ceil` is measured; `N_p`, `n_u` frozen by §2.3.
* **After RQ1/RQ2:** cases **C** / **D** permit *designing* RQ3 (constrained formulation,
  stage-2 enumeration, held-out adapter validation) as a new protocol document. Cases **A**,
  **B without guidance structure**, **E** close the line or reduce it to an analysis note. No
  result here approves a pruning method.

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
