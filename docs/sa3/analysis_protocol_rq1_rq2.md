# SA3 analysis protocol — RQ1 / RQ2 (forward-only)

**Status: ANALYSIS PROTOCOL, written 2026-08-20 23:14 (Montevideo) on Gabriel's "go" for the
protocol only. No GPU has been run. No pruning method, no LoRA recovery, no RQ3/RQ4 design is
part of this document; those are gated on the results below (§11).** This is not a master
plan. It instantiates the adoption rules S1–S7 of
`docs/review/2026-08-20_postmortem_v3_v4.md` §3 (DECISION-RULES-001) for two analysis
questions on Stable Audio 3 `small-sfx-base` / `small-sfx`.

Every number in this document is either (a) read from a cited source (file + line, HF API,
paper) or (b) explicitly marked **TBM** (*to be measured*). There are no estimates presented
as measurements. Upstream code facts were read at `Stability-AI/stable-audio-3` commit
`a0b57f5483c4588f827f3552b7d5c6ca2a9687be` (main, 2026-08-02); line numbers refer to it.

---

## 0. Scientific question and decision

Stable Audio 3 ships, for the same 433 M DiT, a **base** checkpoint (rectified flow, ~50 Euler
steps with CFG) and a **post-trained** checkpoint (distillation warm-up + adversarial
post-training, 8 ping-pong steps, no CFG), and documents that **LoRAs are trained on the base
and applied unchanged to the post-trained model** (`docs/guides/model-overview.md`). The pair
is a controlled perturbation of one architecture. We ask two *analysis* questions about its
structure, block by block, before deciding whether any pruning method is worth building.

* **RQ1 — Does few-step post-training materially change structural importance?** For each of
  the 20 transformer blocks: how much does removing it damage the base field, the
  post-trained field, and — separately — the *post-training transformation* (the difference
  between the two fields)? Is the PT-transformation damage predictable from the plain
  post-trained damage, or from the parameter delta `‖W_P − W_B‖`?
* **RQ2 — Does sensitivity to low-rank adaptation occupy structure distinct from what the
  standalone 8-step generator needs?** For each block: how much of the functional effect of
  low-rank perturbations (LoRA-shaped, unseen, untrained) is carried by the block or disturbed
  by its removal, and do the blocks a *deployment-only* criterion would remove coincide with
  the blocks an *adaptability* criterion would keep?

**What would justify building a method (RQ3 design):** the removal sets chosen at k ∈ {2, 4, 6}
by the deployment criterion and by the adaptability (and/or PT) criterion **differ**, the
difference is above the panel-resampling noise floor, and respecting the adaptability criterion
costs the 8-step generator less than an iso-compute step reduction would (§4, §8 cases C/D).

**What kills the line:** the removal sets coincide at every k within noise and `I_PT` is flat
(case A); or the post-trained 8-step model does not tolerate the removal of even one block
without end-to-end collapse relative to the iso-compute yardstick (case E). Either is reported
as the result; no repair method is invented after seeing it.

---

## 1. Objects (exact)

### 1.1 Checkpoints and configs

| object | HF repo | access | file | size (bytes) | git blob oid (HF tree API, 2026-08-20) | sha256 |
|---|---|---|---|---|---|---|
| base | `stabilityai/stable-audio-3-small-sfx-base` | public, **not gated** | `model.safetensors` | 2 270 384 940 | `da83db9b43a3c2f028d45aa172fb938fd0958d81` | **TBM at download** (recorded in ledger) |
| base config | same | public | `model_config.json` | 8 476 | `bcc43173b590e4ff34e995dff4201d1a0d164d43` | TBM |
| base SVD bases | same | public | `svd_bases.pt` | 1 270 378 123 | `ce26e2ca33f43da49c86f5507aedc9d0cb6a23f1` | TBM |
| post | `stabilityai/stable-audio-3-small-sfx` | **gated = auto** (accept license) | `model.safetensors` | 2 270 384 940 | `eff880f5df27fa7a6289c737595f169028bea1f8` | TBM |
| post config | same | gated | `model_config.json` | 10 454 | `0321320a34e3e0ceda48e47c2c36b70234df746f` | TBM |

Both repos bundle the text encoder `t5gemma-b-b-ul2/`. **No `svd_bases.pt` in the post repo.**
License: Stability AI Community License (+ Gemma license for the text encoder); research use.

The two `model.safetensors` have **identical byte size**, consistent with identical tensor
shapes — *consistent with*, not proof of. **Step 0 (mandatory, CPU, before anything else):**
load both state dicts, assert identical key sets and shapes (report any difference), and diff
the two `model_config.json` files. The post config is 1 978 bytes longer; we could not read it
(HTTP 401 without the gate). Expected from code: `diffusion_objective` = `rectified_flow`
(base, read from the base config) vs `rf_denoiser` (post; `diffusion_cond.py:277` labels
`rf_denoiser` "includes ARC models"), plus sampling-shift options. **Any difference inside
`model.diffusion.config` (depth/width/heads/cond dims/norms) invalidates the 1:1 block mapping
and stops the protocol until resolved.**

### 1.2 Architecture (read from the base `model_config.json`)

`model.diffusion.type = dit`; `config`: `io_channels 256`, `embed_dim 1024`, `depth 20`,
`num_heads 16`, `cond_token_dim 768`, `global_cond_dim 768`, `local_add_cond_dim 257`,
`global_cond_type adaLN`, `timestep_features_type expo`, `timestep_features_logsnr False`,
`attn_kwargs.qk_norm rms`, `norm_type rms_norm` (`force_fp32 True`), `ff_kwargs.mult 4.0`,
`num_memory_tokens 64`. Conditioning: `prompt` (`t5gemma`, cross-attention tokens) and
`seconds_total` (`number`, global). Pretransform: SAME autoencoder, `latent_dim 256`,
`downsampling_ratio 4096`, 44.1 kHz stereo; `sample_size 5 324 800` samples (≈ 120.7 s).
Latent rate = 44 100 / 4 096 ≈ 10.77 frames/s (derived from the two config values).

### 1.3 The 20 blocks and the 1:1 mapping

`DiffusionTransformer.transformer` is a `ContinuousTransformer` whose `self.layers` is an
`nn.ModuleList` of 20 `TransformerBlock`s (`transformer.py:1103,1142-1144`); the forward
iterates them in order (`transformer.py:1235`), with `return_info=True` exposing every
block's hidden state and `exit_layer_ix` supporting early exit (`transformer.py:1169-1171,
1254-1262`). Parameter names are `transformer.layers.{i}.…` (e.g.
`transformer.layers.0.self_attn.to_qkv`, per `docs/workflows/lora.md`). Block `g ∈ {0,…,19}`
in the base is matched to block `g` in the post by name; step 0 asserts the per-block
parameter shapes agree.

Each block is residual (`x = x + residual` around self-attention, cross-attention and
feed-forward in `TransformerBlock.forward`, `transformer.py:1023-1044`; branches are
`zero_init_branch_outputs=True`, `transformer.py:870`), so **removing block g = identity on its
input** — no re-wiring, no
dimension change, and any LoRA slice on a surviving block keeps exactly its geometry.

### 1.4 Fields, samplers, CFG (read from code)

Both checkpoints are called as `v = model(x, t, **cond)` and both samplers use the same
rectified-flow parameterisation `x_t = (1 − t)·x_0 + t·ε`, `t ∈ [0, 1]`, `denoised = x − t·v`
(`sampling.py:147-186` Euler; `sampling.py:308-353` ping-pong: `denoised = x − t_i·v`, then
`x = (1 − t_{i+1})·denoised + t_{i+1}·randn`). **Therefore `F_B` and `F_P` are functions of the
same `(x_t, t, c)` and can be evaluated on identical states.**

* **Base sampler (frozen):** Euler, 50 steps, `cfg_scale = 7.0` — the repo's defaults for
  `rectified_flow` (`diffusion_cond.py:344-351, 380-385`; `docs/workflows/inference.md:52-57`
  "use something like 50", "try 7.0").
* **Post sampler (frozen):** ping-pong, 8 steps, `cfg_scale = 1.0` (`sampling.py:434`
  default for `rf_denoiser`; `diffusion_cond.py:346,350`; `inference.md:56`: CFG parameters
  "have no effect on post-trained checkpoints").
* **CFG (base only):** combined in denoised space and mapped back to velocity
  (`dit.py:579-611`): `d_c = x − t·v_c`, `d_u = x − t·v_u`, `d_cfg = d_c + (s − 1)(d_c − d_u)`,
  `v_cfg = (x − d_cfg)/t`. The **production default is full APG** (`apg_scale = 1.0`: only the
  component of `d_c − d_u` orthogonal to `d_c` is used, `dit.py:370,599-602`; Gradio default
  `diffusion_cond.py:76,377`; `cfg_rescale` default 0.0, `diffusion_cond.py:74`). **We freeze
  the deploy field to the production default (`s = 7`, `apg_scale = 1.0`, `scale_phi = 0.0`,
  `cfg_norm_threshold = 0.0`, interval (0, 1))** because it is the behaviour the post-trained
  model replaces; vanilla CFG (`apg_scale = 0.0`) is computed as a secondary deploy field. Both
  settings are recorded in every output JSON.
* **Schedules:** `build_schedule(steps, sigma_max=1.0, dist_shift, effective_seq_len)`
  (`sampling.py:9-64`) = `linspace(1, 0, steps+1)` warped by the checkpoint's distribution
  shift (`base config: distribution_shift_options.type = full, min_length 256, max_length
  4096`; `use_effective_length_for_schedule = True`). The **8 post noise levels
  `τ_1 > … > τ_8`** are whatever the *post* checkpoint's own shift produces for the panel's
  `seconds_total` — computed by calling `build_schedule` with the post model's
  `sampling_dist_shift`, stored in the panel file, never typed by hand.
* **Padding / duration:** all panel prompts use one fixed `seconds_total` (§2) so the
  effective sequence length — and hence the schedule — is identical across systems.

### 1.5 Common states (the inputs on which all fields are compared)

* **Primary state set `S_traj`:** the states actually visited by the **dense post-trained
  sampler** on the panel: for prompt `p` with initial noise `ε_p` (seed-paired), the 8 pairs
  `(x_i, τ_i)` captured through the sampler `callback` (`sampling.py:347` ping-pong, `:182` Euler expose `x`, `t`, `i`,
  `denoised`). These are the states at which deployment happens; no external audio is needed.
* **Secondary state set `S_noised`:** ARC-style generator inputs `x_τ = (1 − τ)·x̂_0 + τ·ε'`
  at the same 8 levels, with `x̂_0` the dense post model's own final latent for the prompt
  (self-generated; independent noise `ε'`). This removes the "states depend on the post
  model's own trajectory" circularity for the *base*-side quantities.
* **Optional `S_real`:** noised latents of real 44.1 kHz audio. **Not available** — our
  AudioCaps copy is 16 kHz mono (M0-004). Listed so the gap is explicit; not required for
  RQ1/RQ2.
* Base-trajectory states (Euler-50 with CFG 7) are captured too, for the base-side damage
  `D_B` on its own deployment states (`S_traj^B`); they are *not* used for `F_B` vs `F_P`
  comparisons because the two trajectories differ by construction.

---

## 2. Panel, seeds, conditioning

* **Prompts:** a fixed, seeded subset of AudioCaps **test** captions (the evaluation set used
  by the ARC paper, arXiv 2505.08175 §3.3: 881 audios / 4 875 captions; we hold the captions
  in `data/dataset/metadata/audiocaps/`). Size `N_p`: **TBM** from the smoke's s/forward
  (§10); the protocol fixes the *construction* (master seed 20260818, one caption per source
  wav, stratified only by caption length tercile) and records the sha256 of the resulting
  `configs/sa3/panel_prompts.json`. Captions are text; no 16 kHz audio enters any SA3 model.
* **`seconds_total`:** one value for the whole panel, **10 s** (AudioCaps clip length; within
  the model's range; gives ≈ 108 latent frames at 10.77 f/s — derived, not measured).
* **Seeds:** per prompt `p`, one initial noise `ε_p` shared by **every** system (dense base,
  dense post, every block-removed variant, every probe) — seed pairing as in plan v4 §6.
  Ping-pong re-noising uses a second per-prompt, per-step seed stream, also shared across
  systems. The seed table is a file with a hash.
* **Conditioning cache:** T5Gemma token embeddings and the `seconds_total` embedding are
  computed once per prompt (dense model, CPU or GPU) and stored; the DiT is the only module
  evaluated repeatedly. `local_add_cond` (inpainting channels, dim 257) is the model's
  no-inpainting default, identical across systems.

---

## 3. Executable definitions

Notation: `F_B(x, t, c)`, `F_P(x, t, c)` are the **raw conditional velocity fields** of base
and post (`DiffusionTransformer._forward`, `dit.py:179`). `F^{−g}` is the same model with
block `g` replaced by identity. `‖·‖²` is the mean over a state set of the per-state squared
L2 norm over the latent `(256 × T)`, divided by `T` (padding-masked). All quantities are
computed on `S_traj` (primary) and `S_noised` (secondary) and reported for both.

**Deploy field of the base** (the function the post replaces in production):
`F_B^dep = F_B^{cfg}(s = 7, apg = 1.0)` = the production CFG/APG-combined velocity of §1.4 with
the unconditional branch obtained exactly as the DiT does (`dit.py:523-526`: null embedding
for the prompt); `F_B^{dep,vanilla}` (`apg = 0.0`) is the secondary.

**Block damage (RQ1):**

* `D_B(g) = ‖F_B − F_B^{−g}‖² / ‖F_B‖²`
* `D_P(g) = ‖F_P − F_P^{−g}‖² / ‖F_P‖²`

**Post-training delta and its damage (RQ1):**

* `Δ^raw(x,t,c) = F_P − F_B`  `Δ^dep(x,t,c) = F_P − F_B^dep`
* `I_PT^raw(g) = ‖Δ^raw − Δ^{raw,−g}‖² / ‖Δ^raw‖²`, with `Δ^{raw,−g} = F_P^{−g} − F_B^{−g}`
* `I_PT^dep(g)` likewise with `F_B^dep` and `F_B^{dep,−g}` (CFG recomputed on the
  block-removed base; `s = 7` fixed).

The normalisation by `‖Δ‖²` is mandatory: `I_PT` is the *fraction of the post-training
transformation destroyed* by removing `g`. Both versions are reported; their rank agreement is
itself a result (guidance-internalisation structure, see §8).

**Parameter-delta covariate (not an effect estimate, S1):**
`W(g) = Σ_{θ∈g} ‖θ_P − θ_B‖²_F / Σ_{θ∈g} ‖θ_B‖²_F` — reported alongside, used only for
`ρ(W, I_PT)` and `ρ(W, D_P)`.

**Adapter-tangent sensitivity (RQ2).** A probe `u` is a set of low-rank weight perturbations
`{ΔW_ℓ}` on the LoRA-eligible layers (`nn.Linear` / `nn.Conv1d` under `transformer.layers.*`,
the repo's own `--include transformer.layers` target set, `lora.md` "Layer Filtering"),
applied through the official parametrisation (`stable_audio_3/models/lora/model.py`) so
scaling semantics are exactly those of a real adapter (`lora_forward`: `W + scaling·strength·BA`;
`lora_xs_forward`: `W + scaling·strength·U M Vᵀ`, `model.py` lines shown in §1 of the ledger
entry PIVOT-ASSESSMENT-2). Two probe families, `n_u` probes each (`n_u` TBM by the noise-floor
procedure of §4):

* **Generic (`U_gen`):** rank `r = 16` (the `train_lora.py` default), `A` ~ Kaiming-uniform
  as in the repo's init, `B` ~ N(0, 1) (the repo zero-inits `B`; a zero probe has no effect, so
  we draw it), then rescaled per layer so that `‖ΔW_ℓ‖_F / ‖W_ℓ‖_F = κ`.
* **LoRA-XS in the base's bases (`U_xs`):** `ΔW_ℓ = U_ℓ M_ℓ V_ℓᵀ` with `U_ℓ, V_ℓ` = the
  **base checkpoint's** `svd_bases.pt` entries (top-`r` singular vectors of the base `W_ℓ`,
  `model.py:97-111`), `M_ℓ` ~ N(0, 1) `r × r`, rescaled to the same `κ`. **Never the SVD of the
  post weights**: the post repo ships no bases, and the official base→post transfer of an XS
  adapter is only defined in the base's bases (ledger PIVOT-ASSESSMENT-2).
* **`κ` (probe magnitude) is not chosen by us:** it is set so that the dense post model's
  relative output change `‖F_{P+u} − F_P‖ / ‖F_P‖` matches the one produced by the
  *ecological* LoRA of §5.2 (TBM). Until that LoRA exists, `κ` is swept over a small grid and
  all RQ2 quantities are reported as functions of `κ`, with linearity checked
  (`‖δF(2u)‖ ≈ 2‖δF(u)‖`).

Per probe `u` and block `g` define, on the post model:

* dense adapter effect: `δF(u) = F_{P+u} − F_P`
* effect after removing `g`, with the probe restricted to surviving blocks:
  `δF^{−g}(u) = F^{−g}_{P+u_{−g}} − F^{−g}_P`
* **carry share** `A_carry(g) = E_u ‖δF(u_g)‖² / E_u ‖δF(u)‖²` — the fraction of adapter
  effect that lives in block `g`'s own slots (lost with `g`);
* **interference** `A_int(g) = E_u ‖δF(u_{−g}) − δF^{−g}(u_{−g})‖² / E_u ‖δF(u_{−g})‖²` — how
  much removing `g` changes the effect of adapters on the *other* blocks;
* **total adaptability damage** `A(g) = E_u ‖δF(u) − δF^{−g}(u)‖² / E_u ‖δF(u)‖²` (with
  `δF^{−g}(u) := δF^{−g}(u_{−g})`), reported for `U_gen` and `U_xs` separately.

**Rankings and removal sets.** For any per-block score `X ∈ {D_B, D_P, I_PT^raw, I_PT^dep,
A_gen, A_xs}` and `k ∈ {2, 4, 6}`, `R_X(k)` = the `k` blocks with the smallest `X`. Leave-one-out
scores do **not** assume additivity (§7); `R_X(k)` is the *induced* set a naive pruner would
choose, which is exactly what the decision statistics of §4 compare.

**End-to-end damage (validation, §9):** `E(M)` for a removal set `M` = the 8-step ping-pong
generation metrics of §9 relative to the dense post model on the same panel and seeds.

---

## 4. SESOI in decision terms (S7) — and the cheap measurements that fix the numbers

No Spearman threshold is used. Three decision statistics, each with a noise floor measured
before interpretation:

1. **Set divergence** `δ_XY(k) = |R_X(k) △ R_Y(k)| / 2` — number of blocks on which two
   criteria disagree at budget `k`, for the pairs (`D_P`, `A`), (`D_P`, `I_PT`), (`D_B`, `D_P`).
   *Noise floor:* split the panel into two disjoint halves (and, separately, swap the seed
   stream), recompute `R_X(k)` on each half, and take `δ_XX(k)` between halves. A divergence
   between criteria is **real** only if it exceeds the within-criterion half-split divergence
   on every `k` where it is claimed.
2. **Cost of respecting adaptability:** `C_dep(k) = E(R_A(k)) − E(R_{D_P}(k))` — end-to-end
   deployment loss paid for choosing the adaptability-preserving set instead of the
   deployment-only set.
   *Yardstick (not invented):* the **iso-compute step reduction**. Removing `k` of 20 blocks at
   8 steps costs ≈ 8·(20 − k)/20 block-passes; the dense post model at `⌊8·(20 − k)/20⌋` steps
   (k = 2 → 7, k = 4 → 6, k = 6 → 5; a plain FLOP count, one block-pass per block per step) is
   the deployment alternative that needs *no* pruning at all. `E` of that dense-fewer-steps
   system is measured on the same panel; any block-removed system that is *worse* than it is
   dominated for deployment.
3. **Gain in adaptability:** `G_adapt(k) = A_total(R_{D_P}(k)) − A_total(R_A(k))`, with
   `A_total(M)` the total adaptability damage of the set (leave-set-out, computed directly, not
   summed). *Anchor (measured, §5.2):* `L_ceil` = the adapter-effect loss the **official
   dense base → dense post transfer already incurs** (§5.2). A `G_adapt` smaller than `L_ceil`
   is within what the official pipeline already accepts and is **not material**; one that
   exceeds it is.

**Minimal result that would justify a new pruning method (to be confirmed by the numbers
above, not signed here):** at `k = 4`, `δ_{D_P,A}(4) ≥ 1` above the half-split floor,
`G_adapt(4) ≥ L_ceil`, and `C_dep(4)` smaller than the gap between the deployment-only pruned
model and the dense 6-step model (i.e. respecting adaptability does not push us below the
iso-compute alternative). If any of the three anchors (`half-split floor`, `L_ceil`, dense
6-step `E`) cannot be measured, the SESOI is declared *not yet defensible* and the main
experiment is not signed.

---

## 5. Positive controls (S2) — run before any null reading

### 5.1 Synthetic, localised (instrument check for `A`)

Inject one generic probe `u*` **only into blocks 4–7** of the dense post model
(`--include`-style filter `transformer.layers[4-7]` on the parametrisation) at the same `κ`.
Compute `A_carry(g)` for all 20 blocks with the probe family fixed to `{u*}`. **Pass:**
`A_carry` on blocks 4–7 accounts for 100 % of the effect by construction *and* the
leave-one-out `A(g)` for `g ∉ {4..7}` equals the interference term only; the instrument must
rank 4–7 as the top-4 `A_carry` blocks with every other block at the interference floor.
**Fail ⇒ fix the instrument; nothing else runs.** A second synthetic control uses a *trained*
LoRA restricted to `layers[4-7]` once the data of §5.2 exist (stronger, not required to start).

### 5.2 Ecological (does the compatibility contract exist before pruning?)

Measure the dense transfer ceiling of a **real** adapter: train one LoRA on
`small-sfx-base` following the repo recipe (`scripts/train_lora.py`, rank 16, `dora-rows`
default and a second run with `lora`; 20–50 clips per `lora.md` "What You Need"; ~1 000 steps
per its quick-start), apply it to dense base and dense post, and measure the adapter effect on
each: `δF_B(L)`, `δF_P(L)` on the common states, and end-to-end `E` with and without the
adapter. `L_ceil = ‖δF_P(L) − δF_B(L)‖² / ‖δF_B(L)‖²` (field) and its end-to-end analogue.
**Prerequisite (data):** 20–50 clips of 44.1 kHz audio with captions in one SFX domain, CC0
(Freesound) — **we do not have this yet**; it is a separate, small, licensed-data task and the
only external-data dependency of RQ2. If `L_ceil` shows no meaningful transfer (adapter effect
on the post uncorrelated with that on the base), **the adapter axis is dropped** and RQ2 is
reported as "no contract to preserve"; RQ1 proceeds regardless.

---

## 6. The strongest adversary, first (S5)

In the **first GPU job**, before any `I_PT` or `A`:

1. **Block-removal tolerance of the post-trained 8-step model.** For every single block `g`:
   end-to-end `E({g})` (§9). Compare each against the dense post model at 7 steps (the
   iso-compute yardstick for k = 1 rounds to 7.6 → 7). If **every** `E({g})` is worse than the
   dense-7-step model by more than the seed-resampling floor, record **CASE E** (§8): depth
   pruning without repair is dominated by step reduction; the protocol stops there and says so.
2. **Post-only pruning baseline.** `R_{D_P}(k)` and, independently, `R_E(k)` = the `k` single
   blocks with the smallest end-to-end `E({g})` — the "prune the deployed model on its own
   damage" adversary — for k ∈ {2, 4, 6}, with `E(R(k))` measured (leave-set-out, not summed).

Only after these two tables exist are RQ1/RQ2 quantities read.

---

## 7. RQ1 / RQ2 design

* **Stage 1 — leave-one-block-out (this protocol).** 20 variants × {base, post} × states;
  every quantity of §3; plus the single-block end-to-end table of §6. Additivity is **not**
  assumed anywhere: removal sets are evaluated as sets (`E(R(k))`, `A_total(R(k))`) and
  compared with the sum of singles only as a diagnostic (`additivity gap`).
* **Stage 2 — exact enumeration (conditional, specified, not executed).** If and only if
  stage 1 shows real divergence (§4 item 1 above the floor), enumerate all masks for
  k ≤ 4 (C(20,2) = 190, C(20,3) = 1 140, C(20,4) = 4 845) on the field proxies and compute the
  exact deployment/adaptability Pareto front; k = 6 (38 760) by stratified sampling. Its cost
  follows from the measured s/forward (§10) and is costed then. **Stage 2 is not part of this
  document's authorization.**
* **Replication (RQ4, excluded):** the `small-music-base` / `small-music` pair exists (HF);
  not touched here.

---

## 8. Result matrix (frozen before data)

Let "coincide" mean `δ_XY(k)` within the half-split floor at every k ∈ {2, 4, 6}, and "flat"
mean the across-block spread of `I_PT` is within its half-split spread.

| case | pattern | decision |
|---|---|---|
| **A** | `R_{D_B}`, `R_{D_P}`, `R_A` coincide; `I_PT` flat | Close the line. Write the analysis as a negative note (post-training does not reorganise structure; adaptability rides on the same blocks). No RQ3. |
| **B** | `R_{D_P} ≠ R_{D_B}` or `I_PT` structured, **but** `R_A` coincides with `R_{D_P}` | PT-aware analysis is a possible (smaller) paper; the adapter axis adds nothing. RQ3 may be designed **without** the adaptability constraint only if `I_PT^dep` vs `I_PT^raw` shows the guidance-internalisation structure is real. |
| **C** | `R_A ≠ R_{D_P}` above the floor, `G_adapt(k) ≥ L_ceil` for some k, `C_dep(k)` below the iso-compute gap | Adapter-compatible pruning is justified. Proceed to **design** RQ3 (constrained formulation; exact enumeration stage 2). |
| **D** | `I_PT` and `A` both diverge from `D_P` and from each other | Strongest case; RQ3 design with both constraints; stage 2 enumeration on both. |
| **E** | Every single-block removal is dominated by the dense model at 7 steps (§6.1) | Depth pruning without repair is not a deployment option for this few-step model. The current proposal dies; "mask + repair" would enter the territory of 2607.06335 and is **not** pursued from this protocol. Report. |

Mixed patterns (e.g. divergence at k = 2 only) are reported as such and do not trigger RQ3.

Pre-registered secondary readings (no decisions attached): `ρ(W, I_PT)` (is the functional
delta visible in parameter space?), `ρ(I_PT^raw, I_PT^dep)` (guidance internalisation),
`A_carry` vs `A_int` (is adaptability lost by losing slots or by disturbing the rest?),
`U_gen` vs `U_xs` agreement.

---

## 9. End-to-end validation (mechanism ≠ consequence)

Field quantities are proxies for mechanism and screening. **Any statement about pruning
importance is validated end-to-end** with 8-step ping-pong generation on the fixed panel,
seed-paired, decoded through SAME-S, metrics relative to the dense post model:

* **CLAP score** (text–audio cosine, LAION-CLAP as in ARC §3.3) — absolute, per system.
* **KL_passt** between each system's outputs and the dense post outputs, prompt-paired
  (no external reference set needed; the ARC paper's KL metric space).
* **FD_openl3** of each system's outputs against the **dense post model's outputs** on the
  same panel ("drift from dense"), plus, as a secondary reading, against the ARC reference
  protocol only if a 44.1 kHz reference set is obtained (we have none).
* **Optional:** a pre-registered T2A-bench subset (category / count / ordering / timestamp;
  Gemini 2.5 Pro judge, paid API). Secondary; not a gate.
* **Not used as outcome:** PANNs top-10 recall.
* **Seed floor:** the dense post model with a second seed stream on the same panel defines the
  resampling floor for every end-to-end metric.

---

## 10. Implementation plan (concrete; nothing here is run on GPU by this document)

**Environment.** Separate `uv` venv `.venv-sa3` (Python 3.10; `stable-audio-3` pins
`torch 2.7.1` / `torchaudio 2.7.1`), installed from a pinned fork commit
(`a0b57f5483c4588f827f3552b7d5c6ca2a9687be`), `--extra lora`. The frozen AudioLDM
`.venv` (torch 1.13.1) is untouched. T4 (sm_75): **fp16 only** (no bf16, no Flash-Attention;
`small` models do not require FA per the README). `norm_kwargs.force_fp32 = True` is in the
config; fp16-vs-fp32 agreement of `F_P` on CPU-reference states is a smoke assertion.

**Repository layout (new, all under this repo):**

```
research_sa3/
  loading.py        load base/post via stable_audio_3 factory; sha256 + key/shape diff; config diff (step 0)
  blockskip.py      BlockMask: wraps each transformer.layers[i] so skip=True returns its input unchanged;
                    context manager; assert identity == bit-exact original when mask empty
  states.py         capture trajectory states via the sampler callback (x, t, i) for post (8, ping-pong)
                    and base (50, Euler, cfg 7); build S_noised; panel + seed table files with sha256
  fields.py         raw field (DiffusionTransformer._forward), deploy field (CFG s=7 recomputed from cond/uncond raw calls;
                    production APG apg_scale=1.0 primary, vanilla apg_scale=0 secondary; scale_phi=0), padding-masked norms
  probes.py         generic / LoRA-XS(base U,V) probe construction through the official LoRAParametrization;
                    per-layer kappa rescaling; restriction u_{-g}; probe seeds
  metrics.py        D_B, D_P, I_PT^raw, I_PT^dep, W, A_carry, A_int, A; removal sets; half-split floors
  e2e.py            8-step generation per system on the panel; SAME-S decode; CLAP score, KL_passt,
                    FD_openl3 vs dense; seed floor
  report.py         JSON writer (schema below) + markdown table
configs/sa3/
  panel_prompts.json, seed_table.json, schedule_post_10s.json (τ_1..τ_8 as computed), probes.json
scripts/sa3/
  step0_verify_pair.py   (CPU) hashes, key/shape/config diff — MUST pass first
  smoke_t4.py            (GPU, one short job) s/forward at batch {1,4,8}, VRAM, fp16 check, E2E of dense
                         post at 8/7/6/5 steps + one block-removed variant
  rq_adversary.py        §6 tables (single-block E, post-only sets)
  rq1_rq2_loo.py         §3 quantities on S_traj and S_noised; §4 statistics; §8 classification
tests/sa3/
  test_blockskip.py      empty mask == original (bit-exact); skip g changes output; skip-all == project_in/out path
  test_fields.py         our deploy field == DiffusionTransformer.forward(cfg_scale=7, apg_scale={1.0, 0.0}) (bit-exact)
  test_probes.py         XS probe == add_lora('lora-xs') with the same M (bit-exact); kappa rescaling; linearity
  test_metrics.py        synthetic per-block tables -> removal sets, floors, case classification
  test_positive_control.py  §5.1 on a tiny random-weights DiT (architecture-only; structure test), then on real weights in the smoke
```

**Output JSON (every run):** `{git_commit, upstream_commit, ckpt_sha256{base,post,svd_bases},
config_diff, panel_sha256, seed_table_sha256, schedule{tau_1..8}, cfg{s, apg_scale, scale_phi},
kappa, n_u, per_block[20]{D_B, D_P, I_PT_raw, I_PT_dep, W, A_carry_gen, A_int_gen, A_gen,
A_xs…, E_single}, removal_sets{X}{k}, floors{half_split, seed}, e2e{system}{metric},
case, gpu{name, vram_peak_gb, s_per_forward{batch}}, wall_s}`. Results are gitignored under
`artifacts/sa3/`; hashes go to the ledger.

**CPU dry-run (free, before any GPU):** step 0; `BlockMask` identity test on real weights;
field evaluation on 2 prompts × 2 states with the DiT in fp32 (no decode, cached conditioning);
probes on 1 block; metrics on the toy tables; positive-control structure test.

**GPU smoke (one job, TBM cost):** `smoke_t4.py` as listed. Nothing in this protocol is costed
until it reports **s/forward (batch 1/4/8), peak VRAM, and s/generation at 8 steps incl.
decode**. Reference points from the upstream README (H200, not T4, not ours): inference peak
1.69–2.40 GB for `small`; LoRA training ≈ 2.5 GB. They bound VRAM, not time.

**Cost model (filled only from the smoke):** forwards = `N_p × 8 states × [ (2 models) ×
(1 + 20 variants) + CFG unconditional branch for base × 21 + probes n_u × (1 + 20) ]` for the
field stage; `N_p × (1 + 20 + 4 step-variants)` eight-step generations + decodes for §6/§9.
`N_p` and `n_u` are chosen after the smoke so that the half-split floors are estimable; the
numbers are recorded in the ledger with the smoke, not here.

---

## 11. Adoption gate (what this protocol does and does not authorise)

* Authorised by Gabriel's "go" (2026-08-20 23:14): **writing this protocol**. Not authorised:
  any GPU job, any method, any LoRA recovery, RQ3/RQ4.
* **Before the first GPU job** (each needs an explicit go): step 0 passes on CPU; tests in
  `tests/sa3/` pass; the CPU dry-run passes; the panel, seed table and schedule files are
  hashed in the ledger; S1 magnitudes for RQ1 are written down (predicted range of `I_PT` under
  "post-training reorganises structure" vs "it does not", argued from the training-pipeline
  description, not from `W`); the novelty ledger (§12) is re-read.
* **After RQ1/RQ2:** cases **C** or **D** permit *designing* RQ3 (constrained formulation,
  stage-2 enumeration, adapter-held-out validation) as a new protocol document. Cases **A**,
  **B-without-guidance-structure**, and **E** close the line or reduce it to an analysis note.
  No result in this protocol approves a pruning method.

---

## 12. Novelty ledger (operational; re-read at every gate)

| prior work | what they did (one sentence) | what we do differently | claim we will **not** make |
|---|---|---|---|
| **TinyFusion** (CVPR 2025) | Learns a depth-pruning mask for visual DiTs by optimising post-pruning *recoverability* (fine-tuning, incl. LoRA) of **one** model. | We measure, training-free, which blocks carry a *transformation* (base→post) and an *adaptation subspace*, and ask whether one shared subnetwork preserves a **family**; recoverability of one checkpoint is not our objective. | "first recoverability-aware depth pruning of DiTs"; "better than TinyFusion at recovering one model". |
| **2607.06335** (Teacher-aligned repair) | Shows pruning an EDM2-XS teacher then one-step distillation (SiDA) fails without a teacher-matching repair stage (ImageNet-512). | We do not distil and do not repair; we analyse an *already* post-trained few-step model and its base. If case E occurs we stop rather than add a repair stage. | "pruning and few-step distillation commute / do not commute"; any repair method. |
| **Dynamic-in-Few-Step** (2607.06631) | Integrates structured sparsification inside few-step distillation for video (Wan-14B), producing step-specific sub-models. | No distillation, no training; a fixed mask shared across steps and across the base/post pair. | "joint pruning + few-step distillation"; step-specific sparsity. |
| **CAR-LoRA** (ICLR 2026) | Trains a single LoRA to be robust to simulated compressions (quantisation / pruning) and to evolved base versions (LLMs) — **adapter-side**. | **Backbone-side**: we ask which compressed backbone keeps *ordinary, unseen, untouched* adapters working; CAR-LoRA is a conceptual baseline for "pay robustness per adapter vs once in the backbone". | "first work on LoRA robustness to compression / model evolution". |
| **TALL-Masks / NPS** (ICML 2024 / task-vector compression) | Localise and compress the task-specific information in fine-tuned *weights* (task vectors, masks) for merging / storage. | Our quantity is *functional* (`I_PT` on the field, normalised) and our intervention is *structural* (whole blocks); `‖ΔW‖` is a covariate we compare against, and `ρ(W, I_PT) ≈ 0` would itself be a result. | "task-vector-aware pruning"; weight-delta preservation as the method. |
| **Tangent-space task arithmetic** (Ortiz-Jiménez et al., NeurIPS 2023) and linearised fine-tuning | Study fine-tuning / editing in the model's tangent space; weight disentanglement via the Jacobian. | We use the tangent-space view only to *define* `A(g)` (Jacobian restricted to LoRA directions, measured functionally) and to explain why base→post LoRA transfer can work at all; no editing, no merging. | "LoRA tangent-space preservation" as a new concept. |
| **EcoDiff** (2412.02852) | Differentiable-mask structural pruning of SDXL/FLUX, usable on timestep-distilled models (compatibility shown, no analysis). | Analysis of base vs post structure and of adaptability; depth granularity; audio DiT. | "first pruning of a distilled/few-step diffusion model". |
| **Compress-then-Serve** (PMLR v267) | Compresses collections of LoRAs for serving many adapters. | Motivates the many-adapter setting; we compress the backbone, not the adapters. | — |

---

## 13. Inheritance list (S3)

| inherited element | from | why it is necessary here | alternative if not |
|---|---|---|---|
| AudioCaps test **captions** as the prompt panel | v3/v4 repo data; ARC eval set | Text-only, already held, same captions ARC evaluated SA-Open on; the SFX model's domain is sound events | Freesound CC0 descriptions (would also give real 44.1 kHz latents) |
| CLAP score / KL_passt / FD_openl3 | ARC §3.3 | The published metrics for this model family; reference-free or dense-referenced variants avoid the 16 kHz reference problem | T2A-bench subset (secondary) |
| Seed pairing, frozen manifests with hashes, ledger discipline | plan v4 §6, AGENTS.md | Variance reduction and provenance; unchanged | — |
| **Not inherited:** AudioLDM, PANNs top-10, L1 filter pruning, CLAP-swap conditioning, Arshdeep's checkpoints | v3/v4 | None is necessary for the question | — |
