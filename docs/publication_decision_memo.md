# Publication Decision Memo — ICASSP 2027 pivot

**Status: DRAFT IN PROGRESS (2026-08-26 02:31). Verdict: PENDING** — awaiting (a) the M-Full
per-seam inventory and (b) the novelty-collision audit (both in flight). Nothing in this memo
authorizes GPU. Ledger: ICASSP-PIVOT.

**Binding constraints (directive, Gabriel via external reviewer, 2026-08-26):** optimization
target = probability of a defensible ICASSP-2027 paper by **2026-09-16** (CFP-verified; 21 days);
account balance ≈ **6 Lightning cr** (measured T4 rate ~0.89 cr/GPU-h); central figure feasible by
~**Sep 10**; **first falsifying chain ≤ 2.5 cr**; **Gate 0 ≤ 1 cr and precedes any pruning
severity**; phenomenon before method; severity sweep (13–26 cr) **DEFERRED / NOT part of ICASSP**;
no new-substrate bring-up unless it beats the incumbent under 21 days / 6 cr; **no GPU until this
memo + amendment are reviewed**. Verdict must be exactly one of **GO-AUDIOLDM / GO-ALTERNATIVE /
NO-GO-ICASSP + FALLBACK**.

## 1. Candidate thesis under audit (incumbent, Scenario B)

> Structured pruning of a text-to-audio backbone can preserve standalone generation quality while
> disproportionately destroying the utility of **legacy LoRA adapters trained on the dense model**
> and transferred by **deterministic mask-induced slicing** (zero adapter retraining, zero access
> to adapter training data). If that differential fragility exists, can pruning be made
> adapter-compatible without seeing the adapters?

Scenario definitions (frozen):

* **Scenario A** (NOT ours): adapters trained *after* compression, on the compressed backbone —
  collides with TinyFusion (recoverability) and Gordon et al. 2020 (prune-then-transfer).
* **Scenario B** (ours): dense-trained legacy adapters → compressed backbone, **zero retraining,
  zero adapter-data access**. Deterministic mask-induced projection of adapter tensors is allowed
  iff it follows uniquely from the backbone pruning mask; learned repair is NOT allowed.

**Scenario-B transfer core is proven (CPU, commit 482bd2c):** `B'A' == (BA)[K_out,K_in]` exact for
Linear+Conv (fp32/fp16 bit-exact), nested-ladder composition exact
(`tests/research/test_lora_mask_transfer.py` 6/6), and the published pruned checkpoint
`l1_audioldm-m-full_p1.ckpt` is a **bit-exact pure index selection** of the dense checkpoint on all
690 tensors (`test_materialize_channel_mult.py` P4) — so backbone and sliced adapter share the same
selections by construction. Remaining engineering fact to establish: the per-seam inventory of the
M-Full U-Net under the (1,2,3,x) ladder (which attention/proj/FF/conv dims change; whether every
changed dim has recorded selection indices; anything not reducible to pure selection). **PENDING.**

Pre-registered phenomenon criterion (to be frozen in the amendment): differential fragility exists
iff, at a pruning point where **normalized standalone quality** retains ≥ some high threshold, the
**normalized legacy-adapter uplift retained** falls below standalone retention by a margin that is
(i) larger than a pre-registered SESOI and (ii) outside measurement uncertainty (paired bootstrap
CIs). "Adapter score went down" is NOT the criterion. If uplift decays proportionally to standalone
quality → **NO-GO for the compatibility thesis** (pre-registered negative branch).

Cheapest phenomenon test (post-Gate-0): dense + **one moderate + one strong** pruning point from
the existing strict-load-verified ladder — e.g. (1,2,3,4) −23.7 % and (1,2,3,1) −65 % (the −65 %
point equals the published Arshdeep artifact) — no recovery training, no adapter retraining.

Central figure (sketched before compute, per directive): x = parameter/MAC (or measured-latency)
reduction across {0 %, −23.7 %, −65 %}; y1 = normalized standalone quality; y2 = normalized
dense-trained-adapter uplift retained under mask-induced transfer. Interesting regime: y1 high
while y2 falls materially faster. Error bars: paired bootstrap over the frozen prompt set.

## 2. Gate-0 evidence: the independent AudioLDM adapter (VERIFIED 2026-08-26)

Source: Kim et al., "Enhancing Diffusion-Based Music Generation Performance with LoRA," Applied
Sciences 15(15):8646, 2025, DOI 10.3390/app15158646 (CC BY). Deep verification via paper full text,
authors' GitHub, HF APIs (see ledger ICASSP-PIVOT provenance; agent report 2026-08-26).

**Verified facts:**

* **Backbone:** `cvssp/audioldm-s-full-v2` (HF **diffusers** `AudioLDMPipeline`), UNet ≈185 M.
  Our verified pruning ladder/materializer targets **AudioLDM-M-Full** (original ckpt format).
  Architecture delta S→M: **UNet width only** (block_out_channels 128/256/384/640 →
  192/384/576/960, = cross_attention dims; attention_head_dim 8 both); VAE, CLAP text encoder,
  HiFi-GAN vocoder, DDIM scheduler identical components.
* **Recipe (from code, not only paper):** PEFT LoRA on UNet attention **`to_q`,`to_v`** (all
  attention layers, attn1+attn2), text encoder/VAE frozen; r ∈ {1,2,4,8}, α ∈ {r,2r}; AdamW lr
  1e-5, batch 2, **1000 epochs** over **193 four-second MusicCaps hip-hop clips**; checkpoints at
  200/400/600/800/1000 epochs; RTX 3080 Ti 12 GB. Paper-internal inconsistency flagged: §2.3 says
  "text encoder", code+Table 2 say UNet.
* **Reported uplift:** CLAP (scored with `laion/clap-htsat-fused`, cosine, [0,1]-rescaled)
  **0.6497 → 0.6995** best (r8/α16, 800 ep; +0.0498 absolute); their "KAD" (a **custom CLAP-embedding
  MMD**, not the official PANNs-based KAD) 1.1255 → 0.2906 best at a *different* config (r1/α1,
  200 ep); the paper acknowledges an inverse CLAP/KAD trade-off. **No CIs, no statistical tests;
  eval prompt count for the score tables is not stated** (5 seeds averaged per checkpoint).
* **Weights availability: NONE.** Paper has no code/weights link ("inquiries to the corresponding
  author"); the authors' repo (github.com/2025-comprehensive-design/AudioLDM-with-LoRA, unlicensed)
  contains **no checkpoint files and zero releases**; the matching HF repo `Rofla/AudioLDM-with-LoRA`
  is **empty** (usedStorage 0); the repo's `push_to_hub.py` is a placeholder; the demo app's
  LoRA-loading lines are commented out.
* **Training data: PUBLIC.** The exact 193-clip set is hosted:
  `Rofla/AudioLDM-with-LoRA-Hiphop-subgenre` (HF dataset, Apache-2.0, audio+captions, 68 MB).

**Implications for Gate 0 (honest read):**

1. There is **no released adapter to preserve**, so Gate 0 is necessarily **our replication of
   their published task/recipe** — the independent element is the published task, recipe, data,
   and reported effect; the concrete adapter would be ours and must be labeled as such in any
   paper (per directive).
2. The independent evidence is **low-rigor**: modest CLAP effect (+0.0498 absolute), no
   uncertainty quantification, unknown prompt count, contradictory secondary metric, MDPI venue.
   Gate 0 must therefore pre-register ONE primary metric (text-audio CLAP with their scorer, on a
   frozen prompt set), a SESOI, and a power analysis — their +0.0498 is the target effect size.
   Failure to replicate a SESOI-clearing uplift within the ceiling kills the thesis (no adapter
   iteration — the SA3 lesson, now structural).
3. **Backbone options for Gate 0** (decision input, costs DERIVED — anchors: measured M1
   Ttrain(batch 2, 10.24 s, pruned M-Full) = 0.5597 s/step; measured Tgen(10.24 s, S=50) =
   8.4 s/clip; 4-s clips scale ≈ ×0.39; rate 0.89 cr/GPU-h; **every figure must be re-anchored by
   a ≤0.1-cr smoke before authorization**):
   * **G0-M (recipe transposed to M-Full, our pipeline):** LoRA dims track block widths, recipe
     transfers by module name (`to_q`/`to_v`). Full 1000-epoch recipe ≈ 96.5 k steps ≈ **~5 cr —
     DOES NOT FIT** the 1-cr ceiling. The **200-epoch checkpoint** (their first published point,
     where their KAD is best and CLAP already moves) ≈ 19.3 k steps ≈ **~1.0–1.2 cr — borderline**;
     scoring generation (2 systems × ~64 prompts × 3 seeds, 4-s, S=50) ≈ **~0.3 cr**.
   * **G0-S (exact replication on S-Full-v2, diffusers):** faithful to the published backbone and
     ~2.25× smaller UNet (likely ~2× cheaper/step), but **our pruning ladder, materializer,
     strict-load verification and provenance do not exist for S-Full-v2 or for the diffusers
     graph** — rebuilding them inside 21 days competes directly with the phenomenon experiment.
   * The S-vs-M mismatch is a **real Gate-0 feasibility question** (per directive, not
     hand-waved): G0-M trades backbone fidelity for a verified pruning substrate; G0-S trades the
     entire pruning substrate for backbone fidelity. Resolution criteria in §5 (verdict) once the
     seam inventory lands.
4. Alternative independently-published AudioLDM-family adapters found (fallback Gate-0 substrates,
   NOT yet audited to the same depth): **AP-Adapter** (ISMIR 2024, arXiv:2407.16564 — AudioLDM2,
   22 M-param attention adapters, code public; not LoRA, AudioLDM2 ≠ our substrate) and **Guitar
   Tone Morphing** (APSIPA ASC 2025, arXiv:2510.07908 — LoRA r=2 on AudioLDM, code public; weights
   unverified). MusicGen LoRA ecosystems (ylacombe/musicgen-dreamboothing community checkpoints)
   remain the GO-ALTERNATIVE candidate but require full substrate bring-up (autoregressive stack,
   no pruning machinery, unmeasured costs) — prima facie incompatible with 21 days / 6 cr unless
   the incumbent dies.

## 3. Seam inventory (Scenario-B mappability on M-Full) — PENDING

To be filled from the repo audit: per-layer table of dims changed by the (1,2,3,x) ladder
(attention q/k/v/out, proj_in/out, FF, convs, GroupNorm), whether selection indices are recorded
and recoverable per layer, and an executable CPU check extending `test_lora_mask_transfer.py` to
every real LoRA-target seam. Known already: AUDIT-M3-001 found 4 tensors in the published artifact
deviating from Arshdeep's public script + internal inconsistencies at `output_blocks.0/1` and
`output_blocks.2.0.in_layers.2` — the seam inventory must state whether any LoRA-target seam is
affected.

## 4. Novelty-collision audit — COMPLETE (2026-08-26; ~12 query formulations across arXiv,
OpenReview, Semantic Scholar, GitHub, HF; full agent report in ledger provenance)

**Kill-criterion outcome: NO direct collision found.** No published work performs deterministic
mask-induced slicing of dense-trained LoRA adapters onto a structured-pruned backbone with zero
retraining and zero adapter-data access; none asks whether pruning can be made adapter-compatible
without seeing the adapters; and **no study in any domain measures the differential-fragility
phenomenon** (adapter utility vs standalone quality under compression severity).

Closest neighbors (all must be cited and contrasted in the paper):

* **CA-LoRA** (arXiv 2307.07705) — **strongest collision**; owns "reuse existing LoRAs of a dense
  LLM on its compressed version" as a problem statement, but its mechanism is **training**
  (adapted inherited LoRA + trained recovery modules, task data, distillation). Separation: we are
  deterministic, data-free, adapter-blind, generative-audio, and we measure the fragility gap.
* **ASSP** (OpenReview 5Rgn6x9jGn, adapter-tuned SAM pruning) — strongest *mechanical* overlap:
  unified slicing rules for heads/channels/kernels — but the adapter is **visible during pruning**
  (inverts the information assumption that defines Scenario B); discriminative segmentation.
* **LoRA-X** (ICLR 2025) / **ProLoRA** (arXiv 2506.04244) / **LoRASuite** — training-free LoRA
  transfer **across different base models** via subspace similarity; no pruning, no mask-induced
  exact correspondence; LoRA-X additionally requires subspace-restricted source LoRAs.
* **SVDQuant/Nunchaku** (ICLR 2025 Spotlight) — off-the-shelf FLUX LoRAs reused on a **4-bit
  quantized** backbone without retraining; quantization preserves tensor shapes, so no slicing
  problem exists — precisely the structural gap our thesis occupies.
* **X-Adapter** (CVPR 2024) — legacy-plugin motivation almost verbatim, but **trains** a bridge
  network on new paired data and runs a frozen copy of the old model at inference; version
  upgrade between dense models, no pruning.
* **Arshdeep Singh et al.** (arXiv 2607.13330) — same backbone family + compression axis (ℓ1
  filter pruning of AudioLDM U-Net, recovery by 1M-step finetuning; per-event recall analysis);
  LoRA only as future work; entirely on the standalone-quality side. It is the standalone
  baseline to cite, not a collision.
* **TinyFusion** (CVPR 2025) — optimizes pruning for *post-finetuning recoverability* of the
  backbone itself (Scenario A's train-heavy contrast). **Gordon et al. 2020** — prune-then-
  full-finetune transfer in BERT (ancestor citation for adaptation-relevant capacity damage).
  **LoRAPrune / LLM-Pruner** — prune-then-recover with a *fresh* co-trained LoRA on generic data
  (the pipeline our thesis inverts). **CAR-LoRA** — the "make the adapter robust" dual
  (compression-in-the-loop training; complementary, and its brittleness premise *supports* our
  phenomenon claim). **Hooker et al. 1911.05248** — epistemic template ("aggregate metrics hide
  differential harm"), about data subsets not adapters.

**Motivation evidence (community, unpublished):** on Freepik/flux.1-lite-8B (depth-pruned FLUX),
users report LoRAs/ControlNets incompatible and one developer manually performed exactly our
mechanical transfer ("trim the parts that target missing tensors and reorder the remaining, it
kinda works... not as well") — documented user pain, unquantified; motivation gold, not a
collision (HF discussion, Freepik/flux.1-lite-8B #6).

**Ecosystem honesty (verified via HF API 2026-08-26):** third-party adapter ecosystem is **PARTIAL**
— MusicGen: ~196 public LoRA checkpoints (of 590 hub matches; hobbyist-scale downloads);
**AudioLDM: exactly 1 LoRA repo (empty weights)**. Any AudioLDM paper must frame the ecosystem
motivation prospectively (by analogy to MusicGen and to the documented FLUX-lite/SSD-1B breakage),
never imply an existing AudioLDM adapter economy.

## 5. Verdict

**PENDING** — will be exactly one of **GO-AUDIOLDM / GO-ALTERNATIVE / NO-GO-ICASSP + FALLBACK**,
with: exact abstract-level claim; Scenario declaration; Gate-0 spec (backbone choice, primary
metric, SESOI, power, ceiling); cheapest phenomenon test; central figure; mandatory baselines;
novelty non-overlap statements; total budget through Sep 10; writing/evaluation schedule
Sep 10–16; terminal STOPs.
