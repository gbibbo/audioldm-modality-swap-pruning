# Publication Decision Memo — ICASSP 2027 pivot

**Status: COMPLETE (2026-08-26 02:31–02:47). Verdict: GO-AUDIOLDM (§5) — pending Gabriel's
review of `docs/master_plan_v4_amendment_icassp.md`.** All three audits (Gate-0 substrate,
seam inventory + executable proof, novelty collisions) are done. Nothing in this memo
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
**2b. Full score trajectories (fetched 2026-08-26 — decisive for Gate-0 design).** Table 6
(CLAP) per checkpoint shows the published evidence is **extremely noisy**: r8/α16 runs
0.6941 → 0.6451 → 0.5813 → 0.6995 → 0.6644 across 200–1000 epochs (swings ±0.11, including a
point −0.068 *below* baseline at 600 ep); at 400 epochs **no configuration beats the 0.6497
baseline**; only 3/8 configs beat it at 200 ep. The protocol behind these tables (from the paper
+ released code): **a single fixed prompt** ("hip hop music, The subgenre of hip-hop is boom
bap."), **5 seeds averaged**, DDIM S=50, 4-s audio, default guidance; baseline appears once, with
no per-checkpoint baseline row, no CIs, no tests. Table 7 (their custom CLAP-MMD "KAD") always
beats baseline but worsens with rank (inverse to CLAP). **Honest conclusion: the published CLAP
uplift is compatible with single-prompt evaluation noise.** Therefore Gate 0 must be a
**replication with power** — same recipe, but a pre-registered multi-prompt battery, multiple
seeds, paired bootstrap CIs and a SESOI — which either establishes the uplift rigorously (a
small standalone contribution) or kills the substrate cheaply. This is exactly the failure mode
that ended SA3 (modest effects vanishing under powered paired evaluation), now placed FIRST and
cheap instead of last and expensive. Silver lining: their r8/α16 at **200 epochs** (0.6941) ≈
their 800-epoch headline (0.6995), so the affordable 200-epoch replication point is within their
own reported best band.

4. Alternative independently-published AudioLDM-family adapters found (fallback Gate-0 substrates,
   NOT yet audited to the same depth): **AP-Adapter** (ISMIR 2024, arXiv:2407.16564 — AudioLDM2,
   22 M-param attention adapters, code public; not LoRA, AudioLDM2 ≠ our substrate) and **Guitar
   Tone Morphing** (APSIPA ASC 2025, arXiv:2510.07908 — LoRA r=2 on AudioLDM, code public; weights
   unverified). MusicGen LoRA ecosystems (ylacombe/musicgen-dreamboothing community checkpoints)
   remain the GO-ALTERNATIVE candidate but require full substrate bring-up (autoregressive stack,
   no pruning machinery, unmeasured costs) — prima facie incompatible with 21 days / 6 cr unless
   the incumbent dies.

## 3. Seam inventory (Scenario-B mappability on M-Full) — COMPLETE (2026-08-26, CPU repo audit)

**Headline: every seam of the (1,2,3,x) ladder reduces to pure index selection with per-layer
recoverable indices, so the mask-induced LoRA transfer is deterministic and shape-consistent at
every seam — no learned parameters needed anywhere.** Details and honest flags:

* **Ladder structure.** `channel_mult=(1,2,3,m₄)`, base (1,2,3,5): only the deepest U-Net stage
  changes width (960 → 768/576/384/192). 238/690 tensors change shape; only the ten 960-wide
  ranked layers are actually pruned (the 576/384-wide ranking entries keep k=full). Kept sets are
  nested (verified). Params per level exact: 415.95 M → 317.31 M (−23.7 %) / 239.05 M (−42.5 %) /
  182.17 M (−56.2 %) / 145.67 M (−65.0 %). **MACs are not recorded anywhere** → central-figure
  x-axis = parameter reduction (exact) now; a CPU MAC counter is a later nice-to-have.
* **Materialization is pure selection** (`random_masks.py`): ranked rows/cols `perm[:k]` for
  LAYER_MAP tensors, positional truncation `[:k]` otherwise; no rescaling, no re-normalization,
  no group-count change, no fusion. Strict-load (=`strict=True` + forward) committed-verified at
  (1,2,3,4) and (1,2,3,1); (1,2,3,3)/(1,2,3,2) verified in-session only (re-verify before use).
  The published p1 checkpoint is **pruned-only, never finetuned** (2061 same-shape tensors
  bit-identical to dense; materialize == published, 690/690).
* **Seams at LoRA targets.** Attention `to_q/to_k/to_v/to_out.0` (60 Linears), FF GEGLU, and
  transformer `proj_in/out` at the 6 level-3 sites change both dims (960→192 at p1), positional
  selection; ResBlock convs change out/in (ranked, with 3 documented positional exceptions);
  `emb_layers.1` changes out only. Attention truncation is **head-boundary-aligned**
  (`dim_head=32` fixed; heads 30→24/18/12/6) — clean slicing, head *count* changes.
* **Honest flags (inherited from the published artifact's own conventions, not from our
  transfer):** (1) at decoder **concat seams** the artifact's positional truncation takes input
  columns only from the dense `h` segment, so the pruned `skip` segment consumes weights that
  belonged to dense `h` channels — semantically scrambled (AUDIT-M3-001; question pending to
  Arshdeep); (2) **GEGLU value/gate halves**: prefix truncation takes all rows from the dense
  value half → the pruned gate half receives dense value rows; (3) **GroupNorm regrouping**:
  num_groups fixed at 32 ⇒ channels/group 30→6 — normalization statistics repartition (a backbone
  property, unrelated to the adapter map); (4) at **(1,2,3,3)** `input_blocks.10.0.skip_connection`
  becomes `nn.Identity` — a dense LoRA on that conv has no target and is deterministically
  dropped. **Resolution (frozen):** Scenario-B transfer uses *exactly the selections the backbone
  itself used at each tensor* — unique, deterministic, zero learned repair — and therefore
  inherits the artifact's misalignments; the phenomenon experiment measures adapter-uplift
  survival under the *real published compression convention*, flags included and reported.
* **Executable proof (not only analytic): `tests/research/test_scenario_b_seam_transfer.py`
  S1–S6 PASS (CPU, 2026-08-26).** All 238 changed tensors identical across (1,2,3,4)/(1,2,3,1);
  every one receives a deterministic shape-consistent selection (27 ranked / 211 positional);
  the sliced-LoRA identity holds **exactly on every changed 2D (67) and 4D (33) tensor**; all
  138 changed 1D aux tensors transfer by out-selection; the (1,2,3,3) module-drop is detected
  and handled by a deterministic rule; the flattened `LoRAConv2d` layout round-trips. **Zero
  learned parameters needed at any seam** — the directive's mappability precondition is met.
* **Machinery fit.** Our PEFT layer attaches LoRA to exactly the Kim-recipe targets and more
  (type-based injection; 284 modules = 185 Linear + 99 Conv2d proven on the real p1 U-Net;
  merge/unmerge bit-exact; substring filters allow restricting to `to_q`/`to_v` to mirror Kim).
  Engineering note: `LoRAConv2d.lora_A` is stored flattened `(r, in·kh·kw)` — reshape before
  input-dim slicing. Scoring: LAION-CLAP text-audio cosine exists self-tested in `.venv-metrics`
  (must be pointed at the fused checkpoint to mirror Kim's `laion/clap-htsat-fused`); Kim's "KAD"
  is a custom CLAP-embedding MMD, implementable on CPU from cached embeddings if wanted as
  secondary; FAD/FD/KL/PANNs machinery verified in-repo.

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

## 5. Verdict: **GO-AUDIOLDM**

*(Recommendation to Gabriel; the GO authorizes the gated chain below upon his review of the
amendment — it does not pre-authorize the paper or any GPU spend before that review.)*

**Why GO-AUDIOLDM and not the alternatives.** GO-ALTERNATIVE (MusicGen, where ~196 real
third-party LoRA checkpoints exist) is the scientifically richer ecosystem but requires full
substrate bring-up (autoregressive stack, no pruning machinery, no measured costs) — not
credible in 21 days / 6 cr; it is the natural *successor* substrate if the AudioLDM phenomenon
is confirmed (or the fallback consideration if Gate 0 fails). NO-GO-ICASSP is premature: every
precondition that can be established without GPU has been established (§2–§4: no novelty
collision; deterministic transfer proven executable at every seam; public recipe + data; all
scoring/pruning machinery verified in-repo); the two remaining unknowns (real adapter uplift;
differential fragility) are precisely what the ≤2.5-cr falsifying chain tests.

**Exact abstract-level claim (if the chain passes):** "Structured pruning of a text-to-audio
latent diffusion backbone can preserve standalone generation quality while disproportionately
destroying the benefit of legacy LoRA adapters trained on the dense model and transferred by
deterministic mask-induced slicing — without adapter retraining or adapter data. We quantify
this differential fragility across compression severity on AudioLDM, identifying unseen-adapter
compatibility as a distinct, previously unmeasured axis of compression quality." (Phenomenon/
analysis-led; a simple mitigation is added only if it follows naturally and fits the reserve.)

**Scenario: B** (strict). Scenario A appears only as the "retrain-on-pruned" oracle baseline.

**Gate 0 (G0-M, ≤1 cr, precedes any severity work):** replicate the Kim et al. published
task/recipe transposed to dense M-Full in our pipeline — LoRA on `to_q`/`to_v` of all attention
layers (substring filters), **r8/α16** (their headline config), lr 1e-5, AdamW, their public
193-clip hip-hop dataset, 4-s crops, **200 epochs** (their first published checkpoint; within
their own best band, §2b), batch 2 for fidelity (batch 4 + 16-mixed only as a logged deviation
if the smoke requires it). **Powered evaluation replacing their 1-prompt protocol:** frozen
pre-registered battery of 64 hip-hop prompts × 3 seeds, 4-s, DDIM S=50, default guidance
(mirroring their validation code path); primary endpoint **ΔCLAP** = paired mean(adapter −
base) with the LAION-CLAP fused checkpoint (their scorer); paired bootstrap CI, B=10000, frozen
seed. **SESOI = +0.025** (half their claimed +0.0498). **PASS = point ≥ SESOI AND lower CI95 >
0.** The paper labels this "our powered replication of the published recipe" — never "their
adapter". FAIL ⇒ the AudioLDM thesis dies immediately; no adapter iteration.

**Cheapest phenomenon test (post-Gate-0, generation only, no training):** severities {dense,
(1,2,3,4) −23.7 %, (1,2,3,1) −65 % = the published Arshdeep artifact}, systems {backbone,
backbone + mask-sliced legacy LoRA} (dense pair reused from Gate 0) — 2 pruned backbones × 2
systems × 64 prompts × 3 seeds. **Pre-registered differential-fragility criterion:** with
C(s) = standalone CLAP and ΔCLAP(s) = paired adapter uplift at severity s, define
**D(s) = ΔCLAP(0) − ΔCLAP(s)** (adapter benefit lost) and **E(s) = C(0) − C(s)** (standalone
degradation, same scale). The phenomenon exists at s iff **D(s) − E(s) ≥ 0.025 with paired
bootstrap lower CI95 > 0**. "Adapter score went down" alone is nothing. Secondary (non-binding):
FAD vs the 193-clip reference; their CLAP-MMD. If no severity shows the phenomenon ⇒
pre-registered negative ⇒ STOP (no method work, no reframing).

**Central figure:** x = parameter reduction {0, 23.7, 65}% (MACs/latency later if measured);
y1 = C(s) with CI; y2 = ΔCLAP(s) with CI; inset/table: D(s) − E(s) with CI. The interesting
regime: y1 flat while y2 collapses.

**Mandatory baselines:** (a) dense ± adapter; (b) pruned standalone per severity (the Arshdeep
axis); (c) pruned + sliced legacy adapter (the object); (d) **Scenario-A oracle** — LoRA
retrained on the pruned backbone at the moderate point (quantifies "why not just retrain");
optional (e) random-slicing control (slice with random kept-sets; tests that alignment matters).

**Hostile review ("why not spend 10 minutes retraining?"):** the compression provider does not
own the adapters — no training data (rights/privacy; most community LoRAs publish weights, not
datasets), no pipelines, no per-adapter compute across an ecosystem; one compressed backbone
must serve existing adapters (FLUX-lite/SSD-1B breakage is documented user pain). For AudioLDM
specifically the ecosystem motivation is **prospective and must be stated as such** (n=1 LoRA
today; MusicGen ~196 shows the audio trajectory). Independent scientific value: differential
fragility is compression damage invisible to standalone metrics (Hooker-style hidden harm on a
new axis), and the oracle baseline (d) quantifies the retraining comparison instead of dodging
it. Remaining honest weaknesses: single substrate, single task/genre, adapter is our (powered)
replication rather than a wild third-party artifact — all stated as limitations.

**Budget through Sep 10 (DERIVED — every figure re-anchored by a ≤0.1-cr smoke before any
launch; measured anchors: 0.89 cr/GPU-h, Ttrain 0.56 s/step@b2/10-s, Tgen 8.4 s/clip@S=50/10-s,
4-s scale ≈ ×0.39):** smoke 0.1; Gate-0 training ~0.5–1.0 + eval generation ~0.3; phenomenon
generation ~0.6–0.8; Scenario-A oracle ~0.5–0.7; rescoring/contingency ~0.4. **Total ≈ 2.4–3.3
cr, hard cap 3.5 cr** → ≥ 2 cr of the ~6-cr balance stays in reserve. First falsifying chain
(smoke + Gate 0 + phenomenon) ≈ **1.5–2.2 cr ≤ 2.5 ✓**.

**Schedule:** Aug 26–27 amendment review (Gabriel) + prompt battery frozen + CPU wiring (CLAP
scorer path, trainer data path, transfer op) with dry-runs; Aug 28 smoke → Gate-0 launch;
Aug 29–30 Gate-0 scoring + verdict; Aug 31–Sep 2 phenomenon generation + scoring; Sep 3–5
oracle baseline + CIs + figure; **Sep 5 paper skeleton starts** (v3 discipline); **Sep 10
central figure frozen**; Sep 10–16 writing + internal hostile review; submit only if the
phenomenon claim is powered ("do not submit underpowered" stands).

**Terminal STOPs:** **STOP-0** smoke projects Gate 0 > 1 cr → re-cost to Gabriel before launch.
**STOP-1** Gate 0 FAIL → AudioLDM thesis dead; decision GO-ALTERNATIVE vs NO-GO-ICASSP goes to
Gabriel with the evidence then in hand. **STOP-2** phenomenon absent → pre-registered negative;
no method work; venue-fallback decision. **STOP-3** 3.5-cr cap reached → stop and report.
No mitigation/method design before the phenomenon is confirmed.
