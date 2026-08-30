# Compute Budget

Do not invent GPU-hour estimates. Every number below is either **MEASURED** on real
hardware or **DERIVED** from measured values by the master-plan §7.3 formulas, and each is
labelled. Anything still unmeasured says so.

## RECOVERY-REVERSAL-V1 576-generation projection (DERIVED from settled T4 runs; 2026-08-29)

V1 = 3 backbones (dense_ema, p1_pruned_ema_reconstructed, p1_recovered) × 96 prompts × 2 replicates
= **576 WAVs**, SAME operating point (3.84 s / DDIM 50 / eta 0 / guidance 2.5 / FP32 / single-gen)
as the historical settled runs, so their **settled per-WAV cost** is the direct analog:

* phenom analog `gate0-phenom-1`: 1.6020 cr / 768 WAVs = **0.002086 cr/WAV** → ×576 = **1.20 cr**
* dense-gen analog `gate0-gen-1`: 0.8844 cr / 384 WAVs = **0.002303 cr/WAV** → ×576 = **1.33 cr**
* throughput cross-check: 7.14 s/clip measured; settled effective ≈ 7.9–8.9 s/WAV incl. overhead →
  576 WAVs ≈ 76–85 min + **3 checkpoint loads** (dense + pruned + the 4.45 GB recovered ckpt) at
  T4 ≈ 0.94 cr/hr → ≈ 1.25–1.45 cr.

**Point estimate ≈ 1.3 cr** (bracket 1.20–1.33 from the two settled analogs; V1 loads 3 checkpoints
vs the analogs' 1–2, nudging toward the upper end). **Conservative planning figure = 1.5 cr**
(covers load overhead, warmup, and generation-job variance/preemption/retries).

* **Current balance ≈ 0.72 cr** (2026-08-28) → **short by ≈ 0.6 cr (point) to ≈ 0.8 cr (conservative)**.
* **Minimum additional credit to launch = ≈ 0.8 cr** (reach the 1.5-cr conservative cost).
* **Recommended safe balance before launch = ≈ 2.0 cr** (1.5-cr conservative cost + ~0.5-cr margin
  for variance/retries; leaves a small buffer). → top up ≈ **1.3 cr** above the current 0.72.
* **GPU launch BLOCKED** until the balance is raised. This is a budget block only; no design change.

**2026-08-29 — SETTLED (job `reversal-v11-gen-1`, T4).** Authorized after Gabriel's +2.0-cr top-up
(start ≈ 2.72 cr). 576 WAVs, ~91 min, **settled cost = 1.262 cr** (`total_spent` 66.770→68.032; the
API `balance` field stays 5.0 and remains unreliable — use the `total_spent` delta). Under the
1.50-cr hard cap. Projection was ~1.3 cr point → accurate. Ending balance ≈ **1.46 cr**. Primary
result = pre-registered NEGATIVE (PASS=FALSE); GPU stopped, no rescue runs.

## Account balance — AUTHORITATIVE (measured via Lightning SDK, 2026-08-26)

**2026-08-26 16:27 UTC — user credit balance = `5.0` credits** (lifetime `total_spent` = 55.62;
account_id `371a6f4c-1280-40b9-8bb0-d97a49597756`). **Source:** authenticated `LightningClient`
→ `BillingServiceApi.billing_service_get_user_balance()` (SDK `lightning_sdk 2026.06.08`,
`LIGHTNING_USER_ID e15d0a91-ceaa-44c7-813c-41819348adc7`), run read-only from the CPU Studio.
Reproduce: `/home/zeus/miniconda3/envs/cloudspace/bin/python -c "from lightning_sdk.lightning_cloud.rest_client import LightningClient; print(LightningClient().billing_service_get_user_balance())"`.
(`billing_service_get_account_balance()` returned an auth error; the user-balance endpoint is the
authoritative one here.)

This figure **SUPERSEDES the stale "~9.6 credits" line further down** (that reflects the M3-era
state on 2026-08-19 and must not be treated as current) and matches Gabriel's externally-reported
"~6 cr" to within measurement.

**2026-08-28 — current balance ≈ `0.72 credits` (Gabriel-reported).** This supersedes the
5.0-cr figure above for *present* spend decisions. **NB the `0.5332 cr` "remaining" in the ledger
central-chain accounting is HISTORICAL** (leftover capacity of the retired 5-credit central chain),
**NOT the current account balance.** RECOVERY-REVERSAL-V1 GPU envelope for the 576-generation arm
(96×2×3) is ≈ **1.50 cr**, which EXCEEDS 0.72 cr → **GPU launch presently BLOCKED**; the V1 CPU
preflight (reconstruction, sensitivity, waveform panel, Human-CLAP) spent **0 cr, no GPU**.

**Effective ICASSP spend cap = 3.0 cr (BINDING; arithmetic correction DECISION-V4-10).** With a
5.0-cr balance the nominal 3.5-cr hard cap and the ≥2-cr reserve cannot both hold, so the binding
ceiling is **min(3.5, 5.0 − 2.0) = 3.0 cr** and STOP-3 fires there. The nominal 3.5 is kept only as a
reference. This is arithmetic, not a scientific amendment. The falsifying chain (~1.5–2.2 cr) fits
with the reserve intact; the completion tranche (adapter B + full ladder + oracle) is **not
guaranteed** to fit under 3.0 and must be re-costed only after a positive falsifier using actual
smoke costs — do not assume it remains affordable.

**How these numbers were produced.** One Lightning **Job** (the interactive Studio stays on
free CPU — see `docs/HANDOFF.md`):

```bash
lightning job run --name gpu-benchmark-3 --machine T4 \
    --studio gabriel-allgd-deploy-model-devbox \
    --teamspace general --org independentaudioresearch \
    --command "cd audioldm-modality-swap-pruning && .venv/bin/python scripts/research/gpu_benchmark.py \
               --stage all --expect-gpu T4 --expect-commit e6f50f48ce498652bf5e29652aeec3f17113047c \
               --batches 1,2,4,8 --smoke-steps 5 --smoke-warmup 2 --probe-steps 3 --probe-warmup 1 \
               --steps 30 --iters 20 --warmup 5 --out artifacts/m3_pilot/compute_budget_measured.json"
```

* **Job status:** Completed. **Cost: 0.1372 credits.**
* **Code commit:** `e6f50f48ce498652bf5e29652aeec3f17113047c`, clean tree, verified by the
  job's own preflight (`dirty=False`).
* **Upstream patch in effect:** 1 file, 16 insertions / 2 deletions (DECISION-F10).
* **Raw output:** `artifacts/m3_pilot/compute_budget_measured.json`
  (md5 `12f8fef8577bfbff8b053e9ae90dd81e`), also reproduced verbatim in the job log.
* **Model:** the real pruned `(1,2,3,1)` U-Net with the **real published weights**
  (`l1_audioldm-m-full_p1.ckpt`) — the benchmark refuses to run on a fresh-init model,
  because `zero_module(out.2)` would leave the backward graph almost empty (finding R7c).
* **Preflight gate R7a PASSED on the job:** 284/284 LoRA adapters received non-zero
  gradients and 0 frozen base parameters received any, confirming the F9/F10 fixes hold on
  CUDA and not only on CPU.

## First benchmark — MEASURED (Tesla T4, batch 8)

* **GPU_MODEL:** Tesla T4 *(measured)*
* **VRAM_GB:** 14.562 *(measured)*
* **TRAIN_SEC_PER_STEP:** 1.672427 *(measured, batch 8, 30 steps after 5 warmup)*
* **SALIENCY_SEC_PER_GRAD_EVAL_OR_BATCH:** 1.596534 *(measured, batch 8, 20 iters)*
* **FORWARD_SEC_PER_DIAGNOSTIC_BATCH:** 0.465546 *(measured, batch 8, 20 iters)*
* **GEN_SEC_PER_CLIP:** **MEASURED (job `tgen-2`, Tesla T4, 2026-08-20, commit `39d3627`)** —
  real end-to-end `LatentDiffusion.generate_sample` (DDIM + VAE decode + HiFi-GAN vocoder),
  n_gen=1, guidance 3.5, batch 4, 8 clips per point on the disjoint val split:
  **S=50 → 8.435 s/clip; S=200 → 32.337 s/clip.** Raw: job log; `measure_tgen.py`.
* **GEN_BATCH_SIZE:** 4 *(measured; peak VRAM had ~8 GB headroom, can go higher)*
* **PEAK_TRAIN_VRAM_GB:** 4.177 *(measured)*
* **PEAK_SALIENCY_VRAM_GB:** 4.152 *(measured)*
* **PEAK_FORWARD_VRAM_GB:** 1.540 *(measured)*
* **PEAK_GENERATION_VRAM_GB:** 8.293 *(measured, S=50, batch 4; ~57 % of the T4)*

> **KEY FINDING — the DERIVED `Tgen` was ~2.4× too low (2026-08-20).** The earlier
> derivation `Tgen = DDIM_steps × (Tfwd/batch) × 1.15` gave S=50 → 3.35 s and S=200 →
> 13.38 s. **Measured is S=50 → 8.44 s and S=200 → 32.34 s — 2.5×/2.4× higher.** Root cause:
> classifier-free guidance runs **two** U-Net forwards per DDIM step (conditional +
> unconditional), which the ×1 derivation ignored; VAE decode + vocoder add the rest. **Every
> M4/M5-generation cost below that used the derived `Tgen` is therefore ~2.4× optimistic and
> is corrected here.** This is exactly why `Tgen` had to be measured.

### Batch escalation — MEASURED

Probed in ascending order; **no OOM occurred at any rung**.

| Batch | sec/step | s per sample | Peak VRAM (GB) | Headroom (GB) |
|---:|---:|---:|---:|---:|
| 1 | 0.3897 | 0.3897 | 1.048 | 13.514 |
| 2 | 0.5597 | 0.2799 | 1.497 | 13.065 |
| 4 | 0.9586 | 0.2396 | 2.388 | 12.174 |
| 8 | 1.6410 | 0.2051 | 4.177 | 10.385 |

**`MAX_STABLE_BATCH = 8` is the largest batch TESTED, not the largest that fits.** The
ladder ended at 8 by configuration, with **10.385 GB still free**. Throughput per sample
is still improving (0.3897 → 0.2051 s/sample, a 1.90× gain from batch 1 to 8), so a future
job should extend the ladder to 16/32 before any long training run is costed. The earlier
concern that a 16 GB T4 would be tight for this workload was **not** borne out: PEFT
training on the pruned model peaks at 4.18 GB, about 29 % of the card.

## Provider and cost

* **Provider:** Lightning AI, Studio `gabriel-allgd-deploy-model-devbox`,
  teamspace `independentaudioresearch/general`, cluster `lightning-public-prod` (us-east-1).
* **Price per GPU-hour: ~0.89 credits/GPU-hour** *(derived empirically from three
  independent jobs, final settled costs)* — and this is **not** the figure that was quoted
  to us:

  | Job | Settled cost | ~Billed wall | Implied rate |
  |---|---:|---:|---:|
  | `gpu-benchmark-1` | 0.1168 | ~8 min | 0.88 cr/h |
  | `gpu-benchmark-2` | 0.1179 | ~8 min | 0.88 cr/h |
  | `gpu-benchmark-3` | 0.1674 | ~11 min | 0.91 cr/h |

  Three jobs of different durations converge on **~0.89 cr/GPU-h**, so this is the actual
  on-demand T4 rate on this account rather than a provisioning artefact of one short job
  (an earlier version of this file called it a weak upper bound; three data points now say
  otherwise). **A figure of ~0.19 cr/h was quoted to the project — that is 4.7× lower than
  observed.** Before committing to a large run, confirm the published price and check
  whether 0.19 refers to **interruptible** instances; if so, that is a major lever, and one
  this project can safely use because exact training resume is now proven (S4/F11).
  Also note settled costs came in ~10-30 % above the values read while jobs were running.
* **Credits remaining:** ~9.6 of the 10.0 topped up on 2026-08-19 — **0.402 credits spent
  across three jobs** (settled: 0.1168 + 0.1179 + 0.1674; two of the three were failures,
  see the ledger), plus an un-itemised interactive T4 Studio window on 2026-08-19 that does
  not appear in job costs.
* **Paid amount forecast:** cannot be stated until `Tgen` is measured and the M5 step count
  is decided — see Compute Gate CG.
* **Institutional compute assumed:** TBD — **this is now a live question, not a formality.**
* **Contingency:** 20%

## Milestone budget

`Ttrain = 1.672427`, `Tsal = 1.596534`, `Tfwd = 0.465546` s (all measured at batch 8).
Formulas are the master plan's §7.3.

| Milestone | Task | Formula / measured units | GPU-hours | Basis | Evidence |
|---|---|---|---:|---|---|
| M0 | generation/eval smoke | `Neval * Tgen / 3600` | **UNKNOWN** | Tgen not measured | — |
| M1 | 500-step real smoke | `500*Ttrain/3600` | **0.232** | derived from measured | benchmark JSON |
| M2 | paired diagnostics | `batches*Tfwd/3600` | **<0.05** | derived; 48 examples | `docs/condition_swap_validation.md` |
| M3A | random-null diagnostics | `(Krand+1)*2*ceil(Neval/8)*Tfwd/3600`, `Krand=20`, `Neval=200` | **0.136** | derived from measured | benchmark JSON |
| M3B | saliency | `4B*Tsal/3600`, `B=256` (P1 `2B` + P2/P3 `B+B`) | **0.454** | derived; B is a *draft* protocol value | `docs/pilot_protocol.md` |
| M4 | screening + confirmatory generation | `Neval*Tgen/3600` | **UNKNOWN** | Tgen not measured | — |
| M5 | recovery, 100k steps **per model** | `100000*Ttrain/3600` | **46.46 per model** | derived from measured | benchmark JSON |
| M6 | analysis | CPU | ~0 | — | — |

**Measurable subtotal excluding M4 and M5: 0.82 GPU-hours.**

> **CORRECTION (2026-08-20, external review — an earlier version of this file was wrong).**
> That 0.82 figure was previously described as "the entire RQ1/RQ2 programme". **It is
> not.** `docs/claims_matrix.md` states RQ2a's required evidence as **"M3 Gate B + M4"**:
> closing RQ2 scientifically means pruning with P0/P1/P2/P3, **generating audio, and
> evaluating FAD/KL/PANNs** — i.e. M4, whose cost is unknown because `Tgen` is unmeasured.
> What 0.82 GPU-hours actually buys is **the diagnostics of RQ1 and the saliency
> computation for RQ2** — the genuinely novel machinery — not the RQ2 verdict.
> The accurate statement is: *the modality-swap diagnostics and paired-saliency
> computation are surprisingly cheap; the generation-based evaluation is not yet costed;
> the identified bottleneck is M5 recovery at ~46 GPU-h per model.*

**M5 dominates everything by roughly 50×.** At 46.46 GPU-hours per model, four criteria
(P0/P1/P2/P3) would be **185.8 GPU-hours**. At the empirically derived ~0.89 credits/GPU-hour
that is **~42 credits for a single model** against ~9.6 credits available. Even an
order-of-magnitude cheaper marginal rate would not bring four models within the current
balance. This is the finding that Compute Gate CG has to be decided on.

*(Caveat on M5: `100000*Ttrain` uses `Ttrain` at batch 8. Extending the batch ladder would
raise throughput per sample, and the 100k-step figure is the master plan's own number, not
an optimised one. Both are levers, and both are Gabriel's call — see below.)*

## Full-experiment credit estimate (2026-08-20)

**`Tgen` is now MEASURED (2026-08-20), superseding the derivation below.** `Ttrain`, `Tfwd`,
`Tgen` and the credit rate are all measured now.

| DDIM steps | Derived Tgen (old, ×1 fwd) | **MEASURED Tgen** |
|---:|---:|---:|
| 50 | 3.35 s | **8.435 s** |
| 200 | 13.38 s | **32.337 s** |

The derivation was ~2.4× low (CFG = 2 forwards/step; see KEY FINDING above). **Screening uses
S=50 (DECISION-CG-001); confirmatory S=200 is deferred.** `Neval = 200` per the protocol.

Model count for generation assumes the five-criterion split (see the P0 note below):
**P0-published, P0-L1, P1, P2, P3, plus the unpruned base = 6 models.**

| Component | GPU-h | Basis |
|---|---:|---|
| M1 GPU acceptance (500 steps) | 0.23 | measured `Ttrain` |
| M2 paired diagnostics | 0.05 | measured `Tfwd` |
| M3A random null (21 masks × 2 modalities) | 0.14 | measured `Tfwd` |
| M3B saliency (`4B`, B=256) | 0.45 | measured `Tsal` |
| M4 screening generation (6 models × 200 clips) | 4.46 | **derived `Tgen`** |
| M4 confirmatory (3 models × 2 extra seeds) | 4.46 | **derived `Tgen`** |
| M5 recovery | **46.46 per model** | measured `Ttrain`, 100k steps |
| M5 post-recovery generation | 0.74 per model | **derived `Tgen`** |

### Scenarios (including the master plan's 20 % contingency, at ~0.89 cr/GPU-h)

| Scenario | GPU-hours | **Credits** | Notes |
|---|---:|---:|---|
| **A — RQ1 + RQ2 complete** (diagnostics, saliency, pruning, generation, FAD/KL/PANNs; **no recovery**) | 11.8 | **~10** | fits, barely, in the current ~9.6 balance |
| **B — A + RQ3 with 2 recovered models** | 125.0 | **~111** | minimum defensible recovery arm |
| **C — A + RQ3 with 5 recovered models** | 295.0 | **~263** | the full pre-registered programme |

### The levers, in order of impact

1. **Recovery step count.** `100000` is the master plan's number, not an optimised one.
   Parameter-efficient recovery typically converges far sooner. At **20k steps** M5 falls
   from 46.46 to **9.3 GPU-h per model (÷5)**: scenario **B → ~32 credits**, **C → ~64
   credits**. This dwarfs every other lever and should be settled by a short
   loss-vs-steps curve, which is cheap.
2. **Interruptible instances.** Typically ~half price. This project can use them safely
   because **exact training resume is now proven** (F11 fix + regression S4 + the
   `m1_gpu_acceptance.py` resume test), so a preemption costs a restart, not a run.
   Combined with lever 1: **scenario C ≈ 32 credits**.
3. **Batch 16/32.** Per-sample cost was still falling at batch 8 (0.2051 s) with 10.4 GB
   free; a further 15-20 % is plausible on both training and generation.
4. **DDIM steps for screening.** Using S=50 for screening and S=200 only for the
   confirmatory comparison cuts the screening generation ~4×.

### Bottom line

* **RQ1 and RQ2 end to end are affordable today** (~10 credits) — *including* the
  generation-based evaluation, which the earlier version of this file wrongly omitted.
* **RQ3 as pre-registered is not** (~111-263 credits against ~9.6).
* With the two main levers (20k-step recovery + interruptible), **the full programme lands
  near ~32 credits**, which changes the decision from "cut the recovery arm" to "top up
  modestly and validate the step count first".

All of this rests on a derived `Tgen`. Measure it before treating any of these as firm.

## Compute Gate CG

* **Decision:** **RESOLVED (Gabriel, 2026-08-20, ledger DECISION-CG-001) — resolved by
  descoping.** Authorized now: the core RQ1+RQ2 programme (M1 GPU acceptance, M2, M3A,
  M3B, pruning, **screening generation at DDIM S=50** + FAD/KL/PANNs) plus a **~2k-step M5
  convergence probe**. Deferred until top-up: S=200 confirmatory generation and any full
  M5 recovery. **Hard reserve: 2.0 credits untouched** (spendable ≈ 7.6). Interruptible
  instances: trial cheaply first, then use for long jobs (exact resume proven).
* **Evidence date:** 2026-08-19 (measurements); decision 2026-08-20. The table below
  records the state the decision was taken on.

| § | Condition | Status |
|---|---|---|
| 1 | a usable cloud environment is confirmed | **SATISFIED** — Lightning AI, T4 via Jobs, end-to-end verified |
| 2 | M1 benchmark has produced real throughput/VRAM numbers on the selected GPU | **SATISFIED** — this document |
| 3 | projected ICASSP-core cost fits Lightning credits + ~US$50 discretionary and/or institutional compute | **RESOLVED BY DESCOPING (DECISION-CG-001)** — the S=50-screening core fits the balance; RQ3 recovery + S=200 confirmatory deferred to a top-up informed by the M5 probe |
| 4 | projected completion leaves paper writing starting by 2026-09-05 | **SATISFIED under the descoped plan** |

* **Projected ICASSP-core cost:** cannot be stated. Two inputs are missing: `Tgen`
  (blocking M4) and a decision on the M5 step budget / number of recovered models.
* **Projected completion date:** blocked on the same two inputs.
* **Decision rationale (what the numbers actually say):** the diagnostic and saliency
  programme — M1, M2, M3A, M3B, i.e. **RQ1 and RQ2, the modality-swap and paired-saliency
  claims** — costs **under one GPU-hour in total** and is comfortably affordable now. The
  cost is concentrated entirely in **M5 recovery (RQ3)** and in the unmeasured **M4
  generation**. So the gate does not threaten the core scientific contribution; it
  threatens the *recovery* arm and the *generation-based evaluation*.
* **Recommended next measurements, in order:** (a) a GPU job that measures `Tgen`
  (`--with-generation`, once the generation stack is wired) so M4 can be costed at all;
  (b) extend the batch ladder to 16/32, since 10.4 GB was still free and per-sample
  throughput was still improving; (c) only then re-evaluate condition 3 with real numbers
  for both arms.

## SA3 T4 smoke — MEASURED (Tesla T4, 2026-08-21, job `sa3-smoke-1`, 0.156 cr, commit 61dfbfc)

First real GPU numbers for the Stable Audio 3 line (fp16, `small-sfx` post + base). Raw:
`artifacts/sa3/smoke_t4.json`. Engineering smoke on P_smoke (4 prompts); **not scientific**.

* **GPU:** Tesla T4. **Load fp16:** post 19.4 s / **2.28 GB peak**, base 11.8 s / 2.29 GB.
* **Raw DiT s/forward (fp16):** batch 1 **0.0622 s**; batch 4 0.0632 s (0.0158/sample); batch 8
  0.1182 s (**0.0148/sample**). Peak VRAM ~1.79 GB → **enormous headroom on the 14.5 GB T4**;
  batching to 16/32 is free.
* **η (fp16-vs-fp32 post field, per level):** max **6.67e-5** (range 1.8e-5–6.7e-5) — fp16 resolves
  the field; the tangent-regime precision floor `‖δF‖²/‖F‖² ≥ 10η ≈ 6.7e-4` is achievable.
* **S_traj:** 8 states, τ exactly match `schedule_post_10s.json`. **Empty BlockMask bit-exact** (fp16).
* **Generation latency (fp16, incl. SAME-S decode, T4, median):** dense 4-step **0.338 s**, 5
  **0.393**, 6 **0.443**, 7 **0.514**, 8 **0.600 s**; block-skip(5)@8 **0.560 s** (removing 1 of 20
  blocks saves ~0.04 s/gen at 8 steps; nearest-latency dense comparator ≈ 7.5 steps).

### Pilot cost model (from these measurements) — for the budget rule

At ~0.89 cr/GPU-h, T4 wall dominated by DiT forwards (~0.015 s/sample batched) + generations
(0.6 s dense-8). A pilot first rung (N=16, n_u=8) is on the order of tens of minutes of T4
(field forwards + a minimal A_tan probe pass + R=5 dense streams + 8→7 margins), i.e. **~0.2–0.6
cr** — affordable under the 5-cr overnight cap. A_tan's probe×block×κ×family fan-out is the cost
driver; start minimal (U_gen, κ_0) and expand per the pre-registered rules.

## SA3 pilot + adversary — MEASURED (Tesla T4, 2026-08-21)

Overnight-mandate GPU spend (cap 5 cr / target ≤2): **smoke 0.156 + field-pilot 0.289 + adversary
0.195 = 0.640 cr total** (well under target). Stopped job `sa3-pilot-fields-1` = 0.000 cr (caught
in Pending). All on-demand T4, fresh snapshots, `max_runtime` caps, per-job CPU dry-run first.

* **`sa3-pilot-fields-2`** (RQ1 field pilot, N=32, 20 blocks, no-dep): wall 676 s compute, **0.2888 cr**.
  Per-block forward loop with `block_mask` swaps runs ~1.7× slower than the tight-loop s/forward
  (0.062 s) — budget field-stage work at ~0.1 s/effective forward, not 0.06.
* **`sa3-adversary-1`** (single-block E, N=16, 29 systems, 464 gens): wall 255 s compute, **0.195 cr**
  → ~0.55 s/generation incl. block-mask overhead + wav save (vs 0.60 s dense-8 in the tight smoke).
* **Scoring note:** OpenL3/FD on CPU is ~15–30 s/clip → impractical for 464 clips (~2 h); the pilot
  adversary is scored CLAP+KL only (`--no-fd`). A GPU scoring pass (or a subset) is needed for FD.

### Settled costs (final, 2026-08-21 overnight)

Settled ~10–30% above running figures (as noted above). Overnight-mandate total (cap 5, target ≤2):

| job | status | settled cr |
|---|---|---:|
| sa3-smoke-1 | Completed | 0.1560 |
| sa3-pilot-fields-1 | Stopped (in Pending→Running) | 0.0357 |
| sa3-pilot-fields-2 | Completed | 0.3253 |
| sa3-adversary-1 | Completed | 0.2161 |
| **TOTAL** | | **0.7331** |

**Lesson:** a job stopped seconds into Running still settled at **0.036 cr** (provisioning/snapshot is
not free) — cheap, but not zero; prefer getting the launch right over stop-and-relaunch.

## MEASURED — T4 LoRA training (SA3 controls), 2026-08-21 (job sa3-smoke-t4-1)

Infra micro-smoke (synthetic data, `L_6` single-block, standard `lora` r16, backbone block 6,
`16-mixed` + base fp16, 10 s crop, batch 1). **MEASURED on Tesla T4:**

* `TRAIN_SEC_PER_STEP` (LoRA, batch 1) = **0.2534 s/step** (median of 25, warmup dropped)
* `PEAK_VRAM` = **1.81 GB** (of 16 — no OOM; large headroom for batch/steps)
* model load ≈ 13 s; 495 K trainable LoRA params of 568 M.
* **Projection @ ~0.89 cr/GPU-h:** 1000 steps ≈ 0.070 GPU-h ≈ **0.066 cr/control** (incl. load);
  **`L_6` + `L_13` ≈ 0.132 cr**. Synthetic infra numbers only — not a scientific result.
* **Budget conservatively (Gabriel, 2026-08-22): 0.08–0.10 cr per control**, NOT 0.066 as a hard
  expectation — real data adds I/O + step-time variance, and the Lightning **job lifecycle can
  dominate compute cost on short runs** (see SA3-SMOKE-T4-001: a hung job idle-billed past its
  compute). Always attach `scripts/sa3/job_watchdog.py` with a `--max-cost` ceiling to real jobs.
* **Budget accounting:** the SA3 5-cr hard cap is on **TOTAL** billable Lightning (GPU + CPU), not
  GPU-only — CPU jobs bill too. Reconciled counters live in `docs/sa3/budget_reconciliation.md`
  (`scripts/sa3/reconcile_budget.py`).
* **Ops caveat:** the T4 job kept billing after compute finished (wandb non-daemon threads block a
  clean exit); `train_control_loras.py --smoke` now `os._exit()`s to prevent idle billing. For long
  training jobs, poll cost and/or set a max runtime.

## Gate-0 (ICASSP Scenario B) M-Full smoke — MEASURED (Tesla T4, 2026-08-26, job `gate0-smoke-1`, commit `8850417`)

First real GPU numbers for the ICASSP Gate-0 recipe on **DENSE AudioLDM-M-Full + LoRA** (r8/α16 on
`to_q`/`to_v`), **FP32**, EMA convention (V4-12), latent_t=96 (3.84 s). Bounded executions of the
PRODUCTION trainer (`gate0_trainer.train_one_step`) and PRODUCTION generator (`gate0_generator.generate`)
via `scripts/research/gate0_smoke.py`. Raw: `artifacts/icassp_gate0/smoke_t4_measured.json`
(md5 `cd956ee423df5337a39b97b471362845`), reproduced verbatim in the job log; preflight passed
(`--expect-commit 8850417…`, `--expect-gpu T4`, dirty=false). **Settled cost: 0.1835 cr** (running
0.1511; `total_spent` 55.62→55.80 = +0.183). The script `os._exit(0)`ed after persisting, but the job
status flipped Running→Pending and idled ~40 s until the **watchdog stopped it at the 12-min ceiling**
(`job_watchdog.py --max-cost 0.20 --max-minutes 12`, `killed=True`) — the measurement was already
complete; the watchdog prevented open-ended idle billing. **Balance endpoint still returns `balance=5.0`
(a fixed allowance field); `total_spent` is the live usage tracker.**

* **GPU:** Tesla T4 *(measured)*
* **TRAIN_SEC_PER_STEP:** **0.30735** s/step *(measured; batch 2, dense M-Full + LoRA, FP32, 10 timed after 3 warmup)*
* **PEAK_TRAIN_VRAM_GB:** **5.376** *(measured, FP32; 34 % of the 16 GB T4 — the recipe is NOT memory-bound, so mixed precision would buy speed, not headroom)*
* **GEN_SEC_PER_CLIP:** **7.143** s/clip *(measured; 50 DDIM, guidance 2.5, eta 0.0, latent_t 96, dense, batch 1, warmup-discarded: 3 timed after 1 warmup)*
* **PEAK_GENERATION_VRAM_GB:** **5.336** *(measured)*

### Gate-0 cost projection (at 0.89 cr/GPU-h) — STOP-0 TRIPPED

| Component | Units | GPU-h | Credits |
|---|---:|---:|---:|
| Train (adapter) | 19,400 updates × 0.30735 s | 1.656 | **1.474** |
| Generation | 384 clips (64×3×2) × 7.143 s | 0.762 | **0.678** |
| **Gate-0 total** | | **2.418** | **≈ 2.152** |

* **STOP-0 (frozen Gate-0 ceiling 1.0 cr): FAIL — projected 2.152 cr > 1.0 cr.** The FP32 200-epoch
  faithful recipe costs ~2.15 cr for Gate 0 alone. The smoke did exactly its job: caught this before
  the paid run.
* Gate 0 fits under the 3.0-cr effective cap in isolation (leaves ~2.85), **but** Gate 0 (2.15) + the
  phenomenon falsifier (~1.5–2.2) together exceed the 3.0-cr spendable → the programme does not fit as-is.
* Projection is a **floor**: excludes per-job model-load/provisioning (~0.03–0.05 cr) and per-epoch
  dataloading/checkpoint overhead. **CLAP scoring is free** (Studio CPU / `.venv-metrics`; ~2–3 min for
  384 clips; Gate-0 3.84-s clips never reach the CLAP >10-s fusion path) — 0 cr, not in the GPU projection.
* **Dominant driver = training (68 % of the cost).** Levers, **each an EXPLICIT decision, none applied
  here:** (1) interruptible T4 (~½ price, exact resume proven → Gate 0 ≈ 1.08 cr) — no fidelity/recipe
  change; (2) shorter train horizon than Kim's 200 ep (recipe deviation from the published operating
  point); (3) fp16/bf16 (Gabriel: separate decision, numerics-equivalence validation required; buys
  speed not memory); (4) fewer seeds/prompts (battery frozen for statistical power). **No fidelity or
  recipe change was made — STOP-0 is reported and the line stops for Gabriel's decision.**

## Balance reconciliation + PRE-GATE0 ADMINISTRATIVE budget amendment (2026-08-26 ~19:28 MVD)

**Context:** Gabriel reports the account was **topped up to ~10 credits**. This supersedes the budget
premise the old effective cap rested on. This is an **administrative** amendment — **no scientific
parameter changed, and nothing changed because of the smoke result.**

### Authoritative billing query (2026-08-26 ~19:28 MVD)

| Field | Value | Source |
|---|---|---|
| `balance` | **5.0** *(unchanged)* | `LightningClient().billing_service_get_user_balance()` |
| `total_spent` | **55.802705** | same |
| account_id | `371a6f4c-1280-40b9-8bb0-d97a49597756` | same |
| `billing_service_get_account_balance()` | auth error (unchanged) | — |
| **Funded balance (owner-reported)** | **~10 cr** | Gabriel, 2026-08-26 |

**Honest discrepancy, recorded not smoothed over:** the queryable `balance` field returns **5.0** and did
**not** move across either the top-up **or** the 0.183-cr smoke spend — so that field is a static
allowance, **not** the live funded pool. Only `total_spent` moves (55.62 → 55.80 = the smoke). The
funded pool (~10 cr) is **not exposed** by the accessible SDK endpoint; `billing_service_get_account_balance()`
still auth-fails. No "10" reading was fabricated. Operate against Gabriel's **explicit authorization**
below, tracking spend via `total_spent` deltas + settled per-job cost.

### The old effective cap is OBSOLETE

`effective_cap = min(3.5, 5.0 − 2.0) = 3.0 cr` (DECISION-V4-10) rested on a 5.0-cr balance premise that
has changed. **It no longer governs.** (Preserved above for history; not deleted.)

### Amendment (DECISION-V4-13, administrative — NOT scientific)

* Prior **STOP-0 = FAIL** was a **budgetary feasibility** verdict under the previous 1.0-cr Gate-0
  ceiling — **not** a scientific failure. **That history stands and is preserved** (Gate-0 smoke section
  above; ledger GATE0-SMOKE-T4-RUN). The top-up makes that 1.0-cr **feasibility** ceiling obsolete.
* **The scientific Gate-0 recipe is UNCHANGED.** No SESOI, battery, seeds, rank/α, LR, epochs, precision,
  guidance, DDIM, or scorer changed. **Nothing changed because of the smoke result** — the smoke only
  measured throughput/cost.
* **Authorized: up to 5.0 cr from this point for the central ICASSP chain** (Gate 0 + the pre-registered
  phenomenon falsifier), **including lifecycle/provisioning/idle contingency.** Do **not** spend beyond
  5.0 cr without returning for authorization.
* **Path preference (Gabriel):** on-demand T4 + FP32 + exact preregistered 200-epoch recipe. **No**
  interruptible, fp16/bf16, shorter horizon, fewer prompts/seeds, or loss-vs-steps probe. Periodic resume
  checkpoints are engineering insurance only (on-demand ⇒ preemption-resume equivalence is not required).

### Re-cost of the central chain (measured smoke values; 0.89 cr/GPU-h)

| Stage | Work | GPU-h | Credits |
|---|---:|---:|---:|
| Gate-0 train (adapter) | 19,400 × 0.30735 s | 1.656 | **1.474** |
| Gate-0 generation | 384 (64×3×2: dense ± LoRA) × 7.143 s | 0.762 | **0.678** |
| **Gate-0 subtotal** | | 2.418 | **≈ 2.152** |
| Falsifier generation | **768** (4 systems × 64 × 3) × 7.143 s | 1.523 | **≈ 1.356** |
| **Central chain total** | | 3.941 | **≈ 3.508** |

* **WAV reuse (Gabriel):** if Gate 0 PASSES, the **384 dense ± LoRA WAVs are reused** in the phenomenon
  experiment — **do not regenerate.** The falsifier then adds only the 4 downstream systems
  (`p1_pruned_ema_reconstructed` ± sliced LoRA, `p1_recovered` ± sliced LoRA) = **768** generations.
* **≈ 3.51 cr + lifecycle overhead + essentially-free CPU CLAP scoring** fits inside the authorized 5.0-cr
  envelope with headroom for provisioning/idle. CLAP scoring runs on Studio CPU (0 cr).

### Gate-0 ACTUAL settled costs (2026-08-27)

* **Training `gate0-train-1`: settled 1.7969 cr** (projected 1.474; +22% = settlement uplift + model load + epoch-boundary checkpoint writes). 19,400 updates, adapter sha `84a24a38`, commit e349d69, **Completed clean** (normal sys.exit). ~1.9 h wall.
* **Generation `gate0-gen-1`: RUNNING** (projected 0.678 cr, 384 dense±LoRA). Settled cost to be recorded on completion.
* **Spend to date:** smoke 0.1835 + train 1.7969 = **1.98 cr** of the authorized 5.0-cr central-chain envelope.

### Gate-0 FINAL costs + verdict (2026-08-27)

* **Training `gate0-train-1`: 1.7969 cr** (Completed clean). **Generation `gate0-gen-1`: 0.8844 cr** (Completed clean, 384 dense +/- LoRA, ~57 min). **Gate-0 total = 2.6813 cr.** CLAP scoring free (Studio CPU).
* **Total spent (smoke 0.1835 + train 1.7969 + gen 0.8844) = 2.8648 cr; remaining under the 5.0-cr envelope = 2.1352 cr.**
* **Gate 0 verdict: PASS** (ΔCLAP 0.0464, CI95 [0.0221, 0.0720]; `artifacts/icassp_gate0/gate0_verdict.json` md5 62cefa0b6e8e72f567e63bfa6f93f85c).
* **Updated falsifier projection:** 768 gens at the realized gen rate (0.8844 cr / 384 incl. overhead) ~= 1.5-1.8 cr -> central chain ~4.3-4.7 cr, within 5.0. Needs the sliced-adapter generation path built first.

### Phenomenon 768-gen projection — path BUILT (2026-08-27, PHENOM-STAT-D), GPU NOT launched

* **4 systems = {p1_pruned_ema_reconstructed, p1_recovered} × {off, on} = 768 NEW WAVs** (dense 384 reused byte-for-byte, NOT regenerated). Sliced-adapter generation path built + CPU dry-run GREEN on both backbones (sliced adapter sha `5cc0a79a`; loads into pruned + recovered; OFF/ON generate; manifests validate 0 errors).
* **Projected settled cost for exactly 768 WAVs:** pure-gen floor 768 × 7.143 s/clip @ 0.89 cr/GPU-h = **1.356 cr**; at the realized gate0-gen rate (0.8844/384 = 0.002303 cr/WAV incl. overhead) ≈ **1.77 cr**. **Central chain would reach ≈ 4.22–4.63 cr** (2.8648 spent + 1.36–1.77), within the authorized 5.0-cr envelope.
* Entry: `scripts/research/run_phenomenon_gen.sh`. Launch (NOT executed): `lightning job run --machine T4 --studio gabriel-allgd-deploy-model-devbox --teamspace general --org independentaudioresearch "cd audioldm-modality-swap-pruning && bash scripts/research/run_phenomenon_gen.sh"` + `job_watchdog.py`. Falsifier NOT authorized; awaiting Gabriel.

### Phenomenon falsifier ACTUAL settled cost (2026-08-27, job `gate0-phenom-1`)

* **`gate0-phenom-1` COMPLETED clean** (killed=False), settled **1.6020 cr**, ~101 min, T4 FP32, 768 WAVs (watchdog max-cost 2.05 / max-min 150; not triggered). Projected 1.36–1.77; actual 1.602 (within band).
* **Central chain (settled):** smoke 0.1835 + train 1.7969 + Gate-0 gen 0.8844 + phenom gen 1.6020 = **4.4668 / 5.0 cr**; remaining **0.5332 cr**. CLAP scoring free (Studio CPU). GPU stopped after the falsifier (STOP-2 negative; no further runs).

### 2026-08-30 (MVD 01:37) | BUDGET RECONCILIATION before OP-DURATION-DISCRIMINATOR-1 (Arm D) — appended, no history overwritten

Reliable signal = each job's **settled `total_cost`** via `lightning_sdk` in the cloudspace env (the API `balance`/`total_spent` scalar was shown unreliable during V1.1). Do NOT conflate the two counters below.

* **Historical project "central-chain" envelope (5.0 cr, ICASSP plan)** — PRESERVED as historical accounting. Settled central-chain jobs: smoke `gate0-smoke-1` 0.1835 + `gate0-train-1` 1.8304 + `gate0-gen-1` 0.9030 + `gate0-phenom-1` 1.6339 = **4.5508 cr** (minor settlement uplift vs the earlier 4.4668 ledger estimate; do not rewrite the historical entries).
* **Gabriel top-up (+2.0 cr) authorized before V1.1** → project-available ≈ **2.72 cr** at V1.1 start (Gabriel authoritative).
* **V1.1 `reversal-v11-gen-1` settled = 1.2623 cr** (confirmed via `total_cost`; matches the total_spent delta 66.770→68.032 recorded at run time).
* **RECOVERY-METRIC-AUDIT-1 + OP-discriminator design audit + Arm-D preflights = 0 cr** (all CPU, Studio).
* **Current project-available ≈ 2.72 − 1.2623 ≈ `1.46 cr`.** (Teamspace-lifetime settled across all 39 jobs = 16.4431 cr is an account-wide counter spanning other projects/SA3 — NOT the project envelope; recorded only for provenance.)
* **Arm D (OP-DURATION-DISCRIMINATOR-1):** 160 new WAVs @ 10.24 s/DDIM50/single, pruned+recovered only. **Projected ≈ 1.02 cr** (160 × 0.002191 cr/WAV × latent 256/96 + 0.09 fixed). **HARD MAX NEW GPU SPEND = 1.20 cr.** Do NOT launch if projected cost under current pricing exceeds 1.20 cr; do NOT shrink N post-freeze for budget.
* **Expected post-run reserve:** 1.46 − 1.02 ≈ **0.44 cr** (expected); ≥ **0.26 cr** at the 1.20 cap. Meets the ≥0.25 reserve rule.
* Arm-D actual settled cost to be read directly from the job's `total_cost` post-run (as for V1.1).
