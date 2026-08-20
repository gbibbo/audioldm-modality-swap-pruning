# Compute Budget

Do not invent GPU-hour estimates. Every number below is either **MEASURED** on real
hardware or **DERIVED** from measured values by the master-plan §7.3 formulas, and each is
labelled. Anything still unmeasured says so.

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
* **GEN_SEC_PER_CLIP_OR_BATCH:** **NOT MEASURED** — the generation stack is not wired
  (`--with-generation` was not run). This blocks the M4 projection; see Compute Gate CG.
* **GEN_BATCH_SIZE:** **NOT MEASURED**
* **PEAK_TRAIN_VRAM_GB:** 4.177 *(measured)*
* **PEAK_SALIENCY_VRAM_GB:** 4.152 *(measured)*
* **PEAK_FORWARD_VRAM_GB:** 1.540 *(measured)*
* **PEAK_GENERATION_VRAM_GB:** **NOT MEASURED**

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
* **Price per GPU-hour:** **NOT MEASURED DIRECTLY.** *(Derived, weak.)* `gpu-benchmark-3`
  cost **0.1372 credits** for roughly 9 minutes of wall time, implying **~0.91
  credits/GPU-hour**. Treat that as an **upper bound only**: machine provisioning dominates
  a 9-minute job, so the marginal rate for a long run is lower. Do not use this figure for
  a serious projection — read the published price, or derive it from a long job.
* **Credits remaining:** ~9.6 of 10.0 topped up on 2026-08-19 *(10.0 minus 0.107 + 0.091 +
  0.137 across three jobs; see the ledger for the two failures)*.
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

**Measurable subtotal excluding M4 and M5: 0.82 GPU-hours.** M1, M2, M3A and M3B together
are under one GPU-hour — i.e. **the scientific core of the pruning study is essentially
free at this scale.**

**M5 dominates everything by roughly 50×.** At 46.46 GPU-hours per model, four criteria
(P0/P1/P2/P3) would be **185.8 GPU-hours**. At the weakly-derived ~0.91 credits/GPU-hour
that is **~42 credits for a single model** against ~9.6 credits available. Even an
order-of-magnitude cheaper marginal rate would not bring four models within the current
balance. This is the finding that Compute Gate CG has to be decided on.

*(Caveat on M5: `100000*Ttrain` uses `Ttrain` at batch 8. Extending the batch ladder would
raise throughput per sample, and the 100k-step figure is the master plan's own number, not
an optimised one. Both are levers, and both are Gabriel's call — see below.)*

## Compute Gate CG

* **Decision:** **UNRESOLVED — requires Gabriel.** It is a *schedule* decision, not a
  scientific one (master plan §7.4). Condition-by-condition against §7.4:
* **Evidence date:** 2026-08-19 (target decision date was 2026-08-18, already passed).

| § | Condition | Status |
|---|---|---|
| 1 | a usable cloud environment is confirmed | **SATISFIED** — Lightning AI, T4 via Jobs, end-to-end verified |
| 2 | M1 benchmark has produced real throughput/VRAM numbers on the selected GPU | **SATISFIED** — this document |
| 3 | projected ICASSP-core cost fits Lightning credits + ~US$50 discretionary and/or institutional compute | **CANNOT BE SATISFIED AS SPECIFIED** — M5 alone is 46.46 GPU-h per model; `Tgen` is still unmeasured so M4 is unknown |
| 4 | projected completion leaves paper writing starting by 2026-09-05 | **UNDECIDABLE** until 3 is resolved |

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
