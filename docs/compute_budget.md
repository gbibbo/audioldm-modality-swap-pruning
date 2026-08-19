# Compute Budget

Do not invent GPU-hour estimates before the first measured benchmark.

**To populate this file:** on a GPU studio run
`.venv/bin/python scripts/research/gpu_benchmark.py --steps 30 --batch 8 --out docs/compute_budget_measured.json`
(add `--with-generation` once the generation stack is wired). The script measures
every §7.2 variable on the real pruned `(1,2,3,1)` U-Net via the tested M1 PEFT
setup, and **refuses to run without CUDA** so no value here is ever fabricated.
Every field below stays `TBD_MEASURED` until that benchmark writes real numbers.

## First benchmark

* **GPU_MODEL:** TBD_MEASURED
* **VRAM_GB:** TBD_MEASURED
* **TRAIN_SEC_PER_STEP:** TBD_MEASURED
* **SALIENCY_SEC_PER_GRAD_EVAL_OR_BATCH:** TBD_MEASURED
* **FORWARD_SEC_PER_DIAGNOSTIC_BATCH:** TBD_MEASURED
* **GEN_SEC_PER_CLIP_OR_BATCH:** TBD_MEASURED
* **GEN_BATCH_SIZE:** TBD_MEASURED
* **PEAK_TRAIN_VRAM_GB:** TBD_MEASURED
* **PEAK_SALIENCY_VRAM_GB:** TBD_MEASURED
* **PEAK_GENERATION_VRAM_GB:** TBD_MEASURED

## Provider and cost

* **Provider:** Lightning AI
* **Price per GPU-hour:** TBD_MEASURED
* **Credits remaining:** TBD_MEASURED
* **Paid amount forecast:** TBD_MEASURED
* **Institutional compute assumed:** TBD
* **Contingency:** 20%

## Milestone budget

| Milestone | Task | Formula / measured units | GPU-hours | Cost | Evidence |
|---|---|---|---:|---:|---|
| M0 | generation/eval smoke | measured | TBD | TBD | |
| M1 | 500-step real smoke | `500*Ttrain/3600` | TBD | TBD | |
| M2 | paired diagnostics | `batches*Tfwd/3600` | TBD | TBD | |
| M3A | L1 + random-null diagnostics | measured | TBD | TBD | |
| M3B | saliency | `grad_batches*Tsal/3600` | TBD | TBD | |
| M4 | screening + confirmatory generation | measured | TBD | TBD | |
| M5 | recovery runs + evaluation | measured | TBD | TBD | |
| M6 | analysis | measured | TBD | TBD | |

## Compute Gate CG

* **Decision:** UNRESOLVED
* **Evidence date:**
* **Projected ICASSP-core cost:**
* **Projected completion date:**
* **Decision rationale:**
