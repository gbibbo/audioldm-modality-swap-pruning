# Round-2 results: sufficient-stats sizing, I_PT collapse, and the A_tan gate

Follows Gabriel's round-2 review (per-prompt sufficient statistics; no re-GPU for sizing). All
GPU jobs used a clean pushed commit + `--expect-commit` + `max_runtime`, per-prompt stats persisted.

## Sizing (expanded pilot N=64, job `sa3-pilot-n64-1`, 0.485 cr)

Disjoint-pair bootstrap (protocol §2.3, B=1000) over the persisted per-prompt sufficient statistics
(no re-forwards), rungs {16, 32}:

| rung | D_P p95 disagreement (k=2/4/6) | I_PT p95 (k=2/4/6) |
|---|---|---|
| 16 | 0 / **1** / 0 | **1** / 0 / **1** |
| 32 | **0 / 0 / 0** | 0 / 0 / **1** |

* **D_P removal sets are STABLE at N=32** (all k). rung 32 fails ONLY on I_PT (k=6). ⇒ **N_main = 32
  for D_P**; N=16 is underpowered (as the adversary already showed).
* **I_PT is noisier than D_P** (needs > 32) **and** redundant (below). It is dropped.

## I_PT ≈ D_P collapse (floored, N=64)

| k | R_DP (removable) | R_IPT | Δ | verdict |
|---|---|---|---|---|
| 2 | {12,13} | {13,14} | 1 | real (floors 0) — but a 1-block swap among near-tied interiors |
| 4 | {11,12,13,14} | {11,12,13,14} | 0 | identical |
| 6 | {9,10,11,12,13,14} | {10,…,15} | 1 | within floor |

**I_PT provides no decision-relevant criterion distinct from D_P** (identical at k=4; ±1 near-tied
interior at k=2/6) and is strictly less stable. **The PT-specific-criterion idea is closed** (per
Gabriel's rule "si I_PT sigue colapsado sobre D_P, cerrar"). RQ1's structural amplification stays
real (erratum E6); **D_P alone identifies the removable interior blocks {9–14}**, stable at N=32.

## The A_tan decision gate (job `sa3-atan-1`, RUNNING)

The remaining question is the RQ2 one, and it is cheap: **does `A_tan` carry structural information
distinct from `D_P`?** Primary probe family only (U_gen = standard LoRA r16), n_u=16, κ=0.01
(tangent regime confirmed: ‖δF(2u)‖/‖δF(u)‖ ≈ 2.0), N=16 reusing the persisted states.

`size_atan_from_stats.py` computes, with a bootstrap floor, δ(R_{A_tan}, R_{D_P})(k):
* if **not real** (A_tan's removable ranking ≈ D_P's) → the adapter-compatible-pruning hypothesis
  is not supported; **close the line without building CC0 datasets or training LoRAs**;
* if **real and stable** → RQ2 warrants the real held-out-adapter test (case C).

[RESULT PENDING — A_tan job running.]
