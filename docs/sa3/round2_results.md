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
real (erratum E6); **D_P stably identifies the least field-sensitive interior blocks (concentrated around 9–14)**, stable at N=32 — **end-to-end removability remains unresolved** (a field ranking, not a removability claim).

## The A_tan decision gate (job `sa3-atan-1`, RUNNING)

The remaining question is the RQ2 one, and it is cheap: **does `A_tan` carry structural information
distinct from `D_P`?** Primary probe family only (U_gen = standard LoRA r16), n_u=16, κ=0.01
(tangent regime confirmed: ‖δF(2u)‖/‖δF(u)‖ ≈ 2.0), N=16 reusing the persisted states.

`size_atan_from_stats.py` computes, with a bootstrap floor, δ(R_{A_tan}, R_{D_P})(k):
* if **not real** (A_tan's removable ranking ≈ D_P's) → the adapter-compatible-pruning hypothesis
  is not supported; **close the line without building CC0 datasets or training LoRAs**;
* if **real and stable** → RQ2 warrants the real held-out-adapter test (case C).

## A_tan gate RESULT (job `sa3-atan-1`, N=16, n_u=16, 0.459 cr)

* **n_u sizing: `n_u_main = 8`** (probe-bootstrap disagreement 0 at rung 8) — n_u=8 suffices.
* **Linearity:** median ‖δF(2u)‖/‖δF(u)‖ = 1.98 (in range), but NOT all probes ∈ [1.9,2.1] at κ=0.01
  — a smaller κ (or per-probe screen) is advisable for a clean tangent regime; flagged.
* **A_tan removable sets vs D_P:** k=2 {13,14} vs {12,13}; k=4 {12,13,14,15} vs {11,12,13,14};
  k=6 {11–16} vs {9–14}. Divergence δ = 1 / 1 / 2 blocks (A_tan leans to higher-index interiors).
* **Gate verdict — UNDERPOWERED at N=16 (honest).** The initial tool reported "A_tan diverges", but
  that used a bootstrap-**with-replacement** floor (too loose, floor→0). With the **disjoint-pair**
  floor and the fact that **D_P removal sets are not stable below N_main=32** (sizing above), the
  observed 1–2 block divergence is **comparable to D_P's own sampling noise and CANNOT be declared
  real**. δ does not exceed even the looser size-N/2 disjoint floor at k=2,4.

  **⇒ The cheap gate did not resolve at N=16.** The question "does A_tan carry structure distinct
  from D_P?" needs A_tan at **N ≥ 32** (where D_P is stable) with a same-size disjoint floor —
  reusing the already-persisted states (n_u=8 now frozen). Estimated ~0.9 cr (N=16 A_tan was 0.459 cr;
  linear in N). Accumulated so far 1.68 cr ⇒ a N=32 A_tan lands ~2.6 cr, within the ≤2–3 phase target.

## Corrected bottom line (round 2)

* **RQ1:** amplification real (E6); **D_P stably identifies the least field-sensitive interiors (around 9–14), stable at N=32; end-to-end removability unresolved**;
  **I_PT redundant with D_P and noisier → PT-specific criterion closed.**
* **RQ2/A_tan:** machinery validated (batched, verified, linearity ≈ 2.0, n_u=8); **the A_tan-vs-D_P
  gate is UNDERPOWERED at N=16 — not resolved.** Next single step: A_tan at N=32 to decide whether
  the residual RQ2 contribution lives (stable distinct A_tan structure → real held-out adapters) or
  dies (A_tan ≈ D_P → close before building CC0 datasets / training LoRAs).
* No CASE E demonstrated; no main-panel work; N_main(D_P)=32 / n_u=8 are the frozen sizes.

