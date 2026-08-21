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

## A_tan gate RESOLVED at N=32 (job `sa3-atan-n32-1`, 0.482 cr; κ=0.01 frozen)

κ micro-calibrated to **0.01** first (CPU, `kappa_calibration.py`: all 8 probes' mean linearity ∈
[1.956,2.012], precision 7.79e-4 ≥ floor; smaller κ fail precision) — so this is NOT a non-linearity
artifact. N=32, n_u=8, linearity now fully in range (median 1.98). **n_u_main=4** (very stable).

| k | R_Atan (removable) | R_DP | δ | robust? |
|---|---|---|---|---|
| 2 | {13,14} | {12,13} | 1 | floor-sensitive (within size-16 floor) |
| 4 | {12,13,14,15} | {11,12,13,14} | 1 | marginal (floor 0 on the 32 gate prompts, but 1 on the fuller 64-prompt pool) |
| 6 | {11,12,13,14,15,16} | {9,10,11,12,13,14} | 2 | **ROBUST — exceeds the floor on both pools** |

**Key evidence = cross-N stability, not just the floor.** A_tan's removable sets are **IDENTICAL at
N=16 and N=32** (k=2/4/6), and D_P is stable at N=32 (N_main analysis). Two stable rankings that
differ by 2 blocks at k=6 ⇒ the divergence is **real**, not sampling noise. Spearman(A_tan,D_P)=0.86
overall (both rank the boundaries critical) — but the DECISION-RELEVANT removable tail diverges:
**A_tan systematically prefers the higher-index interiors (13–16); D_P prefers the lower ones (9–14).**

**⇒ RQ2 SURVIVES the cheap gate.** This is the anticipated positive outcome: adaptability
(`A_tan`) occupies structural resources **distinct** from standalone field sensitivity (`D_P`) — the
adaptability-aware removable set (13–16) differs stably from the deployment one (9–14), strongest and
robust at k=6. Caveat: a fully same-scale disjoint floor for A_tan would need N=64 A_tan prompts
(Gabriel: do NOT auto-escalate); the cross-N stability + the N=32 D_P stability carry the claim.

**Next step (Gabriel's call — the FIRST expensive step): real held-out adapters (case C).** Train a
few standard `lora` r16 adapters on distinct SFX domains (CC0 44.1 kHz data), and test whether the
A_tan-vs-D_P divergence PREDICTS which blocks a real adapter can survive. Only that closes RQ2 as a
genuine contribution; if the divergence does not predict real-adapter survival, the line still dies —
but now for a scientifically interesting reason, not underpowering.

## Corrected bottom line (round 2)


* **RQ1:** amplification real (E6); **D_P stably identifies the least field-sensitive interiors (around 9–14), stable at N=32; end-to-end removability unresolved**;
  **I_PT redundant with D_P and noisier → PT-specific criterion closed.**
* **RQ2/A_tan:** RESOLVED at N=32 (κ=0.01 frozen) — **A_tan carries structure distinct from D_P**
  (stable removable-tail divergence 9–14 vs 13–16, robust at k=6; identical N=16→N=32). **RQ2 survives.**
  Next (Gabriel's call, first expensive step): real held-out adapters (case C) to test whether the
  divergence predicts real-adapter survival.
* No CASE E demonstrated; no main-panel work; N_main(D_P)=32 / n_u=8 are the frozen sizes.

