# Hostile review — round 4 (2026-08-20, Reviewer B reply)

Output: `docs/master_plan_v4_draft.md` **rc3** (identification layer). Both reviewers vote
DECISION-V4-00 (adopt v4). rc3 is the version proposed for adoption.

## A's seven modifications — verdict and feasibility (checked against the repo)

| # | Modification | Verdict | Feasibility evidence (this session, CPU) |
|---|---|---|---|
| 1 | H-guidance as an event-specific counterfactual `ε(c_full) − ε(c_without_e)` | **Accepted; the decisive fix.** rc2's whole-caption `‖ε_c−ε_∅‖` credited every event in a multi-event caption with the same number. | Text path takes arbitrary strings (`conditioning.py`, `modality="text"` → `list[str]`). Under the strict alias map, **22 697** train clips have exactly one requested event (→ `c_without_e` ≡ ∅), 5 805 have two. Added a second contrast `c_only_e` and the relative form `ΔG/G_F`. Cost negligible (forward-only). |
| 2 | Event-specific acoustic descriptors | **Accepted, with a concrete fallback.** | FineLAP mask if its smoke passes; else restrict H-acoustic to the **single-AudioSet-label subset: 9 637 clips (19.5 %)**, where whole-clip descriptors are event-specific by construction. Never whole-clip descriptors on multi-event clips. |
| 3 | `K_rand = 20` minimum + sentinel panel + power simulation | **Accepted.** Exact test: `p_min = 1/(K+1)`; K=10 cannot reach 0.05, K=20 gives 0.048. Sentinel panel 20 events × 15 prompts stratified by exposure × family; Gate M keeps the broad 50 × 20 set. Power simulation (CPU, from Tier-0 rates, ≥ 80 % at a pre-set MDE) is now milestone M4-1b and a hard prerequisite for Tier 1. | RAND×20 × 300 = 6 000 clips ≈ 14 GPU-h — the single largest Tier-1 item. |
| 4 | P1-placebo | **Accepted as mandatory.** Without it the result is Importance-Aware OBS §4.4 transplanted to audio. | Free at Gate B′ (per-slot saliency recomposition); +500 holdout clips if it reaches generation. |
| 5 | Disjoint mechanism set / intervention holdout | **Accepted.** Partition frozen at source-wav level, also disjoint from both calibration pools; holdout unblinded only after the criterion is frozen. | Event supply is sufficient: **61 events have ≥ 200 strictly requested captions**, 81 have ≥ 100. |
| 6 | Gate I margins, mechanism-general target set, zero/multiple-mechanism rules | **Accepted.** Non-inferiority on non-target events; proposed `δ_target = +5 pp`, `δ_harm = 2 pp`, FAD/FD +5 % rel., KL +0.05 — to be fixed in DECISION-V4-07 before unblinding. | — |
| 7 | Split audio exposure vs calibration-caption exposure | **Accepted; it is also an identification device.** P0 has no calibration: a dependence on caption exposure that appears only in P1 isolates the calibration-sampling mechanism. | Strict map under-counts **"Speech": 1 882 requested vs 20 561 labelled** (captions say *talks/talking*); the expanded map must cover it or the family is excluded from the tail block. |

## Cost consequence (honest)

Tier 1 rises from ≈30 to **≈45 credits** (1a ≈33 incl. RAND×20; 1b ≈3; 1c ≈5; smokes/builds).
Tier 0 unchanged (≈2.7). Tier 0 is screening only: it informs the Tier-1 decision and the
power simulation; it cannot yield the paper, and a positive Tier 0 must not be stretched
into an underpowered 09-16 submission (v3's rule, retained).

## Where I add one thing A did not ask for

The counterfactual deletion of an alias span can leave an ungrammatical caption; CLAP's
text encoder is not guaranteed to treat "a dog barks while a sounds" as "a dog barks".
Hence the second contrast `G_only = ‖ε(c_only_e) − ε(∅)‖` (alias phrase alone), and the
pre-registered rule that the deletion form is primary only on single-requested-event
captions (where it is exactly ∅) and the phrase-only form is primary on multi-event
captions. Both frozen before any pruned model is evaluated.

## Status

No GPU spent; no gate changed. v3 remains the contract until DECISION-V4-00 is recorded.
Eight decisions (V4-00..07) are listed in the draft. Pre-Tier-0 CPU work (FAD/FD fix,
manifests, partition, FineLAP smoke, materializer parameterization) can start the moment
V4-00 is signed.
