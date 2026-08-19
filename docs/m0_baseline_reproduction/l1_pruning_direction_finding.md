# FINDING: the published L1 baseline keeps the LOWEST-magnitude filters

**Date:** 2026-08-19 (autonomous night run, surfaced while validating the P0 machinery).
**Severity:** high — it concerns what the project's published L1/P0 baseline actually is
(bears on RQ2 and RQ3). **Status:** rigorously verified as a fact about the artifact;
the *interpretation* (intentional vs. a direction bug in the reference) is a question for
Gabriel/Arshdeep and should be confirmed with `/auditar`. **No project gate is changed by
this document.**

Reproduce: `.venv/bin/python scripts/research/verify_l1_direction.py` (exit 0 = confirmed).

## The fact

The published pruned checkpoint `l1_audioldm-m-full_p1.ckpt` keeps, in every pruned
layer, the `k` convolution filters of **lowest** output-channel L1 magnitude and removes
the highest. Standard L1 magnitude pruning does the opposite (keeps the highest-magnitude,
most-important filters). So the published "L1-pruned" model is, in effect, a
**keep-least-important** model.

## Four independent, mutually consistent lines of evidence

1. **Rank correlation.** Computing P0 = per-output-channel conv-weight L1 on the real base
   weights (`audioldm-m-full.ckpt`) and sorting descending gives, on **every one of the 28**
   ranked layers, **Spearman = -1.000000** against the published `sorted_indexes_dict.pkl`
   ranking — an exact reversal. The published ranking is ascending in L1 (lowest first).

2. **Raw magnitudes.** The channels the published ranking lists first have low L1
   (e.g. 44.7, 46.6, 47.0) and those it lists last have high L1 (79.9, 86.8, …).

3. **Reference source code** (`_external/PruningAudioLDM/scripts/layerwise_sorted_index_generation.py`,
   ground truth):
   * `l1_imp_index(weights)` returns `scores[i] = sum(|weights[i]|)` per output filter —
     **identical to this project's P0** (up to a max-normalization that does not change order).
   * `sorted_idx = np.argsort(scores)` sorts **ascending** (no `[::-1]`, no negation).
   * The frozen materializer `random_masks.prune_with_indices` keeps `out_idx_full[:out_k]`
     and is **bit-exact to the published checkpoint** (ledger M3-002, 690/690).
   * Ascending argsort **+** keep-first-`k` ⇒ keep the `k` lowest-magnitude filters.

4. **Kept-vs-pruned magnitude.** On all **15/15** actually-pruned layers (`k=192` of 384/576/960),
   the kept set `ranking[:k]` has **lower** mean L1 than the pruned set `ranking[k:]`
   (e.g. `input_blocks.10.0.in_layers.2`: kept 65.3 vs pruned 77.5).

Because the materializer is bit-exact to the published artifact, this is a property of the
**checkpoint itself**, not merely of the reference code — independent of any convention
assumption on our side.

## Why it matters

* **RQ2 (pruning criterion).** The project compares P1/P2/P3 (Taylor) against P0 (L1). If
  the published reference's "L1" keeps the least-important filters, then either (a) the
  project's P0 baseline should follow standard L1 (highest-kept — what `research_pruning.taylor.
  p0_l1_magnitude` implements) and will therefore **differ from the published checkpoint**, or
  (b) to match the published artifact the project must knowingly reproduce an inverted L1.
  These are not the same baseline; the comparison's meaning depends on which is used.
* **RQ3 (recovery).** The published pre-recovery checkpoint (and any recovery built on it)
  starts from a keep-least-important pruning. A model that keeps the lowest-magnitude filters
  is expected to be badly damaged pre-recovery; recovery numbers reported against it describe
  recovery from that inverted starting point, not from a standard L1 prune.

## What is NOT claimed

* This is **not** a claim that the project's own code is wrong: `p0_l1_magnitude` implements
  standard L1 (highest-kept) correctly and is control-tested (M3B-000, C6) and validated here.
* Whether the reference's ascending-argsort-then-keep-`[:k]` is a deliberate design choice or
  an off-by-direction bug is **not** decided here — that is for Gabriel to raise with Arshdeep
  and for `/auditar` to review. Do not change any gate or baseline definition on the strength
  of this document alone.

## Suggested next steps (for Gabriel / an audit session)

1. `/auditar` this finding (re-run the script, re-read the reference source and the M3-002
   materializer independently).
2. Ask Arshdeep whether the published L1 checkpoint intentionally keeps the lowest-magnitude
   filters, or whether the intended baseline keeps the highest.
3. Decide the project's P0 convention: standard L1 (highest-kept) for a correct baseline, vs.
   the published inverted convention for artifact parity — and record the decision in the
   experiment ledger before M3B/M4.
