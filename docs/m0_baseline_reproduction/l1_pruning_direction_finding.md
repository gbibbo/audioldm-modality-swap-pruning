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
  the raw per-channel L1 correctly and is control-tested (M3B-000, C6) and validated here. The
  *direction* in which those magnitudes are used for keep/prune is the P0 convention, decided
  below.
* Whether the reference's ascending-argsort-then-keep-`[:k]` is a deliberate design choice or
  an off-by-direction bug in Arshdeep's code is still not asserted; it does not change the
  decision, because the decision is anchored to *what the published artifact is*, not to why.

## DECISION — 2026-08-19 (Gabriel): adopt the published inverted convention for P0

**Rule Gabriel set:** use the published inverted convention **iff** the published pruning work
is Arshdeep's; otherwise use standard L1.

**It is Arshdeep's.** `_external/PruningAudioLDM/README.md` states *"Official implementation of
our pruning framework for compressing AudioLDM-M-Full"* (`Arshdeep-Singh-Boparai/PruningAudioLDM`,
arXiv 2607.13330). The checkpoint `l1_audioldm-m-full_p1.ckpt` and `sorted_indexes_dict.pkl` are
from **Zenodo record 21376822 (Arshdeep Singh, 2026-07-15)**, md5-verified (`dataset_manifest.md`),
and the finding is derived from Arshdeep's own scripts. So RQ2's L1 baseline **is** that official
artifact.

**Therefore the project's P0 adopts the "published" convention: keep the LOWEST-L1 filters.**
Implemented as `research_pruning.taylor.p0_importance(convs, convention="published")` = `-L1`, so
`keep_topk` keeps the low-L1 channels; `P0_CONVENTION = "published"` is the default. `"standard"`
(keep highest-L1, Li et al. 2017) is retained only for a hypothetical non-Arshdeep baseline and
must not be used for RQ2. `p0_l1_magnitude` (raw magnitudes) is unchanged.

**Verified:** `scripts/research/verify_p0_convention.py` — on the real base `(1,2,3,5)` U-Net,
`keep_topk(p0_importance("published"), k)` reproduces the published kept-set **exactly on 12/12
ranking-driven pruned layers**, and `"standard"` is disjoint from it. Control-model coverage:
`tests/research/test_taylor_saliency.py::C8`. Recorded in the ledger (DECISION-M3B-002).

## Reproduce

1. `.venv/bin/python scripts/research/verify_l1_direction.py` (the finding: Spearman −1, 15/15).
2. `.venv/bin/python scripts/research/verify_p0_convention.py` (the adopted convention reproduces
   the published kept-set 12/12).

Still open for Arshdeep (informational, does not gate the project): confirm whether the inverted
direction was intentional, so the paper can describe the baseline accurately.
