# Draft update to Arshdeep — status, findings, and two questions

**Status: DRAFT for Gabriel to review, edit and send. Not sent.**
Written 2026-08-19. Every number below is traceable to a commit and an artifact; the
reproduction commands are given so nothing has to be taken on trust.

---

## Suggested message

Hi Arshdeep,

Quick update on the modality-swap pruning project, plus two things I'd like your read on —
one of which concerns the published PruningAudioLDM artifacts directly.

**Where the project is.** The infrastructure is complete and reproducible: environment
pinned to your frozen dependency set, both architectures rebuilt and strict-loaded from the
public checkpoints, AudioCaps validated, and a research test suite of 13 modules passing on
CPU. The scientific runs have **not** started — the pilot protocol is pre-registered and
must be frozen before any saliency result is inspected, so I've deliberately kept the
diagnostics away from the pruned checkpoint until then.

**We reproduce your pruning pipeline bit-exactly.** Starting from `audioldm-m-full.ckpt`
and `sorted_indexes_dict.pkl`, our materializer reconstructs
`l1_audioldm-m-full_p1.ckpt` at **690/690 tensors bit-identical**. Along the way I also
confirmed the checkpoint is **pre-recovery**: all 2061 same-shape tensors are bit-identical
to the base model, so it is pure prune-and-merge output, never finetuned. And the base
checkpoint has the same md5 in the official AudioLDM record (Zenodo 7884686) as in yours
(21376822), so the provenance chain is clean end to end.

**Question 1 — the pruning direction.** This is the one I'd most like your view on. The
published L1 checkpoint appears to keep, per pruned layer, the `k` conv filters of
**lowest** output-channel L1 magnitude, i.e. inverted relative to the usual L1 rule of
Li et al. (2017), which keeps the highest. I verified this four independent ways:

1. our per-filter L1 ranking vs the published ranking gives **Spearman = −1.000000 on all
   28 ranked layers** — an exact reversal, not a partial correlation;
2. the published ranking lists low-L1 filters first and high-L1 last;
3. reading your own code: `layerwise_sorted_index_generation.py` computes
   `l1_imp_index` as the per-filter `sum(|w|)` and then `sorted_idx = np.argsort(scores)`,
   which is **ascending**, while `pruned_unet_dict_creation.py:118` keeps
   `out_idx_full[:out_k]` — the first `k`, i.e. the lowest-L1 filters;
4. on the **15/15** layers that are actually pruned, the kept set has a lower mean L1 than
   the removed set.

Because our materializer is bit-exact to the artifact, this is a property of the published
checkpoint itself rather than of our reading of it. I'm **not** asserting it's a bug — it
may be deliberate, or a convention I'm not seeing. But it changes how the baseline should
be described in writing, so I'd rather ask than assume. Reproduce with:
`scripts/research/verify_l1_direction.py` (exits 0 on CONFIRMED).

For our own work we've decided to **adopt your published convention** for the L1 baseline,
since the baseline we compare against *is* your released artifact, and we verified that
reproduces your kept-set exactly on 12/12 ranking-driven layers. We'll additionally report
standard keep-highest-L1 as a secondary reference, and word every comparison as "vs the
published L1 pruning artifact" rather than "vs standard L1 magnitude pruning", so the
direction and the criterion quality don't get conflated.

**Question 2 — four tensors where the public script and the released checkpoint differ.**
Running your public reconstruction script reproduces **686/690** tensors; four need
different conventions to match the release, and the artifact looks internally inconsistent
at those seams:

* `output_blocks.0.0.in_layers.2` and `output_blocks.1.0.in_layers.2` take their output
  channels **positionally** (first 192) while the consumer downstream selects by ranking;
* `output_blocks.2.0.in_layers.2` has a **positional weight but a ranked bias**, so the
  bias values attach to different channels than the weight rows;
* `input_blocks.10.0.in_layers.2` keeps its input columns in identity order, where the
  reference reorders them by the `input_blocks.9.0.op` ranking.

We reproduce all four exactly as released (we're diagnosing that specific checkpoint), but
if any of these were unintended it would be worth knowing before either of us writes about
the pruned model.

**One request.** The recovered, fully finetuned `(1,2,3,1)` checkpoint doesn't appear to be
public — I checked GitHub releases on both repos and Zenodo 21376822, which holds only
pre-recovery artifacts. Our RQ3 (how much a fixed parameter-efficient recovery restores)
would be much stronger with it as the full-finetuning reference. Without it we can only
report a published-reference comparison and can't claim any exact percentage-of-full-FT
recovery. If you can share it, that unblocks the strongest version of that comparison.

**What's coming that may be useful to you.** The diagnostics we've built are
modality-specific: the audio and text CLAP paths both enter the same FiLM interface, so we
can measure per-example how much pruning damage is modality-dependent, against a matched
random-pruning null of 20 masks. If that signal is real, it says something about *what*
pruning destroys in a text-to-audio model, not just how much. Happy to share the code and
the pre-registered protocol before we run it, if you'd like to look at the design first.

Best,
Gabriel

---

## Notes for Gabriel (not part of the message)

**Why question 1 is worth raising carefully.** If the direction is unintended, the
published pruned model is likely worse than a correctly-directed L1 prune would be. That
cuts two ways for us: it makes our P1/P2/P3 comparison look good for a reason that has
nothing to do with the modality-swap hypothesis, which is exactly why we pre-registered
reporting both conventions. Raising it *before* we publish anything protects both sides.

**What we can and cannot claim today.**

| Claim | Status |
|---|---|
| Bit-exact reproduction of the published pruned checkpoint (690/690) | **Verified** |
| Published checkpoint is pre-recovery (2061/2061 tensors identical to base) | **Verified** |
| Published L1 keeps lowest-magnitude filters | **Verified 4 ways** |
| Public script reproduces only 686/690 tensors | **Verified** |
| RQ1 — pruning damage is modality-dependent | **No result. Not started.** |
| RQ2 — paired saliency beats L1 / text-only Taylor | **No result. Not started.** |
| RQ3 — recovery restores residual damage | **No result. Blocked on compute + checkpoint** |

Do not let the message imply any RQ1/RQ2/RQ3 finding. There is none yet, by design: the
protocol has to be frozen first.

**Evidence index.**

* Direction finding + reproduction: `docs/m0_baseline_reproduction/l1_pruning_direction_finding.md`,
  `scripts/research/verify_l1_direction.py`, `scripts/research/verify_p0_convention.py`
* Bit-exactness: `scripts/research/verify_l1_bitexact.py` → `artifacts/m3_pilot/l1_bitexact_check.json`
* Seam conventions: `research_pruning/diagnostics/random_masks.py` docstring; ledger `M3-002`, `AUDIT-M3-001`
* Pre-recovery proof: `artifacts/m0_baseline_reproduction/prerecovery_check.log`
* Provenance/md5: `docs/m0_baseline_reproduction/dataset_manifest.md`
* Conditioning paths: `docs/condition_swap_validation.md`
* Compute: `docs/compute_budget.md`
