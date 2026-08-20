# Draft update to Arshdeep — status, findings, and two questions

**Status: DRAFT for Gabriel to review, edit and send. Not sent.**
Written 2026-08-19. Every number below is traceable to a commit and an artifact; the
reproduction commands are given so nothing has to be taken on trust.

---

## Suggested message (short form — preferred)

An external review judged the first draft too long for an update. This shorter version
says the same load-bearing things. **Two accuracy fixes were applied to the reviewer's
wording** — see "Wording traps" below; do not reintroduce them.

> Hi Arshdeep,
>
> Quick update. We now reproduce the released `(1,2,3,1)` pruning artifact **bit-exactly
> (690/690 tensors)** from the base checkpoint plus `sorted_indexes_dict.pkl`, and our
> modality-swap diagnostics, P0-P3 saliency code and parameter-efficient recovery are all
> implemented and tested. The recovery path is confirmed working **on GPU** (284/284 LoRA
> adapters receiving gradients, no frozen weight updated); the diagnostics and saliency
> criteria are validated on CPU and have not been run on GPU yet.
>
> First T4 profiling is encouraging. PEFT recovery training peaks at only **~4.2 GB at
> batch 8**, and a full-model forward+backward — our cost proxy for a saliency gradient
> evaluation — takes **~1.6 s per batch of 8**. So the modality-aware pruning experiment
> itself looks very affordable. The expensive part is recovery, currently projected at
> **~46 GPU-hours per model** at the 100k-step budget; we still need to benchmark
> generation before fixing the full evaluation budget. We have deliberately **not** run the
> RQ1/RQ2 comparisons yet, because we want to freeze the protocol first.
>
> During the reproduction we found one thing I'd like to check with you. The released L1
> artifact appears to keep the **lowest-L1 filters** in the pruned layers, i.e. the reverse
> of the conventional magnitude rule. We get **Spearman −1.0** between the released ranking
> and the conventional L1 ranking across **all 28 ranked layers**, on the 15 actually-pruned
> layers the kept set has lower mean L1 than the removed set, and the public code looks
> consistent with that (`np.argsort` ascending, then `[:k]`). Is that intentional?
>
> We also found **four tensors** where the public reconstruction script gives a different
> result from the released checkpoint (the script reproduces 686/690), although we can
> reproduce the released checkpoint exactly. Happy to send the details and the reproduction
> script.
>
> Two smaller things that may be useful: the released pruned checkpoint is **pre-recovery**
> — all 2061 same-shape tensors are bit-identical to the base model — and the base
> checkpoint has the **same md5** in the official AudioLDM record (Zenodo 7884686) as in
> yours, so the provenance chain is clean.
>
> Finally, if you have the fully finetuned `(1,2,3,1)` checkpoint, could you share it? It
> would give us an apples-to-apples full-FT reference for the recovery experiment.
>
> Best,
> Gabriel

### Wording traps (fixed above — do not undo)

1. **Do not write "the diagnostics and saliency code are running on GPU."** Only the PEFT
   recovery path has run on CUDA. The `D_gen`/`D_mod`/`R_mod` diagnostics and the P0-P3
   channel-gate Taylor criteria are CPU control-tested only.
2. **The ~1.6 s figure is a cost PROXY, not the saliency path.** `time_saliency` in the
   benchmark runs a generic full-model forward+backward with every parameter requiring
   grad; it does **not** exercise `research_pruning/taylor` channel gates. Describe it as a
   proxy, at batch 8 on a T4, or the number will be read as something we have not measured.

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
