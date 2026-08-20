# Modality-Swap-Aware Structured Pruning for AudioLDM

**Does pruning a text-to-audio model damage it differently depending on whether you
speak to it in audio or in text — and can we prune it better by listening to both?**

This repository is the full, reproducible record of that investigation: code, protocol,
provenance, and an honest log of everything that has and has not been established.

> **Status in one line:** the machinery is built and tested end to end, the compute is
> measured on real hardware, and **the three research questions have deliberately not been
> answered yet** — the analysis protocol must be frozen before any result is inspected.

---

## The idea, in plain terms

[AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning) generates audio from a
text prompt. Inside, there is a quirk that this project is built around:

* During **training**, the model is conditioned on a CLAP embedding of the **audio**.
* At **generation time**, it is conditioned on a CLAP embedding of the **text prompt**.

Both embeddings enter through exactly the same door — we verified this directly: each is a
`[B, 1, 512]` vector feeding the same FiLM interface. So the model lives with a permanent
**modality swap** between how it learned and how it is used.

Now suppose we want to make the model smaller by **pruning** — removing whole convolutional
filters. The standard way to decide what to remove looks at weight magnitude, or at
gradients from a single conditioning signal. The question this project asks is whether that
is leaving something on the table:

> If you decide what to cut while looking only at the text pathway, do you accidentally
> destroy what the audio pathway was holding up — and would looking at **both** signals
> cut better?

An analogy: pruning a bilingual dictionary by reading only the Spanish pages. You might
delete precisely the entries that the English side depended on, and never notice until
someone asks a question in English.

## The three questions

| | Question | How it is answered |
|---|---|---|
| **RQ1** | When we prune, is the damage **modality-dependent** — worse for audio conditioning than for text, or vice versa — beyond what random pruning of the same size would do? | Paired diagnostics `D_gen` / `D_mod` / `R_mod` against a matched null of 20 random masks |
| **RQ2** | At a **matched** architecture, structural budget and gradient budget, does **paired audio+text** saliency preserve text-conditioned generation better than magnitude pruning or a faithful text-only Taylor criterion? | Criteria P0–P3, then generation + FAD/KL/PANNs evaluation |
| **RQ3** | After pruning, how much damage can a **cheap** fix recover — training only adapters, biases and normalisation instead of the whole model? | Parameter-efficient recovery, reported against a full-finetuning reference |

The comparison is deliberately unfair to us: **RQ1's null controls for generic damage.**
Random pruning usually breaks a model *more* overall, so we do not simply compare against
random masks. We fit `R_mod ~ f(D_gen)` across the random controls, read off the expected
random asymmetry *at the same level of generic damage the real pruning caused*, and take
the residual. A modality effect only counts if it survives that.

**On naming:** LoRA is not the novelty here. When biases and GroupNorm affine parameters
are also trainable, we call the mechanism **parameter-efficient recovery**, and report
LoRA, bias, GroupNorm and total trainable parameters separately.

---

## Where the project actually is

Being precise about this matters more than looking finished.

### Built and verified

| | Evidence |
|---|---|
| Environment reproducible from the frozen dependency lock (155 packages, no pin relaxed) | [`docs/environment_report.md`](docs/environment_report.md) |
| Both architectures rebuilt from config and strict-loaded from the public checkpoints — base 415.955 M params, pruned 145.674 M (−65.0 %) | `scripts/research/smoke_load_unet.py` |
| Audio and text conditioning paths proven to reach the same FiLM interface | [`docs/condition_swap_validation.md`](docs/condition_swap_validation.md) |
| Diagnostics, random-mask null and the Gate A / Gate B statistics | `research_pruning/diagnostics/`, `paired_modality/` |
| P0–P3 saliency criteria on the verified 28-layer prunable set | `research_pruning/taylor/` |
| Parameter-efficient recovery: 284 modules wrapped, exact merge/unmerge, exact training resume | `audioldm_peft/` |
| **13 test modules passing** | `.venv/bin/python scripts/research/run_research_tests.py --all` |
| Compute measured on a real T4 | [`docs/compute_budget.md`](docs/compute_budget.md) |

### Not done — and not claimed

* **No result exists for RQ1, RQ2 or RQ3.** No saliency has been computed on the real
  model. This is deliberate: [`docs/pilot_protocol.md`](docs/pilot_protocol.md) is a
  pre-registration and must be frozen and committed *before* any result is inspected.
* The generation stack is not yet benchmarked, so the evaluation budget is not fixed.
* The recovery arm (RQ3) is not funded at the currently projected compute.

---

## What the reproduction turned up

Before running anything new, we reproduced the published pruning baseline exactly. Two
things came out of that, both recorded with full evidence.

**We reproduce the released pruned checkpoint bit-exactly — 690/690 tensors.** Starting
from the base AudioLDM-M-Full weights plus the published ranking file, our materializer
reconstructs [`PruningAudioLDM`](https://github.com/Arshdeep-Singh-Boparai/PruningAudioLDM)'s
released `(1,2,3,1)` checkpoint tensor for tensor. We also confirmed it is a
**pre-recovery** artifact — all 2061 same-shape tensors are bit-identical to the base
model, so it is pure prune-and-merge output — and that the base checkpoint carries the same
md5 in the official AudioLDM Zenodo record as in the pruning record, so the provenance
chain is clean end to end.

**An open question about the pruning direction.** The released artifact appears to keep,
per pruned layer, the filters of *lowest* L1 magnitude — the reverse of the conventional
magnitude rule. We observe Spearman = −1.000000 between the released ranking and a
conventional L1 ranking across all 28 ranked layers, and on the 15 actually-pruned layers
the kept set has lower mean L1 than the removed set. **We are not claiming this is an
error.** It may be deliberate or reflect a convention we have not understood; the question
is with the original authors. Write-up and one-command reproduction:
[`l1_pruning_direction_finding.md`](docs/m0_baseline_reproduction/l1_pruning_direction_finding.md),
`scripts/research/verify_l1_direction.py`.

Because our RQ2 baseline *is* that published artifact, we reproduce its convention exactly,
**and** report conventional keep-highest-L1 alongside it — so that criterion *direction*
and criterion *quality* never get conflated in a result.

---

## Reproducing any of it

Every claim above has a command. Nothing needs to be taken on trust.

```bash
# full research test suite (CPU)
.venv/bin/python scripts/research/run_research_tests.py --all

# the pruning-direction finding
.venv/bin/python scripts/research/verify_l1_direction.py

# bit-exact reconstruction of the released pruned checkpoint
.venv/bin/python scripts/research/verify_l1_bitexact.py

# audio/text conditioning paths
.venv/bin/python tests/research/test_conditioning_paths.py

# review every patch we make to upstream code
git diff upstream-frozen -- audioldm_train/
```

That last command matters. Upstream AudioLDM is merged here unmodified except for **one
deliberate, reviewed patch** (1 file, 16 insertions, 2 deletions): gradient checkpointing
differentiates with respect to frozen parameters, which is incompatible with
parameter-efficient recovery. The patch and its justification are recorded in
[`docs/experiment_ledger.md`](docs/experiment_ledger.md) under `DECISION-F10`.

## Frozen references

| Reference | Commit | Preserved as |
|---|---|---|
| [`haoheliu/AudioLDM-training-finetuning`](https://github.com/haoheliu/AudioLDM-training-finetuning) | `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc` | branch `upstream-frozen`, merged into `main` |
| [`Arshdeep-Singh-Boparai/PruningAudioLDM`](https://github.com/Arshdeep-Singh-Boparai/PruningAudioLDM) | `6f65f628fabc4ad27770753698fc81944e820f9f` | branch `pruning-reference-frozen` |

This work builds directly on both. The pruning baseline, the published `(1,2,3,1)`
checkpoint and the layer ranking are Arshdeep Singh's
([Zenodo 10.5281/zenodo.21376822](https://doi.org/10.5281/zenodo.21376822)); AudioLDM is
Haohe Liu's.

## Layout

```text
audioldm_train/       upstream AudioLDM, one reviewed patch (see DECISION-F10)
audioldm_peft/        parameter-efficient recovery: LoRA, injector, EMA, resume
research_pruning/     diagnostics/ · taylor/ · paired_modality/
scripts/research/     reproducible entrypoints and verification scripts
tests/research/       CPU test suite (13 modules)
docs/                 master plan, pilot protocol, ledger, claims matrix, provenance
data/ artifacts/ _external/                        [gitignored]
```

The documents that carry the project's state:

* [`docs/master_plan_v3.md`](docs/master_plan_v3.md) — the scientific execution contract
* [`docs/pilot_protocol.md`](docs/pilot_protocol.md) — pre-registration (**not yet frozen**)
* [`docs/experiment_ledger.md`](docs/experiment_ledger.md) — every experiment, gate and
  decision, including the failures
* [`docs/claims_matrix.md`](docs/claims_matrix.md) — what may and may not be written in a paper
* [`PROGRESS.md`](PROGRESS.md) — living state
* [`docs/HANDOFF.md`](docs/HANDOFF.md) — the resume point for a new session

## Licensing

Upstream code is MIT. Pretrained AudioLDM checkpoints are **CC-BY-NC-4.0 (no commercial
use)** per the upstream README and are **not** redistributed here — the fetch script
downloads them from their original records and verifies md5. Upstream usage instructions
are preserved verbatim in [`UPSTREAM_README.md`](UPSTREAM_README.md).
