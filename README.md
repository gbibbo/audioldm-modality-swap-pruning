# Modality-Swap-Aware Structured Pruning + Parameter-Efficient Recovery for AudioLDM

Research repository. The scientific execution contract is
[`docs/master_plan_v3.md`](docs/master_plan_v3.md); agent rules are in
[`AGENTS.md`](AGENTS.md); live state is [`PROGRESS.md`](PROGRESS.md).

## Research questions

1. **RQ1** — Does structured magnitude pruning introduce *modality-dependent*
   degradation in AudioLDM beyond generic pruning damage at comparable denoiser
   degradation?
2. **RQ2** — At matched architecture, structural budget and gradient-evaluation
   budget, does paired audio–text saliency preserve text-conditioned generation
   better than L1 and than a faithful text-only Taylor baseline?
3. **RQ3** — How much residual degradation can a *fixed* parameter-efficient
   recovery configuration restore, at what compute/memory cost, and which
   semantic event families stay vulnerable?

LoRA is not the novelty. When biases and GroupNorm affine parameters are also
trainable the mechanism is called **parameter-efficient recovery**, and LoRA,
bias, GroupNorm and total trainable parameters are reported separately.

## Frozen references

| Reference | Commit | Preserved as |
|---|---|---|
| `haoheliu/AudioLDM-training-finetuning` | `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc` | branch `upstream-frozen`, merged into `main` |
| `Arshdeep-Singh-Boparai/PruningAudioLDM` | `6f65f628fabc4ad27770753698fc81944e820f9f` | branch `pruning-reference-frozen` |

Review our patches to upstream code with:

```bash
git diff upstream-frozen -- audioldm_train/
```

Details and verification commands: [`docs/m0_baseline_reproduction/frozen_references.md`](docs/m0_baseline_reproduction/frozen_references.md).

## Layout

```text
audioldm_train/       upstream code, minimal surgical patches (currently: none)
audioldm_peft/        parameter-efficient recovery      [skeleton, no implementation]
research_pruning/     diagnostics/ taylor/ paired_modality/  [skeleton, no implementation]
configs/research/     resolved research configs         [empty]
scripts/research/     reproducible entrypoints
tests/research/       CPU tests for research code       [empty]
docs/                 master plan + provenance documents
data/                 checkpoints and datasets          [gitignored]
artifacts/            run outputs                       [gitignored]
_external/            reference clones                  [gitignored]
```

## Status

M0 is in progress, M1 is **not** verifiable in this repository, M3 is blocked
until the first GPU benchmark populates `docs/compute_budget.md` and Compute
Gate CG is explicitly resolved. See `PROGRESS.md` for what is actually verified.

Upstream AudioLDM usage instructions are preserved verbatim in
[`UPSTREAM_README.md`](UPSTREAM_README.md).

## Licensing

This repository merges MIT-licensed upstream code. Pretrained AudioLDM
checkpoints are **CC-BY-NC-4.0 (no commercial use)** per the upstream README;
downloaded checkpoints are not redistributed here.
