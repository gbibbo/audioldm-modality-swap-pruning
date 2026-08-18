# M0 — Public artifact inventory

Completed 2026-08-18 by querying the Zenodo REST API. Every checksum below is
the value published in the Zenodo record, not a value we invented.

## Architecture mapping (resolved)

`PruningAudioLDM` parameterises pruning by two block-wise scaling factors:
`dp` = Block-3 (b3), `p` = Block-4 (b4). Finetuning replaces, in
`audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`:

```yaml
channel_mult: [1, 2, 3, 5]     # verified present at upstream-frozen, line ~122
```

with `channel_mult: [1, 2, dp, p]`.

Therefore the master plan's mandatory `(1,2,3,1)` budget is **dp = 3, p = 1**,
and corresponds to the Zenodo artifact `l1_audioldm-m-full_p1.ckpt`
("b4 (or p) = 1, {b1,b2,b3} = {1,2,3}"). The out-of-core-scope `(1,2,1,1)`
corresponds to `l1_audioldm-m-full_p1_dp1.ckpt`.

## Record 10.5281/zenodo.21376822 — pruned + baseline AudioLDM-M-Full

Arshdeep Singh, published 2026-07-15, CC-BY-4.0.
*Checkpoints for pruned and Unpruned (baseline) AudioLDM-M-Full model*

| File | Size | md5 | Role | Status |
|---|---|---|---|---|
| `audioldm-m-full.ckpt` | 4.572 GB | `46bad9f176651404b3cf1484942749b9` | AudioLDM-M-Full base | available |
| `Unet_model-m.ckpt` | 1.664 GB | `e44eaa7cbd5a358111d496d1cd246a33` | pretrained U-Net, input to saliency/pruning scripts | available |
| `l1_audioldm-m-full_p1.ckpt` | 3.491 GB | `2666e6fc108a9c4fc0d19bbf26832905` | **L1 `(1,2,3,1)`** merged checkpoint | available |
| `l1_audioldm-m-full_p1_dp1.ckpt` | 3.192 GB | `2427ffb5955d56980f81dadeae189bbc` | L1 `(1,2,1,1)`, outside ICASSP core scope | available, not fetched |
| `l1_unet_pruned_p2_dp2.pt` | 0.540 GB | `9ca234f88784e51c10b9e72620b6432e` | U-Net-only, `(1,2,2,2)`, out of scope | available, not fetched |
| `sorted_indexes_dict.pkl` | 59 112 B | `a4cd11ff83438ee0f9aa5fe0917f39e3` | L1 layer-wise sorted channel indexes | available, fetched |

**Provenance cross-check (verified):** the `audioldm-m-full.ckpt` md5 in this
record is byte-identical to the md5 in the official AudioLDM record
10.5281/zenodo.7884686. The pruning reference is therefore built on the
official AudioLDM-M-Full weights, not a private variant.

**Recorded discrepancy:** the record description lists
`l1_audioldm-m-full_p2_dp2.ckpt`, but no such file exists in the record; the
`(1,2,2,2)` artifact actually published is the U-Net-only
`l1_unet_pruned_p2_dp2.pt`. Not load-bearing for `(1,2,3,1)`.

## Record 10.5281/zenodo.7884686 — official AudioLDM checkpoints

AudioLDM authors, published 2023-01-29, record licensed CC-BY-4.0. Note that the
upstream README states the pretrained AudioLDM checkpoints are **CC-BY-NC-4.0
(no commercial use)**; treat the more restrictive statement as binding.

Relevant files: `audioldm-m-full.ckpt` (4.572 GB, `46bad9f1…`), `VAE.ckpt`,
`hifigan_vocoder.ckpt`, plus S/L variants not used by this project.

## Record 10.5281/zenodo.14342967 — aux checkpoints + preprocessed AudioCaps

AudioLDM authors, published 2024-12-09, CC-BY-4.0. This is the mirror the
upstream README points to (alongside Google Drive).

| File | Size | md5 | Contents |
|---|---|---|---|
| `checkpoints.tar` | 7.816 GB | `d9898f93372582119fa19c6464f59cdc` | pretrained VAE, AudioMAE, CLAP, 16 kHz HiFi-GAN, 48 kHz HiFi-GAN → `data/checkpoints/` |
| `dataset.tar` | 32.288 GB | `1c4e6642754c38f7041efdfeabe6e32d` | preprocessed AudioCaps → `data/dataset/` |

## Verified directly from the downloaded artifacts

Run: `python3 scripts/research/verify_pruned_architecture.py <ckpt>...`
Raw logs: `artifacts/m0_baseline_reproduction/architecture_check.log`,
`artifacts/m0_baseline_reproduction/prerecovery_check.log`.

Structural budget recovered from weight shapes, not from any config or README:

| Checkpoint | tensors | model_channels | level widths | `channel_mult` |
|---|---|---|---|---|
| `audioldm-m-full.ckpt` | 2299 | 192 | 192, 384, 576, 960 | **[1, 2, 3, 5]** |
| `l1_audioldm-m-full_p1.ckpt` | 2299 | 192 | 192, 384, 576, 192 | **[1, 2, 3, 1]** |
| `Unet_model-m.ckpt` | 690 | 192 | 192, 384, 576, 960 | **[1, 2, 3, 5]** |

`l1_audioldm-m-full_p1.ckpt` is confirmed to be the mandatory `(1,2,3,1)`
architecture. The architecture mapping in the section above is therefore
empirically verified, not inferred from the reference README alone.

Measured parameter counts (denoiser = keys under `model.diffusion_model.`):

| Model | U-Net params | full checkpoint params |
|---|---|---|
| AudioLDM-M-Full base | 415.955 M | 1142.629 M |
| L1 `(1,2,3,1)` | 145.674 M | 872.348 M |

The denoiser drops by 65.0% (2.86x fewer parameters). The full checkpoint drops
by only 23.7% because the VAE, CLAP and vocoder are untouched. These are
measured counts and supersede any estimate; they are the starting point for the
EFF claim, but they are parameter counts only — no runtime, VRAM or GPU-hour
number is claimed here.

## Full-FT checkpoint strength gate - RESOLVED 2026-08-18

The master plan requires a public search for a **final recovered full-FT
`(1,2,3,1)` checkpoint** by 2026-08-18.

Search result:

* Neither `haoheliu/AudioLDM-training-finetuning` nor
  `Arshdeep-Singh-Boparai/PruningAudioLDM` publishes any GitHub release.
* The only public checkpoint record for the pruning work is
  10.5281/zenodo.21376822, inventoried above.

**Decisive test.** Level 0 of the U-Net has multiplier 1 in both architectures,
so its tensors are structurally untouched by pruning; any finetuning of the
model would have to change them. Comparing every same-shape tensor between
`audioldm-m-full.ckpt` and `l1_audioldm-m-full_p1.ckpt`:

```text
same-shape tensors compared : 2061
  bit-identical to base     : 2061
  differing from base       :    0
```

All 2061 surviving tensors, including the VAE and the U-Net stem, are
bit-identical to the pretrained base. **No finetuning of any kind was applied.**

**Gate decision:**

* The L1 `(1,2,3,1)` **pre-recovery** checkpoint is publicly available, fetched,
  md5-verified, and proven to be pure prune-and-merge output. The master plan
  task "reconstruct or download the L1 `(1,2,3,1)` pre-recovery checkpoint" is
  **satisfied**, and no reconstruction from `Unet_model-m.ckpt` is required.
* The **recovered full-FT `(1,2,3,1)` checkpoint is NOT publicly available**, and
  this is now proven rather than assumed.

Per the master plan this triggers, today, an immediate request to Arshdeep for
the final recovered full-FT `(1,2,3,1)` checkpoint. Until it is obtained, **RQ3
is downgraded to a published-reference comparison**, and no exact
percentage-of-full-FT recovery may be claimed from cross-pipeline numbers.

## L1 saliency manifest — `sorted_indexes_dict.pkl`

Inspected statically first with `pickletools` (log:
`artifacts/m0_baseline_reproduction/sorted_indexes_inspect.log`). The pickle
contains **no `GLOBAL`/`REDUCE` opcodes**, i.e. it is pure data and executes no
third-party code on load; only then was it loaded.

Contents: `dict[str, list[int]]` over **28 conv layers**. Every value is a full
permutation of `range(n_channels)`, so each entry is a complete L1 importance
ranking of that layer's channels, not a pre-truncated keep-list.

| block family | layers | channel widths present |
|---|---|---|
| `input_blocks.7…11` | 9 | 576, 960 |
| `middle_block` | 4 | 960 |
| `output_blocks.0…6` | 15 | 960, 576, 384 |

The covered layers are those touched by B3/B4 pruning (`pruned_indexes/B3_B4/`),
with one entry reaching into the 384-wide level (`output_blocks.6.0`). The whole
U-Net is **not** covered.

**Consequence for RQ2 (matched comparison).** P1 text-only Taylor, P2 paired
mean and P3 paired max must produce rankings over **exactly this 28-layer set**,
with the same per-layer channel counts, or the comparison against the L1
baseline is not structure-matched. This layer set is therefore a fixed input to
the M3 pilot protocol and must be frozen in `docs/pilot_protocol.md` before
saliency results are inspected.

## Not yet available / still open

* **Recovered full-FT `(1,2,3,1)` checkpoint** - not public, proven absent from
  every public source searched. Request to Arshdeep pending. Blocks the
  same-pipeline RQ3 comparison.
* Intermediate full-FT training logs - useful, not mandatory.
* PANNs top-k semantic pipeline - described in section 5 of the reference
  README; not yet reproduced here.
* FAD/KL pipeline (`audioldm_eval`) - not installed, not verified. Requires the
  environment decision in `environment_report.md`.
* AudioCaps preprocessed dataset - fetching; see `data/dataset.tar`.
* Generation/model-loading smoke tests - require a GPU and the environment
  decision; both pending.
