# M0 — Frozen references

All SHAs below were verified in this repository on 2026-08-18, not copied from
the master plan.

## Upstream training/finetuning codebase

* Repository: `haoheliu/AudioLDM-training-finetuning` (MIT)
* Frozen commit: `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc`
* Commit: *Merge pull request #51 from haoheliu/haoheliu-patch-3*, Haohe Liu, 2024-12-13
* Preserved as: local branch `upstream-frozen`, remote `upstream`
* At fetch time this commit was also the tip of `upstream/main`, and it is an
  ancestor of `upstream/main`. Full history (35 commits) is merged into `main`.

Review our surgical patches with:

```bash
git diff upstream-frozen -- audioldm_train/
```

As of this commit that diff is **empty**: `audioldm_train/` is byte-identical to
`upstream-frozen`. The only file on `main` that differs from `upstream-frozen`
is `.gitignore` (union of the upstream block, kept verbatim, and the agent-kit
research block) plus files we added.

## Pruning reference implementation

* Repository: `Arshdeep-Singh-Boparai/PruningAudioLDM` (MIT)
* Frozen commit: `6f65f628fabc4ad27770753698fc81944e820f9f`
* Commit: *link update*, Arshdeep-Singh-Boparai, 2026-07-16
* Preserved as: local branch `pruning-reference-frozen`, remote `pruning-reference`
* Working copy for reading: `_external/PruningAudioLDM` (gitignored), checked out
  at the frozen SHA
* Associated paper: arXiv:2607.13330 — *Efficient Text-to-Audio Generation via Pruning*
* This history is **not** merged into `main`; it is kept as a reference branch only.

Reference contents at the frozen SHA:

```text
LICENSE
README.md
pruned_indexes/B3_B4/sorted_indexes_dict.pkl
scripts/layerwise_sorted_index_generation.py
scripts/merge_pruned_checkpoint.py
scripts/pruned_unet_dict_creation.py
```

## Verification commands

```bash
git cat-file -t 702a638d023b008a2d9a45cdf1e1f4fcdc590dfc
git cat-file -t 6f65f628fabc4ad27770753698fc81944e820f9f
git merge-base --is-ancestor 702a638d023b008a2d9a45cdf1e1f4fcdc590dfc upstream/main
git diff --stat upstream-frozen HEAD -- audioldm_train/   # must stay reviewable
```
