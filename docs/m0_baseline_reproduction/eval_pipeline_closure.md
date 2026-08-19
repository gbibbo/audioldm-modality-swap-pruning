# M0 closure: FAD/KL and PANNs top-k evaluation pipelines

Records the end-to-end exercise of the two evaluation pipelines the master plan
depends on (M0 remaining items 1 and 2). Both were previously only proven to
**import**; here they are run on real audio and their exact invocations, outputs,
and dependency/behaviour findings are recorded. These are **pipeline smokes on
arbitrary disjoint AudioCaps subsets**, not scientific evaluations — the metric
values are not meaningful comparisons and must not be cited as results.

## 1. FAD / KL / IS / FID pipeline (`audioldm_eval` 0.0.5)

**Script:** `scripts/research/fad_kl_smoke.py`
**Invocation:**

```bash
# build two disjoint 256-clip symlink folders from AudioCaps, then:
.venv/bin/python scripts/research/fad_kl_smoke.py \
    --gen artifacts/m0_baseline_reproduction/fad_kl_smoke/gen \
    --gt  artifacts/m0_baseline_reproduction/fad_kl_smoke/gt \
    --sr 16000 --fresh \
    --out artifacts/m0_baseline_reproduction/fad_kl_smoke/fad_kl_smoke_metrics.json
```

### Dependencies resolved (were undocumented)

1. **VGGish** for FAD is fetched at `EvaluationHelper.__init__` via
   `torch.hub.load("harritaylor/torchvggish", "vggish")` — needs network + GitHub
   reachable; cached under `~/.cache/torch/hub`.
2. **PANNs Cnn14 16 kHz checkpoint** is required at the hardcoded relative path
   `ckpt/Cnn14_16k_mAP=0.438.pth` (`models.py:250`). It is NOT bundled;
   `audioldm_eval` auto-`wget`s both the 16 kHz and 32 kHz PANNs checkpoints only
   if `ckpt/Cnn14_mAP=0.431.pth` is absent. Both were fetched to `ckpt/`
   (gitignored) from Zenodo record 3987831 (16 kHz) and 3576403 (32 kHz).

### Findings (behaviour of the library on this platform)

* **F-eval-1 — CPU deserialize.** `Cnn14.__init__` calls
  `torch.load("ckpt/Cnn14_16k_mAP=0.438.pth")` **without** `map_location`, so on a
  CPU-only machine it raises *"Attempting to deserialize object on a CUDA device"*.
  Fix without patching the dependency: re-save the checkpoint with CPU storage once
  (`torch.load(map_location="cpu")` then `torch.save`). Applied to the 16 kHz file.
* **F-eval-2 — path-keyed feature cache.** `audioldm_eval` writes
  `<folder>_fad_feature_cache.npy` and `<folder>classifier_logits_feature_cache.pkl`
  next to each folder, keyed only on the **folder path**. Re-populating a folder
  with different files silently reuses stale features (observed: an identical FAD
  imaginary component across N=64 and N=256). `--fresh` clears these first.
* **F-eval-3 — FAD numerical failure is mishandled.** The VGGish FAD `sqrtm` has a
  non-negligible imaginary component (~0.0397) exceeding the library's tolerance,
  so `FrechetAudioDistance.score` returns an **int** sentinel and `eval.py` then
  crashes on `out.update(fad_score)` (`'int' object is not iterable`). This is a
  bug in the library's error handling, independent of sample size (same value at
  N=64 and N=256). The smoke wraps the instance method so a failed FAD yields
  `NaN` and the KL / IS / FID metrics (computed afterwards) still run. **A real
  evaluation must supply a FAD implementation that takes the real part of the
  matrix square root (standard pytorch-fid behaviour) rather than raising.**
* **F-eval-4 — Cnn14 backbone IS pretrained.** Contrary to a first reading of
  `eval.py` (no visible load), the Cnn14 loads `state_dict["model"]` from the
  checkpoint inside its own `__init__`; KL/IS/FID therefore use the genuine
  PANNs 16 kHz weights.
* **F-eval-5 — both Frechet metrics fail the same way, and mid-`main` FID crash
  discards KL/IS.** The Cnn14-2048 **FID** (`calculate_fid`) uses the same `sqrtm`
  and raises `AssertionError: Imaginary component 1.95e+287` at N=256 (a 2048-dim
  covariance is hopelessly rank-deficient below ~2048 samples). Because it runs
  *after* KL and IS inside `calculate_metrics`, the crash discards the KL/IS that
  were already computed. Robust closure: compute KL/IS directly from the cached
  classifier features (`scripts/research/fad_kl_from_cache.py`), and compute the
  Frechet distance with the standard `covmean.real` fix — which is finite where the
  library asserts.
* **F-eval-6 — KL needs same-name pairing.** `calculate_kl` with `same_name=False`
  (disjoint filenames) returns the sentinel **-1**; it computes a per-item KL only
  when generated and reference clips share filenames (i.e. same caption). A real
  evaluation must pair generated clips to their reference by caption/filename.

### Outputs (smoke; NOT a scientific comparison)

Computed from the cached PANNs classifier features over two arbitrary disjoint
256-clip AudioCaps subsets (`fad_kl_smoke_metrics.json`) — values are non-scientific:

| Metric | Value | Note |
|---|---|---|
| Inception Score (mean ± std) | **6.230 ± 1.531** | computed from the generated set's logits |
| KL (sigmoid / softmax) | **-1 / -1** | sentinel — needs same-name pairing (F-eval-6) |
| Frechet distance (2048, real-part fix) | **15.361** | finite via the standard fix; audioldm_eval asserts here (F-eval-5) |
| FAD (VGGish) | NaN | audioldm_eval sqrtm assertion (F-eval-3); use standard real-part FAD |

The KL sentinel and the two Frechet failures are library behaviour, not properties
of the audio; they define exactly what a real M4/M5 evaluation must fix (pair by
caption; use a real-part Frechet). IS is the only metric that runs unmodified.

## 2. PANNs Top-K semantic pipeline (PruningAudioLDM README §5)

**Script:** `scripts/research/panns_topk.py`
**Invocation:**

```bash
.venv/bin/python scripts/research/panns_topk.py \
    --dir artifacts/m0_baseline_reproduction/fad_kl_smoke/gt --k 10 --limit 20 \
    --out artifacts/m0_baseline_reproduction/panns_topk_smoke.json
```

Loads the pretrained PANNs Cnn14 (16 kHz), classifies each clip, and returns the
Top-10 AudioSet events (`clipwise_output` = sigmoid over 527 classes) mapped via
`class_labels_indices.csv` (fetched from the AudioSet corpus). This is the exact
machinery §5 uses to compare predicted events before/after pruning and recovery;
run here on real clips to prove it is reproduced (no pruned/generated audio exists
until M4/M5).

**Result:** 20/20 clips classified with semantically coherent Top-10 predictions
(e.g. *Train / Rail transport*, *Waterfall / Stream*, *Neigh, whinny / Horse*,
*Bee, wasp / Insect*, *Applause*), confirming the checkpoint and label mapping are
correct. Full output: `panns_topk_smoke.json`.

## 3. Status

Both evaluation pipelines now run end-to-end on CPU with recorded invocations. The
FAD library bug (F-eval-3) and the CPU-deserialize/cache traps (F-eval-1/2) are
carried into the eval protocol so a real M4/M5 evaluation applies the standard-FAD
fix and fresh-cache discipline. No scientific comparison is claimed here.
