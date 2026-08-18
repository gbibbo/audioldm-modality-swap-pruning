# M0 — Dataset and checkpoint manifest

Everything below is measured on the Lightning Studio. All md5 values were
verified after download against the checksums published in the Zenodo records.

## Archives

| Archive | Size | md5 | Verified | Extracted to |
|---|---|---|---|---|
| `checkpoints.tar` | 7.816 GB | `d9898f93372582119fa19c6464f59cdc` | yes | `data/checkpoints/` |
| `dataset.tar` | 32.288 GB | `1c4e6642754c38f7041efdfeabe6e32d` | yes | `data/dataset/` |

Both from Zenodo 10.5281/zenodo.14342967. Fetch log:
`artifacts/m0_baseline_reproduction/fetch.log` (`fetch rc=0`, all six files OK).

## Standalone checkpoints

| File | Size | md5 | Verified | Source record |
|---|---|---|---|---|
| `audioldm-m-full.ckpt` | 4.572 GB | `46bad9f176651404b3cf1484942749b9` | yes | 7884686 |
| `Unet_model-m.ckpt` | 1.664 GB | `e44eaa7cbd5a358111d496d1cd246a33` | yes | 21376822 |
| `l1_audioldm-m-full_p1.ckpt` | 3.491 GB | `2666e6fc108a9c4fc0d19bbf26832905` | yes | 21376822 |
| `sorted_indexes_dict.pkl` | 59 112 B | `a4cd11ff83438ee0f9aa5fe0917f39e3` | yes | 21376822 |

## Auxiliary checkpoints (from `checkpoints.tar`)

All seven files required by upstream `tests/validate_dataset_checkpoint.py` are
present in `data/checkpoints/`: `audiomae_16k_128bins.ckpt`,
`clap_music_speech_audioset_epoch_15_esc_89.98.pt`, `clap_htsat_tiny.pt`,
`hifigan_16k_64bins.ckpt` + `.json`, `hifigan_48k_256bins.ckpt` + `.json`,
`vae_mel_16k_64bins.ckpt`.

## AudioCaps (from `dataset.tar`)

```text
extracted size    31 GB
audio files       50 961 .wav under data/dataset/audioset/zip_audios/unbalanced_train_segments/
metadata files    25 under data/dataset/metadata/
dataset root      data/dataset/metadata/dataset_root.json
```

Split sizes as reported by `AudioDataset`:

| Split | Size | Metadata file |
|---|---|---|
| train | 49 502 | `metadata/audiocaps/datafiles/audiocaps_train_label.json` |
| test / val | 964 | `metadata/audiocaps/testset_subset/audiocaps_test_nonrepeat_subset_0.json` |

Note that `dataset_root.json` maps **val to the same file as test**
(`audiocaps_test_nonrepeat_subset_0.json`). Any protocol that needs a validation
split disjoint from the test set must define one explicitly; do not assume
upstream provides one.

## Upstream validation

```bash
.venv/bin/python tests/validate_dataset_checkpoint.py
```

```text
All files and directories are present
Checking the validity of the audio datasets
100%|██████████| 49502/49502
100%|██████████|   964/964
All audio files are present. You are good to go!
```

**PASS** — structure complete and every one of the 50 466 referenced audio files
resolves on disk. Log: `artifacts/m0_baseline_reproduction/validate_dataset_checkpoint.log`.

## CPU load smoke test

```bash
.venv/bin/python scripts/research/smoke_load_dataset.py --split train --n 2
```

**PASS** on both splits. Sample tensors, confirming the frozen preprocessing
config (16 kHz, 10.24 s, 64 mel bins, 1024-point FFT, hop 160):

```text
waveform        (1, 163840) float32     # 163840 / 16000 = 10.24 s
log_mel_spec    (1024, 64)  float32
stft            (1024, 512) float32
text            caption string present
```

librosa 0.9.2 emits `FutureWarning`s about positional arguments from upstream
`stft.py` and `dataset.py`. They are upstream code and were deliberately left
untouched.
