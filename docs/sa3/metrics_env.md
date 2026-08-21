# SA3 E-metrics environment (`.venv-metrics`, gitignored)

Isolated venv for the protocol section-9.1 metrics (CLAP / KL_passt / FD_openl3), kept separate
from the frozen AudioLDM `.venv` (torch 1.13.1) and the SA3 `.venv-sa3` (torch 2.7.1) so no pin
is disturbed. Metrics run on **wav files**, so this is a separate process from SA3 generation.

Rebuild (CPU torch is enough — the smoke scores few clips; use CUDA torch for pilot/main):
```bash
uv venv --python 3.10 .venv-metrics
uv pip install --python .venv-metrics/bin/python torch==2.2.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-metrics/bin/python torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-metrics/bin/python "numpy<2" librosa soundfile scipy laion_clap
uv pip install --python .venv-metrics/bin/python "transformers==4.30.2" "tokenizers<0.14" "huggingface_hub<0.26"
uv pip install --python .venv-metrics/bin/python hear21passt torchopenl3
```
Full pin set: `docs/sa3/metrics_requirements.txt`.

Gotchas fixed:
* `laion_clap` needs `torchvision` and an **old** `transformers` (4.30.2) — newer breaks its Roberta path.
* Import **torchvision first** (the scorer does): timm→torchvision has a circular-import bug otherwise.
* Default `CLAP_Module(enable_fusion=False)` is **HTSAT-tiny**, matching the `630k-audioset` ckpt
  from `load_ckpt()` (constructing `HTSAT-base` gives a size-mismatch load error).
* Run under `OPENBLAS_CORETYPE=Haswell` (E-BLAS) — the FD uses numpy linalg on 512-dim covs.

Weights (cached, gitignored): CLAP `630k-audioset-best.pt`; PaSST AudioSet (hear21passt);
OpenL3 `torchopenl3_mel256_env_512.pth.tar`.

Scorer: `scripts/sa3/score_e_metrics.py` (run with `.venv-metrics/bin/python`). Self-test:
`SCRATCH=<dir> OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/sa3/score_e_metrics.py --selftest`
→ reference-self KL==0, FD~0, all finite (PASS).
