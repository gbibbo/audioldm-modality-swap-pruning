# Environment report

Canonical record of the reproducible environment for this project. Everything
below was measured on the Lightning Studio, not copied from upstream docs.

**Decision (2026-08-18):** dedicated Python 3.10, as faithful to `upstream-frozen`
as the platform allows. No pin was relaxed and AudioLDM was not adapted to 3.12.

## Result

`poetry install` resolved and installed the frozen `poetry.lock` with **0 errors**
and **155 packages**. All import smoke tests pass. Both U-Net architectures load
their real weights with `strict=True`.

## Host

```text
os              Linux 6.8.0-1062-aws x86_64
cpus            4
memory_total    15 GB
disk_avail      300G of 387G
gpu             none attached; torch.cuda.is_available() == False
```

## GPU selection is CONSTRAINED by the frozen torch pin (verified 2026-08-19)

The frozen `poetry.lock` pins `torch 1.13.1+cu117`, and `AGENTS.md` forbids relaxing
any pin. That decides which Lightning machine types this project may use — this is a
hard constraint, not a preference.

Extracted from the actual binary in this `.venv`
(`torch/lib/libtorch_cuda_cu.so`, 763 MB, via `strings | grep -oE 'compute_[0-9]+|sm_[0-9]+'`):

```text
SASS compiled : sm_70 (V100), sm_75 (T4), sm_80 (A100)
PTX only      : compute_86
bundled libs  : nvidia-cublas-cu11 11.10.3.66, nvidia-cudnn-cu11 8.5.0.96
```

`torch.cuda.get_arch_list()` returns `[]` on a CPU-only machine (CUDA never
initialises), so it cannot be used for this check — read the binary instead.

| Lightning machine | Arch | Usable with this environment |
|---|---|---|
| `T4`, `T4_SMALL`, `T4_X_*` (16 GB) | sm_75 | **YES** — native SASS |
| `A100`, `A100_80GB`, `A100_X_*` (40/80 GB) | sm_80 | **YES** — native SASS |
| `L4`, `L40S`, `RTXP_6000` (24-48 GB) | sm_89 | **NO** — no SASS; cuBLAS 11.10 and cuDNN 8.5 predate Ada, and they ship precompiled kernels (they do not JIT) |
| `H100`, `H200` | sm_90 | **NO** |
| `B200_X_8` | sm_100 | **NO** |

Do **not** pick an L4 because it looks like the cheap modern default: it would boot,
bill, and then fail or fall back to unusable paths. Only T4 and A100 are valid.

**No reinstall is needed when switching to a GPU machine.** The CUDA wheels are
already present in the `.venv` (torch is 1.8 GB and ships `libtorch_cuda*.so` plus the
`nvidia-*-cu11` runtime wheels), so `torch.cuda.is_available()` flips to `True` on its
own once the machine has a device.

**VRAM caveat for T4 (16 GB).** The base `(1,2,3,5)` U-Net is 415.955 M params
(~1.66 GB fp32) and M3B saliency backpropagates through channel gates on top of the
VAE/CLAP/HiFi-GAN stack. T4 is expected to be adequate for `gpu_benchmark.py` and for
M1 PEFT acceptance on the pruned `(1,2,3,1)` model (149.4 M params, 3.88 M trainable),
but it may be tight for the base-model saliency and for M4 generation. If a run OOMs,
the fallback is `A100` (also native sm_80), not an L4.

**Whatever machine the benchmark runs on becomes part of the record:**
`scripts/research/gpu_benchmark.py` writes `GPU_MODEL` and `VRAM_GB`
(`torch.cuda.get_device_name(0)` / `get_device_properties(0).total_memory`), so
`docs/compute_budget.md` is always traceable to the hardware that produced it. The
projections in it are only valid for that machine — re-benchmark before moving the
scientific runs to a different one.

### Changing the machine on Lightning

The Studio is `gabriel-allgd-deploy-model-devbox` in teamspace
`independentaudioresearch/general`, cluster `lightning-public-prod` (us-east-1). The
`lightning` CLI (2026.06.08) is installed and already authenticated.

```bash
lightning studio switch --machine T4      # or A100; RESTARTS the Studio
lightning machine list                    # available types
lightning job run --studio <name> --machine A100 --command "..."   # GPU job, Studio untouched
```

`lightning studio switch` **restarts the Studio and kills every running process**,
including any agent session. The persistent filesystem
(`/teamspace/studios/this_studio`, 387 G) survives untouched — repo, `.venv`,
checkpoints and the 31 GB AudioCaps dataset are all still there afterwards. Machine
prices are shown in credits/hour in the Studio's machine selector; the SDK does not
expose them offline (`Machine.cost` is `None`).

The `lightning job run --studio` route leaves the Studio (and any running session)
alone, but **it is unverified here whether such a job mounts `data/checkpoints/` and
the dataset** — no job has ever run in this teamspace (`lightning job list` is empty).
Verify that before relying on it.

### Attempted 2026-08-19: refused for insufficient balance

The switch to `T4` was attempted from this terminal and the API refused it:

```text
$ lightning studio switch --name gabriel-allgd-deploy-model-devbox \
      --teamspace independentaudioresearch/general --machine T4
HTTP {"code":9, "message":"Insufficient balance to start the cloud space, top up and try again"}
```

The Studio never restarted — verified afterwards: status `Running`, machine `CPU`,
`torch.cuda.is_available() == False`. Nothing was billed and nothing was lost. **The
blocker is the account balance of the organization `independentaudioresearch` that owns
this teamspace**, not anything technical: the free CPU machine keeps running, while a GPU
machine requires a positive balance.

The personal teamspace `gabriel-allgd/general` also exists and is **completely empty** —
no studios, no jobs (`lightning studio list --teamspace gabriel-allgd/general`). Using it
would mean rebuilding this environment there from scratch: repo from GitHub, artifacts via
`scripts/research/fetch_public_artifacts.sh` (md5-verified, resumable), `.venv` per this
document. That is ~125 GB and several hours, but every step is scripted.

Do not retry the switch until a top-up is confirmed; it will fail identically.

## Platform deviations that were forced on us

Recording these because they are the only differences from the upstream recipe.

### 1. `conda` environment creation is blocked on Lightning

The upstream README says `conda create -n audioldm_train python=3.10`. On this
Studio that is refused, in every form:

```text
$ conda create -n audioldm_train python=3.10
Error: Conda create is not allowed. A Studio has a default conda environment
(max 1 environment). Start a new Studio to create a new environment.

$ conda create -p /teamspace/studios/this_studio/.envs/audioldm_train python=3.10 --dry-run
Error: Conda create is not allowed. ...
```

The prefix form is blocked too, so this is not worked around by choosing a
different location. The Studio's single conda env is `cloudspace` (Python
3.12.11) and must not be mutated.

### 2. `/commands/python3.10` is a decoy

The Studio ships a `/commands/python3.10` that is **not** Python 3.10:

```text
$ /commands/python3.10 -V
Python 3.12.11
```

It is a shell shim that resolves whichever `python` is on PATH. Do not use it.

### 3. Resolution: `uv`-provisioned standalone CPython 3.10 + `venv`

```bash
uv python install 3.10          # cpython-3.10.20-linux-x86_64-gnu
/system/conda/miniconda3/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 -m venv .venv
.venv/bin/pip install poetry    # upstream mechanism, unchanged
POETRY_VIRTUALENVS_CREATE=false VIRTUAL_ENV="$PWD/.venv" .venv/bin/poetry install
```

`POETRY_VIRTUALENVS_CREATE=false` is passed as an environment variable rather
than written with `poetry config --local`, so no `poetry.toml` is added to the
repository and `pyproject.toml`/`poetry.lock` stay byte-identical to
`upstream-frozen`.

This is a *packaging* deviation only: the interpreter is a real, unmodified
CPython 3.10.20 and every dependency version comes from the frozen
`poetry.lock`. `.venv/` is gitignored.

## Resolved toolchain

```text
python          3.10.20  (uv standalone cpython-3.10.20)
venv            .venv  (gitignored)
poetry          
No module named 'packaging.licenses'
packages        155 installed
lockfile        poetry.lock at upstream-frozen, unmodified
```

## Key resolved versions

These are the versions the frozen lock produced. Every number this project
reports must be attributed to this set.

```text
audioldm_eval               0.0.5
einops                      0.7.0
h5py                        3.10.0
hear21passt                 0.0.23
kornia                      0.7.0
librosa                     0.9.2
numpy                       1.23.5
pytorch-lightning           2.1.1
ruamel.yaml                 0.18.5
scipy                       1.9.3
soundfile                   0.12.1
taming-transformers-rom1504 0.0.6
torch                       1.13.1
torchaudio                  0.13.1
torchvision                 0.14.1
transformers                4.30.2
wandb                       0.16.0
webdataset                  0.2.75
```

## Verified in this environment

```bash
.venv/bin/python -c "import audioldm_train"                       # OK
.venv/bin/python scripts/research/smoke_load_unet.py              # SMOKE LOAD: PASS
.venv/bin/python scripts/research/verify_pruned_architecture.py …  # OK
```

Import smoke tests, all passing:

```text
  OK  audioldm_train
  OK  audioldm_train.utilities.data.dataset
  OK  audioldm_train.modules.diffusionmodules.openaimodel
  OK  audioldm_train.modules.latent_diffusion.ddpm
  OK  audioldm_train.conditional_models
  OK  audioldm_eval
```

Note: importing `audioldm_train.modules.latent_diffusion.ddpm` **downloads a
tokenizer from HuggingFace at import time** (`vocab.json`, `merges.txt`,
`tokenizer_config.json`, `config.json`). Imports are therefore not offline-safe;
budget for this on a fresh machine or pre-populate the HF cache.

## Incompatibility found and fixed

`torch.load(..., mmap=True)` does not exist in torch 1.13.1 (added in 2.1). Our
own helper scripts had been written against the Studio's torch 2.8 and failed
here with `TypeError: Unpickler.__init__() got an unexpected keyword argument
'mmap'`. Fixed in `scripts/research/{smoke_load_unet,verify_pruned_architecture}.py`
with a `torch_load()` helper that attempts `mmap=True` and falls back. Both
scripts now run under torch 1.13.1 and torch 2.8. **No upstream or scientific
code was changed**; `git diff upstream-frozen -- audioldm_train/` is still empty.

## Not done, deliberately

* No GPU work. `torch 1.13.1+cu117` is installed and will use CUDA when a GPU is
  attached, but nothing has been benchmarked, so `docs/compute_budget.md`
  remains entirely `TBD_MEASURED` and Compute Gate CG is unresolved.
* No LoRA/PEFT scaffold reconstructed.

## Full resolved package set

```text
absl-py==2.0.0
aiohttp==3.8.6
aiosignal==1.3.1
antlr4-python3-runtime==4.9.3
anyio==4.14.2
appdirs==1.4.4
asttokens==2.4.1
async-timeout==4.0.3
attrs==23.1.0
-e git+https://github.com/gbibbo/audioldm-modality-swap-pruning.git@7039fe6f0e653cb3052eee621e7f5e7f1295684e#egg=audioldm_train
audioldm_eval @ git+https://github.com/haoheliu/audioldm_eval.git@51b7fdd87ad7518d167895f8e18dcfbf93446da7
audioread==3.0.1
backports.tarfile==1.2.0
backports.zstd==1.7.0
braceexpand==0.1.7
build==1.5.0
CacheControl==0.14.4
certifi==2023.7.22
cffi==1.16.0
charset-normalizer==3.3.2
cleo==2.1.0
click==8.1.7
contourpy==1.2.0
crashtest==0.4.1
cryptography==50.0.0
cycler==0.12.1
decorator==5.1.1
distlib==0.4.3
docker-pycreds==0.4.0
dulwich==1.2.12
einops==0.7.0
exceptiongroup==1.1.3
executing==2.0.1
fastjsonschema==2.22.2
filelock==3.13.1
findpython==0.8.0
fonttools==4.44.0
frozenlist==1.4.0
fsspec==2023.10.0
ftfy==6.1.1
gitdb==4.0.11
GitPython==3.1.40
h11==0.16.0
h5py==3.10.0
hear21passt @ git+https://github.com/haoheliu/passt_hear21.git@4dd6b9e426f528e2e8409b9bacecf58a2f464548
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.17.3
idna==3.4
imageio==2.31.5
importlib_metadata==9.0.0
installer==1.0.1
ipdb==0.13.13
ipython==8.17.2
jaraco.classes==3.4.0
jaraco.context==6.1.2
jaraco.functools==4.6.0
jedi==0.19.1
jeepney==0.9.0
joblib==1.3.2
keyring==25.7.0
kiwisolver==1.4.5
kornia==0.7.0
lazy_loader==0.3
librosa==0.9.2
lightning-utilities==0.9.0
llvmlite==0.41.1
matplotlib==3.8.1
matplotlib-inline==0.1.6
more-itertools==11.1.0
msgpack==1.2.1
multidict==6.0.4
networkx==3.2.1
numba==0.58.1
numpy==1.23.5
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
omegaconf==2.3.0
packaging==23.2
pandas==2.1.3
parso==0.8.3
pbs-installer==2026.8.14
pexpect==4.8.0
Pillow==10.1.0
pkginfo==1.12.1.2
platformdirs==4.0.0
poetry==2.4.1
poetry-core==2.4.0
pooch==1.8.0
prompt-toolkit==3.0.41
protobuf==4.25.0
psutil==5.9.6
ptyprocess==0.7.0
pure-eval==0.2.2
pycparser==2.21
Pygments==2.16.1
pyparsing==3.1.1
pyproject_hooks==1.2.0
python-dateutil==2.8.2
python-discovery==1.5.2
pytorch-lightning==2.1.1
pytz==2023.3.post1
PyYAML==6.0.1
RapidFuzz==3.14.5
regex==2023.10.3
requests==2.31.0
requests-toolbelt==1.0.0
resampy==0.4.2
ruamel.yaml==0.18.5
ruamel.yaml.clib==0.2.8
safetensors==0.4.0
scikit-image==0.22.0
scikit-learn==1.3.2
scipy==1.9.3
SecretStorage==3.5.0
sentry-sdk==1.35.0
setproctitle==1.3.3
shellingham==1.5.4
six==1.16.0
smmap==5.0.1
soundfile==0.12.1
ssr-eval==0.0.7
stack-data==0.6.3
taming-transformers-rom1504==0.0.6
threadpoolctl==3.2.0
tifffile==2023.9.26
timm==0.4.12
tokenizers==0.13.3
tomli==2.0.1
tomlkit==0.15.1
torch==1.13.1
torchaudio==0.13.1
torchlibrosa==0.0.9
torchmetrics==1.2.0
torchvision==0.14.1
tqdm==4.66.1
traitlets==5.13.0
transformers==4.30.2
trove-classifiers==2026.6.1.19
typing_extensions==4.8.0
tzdata==2023.3
urllib3==2.1.0
virtualenv==21.7.4
wandb==0.16.0
Wave==0.0.2
wcwidth==0.2.9
webdataset==0.2.75
wget==3.2
yarl==1.9.2
zipp==4.1.0
```
