# M0 — Environment report

Measured on the Lightning Studio on 2026-08-18 01:12 UTC. This is a
record of the environment as found, not of a working AudioLDM environment.

## Host

```text
os              Linux 6.8.0-1062-aws x86_64
cpus            4
memory_total    15 GB
disk_avail      352G of 387G
gpu             none visible (nvidia-smi absent; torch.cuda.is_available() == False)
```

## Active Python environment

```text
conda_env       cloudspace
python          3.12.11  at /home/zeus/miniconda3/envs/cloudspace/bin/python
torch           2.8.0+cu128
pytorch_lightning 2.6.5
pytest          absent
poetry          absent
```

## Dependency gap (blocking)

The active `cloudspace` env is **not** an AudioLDM environment. Import check:

```text
  present  torch  2.8.0+cu128
  present  pytorch_lightning  2.6.5
  MISSING  torchaudio
  MISSING  librosa
  MISSING  transformers
  MISSING  soundfile
  MISSING  omegaconf
  MISSING  einops
  MISSING  taming
  MISSING  laion_clap
  MISSING  h5py
  MISSING  wandb
```

## Decision required before any model code runs

Upstream `pyproject.toml` declares `python = "^3.10"` and pins
`transformers = "4.30.2"`; the upstream README explicitly instructs
`conda create -n audioldm_train python=3.10`. The active env is Python 3.12.11.
`^3.10` formally admits 3.12, but `transformers==4.30.2` and
`taming-transformers-rom1504` are not expected to install cleanly on 3.12.

Two options, both needing an explicit decision because they change every number
this project will report:

1. **Follow upstream exactly** — create a dedicated `audioldm_train` conda env on
   Python 3.10 and `poetry install`. Maximum fidelity to the frozen reference,
   isolated from the Studio's `cloudspace` env.
2. **Modernise** — install on Python 3.12 with relaxed pins. Faster, but any
   numerical difference from the published baseline becomes unattributable.

Option 1 is the default implied by the master plan's reproduction requirement.
This has **not** been done; no environment has been created or modified.

## GPU

No GPU is attached to this Studio right now. Every task performed for M0 so far
was CPU + network only. The first GPU benchmark, and therefore every entry in
`docs/compute_budget.md`, remains pending. Compute Gate CG is unresolved and M3
stays blocked.
