"""M2 conditioning-path diagnostics for AudioLDM modality-swap validation.

This module exposes the audio-conditioned and text-conditioned CLAP paths and the
FiLM injection interface of the diffusion U-Net, so that the modality swap can be
validated *before* any cross-modal scientific claim is made (master plan M2).

It deliberately instantiates ``UNetModel`` and ``CLAPAudioEmbeddingClassifierFreev2``
directly instead of the full ``LatentDiffusion`` object, because the latter is heavy
to build on CPU (VAE, vocoder, EMA, tokenizer downloads). The wiring reproduced here
is byte-for-byte the same path ``LatentDiffusion`` uses at inference; the proof is
recorded in ``docs/condition_swap_validation.md`` and asserted in
``tests/research/test_conditioning_paths.py``:

    conditioner returns [B,1,512]      conditional_models.py:1321  (embed.unsqueeze(1))
    "film" cond -> y = cond.squeeze(1) ddpm.py:1996-2000           (DiffusionWrapper.forward)
    U-Net consumes y via FiLM concat   openaimodel.py:871-872      (film_emb(y) into time emb)
    film_emb = Linear(512, time_emb)   openaimodel.py:555

Everything here is CPU-only, uses the frozen config
``audioldm_original_medium.yaml`` and real weights. No random weight
initialisation and no synthetic audio: embeddings are computed from real
AudioCaps waveforms/captions and real checkpoints.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Literal

import torch
import yaml

from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2
from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
from audioldm_train.modules.latent_encoder.autoencoder import AutoencoderKL
from audioldm_train.utilities.diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
)

Modality = Literal["audio", "text"]

FROZEN_CONFIG = (
    "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
)
DIFFUSION_PREFIX = "model.diffusion_model."


# --------------------------------------------------------------------------- #
# Config / model construction
# --------------------------------------------------------------------------- #
def load_config(config_path: str = FROZEN_CONFIG) -> dict:
    with open(config_path) as handle:
        return yaml.safe_load(handle)


def _torch_load(path: str):
    """CPU checkpoint load; `mmap=True` only exists in torch>=2.1, this env is 1.13.1."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def build_clap(
    config: dict,
    unconditional_prob: float = 0.0,
) -> CLAPAudioEmbeddingClassifierFreev2:
    """Instantiate the CLAP conditioner from the frozen config.

    The upstream default is ``unconditional_prob=0.1`` and the frozen config does
    NOT override it (see ``cond_stage_config.film_clap_cond1.params``). For a
    deterministic diagnostic we force it to 0.0, which disables the stochastic
    unconditional-token dropout in ``CLAPAudioEmbeddingClassifierFreev2.forward``
    (conditional_models.py:1322-1324). The conditioner is put in eval mode; its
    backbone parameters are already frozen (requires_grad=False) upstream.
    """
    params = copy.deepcopy(
        config["model"]["params"]["cond_stage_config"]["film_clap_cond1"]["params"]
    )
    clap = CLAPAudioEmbeddingClassifierFreev2(**params)
    clap.unconditional_prob = float(unconditional_prob)
    clap.eval()
    return clap


def build_unet(
    config: dict,
    ckpt_path: str,
    channel_mult: list[int] | None = None,
    strict: bool = True,
) -> UNetModel:
    """Rebuild the diffusion U-Net from config and strict-load real weights.

    ``channel_mult=None`` keeps the config value ``[1,2,3,5]`` (base
    AudioLDM-M-Full). Pass ``[1,2,3,1]`` for the pruned budget.
    """
    unet_params = copy.deepcopy(config["model"]["params"]["unet_config"]["params"])
    if channel_mult is not None:
        unet_params["channel_mult"] = channel_mult
    unet = UNetModel(**unet_params)

    obj = _torch_load(ckpt_path)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    weights = {
        k[len(DIFFUSION_PREFIX):]: v
        for k, v in state.items()
        if k.startswith(DIFFUSION_PREFIX)
    }
    unet.load_state_dict(weights, strict=strict)
    unet.eval()
    return unet


# --------------------------------------------------------------------------- #
# VAE (first stage) and the diffusion noising schedule  (A1 correction)
# --------------------------------------------------------------------------- #
def read_scale_factor(ckpt_path: str) -> float:
    """Read the DDPM latent `scale_factor` stored in the checkpoint.

    The frozen config sets `scale_by_std: true`, so `scale_factor` is a scalar
    buffer saved inside the checkpoint (ddpm.py:1032,1108) and MUST be read from
    there, never recomputed from a data batch (which would change with the batch).
    """
    obj = _torch_load(ckpt_path)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    sf = state["scale_factor"]
    return float(sf.item() if torch.is_tensor(sf) else sf)


def build_vae(config: dict, ckpt_path: str) -> AutoencoderKL:
    """Build the first-stage AutoencoderKL and load the encoder weights that
    `LatentDiffusion` actually uses at inference: the `first_stage_model.*` tensors
    embedded in `ckpt_path` (loaded last, so they override the standalone
    `reload_from_ckpt` VAE — see `build_val`/report for the measured divergence).

    Construction avoids two CPU-hostile side effects that never touch the encode
    path: the LPIPS discriminator loss (network download) is replaced by
    `torch.nn.Identity`, and the CUDA-pickled vocoder (built only when
    `image_key == "fbank"`, autoencoder.py:65-66) is skipped by constructing with
    `image_key="stft"`. Neither `encode` nor `.mode()` depends on either.
    """
    params = copy.deepcopy(config["model"]["params"]["first_stage_config"]["params"])
    params["lossconfig"] = {"target": "torch.nn.Identity"}
    params["reload_from_ckpt"] = None  # we load the embedded weights explicitly
    params["image_key"] = "stft"  # skip the vocoder in __init__ (autoencoder.py:65)

    vae = AutoencoderKL(**params)

    obj = _torch_load(ckpt_path)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    prefix = "first_stage_model."
    fs = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    # strict=False: the embedded first_stage carries vocoder/loss tensors this
    # trimmed VAE does not define; every encoder/decoder/quant_conv key loads.
    missing, _ = vae.load_state_dict(fs, strict=False)
    encode_missing = [
        k for k in missing
        if k.startswith("encoder.") or k.startswith("quant_conv.")
    ]
    if encode_missing:
        raise RuntimeError(f"encode-path weights missing from ckpt: {encode_missing}")
    vae.eval()
    return vae


@torch.no_grad()
def vae_encode(vae: AutoencoderKL, mel: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Encode a real mel `[B,1,1024,64]` to the scaled latent `z_0`.

    Uses the posterior **mode** (mean), not `.sample()`. `LatentDiffusion` uses
    `.sample()` (ddpm.py:1155) for training-time regularisation, but a diagnostic
    that isolates *pruning* damage needs `z_0` fixed and reproducible: the mode is
    deterministic by construction, is the maximum-likelihood latent, and removes
    VAE posterior-sampling variance that is irrelevant to (and would only add
    noise to) the D_gen/D_mod/R_mod statistics. `z_0 = scale_factor * mode`
    matches `get_first_stage_encoding` (ddpm.py:1153-1162) with `.sample()`
    replaced by `.mode()`.
    """
    posterior = vae.encode(mel)
    return scale_factor * posterior.mode()


class NoiseSchedule:
    """The DDPM forward schedule, reusing the upstream `make_beta_schedule`.

    Reproduces `DDPM.register_schedule` (ddpm.py:201-241) exactly: same beta
    function, `alphas_cumprod = cumprod(1 - betas)`, and the two sqrt buffers.
    `q_sample` uses the upstream `extract_into_tensor` so the noising is
    byte-identical to `DDPM.q_sample` (ddpm.py:430-436).
    """

    def __init__(self, config: dict):
        p = config["model"]["params"]
        self.timesteps = int(p["timesteps"])
        self.linear_start = float(p["linear_start"])
        self.linear_end = float(p["linear_end"])
        betas = make_beta_schedule(
            "linear",
            self.timesteps,
            linear_start=self.linear_start,
            linear_end=self.linear_end,
            cosine_s=8e-3,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.tensor(
            __import__("numpy").cumprod(alphas, axis=0), dtype=torch.float32
        )
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, z_0.shape) * z_0
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape) * noise
        )


# --------------------------------------------------------------------------- #
# Paired diagnostic slots
# --------------------------------------------------------------------------- #
@dataclass
class PairedSlots:
    """Fixed, reproducible diagnostic slots shared by the audio and text paths.

    ``z_t`` and ``t`` are the SAME tensors used for both modalities, so that the
    only difference between eps_a and eps_t is the conditioning modality (master
    plan section 3: same example, noisy latent z_t, timestep t, noise).
    """

    indices: list[int]
    waveforms: torch.Tensor  # [B, 1, T] real AudioCaps waveforms @ 16 kHz
    texts: list[str]  # real AudioCaps captions
    z_t: torch.Tensor  # [B, C, H, W] noised REAL latent  sqrt(a_t) z0 + sqrt(1-a_t) eps
    t: torch.Tensor  # [B] long diffusion timesteps
    seed: int
    noise: torch.Tensor = field(repr=False)  # the noise realisation z_t is built from
    z_0: torch.Tensor = field(default=None, repr=False)  # scaled VAE latent of the real mel

    def audio_items(self) -> torch.Tensor:
        return self.waveforms

    def text_items(self) -> list[str]:
        return self.texts


def build_paired_slots(
    dataset,
    indices: list[int],
    config: dict,
    *,
    vae: AutoencoderKL,
    schedule: "NoiseSchedule",
    scale_factor: float,
    seed: int = 0,
    timesteps: int | None = None,
) -> PairedSlots:
    """Build fixed reproducible (example, noise, t) slots from real AudioCaps items.

    (A1 correction) ``z_t`` is a **noised real latent**, not pure Gaussian noise:

        z_0 = scale_factor * VAE.encode(real_mel).mode()      # deterministic
        z_t = sqrt(a_t) * z_0 + sqrt(1 - a_t) * eps            # DDPM q_sample

    ``eps`` and ``t`` come from a seeded generator, so ``z_0``, ``eps``, ``z_t``
    and ``t`` are all reproducible and, crucially, IDENTICAL across the audio and
    text conditioning paths (the pairing invariants T3/T4). Waveforms, captions and
    mels are all real (no synthetic audio, no random model weights).
    """
    params = config["model"]["params"]
    if timesteps is None:
        timesteps = int(params["timesteps"])

    waveforms, texts, mels = [], [], []
    for idx in indices:
        sample = dataset[idx]
        wav = sample["waveform"]  # [1, T]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        waveforms.append(wav)
        texts.append(sample["text"])
        mels.append(sample["log_mel_spec"])  # [1024, 64]
    waveforms = torch.stack(waveforms, dim=0)  # [B, 1, T]
    fbank = torch.stack(mels, dim=0).unsqueeze(1).float()  # [B,1,1024,64]  (ddpm.py:540)

    z_0 = vae_encode(vae, fbank, scale_factor)  # [B, C, H, W], deterministic (mode)

    gen = torch.Generator().manual_seed(seed)
    batch = len(indices)
    noise = torch.randn(z_0.shape, generator=gen)
    t = torch.randint(0, timesteps, (batch,), generator=gen, dtype=torch.long)
    z_t = schedule.q_sample(z_0, t, noise)

    return PairedSlots(
        indices=list(indices),
        waveforms=waveforms,
        texts=texts,
        z_t=z_t,
        t=t,
        seed=seed,
        noise=noise,
        z_0=z_0,
    )


# --------------------------------------------------------------------------- #
# Embedding + epsilon
# --------------------------------------------------------------------------- #
@torch.no_grad()
def clap_embed(
    clap: CLAPAudioEmbeddingClassifierFreev2,
    items,
    modality: Modality,
) -> torch.Tensor:
    """Return a [B,1,512] CLAP embedding for the given modality.

    ``modality="audio"`` expects a waveform tensor ``[B,1,T]`` (16 kHz); the
    conditioner resamples 16k->48k, mel-transforms and truncates internally.
    ``modality="text"`` expects a list[str] of captions.

    This toggles ``clap.embed_mode`` exactly like ``LatentDiffusion`` does in its
    train/val CLAP switch (ddpm.py:678-683, ddpm.py:712-713).
    """
    if modality not in ("audio", "text"):
        raise ValueError(f"modality must be 'audio' or 'text', got {modality!r}")
    clap.embed_mode = modality
    emb = clap(items)  # [B, 1, 512]
    if emb.dim() != 3 or emb.size(1) != 1 or emb.size(2) != 512:
        raise RuntimeError(f"unexpected CLAP embedding shape {tuple(emb.shape)}")
    return emb


@torch.no_grad()
def eps_pred(
    unet: UNetModel,
    z_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """Predict epsilon through the FiLM interface used by LatentDiffusion.

    ``cond`` is the [B,1,512] CLAP embedding. It is squeezed to [B,512] and passed
    as ``y``, reproducing ``DiffusionWrapper.forward`` (ddpm.py:1998,
    ``y = cond_dict[key].squeeze(1)``) which the U-Net then injects through
    ``film_emb`` (openaimodel.py:871-872).
    """
    y = cond.squeeze(1)  # [B, 512]   -- ddpm.py:1998
    # DiffusionWrapper.forward passes empty context lists when there is no
    # crossattn condition (ddpm.py:1989, then ddpm.py:2039-2041); the U-Net does
    # `[None] + context_list` internally (openaimodel.py:86), so None would fail.
    return unet(
        z_t,
        timesteps=t,
        y=y,
        context_list=[],
        context_attn_mask_list=[],
    )


@torch.no_grad()
def paired_eps(
    unet: UNetModel,
    clap: CLAPAudioEmbeddingClassifierFreev2,
    slots: PairedSlots,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (eps_a, eps_t) on the SAME z_t, t and noise realisation.

    Only the conditioning modality differs between the two predictions.
    """
    e_a = clap_embed(clap, slots.audio_items(), "audio")
    e_t = clap_embed(clap, slots.text_items(), "text")
    eps_a = eps_pred(unet, slots.z_t, slots.t, e_a)
    eps_t = eps_pred(unet, slots.z_t, slots.t, e_t)
    return eps_a, eps_t


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def tensor_hash(tensor: torch.Tensor) -> str:
    """Stable content hash of a tensor (for provenance / pairing proofs)."""
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()
