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
    z_t: torch.Tensor  # [B, C, H, W] noisy latent (fixed noise realisation)
    t: torch.Tensor  # [B] long diffusion timesteps
    seed: int
    noise: torch.Tensor = field(repr=False)  # the noise realisation z_t is built from

    def audio_items(self) -> torch.Tensor:
        return self.waveforms

    def text_items(self) -> list[str]:
        return self.texts


def build_paired_slots(
    dataset,
    indices: list[int],
    config: dict,
    seed: int = 0,
    timesteps: int | None = None,
) -> PairedSlots:
    """Build fixed reproducible (example, noise, t) slots from real AudioCaps items.

    The noisy latent ``z_t`` is drawn from a seeded generator with the latent
    shape declared in the frozen config (channels x latent_t_size x latent_f_size).
    For M2 the *content* of z_t is not a scientific quantity; what matters is that
    it is identical across the audio and text paths and reproducible from ``seed``.
    The waveforms and captions ARE real (no synthetic audio).
    """
    params = config["model"]["params"]
    channels = params["channels"]
    latent_t = params["latent_t_size"]
    latent_f = params["latent_f_size"]
    if timesteps is None:
        timesteps = params["timesteps"]

    waveforms, texts = [], []
    for idx in indices:
        sample = dataset[idx]
        wav = sample["waveform"]  # [1, T]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        waveforms.append(wav)
        texts.append(sample["text"])
    waveforms = torch.stack(waveforms, dim=0)  # [B, 1, T]

    gen = torch.Generator().manual_seed(seed)
    batch = len(indices)
    noise = torch.randn(batch, channels, latent_t, latent_f, generator=gen)
    z_t = noise.clone()
    t = torch.randint(0, timesteps, (batch,), generator=gen, dtype=torch.long)

    return PairedSlots(
        indices=list(indices),
        waveforms=waveforms,
        texts=texts,
        z_t=z_t,
        t=t,
        seed=seed,
        noise=noise,
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
