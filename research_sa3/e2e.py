"""End-to-end generation + wav output (protocol section 9.1), via the OFFICIAL generate path.

We wrap our strictly-loaded ConditionedDiffusionModelWrapper in the upstream StableAudioModel and
call its `generate()` (production sampler, dist_shift, APG, SAME-S decode) rather than reimplement
it. Block-removed variants run inside research_sa3.blockskip.block_mask. Seed pairing: one derived
init seed per prompt, passed as generate(seed=...), seeds the global RNG so the initial noise AND
the ping-pong re-noising draws are identical across systems for that prompt.
"""
from __future__ import annotations
from typing import List, Optional
import os
import torch
from research_sa3.blockskip import block_mask


def wrap_model(model, model_config: dict, device: str, model_half: bool):
    """Wrap a built model in the upstream StableAudioModel (gives the official generate())."""
    from stable_audio_3.model import StableAudioModel
    model.use_lora = False
    model.lora_names = []
    return StableAudioModel(model, model_config, device, model_half)


@torch.inference_mode()
def generate_audio(sa, prompt: str, seconds_total: int, seed: int, steps: int = 8,
                   cfg_scale: float = 1.0, apg_scale: float = 1.0,
                   skip_blocks: Optional[List[int]] = None, callback=None,
                   return_latents: bool = False):
    """Generate one clip (batch 1). Returns the generate() result (audio (1,2,S) @44100, or latents)."""
    skip_blocks = skip_blocks or []
    kwargs = dict(prompt=prompt, duration=seconds_total, steps=steps, cfg_scale=cfg_scale,
                  apg_scale=apg_scale, seed=seed, batch_size=1, return_latents=return_latents)
    if callback is not None:
        kwargs["callback"] = callback
    with block_mask(sa.model, skip_blocks):
        return sa.generate(**kwargs)


def save_wav(audio: torch.Tensor, path: str, sample_rate: int = 44100):
    """audio: (1, 2, S) float in [-1,1] -> wav on disk (soundfile expects (S, C))."""
    import soundfile as sf
    a = audio.squeeze(0).to(torch.float32).cpu().numpy().T  # (S, 2)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, a, sample_rate)
    return path
