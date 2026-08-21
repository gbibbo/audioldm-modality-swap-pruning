"""Common states S_traj / S_traj^B via sampler-callback capture (protocol section 1.5).

S_traj (primary): the states (x_i, tau_i) visited by the DENSE POST ping-pong sampler on the panel
(8 steps, cfg 1.0, apg 1.0). S_traj^B: the DENSE BASE Euler trajectory (50 steps, cfg 7, apg 1) --
used only for D_B^deploy. Both captured through generate(callback=...). x_i is the pre-step latent
at level tau_i; `denoised` is also recorded. Seed-paired via the per-prompt init seed.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
from research_sa3.e2e import generate_audio


def capture_trajectory(sa, prompt: str, seconds_total: int, seed: int, steps: int,
                       cfg_scale: float, apg_scale: float, skip_blocks=None):
    """Return {'states': [(tau_i, x_i)], 'denoised': [d_i], 'final': x_final}. return_latents=True
    (no decode). x_i/d_i are detached CPU tensors (1,256,T)."""
    caps = {"states": [], "denoised": []}

    def cb(info):
        t = info["t"]
        tau = float(t.reshape(-1)[0].item()) if torch.is_tensor(t) else float(t)
        caps["states"].append((tau, info["x"].detach().to("cpu").float().clone()))
        caps["denoised"].append(info["denoised"].detach().to("cpu").float().clone())

    final = generate_audio(sa, prompt, seconds_total, seed, steps=steps, cfg_scale=cfg_scale,
                           apg_scale=apg_scale, skip_blocks=skip_blocks, callback=cb,
                           return_latents=True)
    caps["final"] = final.detach().to("cpu").float()
    return caps
