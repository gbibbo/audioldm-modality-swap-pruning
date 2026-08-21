"""Raw and deployment velocity fields + padding-masked norms (protocol section 1.4, 3.1-3.2).

We call the REAL upstream code, never reimplement it:
  * raw field   F(x,t,c) = DiffusionTransformer._forward(x, t, cross_attn_cond, global_embed, ...)
  * deploy field F^dep    = DiffusionTransformer.forward(..., cfg_scale=7.0, apg_scale=1.0)  (production APG)
Block removal F^{-g} = the same call inside research_sa3.blockskip.block_mask(model, [g]).

Model handle chain: wrapper(ConditionedDiffusionModelWrapper).model = DiTWrapper; .model = DiffusionTransformer.
Conditioning is computed once per prompt (T5Gemma tokens + seconds_total embed) and cached; only the DiT
is re-evaluated. To match production, cross_attn_cond_mask is nulled (forward() does the same, dit.py).
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch


def get_dit(model):
    """Return the DiffusionTransformer backbone from the wrapper (or pass-through if already it)."""
    m = model
    for _ in range(3):
        if m.__class__.__name__ == "DiffusionTransformer":
            return m
        m = getattr(m, "model", None)
        if m is None:
            break
    raise AttributeError("could not locate DiffusionTransformer under model.model.model")


@torch.no_grad()
def prepare_conditioning(model, caption: str, seconds_total: int, device: str,
                         latent_len: int, dtype=None) -> Dict:
    """Compute + cache the DiT conditioning for one prompt. Returns kwargs already named for the DiT
    (`cross_attn_cond`, `cross_attn_cond_mask`, `global_embed`, `local_add_cond`).

    Injects the NO-INPAINTING default local conditioning exactly as the production generate path
    (model.py:293-296): inpaint_mask = zeros(1,1,T), inpaint_masked_input = zeros(1,256,T),
    concatenated by get_conditioning_inputs into the 257-ch local_add_cond. `latent_len` (T) must
    equal the sequence length of the x states the field is evaluated on."""
    meta = [{"prompt": caption, "seconds_total": seconds_total}]
    cond_tensors = model.conditioner(meta, device)
    io_ch = getattr(model, "io_channels", 256)
    cond_tensors["inpaint_mask"] = [torch.zeros((1, 1, latent_len), device=device)]
    cond_tensors["inpaint_masked_input"] = [torch.zeros((1, io_ch, latent_len), device=device)]
    ci = model.get_conditioning_inputs(cond_tensors)
    cc = {
        "cross_attn_cond": ci["cross_attn_cond"],
        "cross_attn_cond_mask": None,   # production forward() nulls this (dit.py); keep raw==deploy consistent
        "global_embed": ci["global_cond"],
        "local_add_cond": ci.get("local_add_cond", None),
    }
    if dtype is not None:
        for k in ("cross_attn_cond", "global_embed", "local_add_cond"):
            if cc[k] is not None:
                cc[k] = cc[k].to(dtype)
    return cc


@torch.no_grad()
def raw_field(model, x: torch.Tensor, t: torch.Tensor, cc: Dict) -> torch.Tensor:
    """F(x,t,c): raw conditional velocity via DiffusionTransformer._forward."""
    dit = get_dit(model)
    return dit._forward(
        x, t,
        cross_attn_cond=cc["cross_attn_cond"],
        cross_attn_cond_mask=cc["cross_attn_cond_mask"],
        global_embed=cc["global_embed"],
        local_add_cond=cc["local_add_cond"],
    )


@torch.no_grad()
def deploy_field(model, x: torch.Tensor, t: torch.Tensor, cc: Dict,
                 cfg_scale: float = 7.0, apg_scale: float = 1.0) -> torch.Tensor:
    """F^dep: production CFG/APG velocity via DiffusionTransformer.forward (default full APG)."""
    dit = get_dit(model)
    return dit.forward(
        x, t,
        cross_attn_cond=cc["cross_attn_cond"],
        cross_attn_cond_mask=cc["cross_attn_cond_mask"],
        global_embed=cc["global_embed"],
        local_add_cond=cc["local_add_cond"],
        cfg_scale=cfg_scale, apg_scale=apg_scale,
    )


def state_sq_norm(v: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-state squared L2 over the latent (C x T), divided by valid T (protocol ||.||^2_S per state).
    v: (B, C, T). padding_mask: (B, T) True=valid. Returns (B,) per-state scalar (float64)."""
    v = v.to(torch.float64)
    if padding_mask is None:
        Tval = v.shape[-1]
        return v.pow(2).sum(dim=(-1, -2)) / Tval
    m = padding_mask.to(torch.float64).unsqueeze(1)  # (B,1,T)
    Tval = padding_mask.to(torch.float64).sum(dim=-1).clamp(min=1.0)  # (B,)
    return (v.pow(2) * m).sum(dim=(-1, -2)) / Tval


def diff_sq_norm(a: torch.Tensor, b: torch.Tensor, padding_mask=None) -> torch.Tensor:
    """Per-state ||a-b||^2_S."""
    return state_sq_norm(a - b, padding_mask)
