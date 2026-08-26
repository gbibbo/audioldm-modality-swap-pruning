"""EMA weight-convention helpers (DECISION-V4-12).

AudioLDM's upstream inference path uses the EMA U-Net (use_ema=True by default; generate_sample ->
ema_scope). raw model.diffusion_model.* and the EMA shadow are materially different (audit: dense
mean-rel ~2.4%, recovered ~5.1%). The study therefore fixes ONE convention: materialize the EMA
(inference) weights into the ordinary U-Net BEFORE any LoRA injection, and disable runtime EMA
(which is structurally invalid after PEFT anyway: LitEma freezes a param-name map pre-injection).

LitEma name mangling: a U-Net parameter "input_blocks.0.0.weight" is stored, relative to self.model,
as "diffusion_model.input_blocks.0.0.weight"; the EMA shadow key drops ALL dots and is prefixed with
"model_ema." -> "model_ema.diffusion_modelinput_blocks00weight".
"""
from __future__ import annotations


def _ema_key_for_unet_param(unet_param_name: str) -> str:
    return "model_ema." + ("diffusion_model." + unet_param_name).replace(".", "")


def ema_unet_state_dict(full_sd: dict) -> dict:
    """Return {unet_relative_name: ema_tensor} for every raw U-Net param whose EMA shadow exists
    with a MATCHING shape. Keys are relative (loadable into a bare UNetModel). Also returns, via the
    second element, the list of raw params whose EMA is missing or shape-mismatched (unusable EMA)."""
    raw = {k[len("model.diffusion_model."):]: v for k, v in full_sd.items()
           if k.startswith("model.diffusion_model.")}
    out, unusable = {}, []
    for name, v in raw.items():
        ek = _ema_key_for_unet_param(name)
        ev = full_sd.get(ek)
        if ev is None or ev.shape != v.shape:
            unusable.append(name)
            continue
        out[name] = ev
    return out, unusable


def materialize_ema_into_unet(unet, full_sd, strict: bool = True):
    """Copy EMA shadows into `unet`'s parameters in place. Returns (n_copied, unusable_names).
    With strict=True, raises if any U-Net param has no usable EMA shadow."""
    import torch
    ema_sd, unusable = ema_unet_state_dict(full_sd)
    if strict and unusable:
        raise ValueError(f"{len(unusable)} U-Net params have no usable EMA shadow (e.g. {unusable[:3]}) "
                         f"— this checkpoint's stored EMA is structurally incomplete for this arch")
    own = dict(unet.named_parameters())
    n = 0
    with torch.no_grad():
        for name, ev in ema_sd.items():
            if name in own and own[name].shape == ev.shape:
                own[name].data.copy_(ev.data)
                n += 1
    return n, unusable
