"""Low-rank probes U_gen / U_xs for the tangent adaptability regime A_tan (protocol section 3.4).

Probes go through the OFFICIAL LoRAParametrization (repo `add_lora`), so a probe is exactly a
LoRA adapter. Two families:
  * U_gen : adapter_type "lora", rank 16; A Kaiming (repo init), B ~ N(0,1) (repo zero-inits B,
            a zero probe is inert), rescaled per layer to ||dW||_F/||W0||_F = kappa.
  * U_xs  : adapter_type "lora-xs", U,V from the BASE checkpoint's svd_bases.pt (frozen), M ~ N(0,1),
            rescaled to kappa.
delta F(u) = F_{P+u} - F_P is obtained by toggling lora_strength: strength=1 -> F_{P+u};
strength=0 -> F_P exactly (lora_forward returns W+0=W bit-exactly). Restriction u_{-g} zeroes the
strength on block g's layers, so the probe acts only on surviving blocks. kappa is never matched
to a real adapter (that is the ecological regime A_eco).
"""
from __future__ import annotations
from functools import partial
from typing import List, Tuple
import re
import torch
import torch.nn as nn

_BLK = re.compile(r"\.layers\.(\d+)\.")


def _probe_params(model) -> List[Tuple[str, object]]:
    """(name, LoRAParametrization) for every parametrized weight under transformer.layers."""
    from stable_audio_3.models.lora.model import LoRAParametrization
    out = []
    for name, mod in model.named_modules():
        plist = getattr(getattr(mod, "parametrizations", None), "weight", None)
        if plist is None:
            continue
        for p in plist:
            if isinstance(p, LoRAParametrization):
                out.append((name, p))
    return out


def build_probe(model, family: str, kappa: float, rank: int = 16, seed: int = 0,
                svd_bases: dict = None, include=("transformer.layers",)):
    """Inject a probe adapter across LoRA-eligible layers; randomize + rescale to kappa.
    Returns the list of (name, parametrization). Call remove_probe() to strip it."""
    from stable_audio_3.models.lora.model import add_lora, LoRAParametrization
    adapter = "lora" if family == "U_gen" else "lora-xs"
    cfg = {
        nn.Linear: {"weight": partial(LoRAParametrization.from_linear, rank=rank, adapter_type=adapter)},
        nn.Conv1d: {"weight": partial(LoRAParametrization.from_conv1d, rank=rank, adapter_type=adapter)},
    }
    add_lora(model, lora_config=cfg, include=list(include), svd_bases=svd_bases)
    params = _probe_params(model)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in params:
        # original weight lives on the owning module: parametrizations.weight.original
        owner = _owner_of(model, name)
        W0 = owner.parametrizations.weight.original
        with torch.no_grad():
            if family == "U_gen":
                p.lora_B.copy_(torch.randn(p.lora_B.shape, generator=g).to(p.lora_B.dtype))
                raw = p.scaling * torch.matmul(*p.swap((p.lora_B, p.lora_A)))
            else:
                p.M_xs.copy_(torch.randn(p.M_xs.shape, generator=g).to(p.M_xs.dtype))
                raw = p.scaling * (p.U @ p.M_xs.to(p.U.dtype) @ p.V.T)
            factor = float(kappa * W0.norm() / (raw.norm() + 1e-12))
            if family == "U_gen":
                p.lora_B.mul_(factor)
            else:
                p.M_xs.mul_(factor)
    return params


def _owner_of(model, name):
    return model.get_submodule(name)


def set_strength(model, value: float, only_block: int = None):
    """Set lora_strength on all probe params (or only those in block `only_block`)."""
    for name, p in _probe_params(model):
        if only_block is not None:
            m = _BLK.search(name)
            if not (m and int(m.group(1)) == only_block):
                continue
        p.lora_strength.fill_(float(value))


def restrict_to_surviving(model, removed_block: int):
    """u_{-g}: zero the probe on block `removed_block`, strength 1 elsewhere."""
    set_strength(model, 1.0)
    set_strength(model, 0.0, only_block=removed_block)


def probe_scale(model, factor: float):
    """Scale the whole probe by `factor` (for the linearity check ||dF(2u)||/||dF(u)||)."""
    for name, p in _probe_params(model):
        with torch.no_grad():
            if p.adapter_type == "lora":
                p.lora_B.mul_(factor)
            elif p.adapter_type == "lora-xs":
                p.M_xs.mul_(factor)


def per_layer_kappa(model):
    """Measured ||dW||_F/||W0||_F per probed layer (strength must be 1)."""
    out = {}
    for name, p in _probe_params(model):
        owner = _owner_of(model, name)
        W0 = owner.parametrizations.weight.original
        with torch.no_grad():
            if p.adapter_type == "lora":
                dW = p.scaling * p.lora_strength * torch.matmul(*p.swap((p.lora_B, p.lora_A)))
            else:
                dW = p.scaling * p.lora_strength * (p.U @ p.M_xs.to(p.U.dtype) @ p.V.T)
            out[name] = float(dW.norm() / (W0.norm() + 1e-12))
    return out


def remove_probe(model):
    from stable_audio_3.models.lora.model import remove_lora
    remove_lora(model)
