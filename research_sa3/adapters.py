"""Apply a TRAINED LoRA adapter onto the base/post model for the ecological regime A_eco (§3.4, §6).

A real held-out adapter `L` is loaded through the SAME official LoRAParametrization machinery the
synthetic probes use (`stable_audio_3.models.lora`), so everything downstream is identical to the
A_tan path — only the weights differ (trained, not random; normal strength, not κ-rescaled):

    δF(L)      = F_{P+L} - F_P                      (adapter strength 1 vs 0)
    δF^{-g}(L) = F^{-g}_{P+L_{-g}} - F^{-g}_P        (adapter restricted to surviving blocks; g removed)

Toggling is done with `research_sa3.probes.set_strength` / `restrict_to_surviving`, which scan every
LoRAParametrization regardless of how it was created. `A_eco(g;L)` then comes from
`research_sa3.metrics.a_eco`. We reuse `load_and_apply_loras` (the officially-tested inference load
path) rather than reimplement key remapping.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set
import re

_BLK = re.compile(r"\.layers\.(\d+)\.")


def _lora_layers(model):
    """(name, LoRAParametrization) for every parametrized weight (any adapter type)."""
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


def apply_trained_lora(model, ckpt_path: str, model_type: str = "diffusion_cond",
                       svd_bases_path: Optional[str] = None, verify: bool = True) -> dict:
    """Attach a trained `.safetensors`/`.ckpt` LoRA to `model` (a ConditionedDiffusionModelWrapper).

    Uses the upstream two-pass loader (resolves adapter type, attaches parametrizations to the
    config's `include` layers, loads weights strict=False onto model.model + model.conditioner).
    Returns a report; raises if nothing attached or the trained weights did not actually load."""
    import torch
    from stable_audio_3.models.lora.loader import load_and_apply_loras
    from stable_audio_3.models.lora.utils import load_lora_checkpoint

    _, config = load_lora_checkpoint(ckpt_path)
    names = load_and_apply_loras(model, [ckpt_path], model_type, svd_bases_path=svd_bases_path)
    layers = _lora_layers(model)
    if not layers:
        raise RuntimeError(f"no LoRA parametrizations attached from {ckpt_path} "
                           f"(config include={config.get('include')!r} matched nothing)")
    blocks = adapter_blocks(model)
    # correctness guard: a TRAINED adapter has non-zero lora_B / M_xs. A failed load leaves the
    # freshly-added B at its zero-init, so a zero max-norm means the weights never landed.
    if verify:
        max_norm = 0.0
        with torch.no_grad():
            for _, p in layers:
                for attr in ("lora_B", "M_xs"):
                    t = getattr(p, attr, None)
                    if t is not None:
                        max_norm = max(max_norm, float(t.detach().float().norm()))
        if not (max_norm > 0):
            raise RuntimeError(f"trained LoRA {ckpt_path} loaded but every lora_B/M_xs is zero -- "
                               f"the checkpoint keys did not map onto the attached layers")
    return {
        "names": names,
        "adapter_type": config.get("adapter_type", "lora"),
        "rank": config.get("rank"),
        "include": config.get("include"),
        "exclude": config.get("exclude"),
        "n_layers": len(layers),
        "blocks": sorted(blocks),
        "config": config,
    }


def adapter_blocks(model) -> Set[int]:
    """The transformer-block indices the attached adapter actually touches."""
    blocks = set()
    for name, _ in _lora_layers(model):
        m = _BLK.search(name)
        if m:
            blocks.add(int(m.group(1)))
    return blocks


def remove_adapter(model):
    """Strip all LoRA parametrizations (mirror of probes.remove_probe for the wrapper)."""
    from stable_audio_3.models.lora.model import remove_lora
    for sub in (getattr(model, "model", None), getattr(model, "conditioner", None), model):
        if sub is None:
            continue
        try:
            remove_lora(sub)
        except Exception:
            pass
    if hasattr(model, "use_lora"):
        model.use_lora = False
        model.lora_names = []
