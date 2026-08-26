#!/usr/bin/env python3
"""Adapter checkpoint round-trip (CPU, no GPU): prove the saved adapter-only artifact reloads exactly.

Before spending ~1 cr on a 19,400-step run we must know the artifact it produces is faithfully
reloadable. This: (1) builds a dense U-Net + EMA + LoRA (gaussian, seeded), perturbs lora_B to
simulate training, saves adapter-only weights (the trainer's format); (2) builds a FRESH dense
U-Net + same LoRA arch, loads the adapter-only weights; (3) asserts lora_A/lora_B match bit-exactly,
delta_weight matches, and a forward at latent_t=96 reproduces the same output within tolerance.
Fast: U-Net only (no CLAP/VAE/vocoder), which is all the adapter round-trip concerns.
"""
import json, os, sys
import torch
from torch import nn
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
PREREG = "configs/research/icassp_gate0_prereg.yaml"
DENSE = "data/checkpoints/audioldm-m-full.ckpt"
OUT = "artifacts/icassp_gate0/adapter_roundtrip"


class Wrap(nn.Module):
    """Minimal container so audioldm_peft root_path 'model.diffusion_model' resolves to the U-Net."""
    def __init__(self, unet):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = unet


def build_unet_lora(config, dense_sd, pre, seed):
    import research_pruning.diagnostics.random_masks as rm
    from research_pruning.eval.ema_weights import materialize_ema_into_unet
    from audioldm_peft import setup_peft
    from scripts.research.gate0_trainer import make_peft_cfg  # reuse the exact recipe cfg
    torch.manual_seed(seed)
    unet = rm.build_pruned_unet(config, [1, 2, 3, 5]).float()
    rel = {k[len("model.diffusion_model."):]: v for k, v in dense_sd.items() if k.startswith("model.diffusion_model.")}
    unet.load_state_dict(rel, strict=True)
    materialize_ema_into_unet(unet, dense_sd, strict=True)
    w = Wrap(unet)
    setup_peft(w, make_peft_cfg(pre))
    return w


def adapter_state_dict(w):
    from audioldm_peft.layers import LoRALinear, LoRAConv2d
    sd = {}
    for name, m in w.named_modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)):
            sd[name + ".lora_A"] = m.lora_A.detach().clone()
            sd[name + ".lora_B"] = m.lora_B.detach().clone()
    return sd


def load_adapter(w, sd):
    own = dict(w.named_parameters())
    n = 0
    with torch.no_grad():
        for k, v in sd.items():
            own[k].data.copy_(v); n += 1
    return n


def fwd(w):
    C, T, F = 8, 96, 16
    x = torch.randn(1, C, T, F); t = torch.tensor([500]); y = torch.randn(1, 512)
    with torch.no_grad():
        return w.model.diffusion_model(x, timesteps=t, y=y, context_list=[], context_attn_mask_list=[])


def main():
    os.makedirs(OUT, exist_ok=True)
    _orig = torch.load
    torch.load = lambda *a, **k: (k.setdefault("map_location", "cpu"), _orig(*a, **k))[1]
    pre = yaml.safe_load(open(PREREG))["gate0"]
    dense_sd = _orig(DENSE, map_location="cpu"); dense_sd = dense_sd.get("state_dict", dense_sd)

    from audioldm_peft.layers import LoRALinear
    A = build_unet_lora(yaml.load(open(CONFIG), Loader=yaml.FullLoader), dense_sd, pre, seed=20260826)
    # simulate a trained adapter: perturb lora_B (init is 0)
    with torch.no_grad():
        for _, m in A.named_modules():
            if isinstance(m, LoRALinear):
                m.lora_B.add_(torch.randn_like(m.lora_B) * 0.02)
    sd = adapter_state_dict(A)
    path = os.path.join(OUT, "roundtrip_adapter.pt"); torch.save(sd, path)

    # fresh model, same arch, load adapter
    B = build_unet_lora(yaml.load(open(CONFIG), Loader=yaml.FullLoader), dense_sd, pre, seed=999)  # different init seed
    n_loaded = load_adapter(B, torch.load(path))

    # compare
    from audioldm_peft.layers import LoRALinear as LL
    max_w = 0.0; max_delta = 0.0
    Amods = {n: m for n, m in A.named_modules() if isinstance(m, LL)}
    Bmods = {n: m for n, m in B.named_modules() if isinstance(m, LL)}
    for n in Amods:
        max_w = max(max_w, (Amods[n].lora_A - Bmods[n].lora_A).abs().max().item(),
                    (Amods[n].lora_B - Bmods[n].lora_B).abs().max().item())
        max_delta = max(max_delta, (Amods[n].delta_weight() - Bmods[n].delta_weight()).abs().max().item())
    torch.manual_seed(7); oa = fwd(A)
    torch.manual_seed(7); ob = fwd(B)
    out_max = (oa - ob).abs().max().item()
    R = {"n_adapter_tensors": len(sd), "n_loaded": n_loaded,
         "max_abs_diff_lora_weights": max_w, "max_abs_diff_delta_weight": max_delta,
         "max_abs_diff_forward": out_max,
         "PASS": max_w == 0.0 and max_delta == 0.0 and out_max < 1e-5}
    json.dump(R, open(os.path.join(OUT, "roundtrip.json"), "w"), indent=1)
    print(json.dumps(R, indent=2))
    print("ADAPTER ROUND-TRIP", "PASS" if R["PASS"] else "FAIL")
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
