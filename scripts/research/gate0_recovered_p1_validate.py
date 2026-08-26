#!/usr/bin/env python3
"""CPU validation of Arshdeep's PUBLIC recovered (1,2,3,1) checkpoint (Zenodo 21977996).

Zero-GPU: (1) inspect structure (weights-only vs optimizer/EMA state), (2) strict-load the recovered
(1,2,3,1) U-Net into our materializer architecture, (3) CPU-forward at latent_t=96 (the Gate-0 gen
length). Confirms the recovered backbone is usable as an in-scope phenomenon-falsifier system.

md5 already verified on download: cfb7ca3f8c712850f5a4bfe2162f5d1c.
"""
import json, os, sys
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
CKPT = "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt"
OUT = "artifacts/icassp_gate0/recovered_p1_validation.json"
UNET_PREFIX = "model.diffusion_model."

R = {}
ck = torch.load(CKPT, map_location="cpu")

# --- (1) structure ---
top = list(ck.keys()) if isinstance(ck, dict) else ["<not a dict>"]
sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
has_opt = isinstance(ck, dict) and any(k in ck for k in ("optimizer_states", "optimizer", "optimizers"))
has_ema = isinstance(ck, dict) and any("ema" in str(k).lower() for k in ck.keys())
gstep = ck.get("global_step") if isinstance(ck, dict) else None
import collections
pref = collections.Counter(k.split(".")[0] for k in sd)
R["structure"] = {
    "top_level_keys": top[:20],
    "has_optimizer_state": bool(has_opt),
    "has_ema_key_at_top": bool(has_ema),
    "global_step": gstep,
    "state_dict_prefixes": dict(pref),
    "n_state_dict_tensors": len(sd),
    "weights_only_guess": (not has_opt),
}

# --- (2) strict-load recovered U-Net into the (1,2,3,1) materializer arch ---
config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
from research_pruning.diagnostics.random_masks import build_pruned_unet
unet = build_pruned_unet(config, channel_mult=[1, 2, 3, 1]).eval()
unet_sd = {k[len(UNET_PREFIX):]: v for k, v in sd.items() if k.startswith(UNET_PREFIX)}
miss, unexp = unet.load_state_dict(unet_sd, strict=False)
R["unet_load"] = {
    "n_unet_tensors_in_ckpt": len(unet_sd),
    "missing": len(miss), "unexpected": len(unexp),
    "strict_ok": len(miss) == 0 and len(unexp) == 0,
    "sample_missing": miss[:5], "sample_unexpected": unexp[:5],
}

# is it actually finetuned (different from the pruned-only p1)? compare a tensor if p1 present
PRUNED = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
if os.path.exists(PRUNED):
    p = torch.load(PRUNED, map_location="cpu"); psd = p.get("state_dict", p)
    common = [k for k in unet_sd if (UNET_PREFIX + k) in psd]
    if common:
        import torch as _t
        diffs = sum(1 for k in common[:200] if not _t.equal(unet_sd[k], psd[UNET_PREFIX + k]))
        R["vs_pruned_only_p1"] = {"compared": min(200, len(common)), "tensors_differing": diffs,
                                  "is_finetuned_not_identical": diffs > 0}

# --- (3) CPU forward at latent_t=96 ---
C = config["model"]["params"]["unet_config"]["params"]["in_channels"]
F = config["model"]["params"]["latent_f_size"]
FILM = config["model"]["params"]["unet_config"]["params"]["extra_film_condition_dim"]
x = torch.randn(1, C, 96, F); t = torch.tensor([500]); y = torch.randn(1, FILM)
with torch.no_grad():
    out = unet(x, timesteps=t, y=y, context_list=[], context_attn_mask_list=[])
R["forward_latent_t_96"] = {"in_shape": [1, C, 96, F], "out_shape": list(out.shape),
                            "ok": list(out.shape) == [1, C, 96, F]}

R["VALID"] = R["unet_load"]["strict_ok"] and R["forward_latent_t_96"]["ok"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(R, open(OUT, "w"), indent=1, default=str)
print(json.dumps(R, indent=2, default=str))
print("\nRECOVERED-P1 VALIDATION", "PASS" if R["VALID"] else "FAIL")
sys.exit(0 if R["VALID"] else 1)
