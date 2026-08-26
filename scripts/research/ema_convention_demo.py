#!/usr/bin/env python3
"""CPU demonstration of the proposed EMA weight convention (DECISION-V4-12, PRE-DATA, no GPU).

Proves, without training:
  (1) dense EMA materializes cleanly into the base U-Net and gives MATERIALLY different output than raw;
  (2) p1_pruned's STORED EMA is unusable (238 tensors dense-shaped) -> we must derive the pruned
      inference backbone by pruning the DENSE EMA (mechanically identical selection, EMA weights);
  (3) p1_recovered's own EMA materializes cleanly into the (1,2,3,1) arch and differs from raw.
Everything forwards at latent_t=96 (the Gate-0 gen length).
"""
import json, os, sys
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
torch.manual_seed(20260826)

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
DENSE = "data/checkpoints/audioldm-m-full.ckpt"
PRUNED = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
RECOV = "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt"
PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
OUT = "artifacts/icassp_gate0/ema_convention_demo.json"

import research_pruning.diagnostics.random_masks as rm
from research_pruning.eval.ema_weights import ema_unet_state_dict, materialize_ema_into_unet

config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
C = config["model"]["params"]["unet_config"]["params"]["in_channels"]
Fd = config["model"]["params"]["latent_f_size"]
FILM = config["model"]["params"]["unet_config"]["params"]["extra_film_condition_dim"]


def fwd(unet):
    x = torch.randn(1, C, 96, Fd); t = torch.tensor([500]); y = torch.randn(1, FILM)
    with torch.no_grad():
        return unet(x, timesteps=t, y=y, context_list=[], context_attn_mask_list=[])


def load_full(path):
    sd = torch.load(path, map_location="cpu"); return sd.get("state_dict", sd)


R = {}
# ---------------- (1) dense: raw vs EMA ----------------
dense_sd = load_full(DENSE)
base = rm.build_pruned_unet(config, [1, 2, 3, 5]).eval()
raw_unet_sd = {k[len("model.diffusion_model."):]: v for k, v in dense_sd.items()
               if k.startswith("model.diffusion_model.")}
base.load_state_dict(raw_unet_sd, strict=True)
out_raw = fwd(base)
n_copied, unusable = materialize_ema_into_unet(base, dense_sd, strict=True)
out_ema = fwd(base)
R["dense"] = {
    "ema_copied": n_copied, "ema_unusable": len(unusable),
    "materialize_clean": len(unusable) == 0,
    "raw_vs_ema_output_max_abs": float((out_raw - out_ema).abs().max()),
    "outputs_differ": bool((out_raw - out_ema).abs().max() > 1e-4),
    "forward_ok": list(out_ema.shape) == [1, C, 96, Fd],
}

# ---------------- (2) p1_pruned stored EMA unusable + prune dense EMA ----------------
pruned_sd = load_full(PRUNED)
_, pruned_unusable = ema_unet_state_dict(pruned_sd)
l1 = rm.load_l1_ranking(PKL)
# dense EMA as a RELATIVE-key base_sd (materialize strips no prefix), then prune it
# (the same (1,2,3,1) L1 selection Arshdeep used, applied to the EMA weights)
dense_ema_base, dense_ema_unusable = ema_unet_state_dict(dense_sd)
pruned_from_ema = rm.materialize(dense_ema_base, l1, config, channel_mult=[1, 2, 3, 1]).eval()
out_pruned_ema = fwd(pruned_from_ema)
R["p1_pruned"] = {
    "stored_ema_unusable_tensors": len(pruned_unusable),
    "stored_ema_is_broken": len(pruned_unusable) > 0,
    "resolution": "prune the DENSE EMA (same (1,2,3,1) L1 selection, EMA weights)",
    "pruned_from_dense_ema_forward_ok": list(out_pruned_ema.shape) == [1, C, 96, Fd],
}

# ---------------- (3) p1_recovered: own EMA materializes into pruned arch ----------------
recov_sd = load_full(RECOV)
prec = rm.build_pruned_unet(config, [1, 2, 3, 1]).eval()
recov_raw = {k[len("model.diffusion_model."):]: v for k, v in recov_sd.items()
             if k.startswith("model.diffusion_model.")}
prec.load_state_dict(recov_raw, strict=True)
out_rraw = fwd(prec)
n_rc, r_unus = materialize_ema_into_unet(prec, recov_sd, strict=True)
out_rema = fwd(prec)
R["p1_recovered"] = {
    "ema_copied": n_rc, "ema_unusable": len(r_unus), "materialize_clean": len(r_unus) == 0,
    "raw_vs_ema_output_max_abs": float((out_rraw - out_rema).abs().max()),
    "outputs_differ": bool((out_rraw - out_rema).abs().max() > 1e-4),
    "forward_ok": list(out_rema.shape) == [1, C, 96, Fd],
}

R["PASS"] = (R["dense"]["materialize_clean"] and R["dense"]["outputs_differ"] and R["dense"]["forward_ok"]
             and R["p1_pruned"]["stored_ema_is_broken"] and R["p1_pruned"]["pruned_from_dense_ema_forward_ok"]
             and R["p1_recovered"]["materialize_clean"] and R["p1_recovered"]["forward_ok"])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps(R, indent=2))
print("\nEMA-CONVENTION DEMO", "PASS" if R["PASS"] else "FAIL")
sys.exit(0 if R["PASS"] else 1)
