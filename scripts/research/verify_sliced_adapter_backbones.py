#!/usr/bin/env python3
"""Item 7: prove the SAME sliced-adapter bytes strict-load onto BOTH (1,2,3,1) backbones
(p1_pruned_ema_reconstructed and p1_recovered) and that they change each real backbone's
forward deterministically, while LoRA injection alone (lora_B=0) is a forward no-op.

CPU only. Read-only w.r.t. weights.
"""
import argparse, hashlib, json, os, sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import torch, yaml

import research_pruning.diagnostics.random_masks as rm
from research_pruning.eval.ema_weights import materialize_ema_into_unet, ema_unet_state_dict
from research_pruning.sliced_adapter import qv_linear_shapes, ADAPTER_PREFIX
from audioldm_peft import PeftConfig, setup_peft
from audioldm_peft.layers import LoRALinear

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
DENSE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
RECOV_CKPT = "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt"
PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
SLICED = "artifacts/icassp_gate0/sliced_adapter/gate0_sliced_adapter_1_2_3_1.pt"
PFX = "model.diffusion_model."


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_reconstructed(config):
    dsd = torch.load(DENSE_CKPT, map_location="cpu"); dsd = dsd.get("state_dict", dsd)
    dense_ema_base, _ = ema_unet_state_dict(dsd)
    l1 = rm.load_l1_ranking(PKL)
    return rm.materialize(dense_ema_base, l1, config, channel_mult=[1, 2, 3, 1]).float().eval()


def build_recovered(config):
    rsd = torch.load(RECOV_CKPT, map_location="cpu"); rsd = rsd.get("state_dict", rsd)
    unet = rm.build_pruned_unet(config, [1, 2, 3, 1]).float()
    rel = {k[len(PFX):]: v for k, v in rsd.items() if k.startswith(PFX)}
    unet.load_state_dict(rel, strict=True)
    materialize_ema_into_unet(unet, rsd, strict=True)
    return unet.eval()


def inject(unet, rank=8, alpha=16):
    cfg = PeftConfig(rank=rank, alpha=alpha, dropout=0.0, target_linear=True, target_conv2d=False,
                     train_bias=False, train_groupnorm_affine=False, train_layernorm_affine=False,
                     root_path="", include_name_substrings=("to_q", "to_v"),
                     init_lora_weights="gaussian")
    setup_peft(unet, cfg)
    return unet


def load_sliced(unet, sliced_sd):
    """Load prefixed sliced-adapter keys (model.diffusion_model.<rel>.lora_A/B) into the bare
    U-Net's LoRALinear modules (relative names). Strict: every module must be found + shape-match."""
    mods = {n: m for n, m in unet.named_modules() if isinstance(m, LoRALinear)}
    loaded = 0
    with torch.no_grad():
        for name, m in mods.items():
            ka, kb = PFX + name + ".lora_A", PFX + name + ".lora_B"
            if ka not in sliced_sd or kb not in sliced_sd:
                raise SystemExit(f"sliced adapter missing keys for {name}")
            a, b = sliced_sd[ka], sliced_sd[kb]
            if tuple(a.shape) != tuple(m.lora_A.shape) or tuple(b.shape) != tuple(m.lora_B.shape):
                raise SystemExit(f"shape mismatch at {name}: adapter {tuple(a.shape)}/{tuple(b.shape)} "
                                 f"vs module {tuple(m.lora_A.shape)}/{tuple(m.lora_B.shape)}")
            m.lora_A.copy_(a); m.lora_B.copy_(b)
            loaded += 1
    if loaded != len(mods):
        raise SystemExit(f"loaded {loaded}/{len(mods)} sliced modules")
    return loaded


def unet_forward(unet, seed=12345):
    # M-Full conditioning is FiLM only (CLAP 512 as y); no crossattn context (context_list=[]),
    # mirroring LatentDiffusion.apply_model for film_clap_cond1.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 8, 96, 16, generator=g)
    t = torch.tensor([10], dtype=torch.long)
    y = torch.randn(1, 512, generator=g)
    with torch.no_grad():
        return unet(x, timesteps=t, y=y, context_list=[], context_attn_mask_list=[])


def check_backbone(name, unet, sliced_sd):
    # bare forward
    out_bare = unet_forward(unet)
    # inject LoRA (lora_B=0) -> forward must be a no-op
    inject(unet)
    out_off = unet_forward(unet)
    off_noop = torch.allclose(out_bare, out_off, atol=0, rtol=0)
    off_maxdiff = float((out_bare - out_off).abs().max())
    # load sliced adapter -> forward must change, deterministically
    load_sliced(unet, sliced_sd)
    out_on1 = unet_forward(unet)
    out_on2 = unet_forward(unet)
    on_changes = not torch.allclose(out_off, out_on1, atol=1e-8)
    on_deterministic = torch.equal(out_on1, out_on2)
    on_maxdiff = float((out_on1 - out_off).abs().max())
    res = {"backbone": name, "off_is_noop": bool(off_noop), "off_maxdiff": off_maxdiff,
           "on_changes_forward": bool(on_changes), "on_deterministic": bool(on_deterministic),
           "on_vs_off_maxdiff": on_maxdiff}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/icassp_gate0/sliced_adapter/backbone_verify.json")
    args = ap.parse_args()
    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    sliced_sd = torch.load(SLICED, map_location="cpu")
    sliced_sha = sha_file(SLICED)

    recon = build_reconstructed(config)
    recov = build_recovered(config)

    # --- shape/name walk: identical names, dims, ordering ---
    s_recon = qv_linear_shapes(recon)
    s_recov = qv_linear_shapes(recov)
    order_recon = [n for n, m in recon.named_modules() if n.endswith(("to_q", "to_v")) and isinstance(m, torch.nn.Linear)]
    order_recov = [n for n, m in recov.named_modules() if n.endswith(("to_q", "to_v")) and isinstance(m, torch.nn.Linear)]
    walk_ok = (s_recon == s_recov) and (order_recon == order_recov) and len(order_recon) == 64
    if not walk_ok:
        raise SystemExit(f"shape/name walk mismatch: shapes_equal={s_recon==s_recov} "
                         f"order_equal={order_recon==order_recov} n={len(order_recon)}")

    r1 = check_backbone("p1_pruned_ema_reconstructed", recon, sliced_sd)
    r2 = check_backbone("p1_recovered", recov, sliced_sd)

    ok = (walk_ok
          and r1["off_is_noop"] and r1["on_changes_forward"] and r1["on_deterministic"]
          and r2["off_is_noop"] and r2["on_changes_forward"] and r2["on_deterministic"])
    report = {
        "sliced_adapter_sha256": sliced_sha,
        "same_sliced_sha_both_conditions": sliced_sha,   # one artifact loaded into both
        "shape_name_walk": {"n_qv": len(order_recon), "shapes_identical": s_recon == s_recov,
                            "order_identical": order_recon == order_recov},
        "reconstructed": r1, "recovered": r2, "PASS": bool(ok),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("SLICED-ADAPTER-BACKBONES", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
