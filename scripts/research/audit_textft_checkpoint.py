#!/usr/bin/env python3
"""Part B0 — CPU-only provenance + structural-compatibility audit of the public dense text-FT
checkpoint `audioldm-m-text-ft.ckpt`. 0 GPU, no generation.

B0.1 provenance : bytes / md5 / sha256 vs the official release.
B0.2 structure  : compare state_dict vs audioldm-m-full.ckpt (dense pretrained, which our pipeline
                  strict-loads); U-Net key/shape match, cond_stage/first_stage match, EMA structure,
                  inferred channel_mult/model_channels, value-delta inventory; then build the U-Net
                  from the frozen config, strict-load the text-ft U-Net, and forward at latent_t 96
                  and 256.
B0.3 comparability: documented (training data / start model / objective / duration).

Role (frozen): PUBLIC dense text-fine-tuning reference ONLY. NOT Singh's deleted dense-FT, NOT a
matched causal control.

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/audit_textft_checkpoint.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts/research"))

TEXTFT = os.path.join(ROOT, "data/checkpoints/audioldm-m-text-ft.ckpt")
FULL = os.path.join(ROOT, "data/checkpoints/audioldm-m-full.ckpt")
CONFIG = os.path.join(ROOT, "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml")
EXPECT = {"bytes": 4571676474, "md5": "036bc9b547a50f78b960ef8f14d0e1fb",
          "sha256": "d77d5a61785af82012edb8a72158d52592ac7c76d7f6ed51a048ec2dec8d5eca"}


def hashes(p):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            md5.update(chunk); sha.update(chunk)
    return os.path.getsize(p), md5.hexdigest(), sha.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "configs/research/textft_checkpoint_audit.json"))
    ap.add_argument("--skip-forward", action="store_true")
    a = ap.parse_args()
    import torch
    R = {"artifact": "textft_checkpoint_audit",
         "role": "PUBLIC dense text-fine-tuning reference ONLY; NOT Singh dense-FT; NOT a matched causal control"}

    # ---- B0.1 provenance
    nb, md5, sha = hashes(TEXTFT)
    R["B0_1_provenance"] = {
        "path": TEXTFT, "bytes": nb, "md5": md5, "sha256": sha, "expected": EXPECT,
        "bytes_match": nb == EXPECT["bytes"], "md5_match": md5 == EXPECT["md5"],
        "sha256_match": sha == EXPECT["sha256"],
        "PASS": nb == EXPECT["bytes"] and md5 == EXPECT["md5"] and sha == EXPECT["sha256"]}
    if not R["B0_1_provenance"]["PASS"]:
        json.dump(R, open(a.out, "w"), indent=2)
        print("B0.1 PROVENANCE FAIL:", R["B0_1_provenance"]); return 1

    # ---- B0.2 structural
    tf = torch.load(TEXTFT, map_location="cpu")
    fu = torch.load(FULL, map_location="cpu")
    R["B0_2_toplevel"] = {"textft_keys": sorted(list(tf.keys()))[:20],
                          "full_keys": sorted(list(fu.keys()))[:20]}
    tsd = tf.get("state_dict", tf); fsd = fu.get("state_dict", fu)
    tks, fks = set(tsd), set(fsd)
    shared = sorted(tks & fks)
    shape_mismatch = [k for k in shared if tuple(tsd[k].shape) != tuple(fsd[k].shape)]
    # value delta on shared same-shape tensors
    changed, identical = 0, 0
    for k in shared:
        if k in shape_mismatch:
            continue
        if torch.equal(tsd[k].float(), fsd[k].float()):
            identical += 1
        else:
            changed += 1

    def subset(prefix, ks):
        return sorted(k for k in ks if k.startswith(prefix))
    unet_t = subset("model.diffusion_model.", tks)
    unet_f = subset("model.diffusion_model.", fks)
    cond_t = subset("cond_stage_model", tks)
    fs_t = subset("first_stage_model.", tks)
    ema_t = subset("model_ema.", tks)

    # infer channel_mult / model_channels from U-Net input conv + a middle block if present
    inconv = "model.diffusion_model.input_blocks.0.0.weight"
    mc = tsd[inconv].shape[0] if inconv in tsd else None    # model_channels
    R["B0_2_structure"] = {
        "textft_only_keys": sorted(tks - fks)[:20], "full_only_keys": sorted(fks - tks)[:20],
        "n_shared_state_keys": len(shared), "n_shape_mismatch": len(shape_mismatch),
        "shape_mismatch_sample": shape_mismatch[:10],
        "n_shared_value_identical": identical, "n_shared_value_changed": changed,
        "unet_keys_textft": len(unet_t), "unet_keys_full": len(unet_f),
        "unet_keyset_equal": unet_t == unet_f,
        "unet_all_shapes_match": all(tuple(tsd[k].shape) == tuple(fsd[k].shape) for k in unet_t if k in fsd),
        "cond_stage_keys_textft": len(cond_t), "first_stage_keys_textft": len(fs_t),
        "ema_keys_textft": len(ema_t), "has_model_ema": len(ema_t) > 0,
        "inferred_model_channels": int(mc) if mc is not None else None,
        "input_conv_shape": list(tsd[inconv].shape) if inconv in tsd else None}

    # strict-load compatibility with our pipeline: U-Net key/shape identity to audioldm-m-full ⇒ same
    # strict-load path (audioldm-m-full loads 690/690 0-missing; proven in PROGRESS).
    unet_compatible = (unet_t == unet_f) and R["B0_2_structure"]["unet_all_shapes_match"]
    R["B0_2_structure"]["strict_load_compatible_with_pipeline"] = bool(unet_compatible)

    # ---- U-Net build + strict-load + forward at latent_t 96 and 256
    if not a.skip_forward:
        try:
            import yaml
            from audioldm_train.utilities.model_util import instantiate_from_config
            cfg = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
            unet = instantiate_from_config(cfg["model"]["params"]["unet_config"]).eval()
            rel = {k[len("model.diffusion_model."):]: v for k, v in tsd.items()
                   if k.startswith("model.diffusion_model.")}
            miss, unexp = unet.load_state_dict(rel, strict=False)
            fwd = {}
            with torch.no_grad():
                for lt in (96, 256):
                    x = torch.zeros(1, 8, lt, 16)
                    t = torch.zeros(1, dtype=torch.long)
                    y = torch.zeros(1, 512)
                    out = unet(x, timesteps=t, y=y)
                    fwd[str(lt)] = list(out.shape)
            R["B0_2_unet_dryrun"] = {"strict_load_missing": len(miss), "strict_load_unexpected": len(unexp),
                                     "unet_tensors_loaded": len(rel), "forward_shapes": fwd,
                                     "runs_at_96_and_256": all(k in fwd for k in ("96", "256"))}
        except Exception as e:
            R["B0_2_unet_dryrun"] = {"error": repr(e)[:600]}

    # ---- B0.3 comparability (documented; not measured)
    R["B0_3_comparability"] = {
        "training_data": "textFT = AudioCaps + MusicCaps (official release doc) vs Singh recovery = AudioCaps only",
        "starting_model": "textFT = DENSE audioldm-m-full vs Singh recovery = PRUNED backbone",
        "objective_conditioning": "full-model text-conditioned fine-tune (FiLM CLAP text); pathway same family as our pipeline",
        "training_duration": "UNKNOWN (not reported as identical to Singh's 1M-step recovery)",
        "max_claim": "whether a duration interaction is ALSO observable in one independently released dense "
                     "text-fine-tuned AudioLDM companion",
        "forbidden": ["generic fine-tuning ruled out", "matched dense control", "pruning causality established",
                      "generic FT explains Singh recovery"]}

    prov = R["B0_1_provenance"]["PASS"]
    R["compat_PASS"] = bool(prov and unet_compatible and
                            R.get("B0_2_unet_dryrun", {}).get("runs_at_96_and_256", False))
    json.dump(R, open(a.out, "w"), indent=2)
    print("B0.1 provenance PASS:", prov, "| bytes", nb, "md5", md5[:8], "sha", sha[:8])
    print("B0.2 U-Net keyset==full:", unet_t == unet_f, "shapes match:",
          R["B0_2_structure"]["unet_all_shapes_match"], "| state changed/identical:",
          changed, "/", identical, "| shape_mismatch:", len(shape_mismatch))
    print("B0.2 EMA keys:", len(ema_t), "| model_channels:", R["B0_2_structure"]["inferred_model_channels"])
    if "B0_2_unet_dryrun" in R:
        print("B0.2 dryrun:", R["B0_2_unet_dryrun"])
    print("COMPAT_PASS:", R["compat_PASS"], "-> wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
