#!/usr/bin/env python3
"""EMA vs raw weight-convention audit (zero-GPU, PRE-DATA).

AudioLDM's upstream inference path (generate_sample -> ema_scope) evaluates with the EMA U-Net, not
the raw model.diffusion_model weights. LitEma freezes a param-name map at construction (BEFORE our
LoRA injection), so runtime ema_scope is invalid after PEFT (KeyError on lora_A/lora_B; base weights
renamed to ...base.weight never receive EMA). This audit quantifies, for each backbone, whether raw
and EMA are materially different, so we can pick ONE convention for the whole study.

For each checkpoint: map every raw U-Net tensor model.diffusion_model.X -> its EMA shadow
model_ema.(X with dots removed) (LitEma's exact mangling), and compare.
"""
import json, os, sys
import torch

CKPTS = [
    ("dense", "data/checkpoints/audioldm-m-full.ckpt"),
    ("p1_pruned", "data/checkpoints/l1_audioldm-m-full_p1.ckpt"),
    ("p1_recovered", "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt"),
]
OUT = "artifacts/icassp_gate0/ema_convention_audit.json"
RAWPFX = "model.diffusion_model."


def ema_key_for(raw_key):
    # raw_key = "model.diffusion_model.X.Y.weight"; LitEma name (rel to self.model) = "diffusion_model.X.Y.weight";
    # shadow name = that with all dots removed; ckpt key = "model_ema." + shadow.
    rel = raw_key[len("model."):]
    return "model_ema." + rel.replace(".", "")


def audit_ckpt(tag, path):
    sd = torch.load(path, map_location="cpu"); sd = sd.get("state_dict", sd)
    raw = {k: v for k, v in sd.items() if k.startswith(RAWPFX)}
    mangled = {k: ema_key_for(k) for k in raw}
    collisions = len(set(mangled.values())) != len(mangled)
    n_diff, n_equal, n_missing = 0, 0, 0
    max_abs, sum_rel, max_rel = 0.0, 0.0, 0.0
    n_cmp = 0
    for k, v in raw.items():
        ek = mangled[k]
        if ek not in sd:
            n_missing += 1; continue
        ev = sd[ek]
        if v.shape != ev.shape:
            n_missing += 1; continue
        d = (v.float() - ev.float()).abs()
        m = d.max().item()
        denom = v.float().abs().max().item() + 1e-12
        rel = m / denom
        if torch.equal(v, ev):
            n_equal += 1
        else:
            n_diff += 1
        max_abs = max(max_abs, m)
        max_rel = max(max_rel, rel)
        sum_rel += rel
        n_cmp += 1
    return {
        "n_raw_unet_tensors": len(raw),
        "mangling_collisions": collisions,
        "n_compared": n_cmp, "n_missing_ema": n_missing,
        "n_bit_identical_raw_eq_ema": n_equal,
        "n_differing": n_diff,
        "raw_vs_ema_max_abs_diff": max_abs,
        "raw_vs_ema_max_rel_diff": max_rel,
        "raw_vs_ema_mean_rel_diff": (sum_rel / n_cmp) if n_cmp else None,
        "materially_different": n_diff > 0 and max_rel > 1e-4,
    }


def main():
    R = {"upstream_inference_uses_ema": True,  # generate_sample -> ema_scope (ddpm.py:306-316)
         "litema_broken_after_peft": True,     # m_name2s_name frozen pre-injection; copy_to KeyErrors on LoRA
         "per_checkpoint": {}}
    for tag, path in CKPTS:
        if not os.path.exists(path):
            R["per_checkpoint"][tag] = {"MISSING": path}; continue
        R["per_checkpoint"][tag] = audit_ckpt(tag, path)
        print(tag, json.dumps(R["per_checkpoint"][tag]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(R, open(OUT, "w"), indent=1)
    print("\nWROTE", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
