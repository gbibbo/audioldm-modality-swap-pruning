#!/usr/bin/env python3
"""Build the ONE deterministic (1,2,3,1) sliced-adapter artifact from the dense Gate-0 LoRA.

No learned mapping, no adapter data, no retraining: the dense-trained LoRA factors are sliced by
the SAME positional kept-index selection the L1 materializer applies to the attention weights.
Emits per-module audit JSON (64/64, 0 missing, 0 unexpected), verifies the float64 restricted-dW
identity, optionally re-verifies the positional convention against the published p1 checkpoint,
and writes the sliced adapter .pt + meta with SHA256.

CPU only. Run:
  OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/build_sliced_adapter.py --verify-p1
"""
import argparse, hashlib, json, os, sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import torch, yaml

import research_pruning.diagnostics.random_masks as rm
from research_pruning.sliced_adapter import (
    qv_linear_shapes, build_sliced_adapter, summarize_audit, ADAPTER_PREFIX)

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
DENSE_ADAPTER = "artifacts/icassp_gate0/gate0_adapter/gate0_adapter.pt"
DENSE_ADAPTER_META = "artifacts/icassp_gate0/gate0_adapter/gate0_adapter_meta.json"
DENSE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
P1_CKPT = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
PFX = "model.diffusion_model."
OUT_DIR = "artifacts/icassp_gate0/sliced_adapter"


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_p1_positional(dense_unet, pruned_unet):
    """Confirm the published p1 to_q/to_v weights are EXACT positional slices of dense."""
    def unet_sd(ckpt):
        obj = torch.load(ckpt, map_location="cpu")
        sd = obj.get("state_dict", obj)
        return {k[len(PFX):]: v for k, v in sd.items() if k.startswith(PFX)}
    dsd, p1 = unet_sd(DENSE_CKPT), unet_sd(P1_CKPT)
    ps = qv_linear_shapes(pruned_unet)
    checked = nonpos = 0
    for rel in sorted(ps):
        wk = rel + ".weight"
        dw, pw = dsd[wk], p1[wk]
        po, pi = pw.shape
        checked += 1
        if not torch.equal(dw[:po, :pi], pw):
            nonpos += 1
    return {"checked": checked, "positional_exact": checked - nonpos, "non_positional": nonpos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-adapter", default=DENSE_ADAPTER)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--verify-p1", action="store_true",
                    help="re-verify positional pruning against the published l1_p1 checkpoint")
    args = ap.parse_args()

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    dense_unet = rm.build_pruned_unet(config, [1, 2, 3, 5])
    pruned_unet = rm.build_pruned_unet(config, [1, 2, 3, 1])
    dense_shapes = qv_linear_shapes(dense_unet)
    pruned_shapes = qv_linear_shapes(pruned_unet)

    dense_sd = torch.load(args.dense_adapter, map_location="cpu")
    dense_meta = json.load(open(DENSE_ADAPTER_META)) if os.path.exists(DENSE_ADAPTER_META) else {}
    dense_sha = dense_meta.get("adapter_sha256") or sha_file(args.dense_adapter)

    # --- accounting: 64/64, 0 missing, 0 unexpected ---
    adapter_modules = {k.rsplit(".", 1)[0] for k in dense_sd}
    adapter_rel = {m[len(PFX):] if m.startswith(PFX) else m for m in adapter_modules}
    unet_targets = set(pruned_shapes)                       # == set(dense_shapes)
    missing = sorted(unet_targets - adapter_rel)            # U-Net targets with no adapter factor
    unexpected = sorted(adapter_rel - unet_targets)         # adapter factors with no U-Net target
    if missing or unexpected:
        print(json.dumps({"missing": missing, "unexpected": unexpected}, indent=2))
        raise SystemExit(f"ACCOUNTING FAIL: {len(missing)} missing, {len(unexpected)} unexpected")

    sliced, audit = build_sliced_adapter(dense_sd, dense_shapes, pruned_shapes, prefix=ADAPTER_PREFIX)
    summ = summarize_audit(audit)

    # hard requirements (item 6)
    assert summ["n_modules"] == 64, f"expected 64 modules, got {summ['n_modules']}"
    assert len(sliced) == 128, f"expected 128 tensors (64x A/B), got {len(sliced)}"
    assert summ["max_restricted_dW_abs_err_float64"] == 0.0, \
        f"restricted-dW identity not exact: {summ['max_restricted_dW_abs_err_float64']}"
    assert summ["all_unchanged_bit_identical"], "unchanged modules not bit-identical to dense"

    p1_check = None
    if args.verify_p1:
        p1_check = verify_p1_positional(dense_unet, pruned_unet)
        assert p1_check["non_positional"] == 0, f"p1 attention not positional: {p1_check}"

    os.makedirs(args.out_dir, exist_ok=True)
    art = os.path.join(args.out_dir, "gate0_sliced_adapter_1_2_3_1.pt")
    torch.save(sliced, art)
    sliced_sha = sha_file(art)
    audit_path = os.path.join(args.out_dir, "gate0_sliced_adapter_1_2_3_1_audit.json")
    json.dump({"summary": summ, "p1_positional_check": p1_check, "modules": audit},
              open(audit_path, "w"), indent=1)
    meta = {
        "artifact": "gate0_sliced_adapter_1_2_3_1",
        "channel_mult": [1, 2, 3, 1],
        "derivation": "positional kept-index slicing of the dense Gate-0 LoRA (no learned mapping, "
                      "no adapter data, no retraining)",
        "dense_adapter_sha256": dense_sha,
        "sliced_adapter_sha256": sliced_sha,
        "n_modules": summ["n_modules"], "n_tensors": len(sliced),
        "n_changed_positional": summ["n_changed_positional"],
        "n_unchanged_identity": summ["n_unchanged_identity"],
        "max_restricted_dW_abs_err_float64": summ["max_restricted_dW_abs_err_float64"],
        "all_unchanged_bit_identical": summ["all_unchanged_bit_identical"],
        "p1_positional_check": p1_check,
        "lora": {"rank": 8, "alpha": 16, "targets": ["to_q", "to_v"]},
    }
    meta_path = os.path.join(args.out_dir, "gate0_sliced_adapter_1_2_3_1_meta.json")
    json.dump(meta, open(meta_path, "w"), indent=1)

    print(json.dumps({"summary": summ, "p1_positional_check": p1_check,
                      "sliced_adapter_sha256": sliced_sha, "dense_adapter_sha256": dense_sha},
                     indent=2))
    print("BUILD-SLICED-ADAPTER PASS ->", art)
    return 0


if __name__ == "__main__":
    sys.exit(main())
