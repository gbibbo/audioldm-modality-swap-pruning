"""Deterministic mask-induced slicing of the dense-trained Gate-0 LoRA onto the (1,2,3,1)
pruned architecture (Scenario B). No learned mapping, no adapter data, no retraining.

The generic factor-slicing identity lives in ``research_pruning.lora_mask_transfer``:

    A' = A[:, keep_in]   B' = B[keep_out, :]   =>   B'@A' == (B@A)[keep_out][:, keep_in]  (exact).

This module supplies the PRODUCTION module->kept-index mapping for the 64 attention to_q/to_v
Linear modules. Empirically (scripts/research/build_sliced_adapter.py --verify-p1):

* 40 of 64 modules are UNCHANGED (levels channel_mult 2 and 3: dims 384/576) -> keep=None ->
  the transferred factors are BIT-IDENTICAL to the dense factors.
* 24 of 64 modules change (960,960)->(192,192) (the channel_mult 5->1 level). The published
  ``l1_audioldm-m-full_p1.ckpt`` prunes these attention weights POSITIONALLY (W_p1 ==
  W_dense[:192,:192]; verified 64/64 exact), because attention weights are NOT in the L1
  materializer's LAYER_MAP and fall to its positional CASE-2 fallback. So the kept-index
  convention for the LoRA is the SAME positional selection: keep_out = keep_in = arange(192).

The same sliced bytes strict-load onto BOTH pruned backbones (p1_pruned_ema_reconstructed and
p1_recovered) because they share the (1,2,3,1) architecture; on p1_recovered the surviving
coordinates carry RECOVERED (non-selection) weights (reported honestly, prereg sliced_lora_note).
"""
from __future__ import annotations

import hashlib

import torch

from research_pruning.lora_mask_transfer import slice_lora_linear, delta_w_linear

ADAPTER_PREFIX = "model.diffusion_model."


def qv_linear_shapes(unet) -> dict:
    """{unet-relative module name -> (out_features, in_features)} for every to_q/to_v Linear."""
    out = {}
    for name, mod in unet.named_modules():
        if name.endswith(("to_q", "to_v")) and isinstance(mod, torch.nn.Linear):
            out[name] = (int(mod.out_features), int(mod.in_features))
    return out


def _positional_keep(dense_dim: int, pruned_dim: int):
    """Positional kept-index vector [0:pruned_dim) if the dim shrinks, else None (unchanged)."""
    if pruned_dim == dense_dim:
        return None
    if pruned_dim > dense_dim:
        raise ValueError(f"pruned dim {pruned_dim} > dense dim {dense_dim}")
    return torch.arange(pruned_dim, dtype=torch.long)


def _idx_hash(idx, dim):
    if idx is None:
        return {"length": dim, "hash": "identity", "changed": False}
    b = idx.to(torch.int64).numpy().tobytes()
    return {"length": int(idx.numel()), "hash": hashlib.sha256(b).hexdigest()[:16], "changed": True}


def build_sliced_adapter(dense_sd: dict, dense_shapes: dict, pruned_shapes: dict,
                         prefix: str = ADAPTER_PREFIX):
    """Slice every dense LoRA (lora_A,lora_B) pair to the pruned kept-set.

    Returns (sliced_state_dict, audit_rows). Raises on any missing/unexpected/misshaped module.
    The identity ``B'@A' == (B@A)[keep_out][:,keep_in]`` is checked in float64 per module.
    """
    modules = sorted({k.rsplit(".", 1)[0] for k in dense_sd})
    sliced: dict = {}
    audit = []
    for mod in modules:
        ka, kb = mod + ".lora_A", mod + ".lora_B"
        if ka not in dense_sd or kb not in dense_sd:
            raise ValueError(f"{mod}: missing lora_A/lora_B in dense adapter")
        rel = mod[len(prefix):] if mod.startswith(prefix) else mod
        if rel not in dense_shapes or rel not in pruned_shapes:
            raise ValueError(f"{mod}: no to_q/to_v Linear named {rel} in the U-Nets")
        A = dense_sd[ka]                          # (r, in)
        B = dense_sd[kb]                           # (out, r)
        do, di = dense_shapes[rel]                 # dense (out, in)
        po, pi = pruned_shapes[rel]                # pruned (out, in)
        if A.shape[1] != di or B.shape[0] != do:
            raise ValueError(f"{mod}: adapter dims {tuple(A.shape)}/{tuple(B.shape)} "
                             f"disagree with dense Linear (out={do}, in={di})")
        keep_out = _positional_keep(do, po)
        keep_in = _positional_keep(di, pi)
        A2, B2 = slice_lora_linear(A, B, keep_in, keep_out)
        # float64 identity: sliced dW equals dense dW restricted by the same indices
        dW = delta_w_linear(A.double(), B.double())
        ko = keep_out if keep_out is not None else torch.arange(do)
        ki = keep_in if keep_in is not None else torch.arange(di)
        dW_restrict = dW.index_select(0, ko).index_select(1, ki)
        dW2 = delta_w_linear(A2.double(), B2.double())
        maxerr = float((dW2 - dW_restrict).abs().max())
        sliced[ka] = A2.contiguous()
        sliced[kb] = B2.contiguous()
        changed = (keep_out is not None) or (keep_in is not None)
        # for unchanged modules the sliced factors must be bit-identical to dense
        bit_identical = bool(torch.equal(A2, A) and torch.equal(B2, B)) if not changed else None
        audit.append({
            "dense_module": mod, "target_module": mod, "unet_relative": rel,
            "dense_A_shape": list(A.shape), "dense_B_shape": list(B.shape),
            "sliced_A_shape": list(A2.shape), "sliced_B_shape": list(B2.shape),
            "keep_in": _idx_hash(keep_in, di), "keep_out": _idx_hash(keep_out, do),
            "in_changed": keep_in is not None, "out_changed": keep_out is not None,
            "mapping_mode": ("positional-slice" if changed else "identity"),
            "mapping_source": ("channel-pruning positional CASE-2 (verified: l1_p1 to_q/to_v "
                               "== dense[:out,:in]) " if changed else "unpruned level (identity)"),
            "restricted_dW_max_abs_err_float64": maxerr,
            "bit_identical_to_dense": bit_identical,
        })
    return sliced, audit


def summarize_audit(audit: list) -> dict:
    changed = [a for a in audit if a["mapping_mode"] == "positional-slice"]
    unchanged = [a for a in audit if a["mapping_mode"] == "identity"]
    max_err = max(a["restricted_dW_max_abs_err_float64"] for a in audit) if audit else 0.0
    unchanged_bit_exact = all(a["bit_identical_to_dense"] for a in unchanged)
    return {
        "n_modules": len(audit),
        "n_changed_positional": len(changed),
        "n_unchanged_identity": len(unchanged),
        "max_restricted_dW_abs_err_float64": max_err,
        "all_unchanged_bit_identical": unchanged_bit_exact,
    }
