"""Validity audit of the published recovered checkpoint (PHENOM-VALIDITY-GEOM).

Two questions, both answerable on CPU without generating audio:

1. Is the ``model_ema`` state of ``l1_p1_finetuned_global_step_999999.ckpt`` a sane
   trailing average of the raw finetuned weights (i.e. was the falsifier's
   ``recovered_ema`` convention a correct reading of the artifact), or stale/pathological?
2. How far did the 1M-step recovery finetuning move the U-Net from its init
   (the published pruned-only ``l1_audioldm-m-full_p1.ckpt``)? "Recovery" as gentle
   repair predicts high cosine to init; wholesale retraining predicts low cosine.

Also extracts the optimizer hyperparameters stored inside the Lightning checkpoint
(lr / betas / weight decay / epoch / global_step) — direct provenance of the authors'
finetuning recipe, not an inference.

Output: artifacts/icassp_gate0/recovered_vs_init_geometry.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")  # E-BLAS guard

import gc
import json

import numpy as np
import torch

RECOVERED = "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt"
P1_INIT = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
OUT = "artifacts/icassp_gate0/recovered_vs_init_geometry.json"


def mangle(raw_key: str) -> str:
    """LitEma buffer name for a raw ``model.``-prefixed parameter key."""
    return "model_ema." + raw_key[len("model."):].replace(".", "")


def main() -> None:
    full = torch.load(RECOVERED, map_location="cpu")
    out: dict = {
        "recovered_ckpt": RECOVERED,
        "p1_init_ckpt": P1_INIT,
        "epoch": full.get("epoch"),
        "global_step": full.get("global_step"),
    }
    try:
        pg = full["optimizer_states"][0]["param_groups"][0]
        out["optimizer"] = {k: pg.get(k) for k in ("lr", "betas", "weight_decay", "eps")}
        out["lr_schedulers"] = full.get("lr_schedulers")
    except Exception as ex:  # provenance-only; absence is a finding, not a failure
        out["optimizer"] = f"unreadable: {ex}"

    sd = full["state_dict"]
    for k in ("model_ema.num_updates", "model_ema.decay"):
        out[k] = sd[k].item() if k in sd else None

    raw_keys = [k for k in sd if k.startswith("model.") and not k.startswith("model_ema.")]
    ema_matched = [k for k in raw_keys if mangle(k) in sd]
    out["n_raw_unet_tensors"] = len(raw_keys)
    out["n_ema_matched"] = len(ema_matched)

    d_re, cos_re = [], []
    for k in ema_matched:
        r = sd[k].double().flatten()
        e = sd[mangle(k)].double().flatten()
        n = r.norm().item() or 1e-12
        d_re.append((r - e).norm().item() / n)
        cos_re.append((torch.dot(r, e) / (r.norm() * e.norm() + 1e-12)).item())
    d_re, cos_re = np.array(d_re), np.array(cos_re)
    out["raw_vs_ema"] = {
        "d_rel_median": float(np.median(d_re)),
        "d_rel_q90": float(np.quantile(d_re, 0.9)),
        "d_rel_max": float(d_re.max()),
        "cos_median": float(np.median(cos_re)),
        "cos_min": float(cos_re.min()),
    }

    raw = {k: sd[k].float().clone() for k in raw_keys}
    del sd, full
    gc.collect()

    p1 = torch.load(P1_INIT, map_location="cpu")["state_dict"]
    cos_rp, d_rp, sizes = [], [], []
    n_missing = n_shape_mismatch = 0
    for k, r in raw.items():
        p = p1.get(k)
        if p is None:
            n_missing += 1
            continue
        if p.shape != r.shape:
            n_shape_mismatch += 1
            continue
        r64, p64 = r.double().flatten(), p.double().flatten()
        n = p64.norm().item() or 1e-12
        d_rp.append((r64 - p64).norm().item() / n)
        cos_rp.append((torch.dot(r64, p64) / (r64.norm() * p64.norm() + 1e-12)).item())
        sizes.append(r64.numel())
    cos_rp, d_rp, sizes = np.array(cos_rp), np.array(d_rp), np.array(sizes, dtype=float)
    w = sizes / sizes.sum()
    out["recovered_raw_vs_p1_init"] = {
        "n_compared": int(cos_rp.size),
        "n_missing": n_missing,
        "n_shape_mismatch": n_shape_mismatch,
        "cos_median": float(np.median(cos_rp)),
        "cos_weighted_mean": float(np.sum(cos_rp * w)),
        "cos_q10": float(np.quantile(cos_rp, 0.1)),
        "cos_q90": float(np.quantile(cos_rp, 0.9)),
        "frac_cos_below_0p5": float((cos_rp < 0.5).mean()),
        "d_rel_median": float(np.median(d_rp)),
        "d_rel_q90": float(np.quantile(d_rp, 0.9)),
        "frac_d_rel_above_1": float((d_rp > 1.0).mean()),
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
