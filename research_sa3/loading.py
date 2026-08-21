"""Step 0 loading + verification utilities (protocol section 1.1, 1.3, 10).

Pure-Python helpers (hashing, safetensors indexing, config/state-dict diffs, block inventory)
work without torch; model construction goes through the upstream stable_audio_3 factory and
is verified STRICTLY by us (the upstream `copy_state_dict` loads with strict=False and only
prints mismatches -- we refuse anything but an exact key/shape match).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, Iterable, Tuple

BLOCK_RE = re.compile(r"(?:^|\.)transformer\.layers\.(\d+)\.")
SA3_UPSTREAM_COMMIT = "a0b57f5483c4588f827f3552b7d5c6ca2a9687be"


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def flatten(d, prefix: str = "") -> Dict[str, object]:
    """Flatten nested dict/list into dotted keys (lists by index)."""
    out: Dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]."))
    else:
        out[prefix[:-1]] = d
    return out


def diff_configs(a: dict, b: dict) -> Dict[str, list]:
    """Keys only in a, only in b, and differing values (flattened dotted keys)."""
    fa, fb = flatten(a), flatten(b)
    only_a = sorted(k for k in fa if k not in fb)
    only_b = sorted(k for k in fb if k not in fa)
    changed = sorted((k, fa[k], fb[k]) for k in fa if k in fb and fa[k] != fb[k])
    return {"only_a": only_a, "only_b": only_b, "changed": changed}


def diffusion_config_subtree(config: dict) -> dict:
    """The part of model_config.json whose equality defines the 1:1 block mapping."""
    return config["model"]["diffusion"]["config"]


def safetensors_index(path: str) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """{key: (shape, dtype)} read from the safetensors header, without loading tensors."""
    from safetensors import safe_open
    idx = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            idx[k] = (tuple(sl.get_shape()), str(sl.get_dtype()))
    return idx


def diff_indices(a: Dict[str, Tuple], b: Dict[str, Tuple]) -> Dict[str, list]:
    only_a = sorted(k for k in a if k not in b)
    only_b = sorted(k for k in b if k not in a)
    shape = sorted((k, a[k][0], b[k][0]) for k in a if k in b and a[k][0] != b[k][0])
    dtype = sorted((k, a[k][1], b[k][1]) for k in a if k in b and a[k][1] != b[k][1])
    return {"only_a": only_a, "only_b": only_b, "shape_mismatch": shape, "dtype_mismatch": dtype}


def block_inventory(index: Dict[str, Tuple]) -> Dict[int, dict]:
    """Per transformer block: number of tensors and parameters (from shapes)."""
    inv: Dict[int, dict] = {}
    for k, (shape, _) in index.items():
        m = BLOCK_RE.search(k)
        if not m:
            continue
        g = int(m.group(1))
        n = 1
        for s in shape:
            n *= int(s)
        e = inv.setdefault(g, {"tensors": 0, "params": 0})
        e["tensors"] += 1
        e["params"] += n
    return dict(sorted(inv.items()))


def total_params(index: Dict[str, Tuple], prefix: str = "") -> int:
    tot = 0
    for k, (shape, _) in index.items():
        if not k.startswith(prefix):
            continue
        n = 1
        for s in shape:
            n *= int(s)
        tot += n
    return tot


def patch_text_encoder_path(config: dict, local_dir: str) -> dict:
    """Point the t5gemma conditioner at a LOCAL copy (the shipped config references the
    gated post repo `stabilityai/stable-audio-3-small-sfx` even for the base model)."""
    import copy
    cfg = copy.deepcopy(config)
    for c in cfg["model"]["conditioning"]["configs"]:
        if c.get("type") == "t5gemma":
            c["config"]["model_path"] = local_dir
            c["config"].pop("repo_id", None)
            c["config"].pop("subfolder", None)
    return cfg


def build_model_strict(config: dict, ckpt_path: str, device: str = "cpu"):
    """Build via the upstream factory and load the checkpoint with an EXACT key/shape match.

    Returns (model, report). Raises if the checkpoint does not match the model's state_dict
    exactly (after the upstream key remapping, which we also report)."""
    import torch
    from safetensors.torch import load_file
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.loading_utils import remap_state_dict_keys

    model = create_diffusion_cond_from_config(config)
    msd = model.state_dict()
    sd = load_file(ckpt_path)
    remapped = remap_state_dict_keys(sd, msd)
    renamed = sorted(k for k in sd if k not in remapped)
    a = {k: (tuple(v.shape), str(v.dtype)) for k, v in remapped.items()}
    b = {k: (tuple(v.shape), str(v.dtype)) for k, v in msd.items()}
    d = diff_indices(a, b)
    report = {"renamed_keys": renamed, "ckpt_only": d["only_a"], "model_only": d["only_b"],
              "shape_mismatch": d["shape_mismatch"], "dtype_mismatch": d["dtype_mismatch"],
              "n_ckpt": len(sd), "n_model": len(msd)}
    if d["only_a"] or d["only_b"] or d["shape_mismatch"]:
        raise RuntimeError(f"checkpoint/model mismatch: {json.dumps(report)[:2000]}")
    model.load_state_dict(remapped, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, report
