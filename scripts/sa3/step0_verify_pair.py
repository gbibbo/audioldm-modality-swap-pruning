#!/usr/bin/env python3
"""Step 0 of docs/sa3/analysis_protocol_rq1_rq2.md: verify the base / post pair on CPU.

For each available repo dir (base mandatory, post optional until the gate is accepted):
  * sha256 of model.safetensors, model_config.json, svd_bases.pt (base only), t5gemma files
  * safetensors header index: tensor count, parameter totals, per-block inventory
  * model_config.json summary (diffusion config subtree)
If both are present: config diff (full + diffusion subtree), state-dict key/shape diff,
t5gemma copy comparison. Optionally (--build) construct the base model through the upstream
factory with an EXACT key/shape load and run (i) a tiny DiT forward and (ii) the BlockMask
identity check on real weights. Exit code 1 if any mandatory verification fails.

    .venv-sa3/bin/python scripts/sa3/step0_verify_pair.py --base-dir data/sa3/small-sfx-base \
        [--post-dir data/sa3/small-sfx] [--build] --out artifacts/sa3/step0.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from research_sa3 import loading as L  # noqa: E402

ARCH_PREFIXES = ("model.diffusion.config.", "model.pretransform.", "model.conditioning.", "model.io_channels",
                 "sample_rate", "audio_channels", "model_type", "model.diffusion.type")
OBJECTIVE_SAMPLING_KEYS = ("model.diffusion.diffusion_objective", "distribution_shift", "use_effective_length_for_schedule",
                           "mask_padding_attention", "sample_size", "training.", "sampling")


def classify_config_diff(cd: dict) -> dict:
    """Split a flattened config diff into ARCHITECTURE (invalidates the block mapping) vs
    OBJECTIVE/SAMPLING (legitimate base-vs-post differences) vs OTHER (must be inspected)."""
    def bucket(key: str) -> str:
        if key.startswith("model.conditioning.") and ("repo_id" in key or "subfolder" in key or "model_path" in key):
            return "text_encoder_location"          # where t5gemma is fetched from: not architecture
        if any(key.startswith(p) or key == p for p in ARCH_PREFIXES):
            return "architecture"
        if any(k in key for k in OBJECTIVE_SAMPLING_KEYS):
            return "objective_or_sampling"
        return "other"
    out = {"architecture": [], "objective_or_sampling": [], "text_encoder_location": [], "other": []}
    for k in cd["only_a"]:
        out[bucket(k)].append(("only_base", k))
    for k in cd["only_b"]:
        out[bucket(k)].append(("only_post", k))
    for k, va, vb in cd["changed"]:
        out[bucket(k)].append(("changed", k, va, vb))
    return out


T5_FILES = ["config.json", "generation_config.json", "special_tokens_map.json", "tokenizer.json",
            "tokenizer.model", "tokenizer_config.json", "model.safetensors"]
EXPECTED_DEPTH = 20  # base model_config.json (read 2026-08-20): model.diffusion.config.depth


def git_commit(path="."):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def describe_repo(d: str, with_svd: bool) -> dict:
    out = {"dir": d, "files": {}}
    for f in ["model.safetensors", "model_config.json"] + (["svd_bases.pt"] if with_svd else []):
        p = os.path.join(d, f)
        out["files"][f] = {"bytes": os.path.getsize(p), "sha256": L.sha256_file(p)} if os.path.exists(p) else None
    t5 = {}
    for f in T5_FILES:
        p = os.path.join(d, "t5gemma-b-b-ul2", f)
        t5[f] = {"bytes": os.path.getsize(p), "sha256": L.sha256_file(p)} if os.path.exists(p) else None
    out["t5gemma"] = t5
    cfg = L.load_json(os.path.join(d, "model_config.json"))
    out["config"] = {"top": {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))},
                     "diffusion_objective": cfg["model"]["diffusion"].get("diffusion_objective"),
                     "diffusion_config": L.diffusion_config_subtree(cfg),
                     "diffusion_keys": sorted(cfg["model"]["diffusion"].keys()),
                     "conditioning": cfg["model"]["conditioning"]}
    idx = L.safetensors_index(os.path.join(d, "model.safetensors"))
    inv = L.block_inventory(idx)
    prefixes = sorted({k.split(".")[0] for k in idx})
    out["index"] = {"n_tensors": len(idx), "total_params": L.total_params(idx),
                    "top_prefixes": prefixes,
                    "params_by_prefix": {p: L.total_params(idx, p + ".") for p in prefixes},
                    "n_blocks": len(inv), "block_inventory": inv,
                    "dit_params_sample_keys": sorted(k for k in idx if "transformer.layers.0." in k)[:8]}
    return out, cfg, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--post-dir", default=None)
    ap.add_argument("--build", action="store_true", help="construct the base model (strict) + tiny forward + BlockMask identity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    result = {"script": "step0_verify_pair.py", "git_commit": git_commit(), "upstream_commit": L.SA3_UPSTREAM_COMMIT,
              "checks": {}}
    base, base_cfg, base_idx = describe_repo(args.base_dir, with_svd=True)
    result["base"] = base
    ok = True

    # mandatory base checks
    c = result["checks"]
    c["base_depth_20"] = base["config"]["diffusion_config"].get("depth") == EXPECTED_DEPTH
    c["base_blocks_20"] = base["index"]["n_blocks"] == EXPECTED_DEPTH
    bp = [v["params"] for v in base["index"]["block_inventory"].values()]
    c["base_blocks_equal_params"] = len(set(bp)) == 1
    c["base_model_bytes_match_hf_api"] = base["files"]["model.safetensors"]["bytes"] == 2270384940
    ok &= all(c[k] for k in ["base_depth_20", "base_blocks_20", "base_model_bytes_match_hf_api"])

    if os.path.exists(os.path.join(args.base_dir, "svd_bases.pt")):
        import torch
        sb = torch.load(os.path.join(args.base_dir, "svd_bases.pt"), map_location="cpu", weights_only=True)
        keys = sorted(sb.keys())
        ex = sb[keys[0]]
        result["base"]["svd_bases"] = {"n_layers": len(keys), "first_key": keys[0],
                                       "entry_keys": sorted(ex.keys()) if isinstance(ex, dict) else str(type(ex)),
                                       "U_shape": list(ex["U"].shape) if isinstance(ex, dict) and "U" in ex else None,
                                       "V_shape": list(ex["V"].shape) if isinstance(ex, dict) and "V" in ex else None,
                                       "model_keys": sum(k.startswith("model.") for k in keys),
                                       "conditioner_keys": sum(k.startswith("conditioner.") for k in keys)}

    if args.post_dir:
        post, post_cfg, post_idx = describe_repo(args.post_dir, with_svd=False)
        result["post"] = post
        cd = L.diff_configs(base_cfg, post_cfg)
        dd = L.diff_configs(L.diffusion_config_subtree(base_cfg), L.diffusion_config_subtree(post_cfg))
        idd = L.diff_indices(base_idx, post_idx)
        result["pair"] = {"config_diff": cd, "diffusion_config_diff": dd, "index_diff": idd,
                          "t5gemma_identical": all(
                              base["t5gemma"][f] and post["t5gemma"][f] and base["t5gemma"][f]["sha256"] == post["t5gemma"][f]["sha256"]
                              for f in T5_FILES)}
        cls = classify_config_diff(cd)
        result["pair"]["config_diff_classified"] = cls
        result["pair"]["t5gemma_per_file_identical"] = {
            f: (base["t5gemma"][f] is not None and post["t5gemma"][f] is not None
                and base["t5gemma"][f]["sha256"] == post["t5gemma"][f]["sha256"]) for f in T5_FILES}
        bi, pi = base["index"]["block_inventory"], post["index"]["block_inventory"]
        c["pair_block_mapping_20_20"] = (len(bi) == EXPECTED_DEPTH and len(pi) == EXPECTED_DEPTH
                                        and all(bi[g] == pi[g] for g in bi))
        c["pair_diffusion_config_identical"] = not (dd["only_a"] or dd["only_b"] or dd["changed"])
        c["pair_no_architecture_config_diff"] = len(cls["architecture"]) == 0
        c["pair_other_config_diff_empty"] = len(cls["other"]) == 0
        c["pair_state_dict_keys_shapes_identical"] = not (idd["only_a"] or idd["only_b"] or idd["shape_mismatch"])
        c["pair_objectives"] = [base["config"]["diffusion_objective"], post["config"]["diffusion_objective"]]
        c["pair_t5gemma_identical"] = result["pair"]["t5gemma_identical"]
        ok &= (c["pair_diffusion_config_identical"] and c["pair_no_architecture_config_diff"]
               and c["pair_state_dict_keys_shapes_identical"] and c["pair_block_mapping_20_20"])

    if args.build:
        import torch
        from research_sa3.blockskip import block_mask, depth
        cfg = L.patch_text_encoder_path(base_cfg, os.path.join(args.base_dir, "t5gemma-b-b-ul2"))
        tb = time.time()
        model, rep = L.build_model_strict(cfg, os.path.join(args.base_dir, "model.safetensors"), device="cpu")
        result["build"] = {"strict_load_report": rep, "build_s": time.time() - tb}
        dit = model.model.model  # ConditionedDiffusionModelWrapper -> DiTWrapper -> DiffusionTransformer
        result["build"]["depth"] = depth(dit)
        result["build"]["dit_params"] = sum(p.numel() for p in dit.parameters())
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 256, 16, generator=g)
        t = torch.full((1,), 0.5)
        ctx = torch.randn(1, 4, 768, generator=g)
        glob = torch.randn(1, 768, generator=g)
        with torch.no_grad():
            tf = time.time()
            y = dit._forward(x, t, cross_attn_cond=ctx, global_embed=glob)
            result["build"]["tiny_forward_s_cpu_fp32"] = time.time() - tf
            with block_mask(dit, []):
                y0 = dit._forward(x, t, cross_attn_cond=ctx, global_embed=glob)
            with block_mask(dit, [5]):
                y5 = dit._forward(x, t, cross_attn_cond=ctx, global_embed=glob)
        c["build_strict_load"] = True
        c["build_depth_20"] = result["build"]["depth"] == EXPECTED_DEPTH
        c["build_forward_finite"] = bool(torch.isfinite(y).all())
        c["build_blockmask_identity_bitexact"] = bool(torch.equal(y, y0))
        c["build_blockmask_skip5_changes"] = not torch.equal(y, y5)
        result["build"]["rel_change_skip5"] = float(((y - y5).norm() / y.norm()).item())
        ok &= all(c[k] for k in ["build_depth_20", "build_forward_finite", "build_blockmask_identity_bitexact", "build_blockmask_skip5_changes"])
        del model

        if args.post_dir:
            cfg_p = L.patch_text_encoder_path(post_cfg, os.path.join(args.post_dir, "t5gemma-b-b-ul2"))
            tb = time.time()
            model_p, rep_p = L.build_model_strict(cfg_p, os.path.join(args.post_dir, "model.safetensors"), device="cpu")
            result["build_post"] = {"strict_load_report": rep_p, "build_s": time.time() - tb,
                                    "diffusion_objective": model_p.model.diffusion_objective}
            dit_p = model_p.model.model
            result["build_post"]["depth"] = depth(dit_p)
            result["build_post"]["dit_params"] = sum(p.numel() for p in dit_p.parameters())
            with torch.no_grad():
                yp = dit_p._forward(x, t, cross_attn_cond=ctx, global_embed=glob)
                with block_mask(dit_p, []):
                    yp0 = dit_p._forward(x, t, cross_attn_cond=ctx, global_embed=glob)
                with block_mask(dit_p, [5]):
                    yp5 = dit_p._forward(x, t, cross_attn_cond=ctx, global_embed=glob)
            c["build_post_strict_load"] = True
            c["build_post_depth_20"] = result["build_post"]["depth"] == EXPECTED_DEPTH
            c["build_post_forward_finite"] = bool(torch.isfinite(yp).all())
            c["build_post_blockmask_identity_bitexact"] = bool(torch.equal(yp, yp0))
            c["build_post_blockmask_skip5_changes"] = not torch.equal(yp, yp5)
            # same random input through both checkpoints: they must differ (post-training happened)
            result["build_post"]["rel_diff_base_vs_post_same_input"] = float(((y - yp).norm() / y.norm()).item())
            c["build_pair_fields_differ"] = not torch.equal(y, yp)
            ok &= all(c[k] for k in ["build_post_depth_20", "build_post_forward_finite",
                                     "build_post_blockmask_identity_bitexact", "build_post_blockmask_skip5_changes",
                                     "build_pair_fields_differ"])

    result["ok"] = bool(ok)
    result["wall_s"] = time.time() - t0
    txt = json.dumps(result, indent=1, default=str)
    print(txt[:6000])
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(txt)
        print(f"\nwrote {args.out}")
    print("\nSTEP0", "PASS" if ok else "FAIL", json.dumps({k: v for k, v in c.items()}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
