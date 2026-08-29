#!/usr/bin/env python3
"""CPU generation preflight for RECOVERY-REVERSAL-V1 (NO GPU, NO generation).

Proves the frozen common-random-number design for the future 576-WAV run WITHOUT generating:
the initial latent x_T is a pure function of (ytid, replicate) via GENERATION_SALT, so all three
backbones (dense_ema, p1_pruned_ema_reconstructed, p1_recovered) share the SAME x_T per prompt/
replicate, replicates 0 and 1 differ, and repetition is bit-identical. Also expands and validates
the 3x96x2 = 576 generation plan: 192 WAVs/system, shared seeds across backbones, and the exact
frozen operating point.

x_T shape = (1, latent_c=8, latent_t=96, latent_f=16), matching gate0_generator.make_x_T; only the
GENERATION_SALT differs (V1 namespace), so Gate-0 reproducibility is untouched.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/reversal_v1_gen_preflight.py \
        --manifest configs/research/reversal_v1_audiocaps_manifest.json \
        --out artifacts/icassp_gate0/reversal_v1_gen_preflight.json
     (or --self-test for the CPU latent-determinism checks with synthetic ytids)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import (  # noqa: E402
    BACKBONES_V1, N_PROMPTS_V1, N_REPLICATES_V1, OPERATING_POINT_V1, generation_seed)

LATENT_C, LATENT_T, LATENT_F = 8, 96, 16
# expected standalone checkpoint identities (from the historical phenom provenance)
CHECKPOINTS = {
    "dense_ema": {"path": "data/checkpoints/audioldm-m-full.ckpt",
                  "source_sha256": "936914a388905e1fc179c148a41a2b1552dba322ce474160b1cfa0f01ac26f8f",
                  "convention": "dense EMA (descriptive anchor)"},
    "p1_pruned_ema_reconstructed": {"path": "data/checkpoints/audioldm-m-full.ckpt",
                  "source_sha256": "936914a388905e1fc179c148a41a2b1552dba322ce474160b1cfa0f01ac26f8f",
                  "convention": "prune(dense_ema)"},
    "p1_recovered": {"path": "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt",
                  "source_sha256": "c6997e83c3deac43ef2e45118cf359ba19389a3db4c94415632372fcd966dbd0",
                  "convention": "recovered_ema"},
}


def make_x_T(ytid, replicate):
    import torch
    g = torch.Generator().manual_seed(generation_seed(ytid, replicate))
    return torch.randn(1, LATENT_C, LATENT_T, LATENT_F, generator=g)


def latent_checks(ytids) -> dict:
    import torch
    checks = {}
    y0 = ytids[0]
    a = make_x_T(y0, 0); b = make_x_T(y0, 0); c = make_x_T(y0, 1)
    checks["deterministic_repeat"] = bool(torch.equal(a, b))
    checks["replicate0_ne_replicate1"] = bool(not torch.equal(a, c))
    checks["x_T_shape"] = list(a.shape)
    # backbone independence: x_T is built with NO backbone argument -> identical object reused.
    checks["backbone_independent_by_construction"] = True
    # distinctness across a few prompts
    distinct = len({make_x_T(y, 0).sum().item() for y in ytids[:8]}) == len(ytids[:8])
    checks["distinct_across_prompts"] = bool(distinct)
    return checks


def build_plan(prompts) -> list:
    plan = []
    for bk in BACKBONES_V1:
        for p in prompts:
            for r in p["replicate_indices"]:
                plan.append({"backbone": bk, "prompt_index": p["prompt_index"], "ytid": p["ytid"],
                             "replicate_index": r, "seed": generation_seed(p["ytid"], r)})
    return plan


def validate(manifest_path: str) -> dict:
    manifest = json.load(open(manifest_path))
    prompts = manifest["prompts"]
    # manifest seeds must equal the frozen derivation
    for p in prompts:
        for r in p["replicate_indices"]:
            if p["generation_seeds"][r] != generation_seed(p["ytid"], r):
                raise SystemExit(f"seed mismatch at prompt {p['prompt_index']} r{r}")
    ytids = [p["ytid"] for p in prompts]
    lc = latent_checks(ytids)
    plan = build_plan(prompts)
    per_system = {bk: sum(1 for x in plan if x["backbone"] == bk) for bk in BACKBONES_V1}
    # shared seed across backbones for each (prompt, replicate)
    from collections import defaultdict
    seeds_by_pr = defaultdict(set)
    for x in plan:
        seeds_by_pr[(x["prompt_index"], x["replicate_index"])].add(x["seed"])
    shared = all(len(s) == 1 for s in seeds_by_pr.values())
    assert all(v == N_PROMPTS_V1 * N_REPLICATES_V1 for v in per_system.values()), "192/system"
    assert len(plan) == N_PROMPTS_V1 * N_REPLICATES_V1 * len(BACKBONES_V1), "576 total"
    assert shared, "seeds not shared across backbones"
    ckpt = {bk: {**meta, "exists": os.path.exists(meta["path"])} for bk, meta in CHECKPOINTS.items()}
    out = {"artifact": "reversal_v1_gen_preflight", "manifest": manifest_path,
           "manifest_sha256": manifest.get("manifest_sha256"),
           "latent_checks": lc, "operating_point": OPERATING_POINT_V1,
           "per_system_wavs": per_system, "total_wavs": len(plan),
           "seeds_shared_across_backbones": shared, "checkpoints": ckpt,
           "all_ok": bool(all(lc[k] for k in ("deterministic_repeat", "replicate0_ne_replicate1",
                                              "distinct_across_prompts")) and shared)}
    return out


def _self_test() -> int:
    y = [f"ytid_{i:07d}" for i in range(96)]
    lc = latent_checks(y)
    plan = build_plan([{"prompt_index": i, "ytid": y[i], "replicate_indices": [0, 1]} for i in range(96)])
    from collections import defaultdict
    seeds_by_pr = defaultdict(set)
    for x in plan:
        seeds_by_pr[(x["prompt_index"], x["replicate_index"])].add(x["seed"])
    shared = all(len(s) == 1 for s in seeds_by_pr.values())
    ok = (lc["deterministic_repeat"] and lc["replicate0_ne_replicate1"] and lc["distinct_across_prompts"]
          and lc["x_T_shape"] == [1, LATENT_C, LATENT_T, LATENT_F]
          and len(plan) == 576 and shared)
    print(json.dumps({"latent_checks": lc, "plan_total": len(plan), "seeds_shared": shared}, indent=2))
    print("GEN-PREFLIGHT SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/research/reversal_v1_audiocaps_manifest.json")
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_v1_gen_preflight.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    out = validate(args.manifest)
    out["artifact_sha256"] = hashlib.sha256(
        json.dumps(out, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("latent_checks", "per_system_wavs", "total_wavs",
                      "seeds_shared_across_backbones", "all_ok")}, indent=2))
    print("V1 gen preflight written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
