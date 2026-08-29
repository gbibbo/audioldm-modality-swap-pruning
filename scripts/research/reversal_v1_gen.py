#!/usr/bin/env python3
"""RECOVERY-REVERSAL-V1.1 production generator (GPU job entry; reuses the frozen gate0 core).

Generates the 192 standalone (no-adapter) WAVs for ONE V1.1 system over the frozen 96-prompt
AudioCaps battery. Reuses gate0_generator.build_backbone / generate / EMA convention UNCHANGED;
the ONLY differences from Gate-0 are (a) the prompt source = the frozen V1.1 manifest, (b) the
initial-latent seed namespace = GENERATION_SALT (V1.1 CRN), and (c) no adapter (standalone).

Operating point is the frozen V1 contract: latent_t=96 (3.84 s), DDIM 50, eta 0, guidance 2.5,
FP32, single generation, no best-of-3, no generation-time scoring. x_T = make_x_T(ytid, r) is a pure
function of (ytid, replicate) via GENERATION_SALT -> identical across dense_ema / pruned / recovered.

WAV names match the frozen scorer convention (reversal_v1_score.PREFIX):
  dense_noadapter_p{p}_r{r}.wav / p1_pruned_ema_reconstructed_noadapter_... / p1_recovered_noadapter_...

DO NOT run in the CPU Studio (except --verify-paired-noise / --dry-run-cpu). This is a Lightning
T4 Job entry. Run one system per invocation; the entry script run_reversal_v1_1_gen.sh loops all 3.
"""
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import gate0_generator as G0  # reuse build_backbone / generate / provenance UNCHANGED
from research_pruning.eval.reversal import (  # noqa: E402
    GENERATION_SALT_V1, N_PROMPTS_V1, N_REPLICATES_V1, OPERATING_POINT_V1, generation_seed)

MANIFEST = "configs/research/reversal_v1_1_audiocaps_manifest.json"
CONFIG = G0.CONFIG
PREREG = G0.PREREG
# V1.1 system -> (gate0 build_backbone id, WAV prefix matching reversal_v1_score.PREFIX)
SYSTEMS = {
    "dense_ema": ("dense", "dense_noadapter"),
    "p1_pruned_ema_reconstructed": ("p1_pruned_ema_reconstructed", "p1_pruned_ema_reconstructed_noadapter"),
    "p1_recovered": ("p1_recovered", "p1_recovered_noadapter"),
}


def make_x_T(ytid, replicate, C, T, F):
    g = torch.Generator().manual_seed(generation_seed(ytid, replicate))
    return torch.randn(1, C, T, F, generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEMS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_v1_1_gen")
    ap.add_argument("--dry-run-cpu", action="store_true", help="1 prompt x1 rep, ddim 6, CPU end-to-end")
    ap.add_argument("--verify-paired-noise", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    pre = yaml.safe_load(open(PREREG))
    g0 = pre["gate0"]
    C = g0["data"].get("latent_c", 8)
    T = g0["data"]["latent_t_size"]
    F = g0["data"]["latent_f_size"] if "latent_f_size" in g0["data"] else 16
    op = OPERATING_POINT_V1
    ddim, guidance, eta = op["ddim_steps"], op["guidance"], op["eta"]
    prompts = json.load(open(MANIFEST))["prompts"]
    reps = N_REPLICATES_V1
    if args.dry_run_cpu:
        prompts = prompts[:1]; ddim = 6; reps = 1

    build_id, prefix = SYSTEMS[args.system]

    if args.verify_paired_noise:
        a = make_x_T(prompts[0]["ytid"], 0, C, T, F); b = make_x_T(prompts[0]["ytid"], 0, C, T, F)
        c = make_x_T(prompts[0]["ytid"], 1, C, T, F)
        # cross-system identity: x_T is independent of `--system` (no backbone term)
        ok = bool(torch.equal(a, b)) and bool(not torch.equal(a, c))
        exp_seed = generation_seed(prompts[0]["ytid"], 0)
        manifest_seed = prompts[0]["generation_seeds"][0]
        ok = ok and exp_seed == manifest_seed
        print(json.dumps({"deterministic_same_ytid_r": bool(torch.equal(a, b)),
                          "differs_across_replicate": bool(not torch.equal(a, c)),
                          "seed_matches_manifest": exp_seed == manifest_seed,
                          "x_T_shape": list(a.shape), "salt": GENERATION_SALT_V1}, indent=2))
        print("V1.1 PAIRED-NOISE", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    dev = (torch.device("cpu") if args.dry_run_cpu else
           (torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
            else torch.device(args.device)))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PREFLIGHT FAIL: --device cuda but no CUDA available")
    if args.device == "cuda" and dev.type != "cuda":
        raise SystemExit("PREFLIGHT FAIL: requested cuda unavailable")

    os.makedirs(args.out, exist_ok=True)
    torch.load = G0._cpu_load
    import audioldm_train.modules.latent_diffusion.ddim as _ddim
    _oi = _ddim.DDIMSampler.__init__
    _ddim.DDIMSampler.__init__ = lambda s, m, schedule="linear", device=None, **k: _oi(
        s, m, schedule=schedule, device=dev, **k)

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = g0["data"]["train_clip_seconds"]
    model, ck_sha = G0.build_backbone(build_id, config, dev)
    model._gate0_config = config
    model.eval()

    import soundfile as sf
    rows = []
    for pi, p in enumerate(prompts):
        for r in range(reps):
            x_T = make_x_T(p["ytid"], r, C, T, F).to(dev)
            w = G0.generate(model, p["caption"], x_T, ddim, guidance, eta)
            path = os.path.join(args.out, f"{prefix}_p{pi}_r{r}.wav")
            sf.write(path, np.asarray(w).squeeze().astype(np.float32), 16000, subtype="PCM_16")
            rows.append({"ytid": p["ytid"], "prompt_index": pi, "replicate_index": r,
                         "seed": generation_seed(p["ytid"], r), "system": args.system,
                         "backbone_build_id": build_id, "adapter_state": "off", "checkpoint": ck_sha,
                         "ddim_steps": ddim, "eta": eta, "guidance": guidance, "latent_t": T,
                         "device": str(dev), "wav": path, "wav_sha256": G0.sha_file(path)})

    prov = {**G0._git_info(), **G0._env_info(dev)}
    src = G0.SOURCE_CKPT.get(build_id)
    prov["source_checkpoint"] = src
    if src and os.path.exists(src) and not args.dry_run_cpu:
        prov["source_checkpoint_sha256"] = G0.sha_file(src)
    prov["checkpoint_convention"] = ck_sha
    man = {"artifact": "reversal_v1_1_gen", "system": args.system,
           "manifest_source": MANIFEST, "device": str(dev),
           "recipe": {"ddim": ddim, "eta": eta, "guidance": guidance, "latent_t": T,
                      "replicates": reps, "weight_convention": "ema",
                      "generation_salt": GENERATION_SALT_V1, "adapter": "none", "best_of": 1},
           "provenance": prov, "n": len(rows), "rows": rows}
    outman = os.path.join(args.out, f"gen_manifest_{args.system}.json")
    json.dump(man, open(outman, "w"), indent=1)
    exp = 1 if args.dry_run_cpu else N_PROMPTS_V1 * N_REPLICATES_V1
    if not args.dry_run_cpu and len(rows) != exp:
        raise SystemExit(f"expected {exp} WAVs, wrote {len(rows)}")
    if args.validate and not args.dry_run_cpu:
        cells = {(r["prompt_index"], r["replicate_index"]) for r in rows}
        if len(cells) != N_PROMPTS_V1 * N_REPLICATES_V1:
            raise SystemExit("duplicate/missing (prompt,replicate) cells")
        for r in rows:
            if r["seed"] != generation_seed(r["ytid"], r["replicate_index"]):
                raise SystemExit(f"seed mismatch at p{r['prompt_index']} r{r['replicate_index']}")
        print("MANIFEST-VALIDATED", json.dumps({"system": args.system, "n": len(rows),
              "unique_cells": len(cells), "seeds_match": True}))
    print(f"generated {len(rows)} wavs -> {args.out}; manifest {outman}")
    print("V1.1-GENERATOR PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
