#!/usr/bin/env python3
"""OP-DURATION-DISCRIMINATOR-1 (Arm D) ALT generator — 10.24 s / latent_t 256 (GPU T4 job entry).

Generates the 80 standalone (no-adapter) r0 WAVs for ONE system over the frozen 80-ytid Arm-D subset,
at the ALT operating point: duration 10.24 s (latent_t 256), DDIM 50, guidance 2.5, eta 0, fp32, single,
EMA. The ONLY change vs the frozen V1.1 generator is duration/latent_t (3.84 s/96 -> 10.24 s/256). x_T is
built from the REUSED V1.1 r0 generation seed (generation_seed(ytid, 0)); shape (1,8,256,16) so it is NOT
identical to the 3.84 s control x_T (documented). Same x_T across pruned/recovered per ytid (CRN).

Reuses gate0_generator.build_backbone / generate / EMA UNCHANGED. Run one system per invocation; entry
script run_reversal_armd_gen.sh loops pruned + recovered. CPU only for --dry-run-cpu / --verify.

Run (GPU):  .venv/bin/python scripts/research/reversal_armd_gen.py --system p1_recovered
"""
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import yaml

import gate0_generator as G0
from research_pruning.eval.reversal import GENERATION_SALT_V1, generation_seed

SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
CONFIG = G0.CONFIG
PREREG = G0.PREREG
ALT_LATENT_T = 256          # 10.24 s
ALT_DURATION = 10.24
DDIM, GUIDANCE, ETA = 50, 2.5, 0.0
SYSTEMS = {
    "p1_pruned_ema_reconstructed": ("p1_pruned_ema_reconstructed", "p1_pruned_ema_reconstructed_noadapter_alt10s"),
    "p1_recovered": ("p1_recovered", "p1_recovered_noadapter_alt10s"),
}


def make_x_T(ytid, C, T, F):
    g = torch.Generator().manual_seed(generation_seed(ytid, 0))   # reuse frozen V1.1 r0 seed
    return torch.randn(1, C, T, F, generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEMS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_armd_gen")
    ap.add_argument("--dry-run-cpu", action="store_true", help="1 prompt, ddim 6, CPU end-to-end")
    ap.add_argument("--verify-seeds", action="store_true")
    args = ap.parse_args()

    pre = yaml.safe_load(open(PREREG))
    g0 = pre["gate0"]
    C = g0["data"].get("latent_c", 8)
    F = g0["data"]["latent_f_size"] if "latent_f_size" in g0["data"] else 16
    T = ALT_LATENT_T
    prompts = json.load(open(SUBSET))["prompts"]
    ddim = DDIM
    if args.dry_run_cpu:
        prompts = prompts[:1]; ddim = 6

    build_id, prefix = SYSTEMS[args.system]

    if args.verify_seeds:
        ok = True
        for p in prompts:
            exp = generation_seed(p["ytid"], 0)
            if exp != p["generation_seed_r0"]:
                ok = False; print("SEED MISMATCH", p["ytid"], exp, p["generation_seed_r0"])
        a = make_x_T(prompts[0]["ytid"], C, T, F); b = make_x_T(prompts[0]["ytid"], C, T, F)
        ok = ok and bool(torch.equal(a, b)) and list(a.shape) == [1, C, T, F]
        print(json.dumps({"seeds_match_subset": ok, "x_T_shape": list(a.shape),
                          "salt": GENERATION_SALT_V1, "n": len(prompts)}, indent=2))
        print("ARM-D SEED-VERIFY", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    dev = (torch.device("cpu") if args.dry_run_cpu else
           (torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
            else torch.device(args.device)))
    if args.device == "cuda" and dev.type != "cuda":
        raise SystemExit("PREFLIGHT FAIL: requested cuda unavailable")

    os.makedirs(args.out, exist_ok=True)
    torch.load = G0._cpu_load
    import audioldm_train.modules.latent_diffusion.ddim as _ddim
    _oi = _ddim.DDIMSampler.__init__
    _ddim.DDIMSampler.__init__ = lambda s, m, schedule="linear", device=None, **k: _oi(
        s, m, schedule=schedule, device=dev, **k)

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = ALT_DURATION            # 10.24 s (ALT)
    model, ck_sha = G0.build_backbone(build_id, config, dev)
    model._gate0_config = config
    model.latent_t_size = ALT_LATENT_T                                     # override 96 -> 256
    model.eval()

    import soundfile as sf
    rows = []
    for p in prompts:
        pi = p["subset_prompt_index"]
        x_T = make_x_T(p["ytid"], C, T, F).to(dev)
        w = G0.generate(model, p["caption"], x_T, ddim, GUIDANCE, ETA)
        path = os.path.join(args.out, f"{prefix}_p{pi}_r0.wav")
        sf.write(path, np.asarray(w).squeeze().astype(np.float32), 16000, subtype="PCM_16")
        rows.append({"ytid": p["ytid"], "subset_prompt_index": pi, "v1_1_prompt_index": p["v1_1_prompt_index"],
                     "replicate_index": 0, "seed": generation_seed(p["ytid"], 0), "system": args.system,
                     "backbone_build_id": build_id, "adapter_state": "off", "checkpoint": ck_sha,
                     "ddim_steps": ddim, "eta": ETA, "guidance": GUIDANCE, "latent_t": T,
                     "duration_s": ALT_DURATION, "n_samples": int(np.asarray(w).squeeze().shape[-1]),
                     "device": str(dev), "wav": path, "wav_sha256": G0.sha_file(path)})

    prov = {**G0._git_info(), **G0._env_info(dev)}
    src = G0.SOURCE_CKPT.get(build_id)
    prov["source_checkpoint"] = src
    if src and os.path.exists(src) and not args.dry_run_cpu:
        prov["source_checkpoint_sha256"] = G0.sha_file(src)
    prov["checkpoint_convention"] = ck_sha
    man = {"artifact": "reversal_armd_gen", "system": args.system, "operating_point": "ALT_10.24s",
           "subset_source": SUBSET, "device": str(dev),
           "recipe": {"ddim": ddim, "eta": ETA, "guidance": GUIDANCE, "latent_t": T, "duration_s": ALT_DURATION,
                      "replicates": 1, "weight_convention": "ema", "generation_salt": GENERATION_SALT_V1,
                      "adapter": "none", "best_of": 1},
           "provenance": prov, "n": len(rows), "rows": rows}
    outman = os.path.join(args.out, f"gen_manifest_{args.system}.json")
    json.dump(man, open(outman, "w"), indent=1)
    exp = 1 if args.dry_run_cpu else 80
    if len(rows) != exp:
        raise SystemExit(f"expected {exp} WAVs, wrote {len(rows)}")
    # sanity: 10.24 s => ~163840 samples at 16 kHz (dry-run ddim6 still full-length)
    if not args.dry_run_cpu:
        bad = [r for r in rows if abs(r["n_samples"] - 163840) > 2000]
        if bad:
            raise SystemExit(f"unexpected sample length (not ~10.24 s): {bad[0]['n_samples']}")
    print(f"generated {len(rows)} ALT wavs -> {args.out}; sample_len {rows[0]['n_samples']}; manifest {outman}")
    print("ARM-D-GENERATOR PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
