#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 generator (GPU T4 job entry). One (system, context) per invocation.

Systems: pruned2_A / pruned2_B (prune_operator on dense EMA @[1,2,1,1]), recovered2 (public dp1
recovered, EMA), dense (severity-1 dense EMA, NATIVE control only). Contexts: ac_short (192×1, 3.84s/
latent96), ac_native (192×1, 10.24s/latent256), music (64×3, 3.84s/latent96), dense_native (Arm-D 80×1,
10.24s/latent256), music_native (64×1, 10.24s/latent256; XSEV-MUSIC-NATIVE-1). CRN: x_T = f(ytid,rep) via the frozen generation salt, shared across systems within a
(context,ytid,rep). EMA convention; DDIM50/g2.5/eta0/fp32/single. Reuses gate0_generator core.

Run (GPU): .venv/bin/python scripts/research/reversal_xsev_gen.py --system pruned2_A --context ac_native
"""
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import numpy as np, torch, yaml
import gate0_generator as G0
from research_pruning.diagnostics import prune_operator as PO
import research_pruning.diagnostics.random_masks as rm
from research_pruning.eval.ema_weights import materialize_ema_into_unet, ema_unet_state_dict
from research_pruning.eval.reversal import derive_paired_seed

CONFIG, PREREG, PKL = G0.CONFIG, G0.PREREG, G0.PKL
DENSE = "data/checkpoints/audioldm-m-full.ckpt"
REC_DP1 = "data/checkpoints/l1_p1_dp1_finetuned_global_step_999999.ckpt"
GEN_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30"
AC_MANIFEST = "configs/research/xsev_audiocaps_manifest.json"
MUSIC_MANIFEST = "configs/research/xsev_music_manifest.json"
ARMD_SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
CM = [1, 2, 1, 1]
# context -> (manifest, n_reps, latent_t, duration, gen_salt_for_seed, ytid_key, idx_key)
CTX = {
    "ac_short":     (AC_MANIFEST, 1, 96,  3.84,  GEN_SALT, "ytid", "prompt_index"),
    "ac_native":    (AC_MANIFEST, 1, 256, 10.24, GEN_SALT, "ytid", "prompt_index"),
    "music":        (MUSIC_MANIFEST, 3, 96, 3.84, GEN_SALT, "ytid", "prompt_index"),
    "dense_native": (ARMD_SUBSET, 1, 256, 10.24, "ARMD", "ytid", "subset_prompt_index"),
    # XSEV-MUSIC-NATIVE-1 (docs/xsev_music_native_1.md): the missing factorial cell, music @10.24 s.
    # 64 prompts x replicate 0 only; integer seed = the frozen music replicate-0 seed (same convention
    # as ac_short/ac_native: same integer seed per ytid, x_T shape (1,8,256,16) differs from 3.84 s).
    "music_native": (MUSIC_MANIFEST, 1, 256, 10.24, GEN_SALT, "ytid", "prompt_index"),
}


def gen_seed(context, ytid, rep):
    if context == "dense_native":                 # reuse the frozen Arm-D r0 seed convention
        from research_pruning.eval.reversal import generation_seed as v1_seed
        return v1_seed(ytid, rep)
    return derive_paired_seed(GEN_SALT, ytid, rep)


def make_x_T(context, ytid, rep, C, T, F):
    g = torch.Generator().manual_seed(gen_seed(context, ytid, rep))
    return torch.randn(1, C, T, F, generator=g)


def build_backbone(system, config, dev):
    """Return (unet, ck_sha) for a severity-2 / dense system, EMA convention."""
    dsd = G0._orig_load(DENSE, map_location="cpu"); dsd = dsd.get("state_dict", dsd)
    if system == "dense":
        from measure_tgen import build_model
        model, _ = build_model(config, dev); model = model.float()
        materialize_ema_into_unet(model.model.diffusion_model, dsd, strict=True)
        return model.model.diffusion_model, "dense_ema"
    ema_base, _ = ema_unet_state_dict(dsd)
    ranking = rm.load_l1_ranking(PKL)
    if system in ("pruned2_A", "pruned2_B"):
        conv = "A" if system.endswith("_A") else "B"
        unet = PO.build_pruned_ema(ema_base, ranking, config, CM, conv)
        return unet, f"prune{conv}(dense_ema)[1,2,1,1]"
    if system == "recovered2":
        rsd = G0._orig_load(REC_DP1, map_location="cpu"); rsd = rsd.get("state_dict", rsd)
        unet = rm.build_pruned_unet(config, CM).float()
        rel = {k[len("model.diffusion_model."):]: v for k, v in rsd.items() if k.startswith("model.diffusion_model.")}
        unet.load_state_dict(rel, strict=True)
        materialize_ema_into_unet(unet, rsd, strict=True)
        return unet, "recovered2_dp1_ema"
    raise SystemExit(f"unknown system {system}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["pruned2_A", "pruned2_B", "recovered2", "dense"])
    ap.add_argument("--context", required=True, choices=list(CTX))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_xsev_gen")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--indices", default="", help="resume: comma-list of prompt_index (ikey) to (re)generate; "
                    "seeds/x_T/everything else unchanged. Writes an index-suffixed manifest. Empty = full set.")
    args = ap.parse_args()
    if args.context == "dense_native" and args.system != "dense":
        raise SystemExit("dense_native context is for --system dense only")
    if args.context != "dense_native" and args.system == "dense":
        raise SystemExit("dense only generates the dense_native control")

    manifest_path, reps, T, duration, _salt, ykey, ikey = CTX[args.context]
    prompts = json.load(open(manifest_path))["prompts"]
    ddim = 50
    idx_suffix = ""
    if args.indices:
        want = {int(x) for x in args.indices.split(",") if x.strip() != ""}
        prompts = [p for p in prompts if p[ikey] in want]
        got = {p[ikey] for p in prompts}
        if got != want:
            raise SystemExit(f"PREFLIGHT FAIL: requested indices {sorted(want)} but manifest has {sorted(got)}")
        idx_suffix = f"_idx{min(want)}-{max(want)}"
    if args.dry_run_cpu:
        prompts = prompts[:1]; reps = 1; ddim = 6

    dev = (torch.device("cpu") if args.dry_run_cpu else
           (torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
            else torch.device(args.device)))
    if args.device == "cuda" and dev.type != "cuda":
        raise SystemExit("PREFLIGHT FAIL: requested cuda unavailable")

    os.makedirs(args.out, exist_ok=True)
    torch.load = G0._cpu_load
    import audioldm_train.modules.latent_diffusion.ddim as _ddim
    _oi = _ddim.DDIMSampler.__init__
    _ddim.DDIMSampler.__init__ = lambda s, m, schedule="linear", device=None, **k: _oi(s, m, schedule=schedule, device=dev, **k)

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = duration
    pre = yaml.safe_load(open(PREREG))["gate0"]["data"]
    C = pre.get("latent_c", 8); F = pre.get("latent_f_size", 16)

    # build full pipeline once, swap in the backbone
    from measure_tgen import build_model
    model, _ = build_model(config, dev); model = model.float()
    unet, ck_sha = build_backbone(args.system, config, dev)
    model.model.diffusion_model = unet.to(dev).eval()
    model._gate0_config = config
    model.latent_t_size = T
    model.use_ema = False; model.eval()

    import soundfile as sf
    rows = []
    for p in prompts:
        pi = p[ikey]; ytid = p[ykey]; caption = p["caption"]
        for r in range(reps):
            x_T = make_x_T(args.context, ytid, r, C, T, F).to(dev)
            w = G0.generate(model, caption, x_T, ddim, 2.5, 0.0)
            w = np.asarray(w).squeeze().astype(np.float32)
            path = os.path.join(args.out, f"{args.system}_{args.context}_p{pi}_r{r}.wav")
            sf.write(path, w, 16000, subtype="PCM_16")
            rows.append({"ytid": ytid, "prompt_index": pi, "replicate_index": r,
                         "seed": gen_seed(args.context, ytid, r), "system": args.system, "context": args.context,
                         "checkpoint": ck_sha, "ddim": ddim, "guidance": 2.5, "eta": 0.0, "latent_t": T,
                         "duration_s": duration, "n_samples": int(w.shape[-1]), "device": str(dev),
                         "wav": path, "wav_sha256": G0.sha_file(path)})
    exp = 1 if args.dry_run_cpu else len(prompts) * reps
    if len(rows) != exp:
        raise SystemExit(f"expected {exp} WAVs, wrote {len(rows)}")
    prov = {**G0._git_info(), **G0._env_info(dev), "checkpoint_convention": ck_sha}
    man = {"artifact": "reversal_xsev_gen", "system": args.system, "context": args.context,
           "manifest": manifest_path, "recipe": {"ddim": ddim, "guidance": 2.5, "eta": 0.0, "latent_t": T,
           "duration_s": duration, "reps": reps, "gen_salt": GEN_SALT, "weight_convention": "ema"},
           "provenance": prov, "n": len(rows), "rows": rows}
    outman = os.path.join(args.out, f"gen_manifest_{args.system}_{args.context}{idx_suffix}.json")
    json.dump(man, open(outman, "w"), indent=1)
    print(f"generated {len(rows)} wavs [{args.system}/{args.context}] len {rows[0]['n_samples']} -> {args.out}")
    print("XSEV-GENERATOR PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
