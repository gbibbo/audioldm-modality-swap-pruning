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
# REVIEWER2-FOLLOWUP (docs/reviewer2_followup.md): new batteries / systems, frozen before generation.
TEXTFT = "data/checkpoints/audioldm-m-text-ft.ckpt"          # public dense text-FT reference (TEXTFT-CHECKPOINT-AUDIT)
REC_P1 = "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt"   # severity-1 recovered (1,2,3,1)
R2_GEN_SALT = "REVIEWER2-FOLLOWUP|GENERATION|2026-09-05"
CLOTHO_MANIFEST = "configs/research/r2_clotho_manifest.json"
MUSIC_EXT_MANIFEST = "configs/research/r2_music_ext_manifest.json"
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
    # DRAFT5-OPSWEEP-1 (docs/draft5_opsweep.md): two intermediate durations so the duration
    # "interaction" stops being a two-point slope. Same 192 AudioCaps prompts, replicate 0, same
    # integer seed per ytid as ac_short/ac_native (x_T shape differs with the latent length) --
    # identical convention to music_native.
    "ac_d128": (AC_MANIFEST, 1, 128, 5.12, GEN_SALT, "ytid", "prompt_index"),
    "ac_d192": (AC_MANIFEST, 1, 192, 7.68, GEN_SALT, "ytid", "prompt_index"),
    # REVIEWER2-FOLLOWUP: one point BEYOND the fine-tuning duration (15.36 s, latent 384), same seed
    # convention as the sweep (same integer seed per ytid; x_T shape follows the latent length).
    "ac_d384": (AC_MANIFEST, 1, 384, 15.36, GEN_SALT, "ytid", "prompt_index"),
    # REVIEWER2-FOLLOWUP: Clotho evaluation-split battery (96 x r0) at both durations; new salt.
    "clotho_short":  (CLOTHO_MANIFEST, 1, 96,  3.84,  R2_GEN_SALT, "clip_id", "prompt_index"),
    "clotho_native": (CLOTHO_MANIFEST, 1, 256, 10.24, R2_GEN_SALT, "clip_id", "prompt_index"),
    # REVIEWER2-FOLLOWUP: hip-hop battery extension (all 63 remaining eligible prompts x r0); new salt.
    "music_ext":        (MUSIC_EXT_MANIFEST, 1, 96,  3.84,  R2_GEN_SALT, "ytid", "prompt_index"),
    "music_ext_native": (MUSIC_EXT_MANIFEST, 1, 256, 10.24, R2_GEN_SALT, "ytid", "prompt_index"),
}
DENSE_OK = ("dense_native", "ac_short", "ac_native", "ac_d128", "ac_d192",
            # REVIEWER2-FOLLOWUP: dense anchors on the hip-hop cells (E6/E7), Clotho (E5), beyond-native (E1c)
            "music", "music_native", "music_ext", "music_ext_native", "clotho_short", "clotho_native", "ac_d384")


def gen_seed(context, ytid, rep):
    if context == "dense_native":                 # reuse the frozen Arm-D r0 seed convention
        from research_pruning.eval.reversal import generation_seed as v1_seed
        return v1_seed(ytid, rep)
    salt = CTX[context][4]                        # frozen contexts: GEN_SALT (unchanged); R2 contexts: R2_GEN_SALT
    return derive_paired_seed(salt, ytid, rep)


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
    # ---- REVIEWER2-FOLLOWUP systems (docs/reviewer2_followup.md)
    if system == "textft":                        # public dense text-FT reference: dense architecture, its own EMA
        from measure_tgen import build_model
        model, _ = build_model(config, dev); model = model.float()
        tsd = G0._orig_load(TEXTFT, map_location="cpu"); tsd = tsd.get("state_dict", tsd)
        unet = model.model.diffusion_model
        rel = {k[len("model.diffusion_model."):]: v for k, v in tsd.items() if k.startswith("model.diffusion_model.")}
        unet.load_state_dict(rel, strict=True)
        materialize_ema_into_unet(unet, tsd, strict=True)
        return unet, "textft_ema"
    if system == "p1_pruned":                     # severity-1 P: L1 selection on the dense EMA, [1,2,3,1] (gate0 convention)
        unet = rm.materialize(ema_base, ranking, config, channel_mult=[1, 2, 3, 1]).float()
        return unet, "prune(dense_ema)[1,2,3,1]"
    if system == "p1_recovered":                  # severity-1 P+FT: public recovered (1,2,3,1), its own EMA
        rsd = G0._orig_load(REC_P1, map_location="cpu"); rsd = rsd.get("state_dict", rsd)
        unet = rm.build_pruned_unet(config, [1, 2, 3, 1]).float()
        rel = {k[len("model.diffusion_model."):]: v for k, v in rsd.items() if k.startswith("model.diffusion_model.")}
        unet.load_state_dict(rel, strict=True)
        materialize_ema_into_unet(unet, rsd, strict=True)
        return unet, "recovered1_p1_ema"
    if system in ("shortft", "longft"):           # E3 / REVIEWER2-FOLLOWUP-EXT: this project's full fine-tunes of pruned2_A (raw weights)
        env = "SHORTFT_UNET" if system == "shortft" else "LONGFT_UNET"
        path = os.environ.get(env)
        if not path or not os.path.exists(path):
            raise SystemExit(f"PREFLIGHT FAIL: system {system} needs {env}=<path to the saved U-Net state_dict>")
        ssd = G0._orig_load(path, map_location="cpu")
        unet = rm.build_pruned_unet(config, CM).float()
        unet.load_state_dict(ssd["unet"] if "unet" in ssd else ssd, strict=True)
        return unet, f"{system}_raw:{G0.sha_file(path)[:16]}"
    if system == "denseft":                        # REVIEWER2-FOLLOWUP-EXT: this project's full fine-tune of the DENSE model (raw weights)
        path = os.environ.get("DENSEFT_UNET")
        if not path or not os.path.exists(path):
            raise SystemExit("PREFLIGHT FAIL: system denseft needs DENSEFT_UNET=<path to the saved dense U-Net state_dict>")
        from measure_tgen import build_model
        model, _ = build_model(config, dev); model = model.float()
        dsd = G0._orig_load(path, map_location="cpu")
        model.model.diffusion_model.load_state_dict(dsd["unet"] if "unet" in dsd else dsd, strict=True)
        return model.model.diffusion_model, f"denseft_raw:{G0.sha_file(path)[:16]}"
    raise SystemExit(f"unknown system {system}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["pruned2_A", "pruned2_B", "recovered2", "dense",
                                                       "textft", "p1_pruned", "p1_recovered", "shortft", "longft", "denseft"])
    ap.add_argument("--context", required=True, choices=list(CTX))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_xsev_gen")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--indices", default="", help="resume: comma-list of prompt_index (ikey) to (re)generate; "
                    "seeds/x_T/everything else unchanged. Writes an index-suffixed manifest. Empty = full set.")
    # DRAFT5-OPSWEEP-1: sampler recipe. "frozen" reproduces every earlier run bit-for-bit and stays
    # the default; "published" is Singh et al.'s reported recipe (DDIM 200, guidance 3.5), used only
    # by the E2b spot check. A non-frozen recipe MUST carry --tag so its WAVs cannot collide with,
    # or be mistaken for, the frozen ones.
    ap.add_argument("--recipe", default="frozen", choices=["frozen", "published"])
    ap.add_argument("--first-n", type=int, default=0,
                    help="outcome-blind subset: the first N prompts in frozen manifest order (0 = all)")
    ap.add_argument("--tag", default="", help="suffix for WAV and manifest names (keeps runs separate)")
    args = ap.parse_args()
    if args.recipe != "frozen" and not args.tag:
        raise SystemExit("PREFLIGHT FAIL: a non-frozen recipe requires --tag")
    if args.first_n and args.indices:
        raise SystemExit("PREFLIGHT FAIL: --first-n and --indices are mutually exclusive")
    if args.context == "dense_native" and args.system != "dense":
        raise SystemExit("dense_native context is for --system dense only")
    if args.system == "dense" and args.context not in DENSE_OK:
        # dense: the Arm-D native control (frozen), the XSEV-DENSE-192-CONTROL cells
        # (docs/xsev_dense_192_control.md), the DRAFT5-OPSWEEP-1 sweep points (docs/draft5_opsweep.md)
        # or the REVIEWER2-FOLLOWUP anchors (docs/reviewer2_followup.md)
        raise SystemExit(f"dense generates only {DENSE_OK}")
    if args.system in ("textft", "p1_pruned", "p1_recovered", "shortft", "longft", "denseft") and args.context not in ("ac_short", "ac_native"):
        raise SystemExit("textft / p1_* / shortft / longft / denseft generate only ac_short and ac_native (docs/reviewer2_followup*.md)")

    manifest_path, reps, T, duration, _salt, ykey, ikey = CTX[args.context]
    prompts = json.load(open(manifest_path))["prompts"]
    ddim, guidance = (50, 2.5) if args.recipe == "frozen" else (200, 3.5)
    idx_suffix = args.tag
    if args.first_n:
        prompts = sorted(prompts, key=lambda q: q[ikey])[:args.first_n]
        if len(prompts) != args.first_n:
            raise SystemExit(f"PREFLIGHT FAIL: asked for the first {args.first_n} prompts, got {len(prompts)}")
    if args.indices:
        want = {int(x) for x in args.indices.split(",") if x.strip() != ""}
        prompts = [p for p in prompts if p[ikey] in want]
        got = {p[ikey] for p in prompts}
        if got != want:
            raise SystemExit(f"PREFLIGHT FAIL: requested indices {sorted(want)} but manifest has {sorted(got)}")
        idx_suffix = f"{args.tag}_idx{min(want)}-{max(want)}"
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
            w = G0.generate(model, caption, x_T, ddim, guidance, 0.0)
            w = np.asarray(w).squeeze().astype(np.float32)
            path = os.path.join(args.out, f"{args.system}_{args.context}{args.tag}_p{pi}_r{r}.wav")
            sf.write(path, w, 16000, subtype="PCM_16")
            rows.append({"ytid": ytid, "prompt_index": pi, "replicate_index": r,
                         "seed": gen_seed(args.context, ytid, r), "system": args.system, "context": args.context,
                         "checkpoint": ck_sha, "ddim": ddim, "guidance": guidance, "eta": 0.0, "latent_t": T,
                         "duration_s": duration, "n_samples": int(w.shape[-1]), "device": str(dev),
                         "wav": path, "wav_sha256": G0.sha_file(path)})
    exp = 1 if args.dry_run_cpu else len(prompts) * reps
    if len(rows) != exp:
        raise SystemExit(f"expected {exp} WAVs, wrote {len(rows)}")
    prov = {**G0._git_info(), **G0._env_info(dev), "checkpoint_convention": ck_sha}
    man = {"artifact": "reversal_xsev_gen", "system": args.system, "context": args.context,
           "manifest": manifest_path, "recipe": {"name": args.recipe, "ddim": ddim, "guidance": guidance,
           "eta": 0.0, "latent_t": T, "duration_s": duration, "reps": reps, "gen_salt": CTX[args.context][4],
           "weight_convention": "ema", "first_n": args.first_n or None, "tag": args.tag or None},
           "provenance": prov, "n": len(rows), "rows": rows}
    outman = os.path.join(args.out, f"gen_manifest_{args.system}_{args.context}{idx_suffix}.json")
    json.dump(man, open(outman, "w"), indent=1)
    print(f"generated {len(rows)} wavs [{args.system}/{args.context}] len {rows[0]['n_samples']} -> {args.out}")
    print("XSEV-GENERATOR PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
