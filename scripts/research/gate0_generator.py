#!/usr/bin/env python3
"""Gate-0 / phenomenon PRODUCTION generator (EMA convention, common-random-number design).

Reads the frozen prereg v4 and generates exactly the pre-registered experiment:
  latent_t=96 (3.84 s), DDIM steps=50, guidance_scale=2.5, **DDIM eta=0.0**, n_gen=1,
  weight_convention=EMA (materialized pre-LoRA, runtime EMA disabled).

Paired seeds: for each (battery ytid, replicate r in 0..2) a deterministic seed is derived as
sha256(SALT | ytid | r); the SAME initial latent x_T is used for EVERY compared system
(dense, dense+LoRA, p1-pruned, p1-pruned+LoRA, p1-recovered, p1-recovered+LoRA). With eta=0 this is a
clean common-random-number design: the only thing that varies across systems is the weights.

Every generated WAV manifest row carries: ytid, prompt_index, replicate_index, seed, backbone_id,
adapter_id/sha, checkpoint_sha, ddim_steps, eta, guidance, latent_t, wav_sha256.

`--verify-paired-noise` proves (CPU, no generation) that x_T is deterministic per (ytid, r) and
identical across systems. `--dry-run-cpu` generates a tiny bounded set to validate the full path.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
PREREG = "configs/research/icassp_gate0_prereg.yaml"
BATTERY = "configs/research/icassp_gate0_battery.json"
PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
DENSE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
RECOV_CKPT = "data/checkpoints/l1_p1_finetuned_global_step_999999.ckpt"
SEED_SALT = "icassp-gate0-noise-20260826"

_orig_load = torch.load
def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu"); return _orig_load(*a, **k)


def paired_seed(ytid, replicate):
    h = hashlib.sha256(f"{SEED_SALT}|{ytid}|{replicate}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def make_x_T(ytid, replicate, C, T, F):
    g = torch.Generator().manual_seed(paired_seed(ytid, replicate))
    return torch.randn(1, C, T, F, generator=g)


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# Source checkpoint per backbone (item 8: stamp the real source checkpoint SHA256).
SOURCE_CKPT = {
    "dense": DENSE_CKPT,
    "p1_pruned_ema_reconstructed": DENSE_CKPT,   # derived from the DENSE EMA (L1 selection)
    "p1_recovered": RECOV_CKPT,
}


def _git_info():
    import subprocess
    def _q(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    sha = _q(["git", "rev-parse", "HEAD"])
    dirty = _q(["git", "status", "--porcelain"])
    return {"git_sha": sha, "git_dirty": bool(dirty) if dirty is not None else None}


def _env_info(dev):
    import platform
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None), "device": str(dev), "gpu": None}
    try:
        if dev.type == "cuda" and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def build_backbone(backbone_id, config, device):
    """Full LatentDiffusion pipeline with the backbone's U-Net under the EMA convention."""
    from measure_tgen import build_model
    from research_pruning.eval.ema_weights import materialize_ema_into_unet, ema_unet_state_dict
    import research_pruning.diagnostics.random_masks as rm
    model, _ = build_model(config, device)      # dense pipeline (VAE/CLAP/vocoder shared)
    model = model.float()
    dsd = _orig_load(DENSE_CKPT, map_location="cpu"); dsd = dsd.get("state_dict", dsd)
    ck_sha = None
    if backbone_id == "dense":
        materialize_ema_into_unet(model.model.diffusion_model, dsd, strict=True)
        ck_sha = "dense_ema"
    elif backbone_id == "p1_pruned_ema_reconstructed":
        l1 = rm.load_l1_ranking(PKL)
        dense_ema_base, _ = ema_unet_state_dict(dsd)          # relative-key dense EMA
        pruned = rm.materialize(dense_ema_base, l1, config, channel_mult=[1, 2, 3, 1]).float()
        model.model.diffusion_model = pruned.to(device)
        ck_sha = "prune(dense_ema)"
    elif backbone_id == "p1_recovered":
        rsd = _orig_load(RECOV_CKPT, map_location="cpu"); rsd = rsd.get("state_dict", rsd)
        pruned = rm.build_pruned_unet(config, [1, 2, 3, 1]).float()
        rel = {k[len("model.diffusion_model."):]: v for k, v in rsd.items() if k.startswith("model.diffusion_model.")}
        pruned.load_state_dict(rel, strict=True)
        materialize_ema_into_unet(pruned, rsd, strict=True)   # recovered's own EMA
        model.model.diffusion_model = pruned.to(device)
        ck_sha = "recovered_ema"
    else:
        raise SystemExit(f"unknown backbone {backbone_id}")
    model.model.diffusion_model.eval()
    model.latent_t_size = yaml.safe_load(open(PREREG))["gate0"]["data"]["latent_t_size"]
    model.use_ema = False
    model.eval()
    return model, ck_sha


def generate(model, caption, x_T, ddim_steps, guidance, eta):
    from audioldm_train.utilities.data.dataset import AudioDataset
    from torch.utils.data import DataLoader
    kim = json.load(open("artifacts/icassp_gate0/kim193_train_manifest.json"))["data"][0]["wav"]
    ds = AudioDataset(config=model._gate0_config, split="test", waveform_only=False,
                      dataset_json={"data": [{"wav": kim, "caption": caption}]})
    batch = next(iter(DataLoader(ds, batch_size=1)))
    with torch.no_grad():
        _, c = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
        uc = None
        if guidance != 1.0:
            uc = {}
            for key in model.cond_stage_model_metadata:
                idx = model.cond_stage_model_metadata[key]["model_idx"]
                uc[key] = model.cond_stage_models[idx].get_unconditional_condition(1)
        samples, _ = model.sample_log(cond=c, batch_size=1, ddim=True, ddim_steps=ddim_steps,
                                      eta=eta, x_T=x_T, unconditional_guidance_scale=guidance,
                                      unconditional_conditioning=uc, use_plms=False)
        mel = model.decode_first_stage(samples)
        wav = model.mel_spectrogram_to_waveform(mel)
    return np.asarray(wav).squeeze()


def _select_device(dry_run_cpu):
    if dry_run_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_lora(model, pre):
    """Inject the frozen Gate-0 LoRA (to_q/to_v r8/alpha16) into the backbone U-Net (lora_B=0 => identity)."""
    import gate0_trainer as GT
    from audioldm_peft import setup_peft
    setup_peft(model, GT.make_peft_cfg(pre["gate0"]))


def load_adapter(model, adapter_path):
    """Load a trained adapter-only state dict (lora_A/lora_B keyed by module name) into the injected model."""
    from audioldm_peft.layers import LoRALinear, LoRAConv2d
    sd = _orig_load(adapter_path, map_location="cpu")
    mods = {n: m for n, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv2d))}
    loaded = 0
    with torch.no_grad():
        for name, m in mods.items():
            ka, kb = name + ".lora_A", name + ".lora_B"
            if ka in sd and kb in sd:
                m.lora_A.copy_(sd[ka].to(m.lora_A.device, m.lora_A.dtype))
                m.lora_B.copy_(sd[kb].to(m.lora_B.device, m.lora_B.dtype))
                loaded += 1
    if loaded != len(mods) or loaded == 0:
        raise SystemExit(f"adapter load mismatch: {loaded}/{len(mods)} modules loaded from {adapter_path}")
    return loaded


def adapter_sha_for(adapter_path):
    """Prefer the sha recorded in the sibling meta; else hash the file."""
    meta_p = os.path.join(os.path.dirname(adapter_path), "gate0_adapter_meta.json")
    if os.path.exists(meta_p):
        s = json.load(open(meta_p)).get("adapter_sha256")
        if s:
            return s
    return sha_file(adapter_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dense",
                    choices=["dense", "p1_pruned_ema_reconstructed", "p1_recovered"])
    ap.add_argument("--adapter", default=None, help="path to a trained gate0_adapter.pt (lora_A/lora_B)")
    ap.add_argument("--adapter-mode", default="off", choices=["off", "on", "both"],
                    help="off=backbone only; on=backbone+adapter; both=paired backbone AND backbone+adapter")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="artifacts/icassp_gate0/gen")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--verify-paired-noise", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="after writing, validate the manifest with the shared parametric validator")
    args = ap.parse_args()

    pre = yaml.safe_load(open(PREREG))
    g0, bat = pre["gate0"], pre["battery"]
    C = pre["gate0"]["data"].get("latent_c", 8)
    T = g0["data"]["latent_t_size"]; F = g0["data"]["latent_f_size"] if "latent_f_size" in g0["data"] else 16
    ddim = bat["ddim_steps"]; guidance = bat["guidance_scale"]; eta = bat.get("ddim_eta", 0.0)
    replicates = args.replicates or bat["n_seeds"]
    prompts = json.load(open(BATTERY))["prompts"]
    if args.dry_run_cpu:
        prompts = prompts[: (args.n_prompts or 2)]; ddim = 6; replicates = min(replicates, 2)
    elif args.n_prompts:
        prompts = prompts[: args.n_prompts]

    # --- paired-noise proof (no model build) ---
    if args.verify_paired_noise:
        x0a = make_x_T(prompts[0]["ytid"], 0, C, T, F)
        x0b = make_x_T(prompts[0]["ytid"], 0, C, T, F)   # same (ytid,r) twice -> identical
        x1 = make_x_T(prompts[0]["ytid"], 1, C, T, F)    # different replicate -> differs
        res = {"deterministic_same_ytid_r": bool(torch.equal(x0a, x0b)),
               "differs_across_replicate": bool(not torch.equal(x0a, x1)),
               "x_T_shape": list(x0a.shape),
               "seed_p0_r0": paired_seed(prompts[0]["ytid"], 0),
               "note": "x_T depends ONLY on (ytid, replicate) -> identical across all systems"}
        res["PASS"] = res["deterministic_same_ytid_r"] and res["differs_across_replicate"]
        print(json.dumps(res, indent=2)); print("PAIRED-NOISE", "PASS" if res["PASS"] else "FAIL")
        return 0 if res["PASS"] else 1

    os.makedirs(args.out, exist_ok=True)
    dev = (torch.device("cpu") if args.dry_run_cpu else
           (torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
            else torch.device(args.device)))
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PREFLIGHT FAIL: --device cuda but no CUDA available")
    torch.load = _cpu_load
    import audioldm_train.modules.latent_diffusion.ddim as _ddim
    _oi = _ddim.DDIMSampler.__init__
    _ddim.DDIMSampler.__init__ = lambda s, m, schedule="linear", device=None, **k: _oi(s, m, schedule=schedule, device=dev, **k)

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = g0["data"]["train_clip_seconds"]
    model, ck_sha = build_backbone(args.backbone, config, dev)
    model._gate0_config = config

    # adapter modes: off = backbone only; on = backbone+adapter; both = paired backbone AND backbone+adapter.
    # Inject LoRA once (lora_B=0 => the 'off' pass is bit-identical to the pure backbone); the 'on' pass
    # loads the trained adapter. x_T is deterministic per (ytid, replicate), so off/on are paired.
    if args.adapter_mode in ("on", "both"):
        if not args.adapter:
            raise SystemExit("--adapter <path> is required for --adapter-mode on/both")
        inject_lora(model, pre)
        model = model.to(dev)
        a_sha = adapter_sha_for(args.adapter)
        passes = [("off", "none"), ("on", a_sha)] if args.adapter_mode == "both" else [("on", a_sha)]
    else:
        passes = [("off", "none")]

    import soundfile as sf
    rows = []
    for state, aid in passes:
        if state == "on":
            load_adapter(model, args.adapter)
        model.eval()
        for pi, p in enumerate(prompts):
            for r in range(replicates):
                x_T = make_x_T(p["ytid"], r, C, T, F).to(dev)
                w = generate(model, p["caption"], x_T, ddim, guidance, eta)
                tag = "adapter" if state == "on" else "noadapter"
                fn = f"{args.backbone}_{tag}_p{pi}_r{r}.wav"
                path = os.path.join(args.out, fn)
                sf.write(path, w.astype(np.float32), 16000, subtype="PCM_16")
                rows.append({"ytid": p["ytid"], "prompt_index": pi, "replicate_index": r,
                             "seed": paired_seed(p["ytid"], r), "backbone_id": args.backbone,
                             "adapter_state": state, "adapter_id": aid, "checkpoint": ck_sha,
                             "ddim_steps": ddim, "eta": eta, "guidance": guidance, "latent_t": T,
                             "device": str(dev), "wav": path, "wav_sha256": sha_file(path)})
    tag = f"{args.backbone}_{args.adapter_mode}"
    # --- hardened provenance (item 8): git, env/GPU, source-ckpt + dense/sliced adapter SHAs ---
    prov = {**_git_info(), **_env_info(dev)}
    src_ckpt = SOURCE_CKPT.get(args.backbone)
    prov["source_checkpoint"] = src_ckpt
    if src_ckpt and os.path.exists(src_ckpt) and not args.dry_run_cpu:
        prov["source_checkpoint_sha256"] = sha_file(src_ckpt)   # heavy hash; skipped on CPU dry-run
    prov["checkpoint_convention"] = ck_sha
    if args.adapter:
        prov["adapter_path"] = args.adapter
        prov["adapter_sha256"] = adapter_sha_for(args.adapter)
        ameta = os.path.join(os.path.dirname(args.adapter),
                             os.path.basename(args.adapter).replace(".pt", "_meta.json"))
        # sliced-adapter meta records the dense ancestor; dense-adapter meta records itself.
        for mp in (ameta, os.path.join(os.path.dirname(args.adapter), "gate0_adapter_meta.json")):
            if os.path.exists(mp):
                m = json.load(open(mp))
                if m.get("dense_adapter_sha256"):
                    prov["dense_adapter_sha256"] = m["dense_adapter_sha256"]
                if m.get("sliced_adapter_sha256"):
                    prov["sliced_adapter_sha256"] = m["sliced_adapter_sha256"]
                if m.get("adapter_sha256") and "dense_adapter_sha256" not in prov:
                    prov["dense_adapter_sha256"] = m["adapter_sha256"]
                break
    man = {"backbone": args.backbone, "adapter": args.adapter, "adapter_mode": args.adapter_mode,
           "device": str(dev),
           "recipe": {"ddim": ddim, "eta": eta, "guidance": guidance, "latent_t": T,
                      "replicates": replicates, "weight_convention": "ema", "seed_salt": SEED_SALT},
           "provenance": prov,
           "n": len(rows), "rows": rows}
    outman = os.path.join(args.out, f"gen_manifest_{tag}.json")
    json.dump(man, open(outman, "w"), indent=1)
    print(f"generated {len(rows)} wavs -> {args.out}; manifest {outman}")

    if args.validate:
        from research_pruning.manifest_validator import ManifestSpec, assert_valid
        yt = {pi: p["ytid"] for pi, p in enumerate(prompts)}
        state_ids = {st: ({"none"} if aid == "none" else {aid}) for st, aid in passes}
        spec = ManifestSpec(
            n_prompts=len(prompts), replicates=replicates, battery_ytids=yt,
            backbones={args.backbone}, adapter_state_ids=state_ids,
            recipe={"ddim_steps": ddim, "guidance": guidance, "eta": eta, "latent_t": T},
            seed_salt=SEED_SALT)
        summary = assert_valid(man, spec)
        print("MANIFEST-VALIDATED", json.dumps(summary))
    print("GENERATOR", "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
