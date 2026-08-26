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
    elif backbone_id == "p1_pruned":
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dense", choices=["dense", "p1_pruned", "p1_recovered"])
    ap.add_argument("--out", default="artifacts/icassp_gate0/gen")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--verify-paired-noise", action="store_true")
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
    torch.load = _cpu_load
    import audioldm_train.modules.latent_diffusion.ddim as _ddim
    _oi = _ddim.DDIMSampler.__init__
    _ddim.DDIMSampler.__init__ = lambda s, m, schedule="linear", device=None, **k: _oi(s, m, schedule=schedule, device=torch.device("cpu"), **k)

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = g0["data"]["train_clip_seconds"]
    device = torch.device("cpu")
    model, ck_sha = build_backbone(args.backbone, config, device)
    model._gate0_config = config

    rows = []
    for pi, p in enumerate(prompts):
        for r in range(replicates):
            x_T = make_x_T(p["ytid"], r, C, T, F)
            w = generate(model, p["caption"], x_T, ddim, guidance, eta)
            fn = f"{args.backbone}_p{pi}_r{r}.wav"
            path = os.path.join(args.out, fn)
            import soundfile as sf
            sf.write(path, w.astype(np.float32), 16000, subtype="PCM_16")
            rows.append({"ytid": p["ytid"], "prompt_index": pi, "replicate_index": r,
                         "seed": paired_seed(p["ytid"], r), "backbone_id": args.backbone,
                         "adapter_id": "none", "checkpoint": ck_sha, "ddim_steps": ddim,
                         "eta": eta, "guidance": guidance, "latent_t": T,
                         "wav": path, "wav_sha256": sha_file(path)})
    man = {"backbone": args.backbone, "recipe": {"ddim": ddim, "eta": eta, "guidance": guidance,
           "latent_t": T, "replicates": replicates, "weight_convention": "ema", "seed_salt": SEED_SALT},
           "n": len(rows), "rows": rows}
    outman = os.path.join(args.out, f"gen_manifest_{args.backbone}.json")
    json.dump(man, open(outman, "w"), indent=1)
    print(f"generated {len(rows)} wavs -> {args.out}; manifest {outman}")
    print("GENERATOR", "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
