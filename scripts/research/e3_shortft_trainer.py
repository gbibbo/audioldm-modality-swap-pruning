#!/usr/bin/env python3
"""REVIEWER2-FOLLOWUP E3 — short-duration FULL fine-tune of the severity-2 pruned checkpoint (T4 job entry).

Protocol: docs/reviewer2_followup.md §8 (frozen + sha256 sidecar before any output). Recipe = Singh's /
upstream `audioldm_original_medium.yaml` with ONE change, the training duration (10.24 s -> 3.84 s,
latent 96): start from pruned2_A (A' L1 selection on the dense EMA, [1,2,1,1]); train ALL U-Net parameters
(VAE, CLAP conditioner, vocoder frozen); AdamW lr 1e-4 constant, torch defaults (betas 0.9/0.999,
weight decay 0.01, eps 1e-8), no scheduler, no grad clip (none upstream); batch 2; CFG dropout 0.1
(`unconditional_prob_cfg`); FP32; data = preprocessed AudioCaps TRAIN split (49 502 clips), random 3.84-s
crops (the dataset's own random_segment_wav). N = 20 000 optimizer updates.

Self-gate (protocol §8): the first 200 steps are timed; if sec/step * 20000 / 3600 * 0.89 cr/h > 2.0 cr the
job writes the benchmark and STOPS without training further ("BENCH-ONLY STOP").

Outputs (gitignored): <out>/shortft_unet.pt {"unet": state_dict (raw weights), "meta": {...}} + sha256,
<out>/bench.json, <out>/train_log.jsonl, resume checkpoints every 5000 steps.

  .venv/bin/python scripts/research/e3_shortft_trainer.py --dry-run-cpu     # 3 steps on CPU + assertions
  .venv/bin/python scripts/research/e3_shortft_trainer.py                   # full run (needs CUDA)
"""
import argparse, hashlib, json, math, os, sys, time
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import torch, yaml

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
TRAIN_JSON = "data/dataset/metadata/audiocaps/datafiles/audiocaps_train_label.json"
DATA_ROOT = "data/dataset/audioset"
OUT_DEFAULT = "artifacts/icassp_gate0/r2_shortft"
N_STEPS = 20000
BENCH_STEPS = 200
BENCH_CAP_CR = 2.0
CR_PER_GPU_H = 0.89
LR, BATCH, UNCOND_P, DURATION, LATENT_T = 1e-4, 2, 0.1, 3.84, 96
SEED = 20260905


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_head():
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return None


def build_loader(config, n_limit=None):
    from torch.utils.data import DataLoader
    from audioldm_train.utilities.data.dataset import AudioDataset
    data = json.load(open(TRAIN_JSON))["data"]
    rows = [{"wav": os.path.join(DATA_ROOT, d["wav"]), "caption": d["caption"]} for d in data]
    if n_limit:
        rows = rows[:n_limit]
    ds = AudioDataset(config=config, split="train", waveform_only=False, dataset_json={"data": rows})
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=2, generator=g), len(rows)


def train_one_step(model, batch, opt):
    with torch.no_grad():                                    # VAE / CLAP frozen: no graph through them
        z, cond = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=UNCOND_P)
    t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=z.device).long()
    loss, _ = model.p_losses(z, cond, t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss)


def save_unet(unet, out, meta, name="shortft_unet.pt"):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    torch.save({"unet": {k: v.detach().cpu() for k, v in unet.state_dict().items()}, "meta": meta}, path)
    return path, sha_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="bounded run (smoke); full run leaves it None")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--resume", action="store_true", help="resume from <out>/resume/latest.pt if present")
    a = ap.parse_args()
    if not a.dry_run_cpu and not torch.cuda.is_available():
        raise SystemExit("refusing to run the full trainer without CUDA (use --dry-run-cpu)")
    dev = torch.device("cpu" if a.dry_run_cpu else "cuda")
    torch.manual_seed(SEED)
    import gate0_generator as G0
    torch.load = G0._cpu_load
    import reversal_xsev_gen as XG
    from measure_tgen import build_model

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = DURATION
    model, _ = build_model(config, dev); model = model.float()
    unet, ck = XG.build_backbone("pruned2_A", config, dev)   # A' L1 selection on the dense EMA, [1,2,1,1]
    model.model.diffusion_model = unet.to(dev)
    model.latent_t_size = LATENT_T
    model.use_ema = False
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.diffusion_model.parameters():
        p.requires_grad_(True)
    model.model.diffusion_model.train()
    n_train = sum(p.numel() for p in model.model.diffusion_model.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.model.diffusion_model.parameters(), lr=LR)

    R = {"assertions": {}, "report": {"start_checkpoint": ck, "n_unet_params": n_train, "n_pipeline_params": n_all,
                                       "trainable_only_unet": True, "lr": LR, "batch": BATCH, "uncond_p": UNCOND_P,
                                       "duration": DURATION, "latent_t": LATENT_T, "n_steps_planned": N_STEPS,
                                       "seed": SEED, "commit": git_head(), "device": str(dev)}}
    R["assertions"]["only_unet_trainable"] = all((p.requires_grad == (id(p) in {id(q) for q in model.model.diffusion_model.parameters()}))
                                                 for p in model.parameters())
    R["assertions"]["unet_is_71M"] = 70e6 < n_train < 72e6
    loader, n_data = build_loader(config, n_limit=(8 if a.dry_run_cpu else None))
    R["report"]["n_data"] = n_data
    vae_ref = next(model.first_stage_model.parameters()).detach().clone()
    w_ref = next(model.model.diffusion_model.parameters()).detach().clone()

    # resume (insurance only; approximate: shuffle order differs after a resume)
    start_step = 0
    rpath = os.path.join(a.out, "resume", "latest.pt")
    if a.resume and os.path.exists(rpath):
        ck_r = G0._orig_load(rpath, map_location="cpu")
        model.model.diffusion_model.load_state_dict(ck_r["unet"]); opt.load_state_dict(ck_r["optimizer"])
        start_step = int(ck_r["step"]); R["report"]["resumed_from_step"] = start_step

    cap = a.max_steps if a.max_steps is not None else (3 if a.dry_run_cpu else N_STEPS)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
    step, losses, t0, bench_done = start_step, [], time.perf_counter(), start_step > 0
    stop = False
    while not stop:
        for batch in loader:
            if a.dry_run_cpu and step == start_step:
                R["assertions"]["waveform_61440"] = batch["waveform"].shape[-1] == int(DURATION * 16000)
                with torch.no_grad():
                    z0, _ = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
                R["assertions"]["latent_time_96"] = z0.shape[2] == LATENT_T
            loss = train_one_step(model, batch, opt); step += 1; losses.append(loss)
            if step % 100 == 0 or a.dry_run_cpu:
                logf.write(json.dumps({"step": step, "loss": loss, "t": time.time()}) + "\n"); logf.flush()
            if not bench_done and step - start_step == BENCH_STEPS:
                sps = (time.perf_counter() - t0) / BENCH_STEPS
                proj_cr = sps * N_STEPS / 3600 * CR_PER_GPU_H
                bench = {"sec_per_step": sps, "projected_train_cr_20000": proj_cr, "cap_cr": BENCH_CAP_CR,
                         "vram_peak_gb": (torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None),
                         "mean_loss_first_200": sum(losses) / len(losses)}
                json.dump(bench, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
                print("BENCH", json.dumps(bench), flush=True)
                bench_done = True
                if proj_cr > BENCH_CAP_CR and not a.dry_run_cpu:
                    print("BENCH-ONLY STOP: projected training cost exceeds the protocol gate", flush=True)
                    R["report"]["bench_only_stop"] = True; stop = True; break
            if step % 5000 == 0 and not a.dry_run_cpu:
                os.makedirs(os.path.join(a.out, "resume"), exist_ok=True)
                tmp = rpath + ".tmp"
                torch.save({"unet": model.model.diffusion_model.state_dict(), "optimizer": opt.state_dict(), "step": step}, tmp)
                os.replace(tmp, rpath); print(f"resume checkpoint @ {step}", flush=True)
            if step >= cap:
                stop = True; break
        if a.dry_run_cpu and not stop:
            stop = True
    wall = time.perf_counter() - t0
    R["report"]["ran_steps"] = step - start_step; R["report"]["sec_per_step"] = wall / max(1, step - start_step)
    R["report"]["last_losses"] = losses[-5:]
    R["assertions"]["loss_finite"] = all(math.isfinite(l) for l in losses)
    R["assertions"]["unet_updated"] = bool((next(model.model.diffusion_model.parameters()).detach() - w_ref).abs().max() > 0)
    R["assertions"]["vae_unchanged"] = bool((next(model.first_stage_model.parameters()).detach() - vae_ref).abs().max() == 0)
    if not R["report"].get("bench_only_stop"):
        meta = {**R["report"], "protocol": "docs/reviewer2_followup.md#8", "weights": "raw (no EMA)", "final_step": step}
        path, sha = save_unet(model.model.diffusion_model, a.out, meta)
        R["report"]["saved_unet"] = {"path": path, "sha256": sha}
        print("SAVED", path, sha, flush=True)
    all_ok = all(R["assertions"].values()); R["ALL_ASSERTIONS_PASS"] = all_ok
    R["mode"] = "dry-run-cpu" if a.dry_run_cpu else ("bounded" if a.max_steps else "full")
    json.dump(R, open(os.path.join(a.out, "trainer_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "report"}, indent=1))
    print("TRAINER", "PASS" if all_ok else "FAIL", "| mode", R["mode"], "| steps", step)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
