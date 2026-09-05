#!/usr/bin/env python3
"""REVIEWER2-FOLLOWUP E3 / REVIEWER2-FOLLOWUP-EXT — full fine-tune of a pruned or dense AudioLDM-M backbone (T4 job entry).

Protocol: docs/reviewer2_followup.md §8 (E3, frozen) and docs/reviewer2_followup_ext.md (the 2x2, frozen + sha256
sidecar before any output). Recipe = Singh's / upstream `audioldm_original_medium.yaml` with the training DURATION as
the deliberate factor: start from a backbone (pruned2_A = A' L1 selection on the dense EMA, [1,2,1,1]; or dense =
AudioLDM-M-Full EMA); train ALL U-Net parameters (VAE, CLAP conditioner, vocoder frozen); AdamW lr 1e-4 constant,
torch defaults (betas 0.9/0.999, weight decay 0.01, eps 1e-8), no scheduler, no grad clip (none upstream); effective
batch 2 (batch x accum); CFG dropout 0.1 (`unconditional_prob_cfg`); FP32; data = the preprocessed AudioCaps TRAIN
split (49 502 clips), random `duration`-second crops (the dataset's own random_segment_wav). N = 20 000 optimizer
updates. Gradient accumulation (batch 1 x accum 2) keeps the dense 10.24-s arm inside a 16 GB T4 while holding the
effective batch at 2 (GroupNorm, not BatchNorm, so accumulation is numerically equivalent).

Self-gate (protocol): the first 200 optimizer updates are timed; if sec/step * N_STEPS / 3600 * 0.89 cr/h > --cap-cr
the job writes the benchmark and STOPS without training further ("BENCH-ONLY STOP") -> no checkpoint, no eval.

E3 reproducibility: every new flag DEFAULTS to E3's frozen value (duration 3.84, latent 96, backbone pruned2_A,
batch 2, accum 1, save-name shortft_unet.pt, cap 2.0), so the original invocation is unchanged.

Outputs (gitignored): <out>/<save-name> {"unet": state_dict (raw weights), "meta": {...}} + sha256,
<out>/bench.json, <out>/train_log.jsonl, resume checkpoints every 5000 steps (+ optional --mid-step insurance).

  .venv/bin/python scripts/research/e3_shortft_trainer.py --dry-run-cpu                                  # E3 path, 3 steps
  .venv/bin/python scripts/research/e3_shortft_trainer.py --backbone dense --duration 10.24 --latent-t 256 \
      --batch 1 --accum 2 --save-name denseft_native_unet.pt --cap-cr 5.6 --dry-run-cpu                  # dense 10.24 arm
"""
import argparse, hashlib, json, math, os, sys, time
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import torch, yaml

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
TRAIN_JSON = "data/dataset/metadata/audiocaps/datafiles/audiocaps_train_label.json"
DATA_ROOT = "data/dataset/audioset"
OUT_DEFAULT = "artifacts/icassp_gate0/r2_shortft"
N_STEPS = 20000                     # optimizer updates
BENCH_STEPS = 200
CR_PER_GPU_H = 0.89
LR, UNCOND_P = 1e-4, 0.1
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


def build_loader(config, batch, n_limit=None):
    from torch.utils.data import DataLoader
    from audioldm_train.utilities.data.dataset import AudioDataset
    data = json.load(open(TRAIN_JSON))["data"]
    rows = [{"wav": os.path.join(DATA_ROOT, d["wav"]), "caption": d["caption"]} for d in data]
    if n_limit:
        rows = rows[:n_limit]
    ds = AudioDataset(config=config, split="train", waveform_only=False, dataset_json={"data": rows})
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=2, generator=g), len(rows)


def forward_loss(model, batch):
    with torch.no_grad():                                    # VAE / CLAP frozen: no graph through them
        z, cond = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=UNCOND_P)
    t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=z.device).long()
    loss, _ = model.p_losses(z, cond, t)
    return loss


def save_unet(unet, out, meta, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    torch.save({"unet": {k: v.detach().cpu() for k, v in unet.state_dict().items()}, "meta": meta}, path)
    return path, sha_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="bounded run (smoke); full run leaves it None")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--backbone", default="pruned2_A", choices=["pruned2_A", "dense"])
    ap.add_argument("--duration", type=float, default=3.84)
    ap.add_argument("--latent-t", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="gradient-accumulation micro-batches per optimizer update; effective batch = batch*accum")
    ap.add_argument("--save-name", default="shortft_unet.pt")
    ap.add_argument("--cap-cr", type=float, default=2.0, help="self-gate: stop before training if the 200-step bench projects training cost above this")
    ap.add_argument("--mid-step", type=int, default=0, help="also save an extra raw-weight checkpoint at this optimizer step (0 = off)")
    ap.add_argument("--resume", action="store_true", help="resume from <out>/resume/latest.pt if present")
    a = ap.parse_args()
    if not a.dry_run_cpu and not torch.cuda.is_available():
        raise SystemExit("refusing to run the full trainer without CUDA (use --dry-run-cpu)")
    dev = torch.device("cpu" if a.dry_run_cpu else "cuda")
    eff_batch = a.batch * a.accum
    torch.manual_seed(SEED)
    import gate0_generator as G0
    torch.load = G0._cpu_load
    import reversal_xsev_gen as XG
    from measure_tgen import build_model

    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = a.duration
    model, _ = build_model(config, dev); model = model.float()
    if a.backbone == "dense":
        # reuse the dense pipeline build_model already made: materialize the AudioLDM-M-Full EMA into its own U-Net.
        # (build_backbone("dense") would build a SECOND full pipeline -> OOM on a 16 GB CPU / 15 GB T4; avoided here.)
        from research_pruning.eval.ema_weights import materialize_ema_into_unet
        dsd = G0._orig_load(XG.DENSE, map_location="cpu"); dsd = dsd.get("state_dict", dsd)
        materialize_ema_into_unet(model.model.diffusion_model, dsd, strict=True)
        ck = "dense_ema"
    else:
        unet, ck = XG.build_backbone(a.backbone, config, dev)     # pruned2_A -> A' L1 selection on the dense EMA, [1,2,1,1]
        model.model.diffusion_model = unet.to(dev)
    model.latent_t_size = a.latent_t
    model.use_ema = False
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.diffusion_model.parameters():
        p.requires_grad_(True)
    model.model.diffusion_model.train()
    n_train = sum(p.numel() for p in model.model.diffusion_model.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.model.diffusion_model.parameters(), lr=LR)

    exp_params = (70e6, 72e6) if a.backbone == "pruned2_A" else (410e6, 420e6)
    R = {"assertions": {}, "report": {"start_checkpoint": ck, "backbone": a.backbone, "n_unet_params": n_train,
                                       "n_pipeline_params": n_all, "trainable_only_unet": True, "lr": LR, "batch": a.batch,
                                       "accum": a.accum, "effective_batch": eff_batch, "uncond_p": UNCOND_P,
                                       "duration": a.duration, "latent_t": a.latent_t, "n_steps_planned": N_STEPS,
                                       "cap_cr": a.cap_cr, "seed": SEED, "commit": git_head(), "device": str(dev)}}
    R["assertions"]["only_unet_trainable"] = all((p.requires_grad == (id(p) in {id(q) for q in model.model.diffusion_model.parameters()}))
                                                 for p in model.parameters())
    R["assertions"]["unet_param_count_in_range"] = exp_params[0] < n_train < exp_params[1]
    loader, n_data = build_loader(config, a.batch, n_limit=(8 if a.dry_run_cpu else None))
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

    cap_updates = a.max_steps if a.max_steps is not None else (3 if a.dry_run_cpu else N_STEPS)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
    step, losses, t0, bench_done = start_step, [], time.perf_counter(), start_step > 0
    stop, micro, last_loss = False, 0, None
    opt.zero_grad(set_to_none=True)
    while not stop:
        for batch in loader:
            if a.dry_run_cpu and step == start_step and micro == 0:
                R["assertions"]["waveform_ok"] = batch["waveform"].shape[-1] == int(a.duration * 16000)
                with torch.no_grad():
                    z0, _ = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
                R["assertions"]["latent_time_ok"] = z0.shape[2] == a.latent_t
            loss = forward_loss(model, batch)
            (loss / a.accum).backward()
            micro += 1; last_loss = float(loss)
            if micro < a.accum:
                continue
            opt.step(); opt.zero_grad(set_to_none=True); micro = 0
            step += 1; losses.append(last_loss)
            if step % 100 == 0 or a.dry_run_cpu:
                logf.write(json.dumps({"step": step, "loss": last_loss, "t": time.time()}) + "\n"); logf.flush()
            if not bench_done and step - start_step == BENCH_STEPS:
                sps = (time.perf_counter() - t0) / BENCH_STEPS
                proj_cr = sps * N_STEPS / 3600 * CR_PER_GPU_H
                bench = {"sec_per_step": sps, "effective_batch": eff_batch, "projected_train_cr_20000": proj_cr, "cap_cr": a.cap_cr,
                         "vram_peak_gb": (torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None),
                         "mean_loss_first_200": sum(losses) / len(losses)}
                json.dump(bench, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
                print("BENCH", json.dumps(bench), flush=True)
                bench_done = True
                if proj_cr > a.cap_cr and not a.dry_run_cpu:
                    print("BENCH-ONLY STOP: projected training cost exceeds the protocol gate", flush=True)
                    R["report"]["bench_only_stop"] = True; stop = True; break
            if a.mid_step and step == a.mid_step and not a.dry_run_cpu:
                mp, ms = save_unet(model.model.diffusion_model, a.out, {"note": "mid-step insurance", "step": step},
                                   a.save_name.replace(".pt", f"_step{step}.pt"))
                R["report"]["mid_checkpoint"] = {"path": mp, "sha256": ms, "step": step}; print("MID SAVE", mp, flush=True)
            if step % 5000 == 0 and not a.dry_run_cpu:
                os.makedirs(os.path.join(a.out, "resume"), exist_ok=True)
                tmp = rpath + ".tmp"
                torch.save({"unet": model.model.diffusion_model.state_dict(), "optimizer": opt.state_dict(), "step": step}, tmp)
                os.replace(tmp, rpath); print(f"resume checkpoint @ {step}", flush=True)
            if step >= cap_updates:
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
        meta = {**R["report"], "protocol": "docs/reviewer2_followup_ext.md", "weights": "raw (no EMA)", "final_step": step}
        path, sha = save_unet(model.model.diffusion_model, a.out, meta, a.save_name)
        R["report"]["saved_unet"] = {"path": path, "sha256": sha}
        print("SAVED", path, sha, flush=True)
    all_ok = all(R["assertions"].values()); R["ALL_ASSERTIONS_PASS"] = all_ok
    R["mode"] = "dry-run-cpu" if a.dry_run_cpu else ("bounded" if a.max_steps else "full")
    json.dump(R, open(os.path.join(a.out, "trainer_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "report"}, indent=1))
    print("TRAINER", "PASS" if all_ok else "FAIL", "| mode", R["mode"], "| steps", step, "| backbone", a.backbone)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
