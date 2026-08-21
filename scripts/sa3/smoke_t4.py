#!/usr/bin/env python3
"""SA3 T4 engineering smoke (protocol section 10; overnight mandate Phase 1). RUN ON P_smoke ONLY.

Measures (median + dispersion, warm-up + cuda.synchronize):
  A load base/post fp16 + param counts
  B s/forward at batch {1,4,8} (raw field), peak VRAM, OOM-safe
  C S_traj capture (dense post 8-step); tau vs frozen schedule
  D eta_i (fp16 vs fp32 post field per level; eta=max_i)  -- the section-3.2 denominator guard
  E empty BlockMask leaves the field unchanged within fp tolerance
  F latency: dense generate at steps {4,5,6,7,8} + one block-skip(5)@8 (+ decode)
  G E-panel wavs: dense {8,7,6,5} + skip(5)@8 for every P_smoke prompt (scored on the Studio after)

Writes artifacts/sa3/smoke_t4.json + wavs under artifacts/sa3/smoke_wavs/. P_smoke scores are NOT
scientific. Per-measurement try/except so one failure still yields the rest.

GPU job:   _external/stable-audio-3/.venv/bin/python scripts/sa3/smoke_t4.py --device cuda \
               --expect-gpu T4 --expect-commit <sha>
CPU dry:   OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/smoke_t4.py --dry-run-cpu
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, fields as F, e2e, states, latency
from research_sa3 import seeds as S
from research_sa3.blockskip import block_mask

SECONDS = 10; T_LATENT = 108  # for batched s/forward synthetic states (10s effective)


def sha256_file(p):
    import hashlib; return hashlib.sha256(open(p, "rb").read()).hexdigest()


def vram_peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 4) if torch.cuda.is_available() else None


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()


def load(which, device, half):
    d = "data/sa3/small-sfx-base" if which == "base" else "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device=device)
    if half:
        model = model.half()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--out", default="artifacts/sa3/smoke_t4.json")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.dry_run_cpu:
        a.device = "cpu"
    dry = a.dry_run_cpu
    half = (a.device == "cuda")
    reps = 2 if dry else a.reps
    batches = [1] if dry else [1, 4, 8]
    steps_ladder = [8, 6] if dry else [4, 5, 6, 7, 8]
    n_prompts = 1 if dry else None
    dev = a.device

    R = {"phase": "smoke", "device": dev, "dry_run_cpu": dry, "seconds_total": SECONDS,
         "git_commit": subprocess.getoutput("git rev-parse HEAD"),
         "upstream_commit": loading.SA3_UPSTREAM_COMMIT,
         "panel_smoke_sha256": sha256_file("configs/sa3/panel_smoke.json"),
         "schedule_post_sha256": sha256_file("configs/sa3/schedule_post_10s.json"),
         "seed_table_sha256": sha256_file("configs/sa3/seed_table.json"),
         "measurements": {}, "errors": {}}

    # ---- preflight ----
    if not dry:
        assert torch.cuda.is_available(), "CUDA not available"
        name = torch.cuda.get_device_name(0)
        R["gpu_name"] = name
        if a.expect_gpu and a.expect_gpu.lower() not in name.lower():
            print(f"WARNING expected {a.expect_gpu}, got {name}")
        if a.expect_commit:
            cur = subprocess.getoutput("git rev-parse HEAD")
            assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur} != {a.expect_commit}"
            dirty = subprocess.getoutput("git status --porcelain")
            assert not dirty, f"dirty tree:\n{dirty}"

    panel = json.load(open("configs/sa3/panel_smoke.json"))
    prompts = sorted(panel["items"], key=lambda x: int(x["audiocap_id"]))
    if n_prompts:
        prompts = prompts[:n_prompts]
    sched = json.load(open("configs/sa3/schedule_post_10s.json"))["tau_levels"]

    def measure(key, fn):
        try:
            reset_vram(); t0 = time.time()
            out = fn()
            out = out if isinstance(out, dict) else {"value": out}
            out["vram_peak_gb"] = vram_peak_gb(); out["wall_s"] = round(time.time() - t0, 2)
            R["measurements"][key] = out
            print(f"[smoke] {key}: {json.dumps({k: v for k, v in out.items() if k != 'per_level'})[:200]}")
        except Exception as e:
            R["errors"][key] = f"{type(e).__name__}: {e}"
            print(f"[smoke] {key} ERROR: {e}\n{traceback.format_exc()[-800:]}")

    # ---- A: load post ----
    t0 = time.time()
    post, cfg = load("post", dev, half)
    R["measurements"]["A_load_post"] = {"load_s": round(time.time() - t0, 2), "half": half,
                                        "vram_peak_gb": vram_peak_gb()}
    print(f"[smoke] A_load_post: {R['measurements']['A_load_post']}")
    sa = e2e.wrap_model(post, cfg, dev, model_half=half)
    dtype = torch.float16 if half else torch.float32

    # ---- B: s/forward batch {1,4,8} ----
    def bfwd(bs):
        cc = F.prepare_conditioning(post, prompts[0]["caption"], SECONDS, dev, latent_len=T_LATENT, dtype=dtype)
        cc = {k: (v.repeat(bs, *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v) for k, v in cc.items()}
        g = torch.Generator().manual_seed(7)
        x = torch.randn(bs, 256, T_LATENT, generator=g).to(dev, dtype)
        t = torch.full((bs,), 0.5, device=dev, dtype=dtype)
        st = latency.time_call(lambda: F.raw_field(post, x, t, cc), device=dev, warmup=1, reps=reps)
        return {"batch": bs, "median_s": round(st["median_s"], 5), "per_sample_s": round(st["median_s"] / bs, 5),
                "iqr_s": round(st["iqr_s"], 5)}
    for bs in batches:
        measure(f"B_sforward_b{bs}", (lambda b=bs: bfwd(b)))

    # ---- C: S_traj capture ----
    def cap():
        aid = prompts[0]["audiocap_id"]; seed = S.derive_seed(0, aid, "init", 0)
        tr = states.capture_trajectory(sa, prompts[0]["caption"], SECONDS, seed, steps=8,
                                       cfg_scale=1.0, apg_scale=1.0)
        taus = [round(s[0], 5) for s in tr["states"]]
        match = all(abs(taus[i] - round(sched[i], 5)) < 1e-3 for i in range(min(8, len(taus))))
        # stash states for eta/empty-mask
        cap.states = tr["states"]
        return {"n_states": len(tr["states"]), "tau": taus, "schedule_match": match}
    measure("C_S_traj", cap)

    # ---- D: eta_i (fp16 vs fp32 post field per level) ----
    def eta():
        sts = getattr(cap, "states", None)
        if not sts:
            aid = prompts[0]["audiocap_id"]; seed = S.derive_seed(0, aid, "init", 0)
            sts = states.capture_trajectory(sa, prompts[0]["caption"], SECONDS, seed, 8, 1.0, 1.0)["states"]
        cap_ = prompts[0]["caption"]
        # fp32 fields at the 8 states
        post.float()
        cc32 = F.prepare_conditioning(post, cap_, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=torch.float32)
        f32 = []
        for tau, x in sts:
            xt = x.to(dev, torch.float32); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=torch.float32)
            f32.append(F.raw_field(post, xt, tt, cc32).float())
        # fp16 fields
        etas = []
        if half or torch.cuda.is_available():
            post.half()
            cc16 = F.prepare_conditioning(post, cap_, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=torch.float16)
            for i, (tau, x) in enumerate(sts):
                xt = x.to(dev, torch.float16); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=torch.float16)
                f16 = F.raw_field(post, xt, tt, cc16).float()
                num = (f16 - f32[i]).pow(2).sum().item(); den = f32[i].pow(2).sum().item()
                etas.append(num / den if den > 0 else float("nan"))
            post.half() if half else post.float()
        else:
            etas = [None] * len(sts)
        return {"per_level": [None if e is None else round(e, 8) for e in etas],
                "eta_max": (None if any(e is None for e in etas) else max(etas))}
    measure("D_eta", eta)

    # ---- E: empty BlockMask ----
    def emptymask():
        sts = getattr(cap, "states", None)
        cap_ = prompts[0]["caption"]
        cc = F.prepare_conditioning(post, cap_, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=dtype)
        tau, x = sts[0]; xt = x.to(dev, dtype); tt = torch.full((xt.shape[0],), tau, device=dev, dtype=dtype)
        base_out = F.raw_field(post, xt, tt, cc)
        with block_mask(post, []):
            masked = F.raw_field(post, xt, tt, cc)
        bit = bool(torch.equal(base_out, masked))
        rel = (base_out.float() - masked.float()).pow(2).sum().item() / base_out.float().pow(2).sum().item()
        return {"bit_exact": bit, "rel_diff": rel}
    measure("E_empty_mask", emptymask)

    # ---- F/G: latency + E-panel wavs ----
    wavdir = "artifacts/sa3/smoke_wavs"; os.makedirs(wavdir, exist_ok=True)
    manifest = {"reference_system": "dense8", "prompts": {}, "systems": {}}
    def latency_and_wavs():
        lat = {}
        # dense at each step count on prompt 0 (timed) + save the E-panel wavs for all prompts
        for st in steps_ladder:
            sid = f"dense{st}"
            aid0 = prompts[0]["audiocap_id"]; seed0 = S.derive_seed(0, aid0, "init", 0)
            timing = latency.time_call(
                lambda st=st, seed0=seed0: e2e.generate_audio(sa, prompts[0]["caption"], SECONDS, seed0,
                                                              steps=st, cfg_scale=1.0, apg_scale=1.0),
                device=dev, warmup=1, reps=max(2, reps - 2))
            lat[sid] = round(timing["median_s"], 4)
        # block-skip(5)@8 latency
        aid0 = prompts[0]["audiocap_id"]; seed0 = S.derive_seed(0, aid0, "init", 0)
        tsk = latency.time_call(
            lambda: e2e.generate_audio(sa, prompts[0]["caption"], SECONDS, seed0, steps=8,
                                       cfg_scale=1.0, apg_scale=1.0, skip_blocks=[5]),
            device=dev, warmup=1, reps=max(2, reps - 2))
        lat["skip5@8"] = round(tsk["median_s"], 4)
        # G: E-panel wavs (dense 8/7/6/5 + skip5@8) for ALL smoke prompts (scored on Studio after)
        e_steps = [8] if dry else [8, 7, 6, 5]
        for st in e_steps:
            manifest["systems"].setdefault(f"dense{st}", {})
        manifest["systems"].setdefault("skip5", {})
        for it in prompts:
            aid = it["audiocap_id"]; cap_ = it["caption"]; seed = S.derive_seed(0, aid, "init", 0)
            manifest["prompts"][aid] = cap_
            for st in e_steps:
                aud = e2e.generate_audio(sa, cap_, SECONDS, seed, steps=st, cfg_scale=1.0, apg_scale=1.0)
                wp = f"{wavdir}/dense{st}_{aid}.wav"; e2e.save_wav(aud, wp); manifest["systems"][f"dense{st}"][aid] = wp
            aud = e2e.generate_audio(sa, cap_, SECONDS, seed, steps=8, cfg_scale=1.0, apg_scale=1.0, skip_blocks=[5])
            wp = f"{wavdir}/skip5_{aid}.wav"; e2e.save_wav(aud, wp); manifest["systems"]["skip5"][aid] = wp
        json.dump(manifest, open("artifacts/sa3/smoke_e_manifest.json", "w"), indent=2)
        # nearest-latency comparator demo for skip5@8
        comp = latency.nearest_latency_comparator(lat["skip5@8"], {int(k.replace("dense","")): v for k, v in lat.items() if k.startswith("dense")})
        return {"latency_s": lat, "skip5_comparator": comp, "n_wavs": sum(len(v) for v in manifest["systems"].values())}
    measure("F_latency_and_G_wavs", latency_and_wavs)

    # ---- optional base load ----
    if not dry:
        try:
            del post, sa; import gc; gc.collect(); reset_vram()
            t0 = time.time(); base, bcfg = load("base", dev, half)
            R["measurements"]["A_load_base"] = {"load_s": round(time.time() - t0, 2), "vram_peak_gb": vram_peak_gb()}
            print(f"[smoke] A_load_base: {R['measurements']['A_load_base']}")
        except Exception as e:
            R["errors"]["A_load_base"] = str(e)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(f"[smoke] wrote {a.out}; errors={list(R['errors'])}")
    print("SMOKE_JSON_BEGIN"); print(json.dumps(R)); print("SMOKE_JSON_END")
    return 0 if not R["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
