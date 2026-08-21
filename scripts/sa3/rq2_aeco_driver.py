#!/usr/bin/env python3
"""Ecological regime driver: A_eco (field) + ΔT_L (task) for a trained held-out LoRA (§3.4, §4.3, §6).

Staged per rc1 (§5): FIELD A_eco FIRST; end-to-end ΔT_L generation only if the field link survives.

FIELD (--mode field):  attach the trained adapter `L`, toggle strength for
    δF(L)      = F_{P+L} - F_P                         (strength 1 vs 0)
    δF^{-g}(L) = F^{-g}_{P+L_{-g}} - F^{-g}_P           (restrict to surviving blocks; g removed)
  A_eco(g;L) = Σ_states ||δF(L) - δF^{-g}(L)||² / Σ_states ||δF(L)||²   (research_sa3.metrics.a_eco)
  Two readings (rc1): --panel generic (reuse persisted S_traj) | --panel domain (new post states on
  prompts_L). A signal/precision guard (metrics.precision_ok on ||δF(L)||² vs ||F_P||²) flags an
  adapter with no measurable field effect — its ranking is not interpreted.
  If A_tan / D_P per-block arrays are supplied (--atan-json / --dp-json), runs the frozen §6
  prediction check (aeco_predict.prediction_check) at k=6.

TASK (--mode task):  generate with/without L on prompts (strength 1 vs 0), write a score_e_metrics
  manifest for the uplift ΔT_L = T(S+L) - T(S). Scoring is a separate `.venv-metrics` step.

CPU dry:  OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/rq2_aeco_driver.py --dry-run-cpu \
              --lora artifacts/sa3/control_dry/L_6.safetensors --state-store artifacts/sa3/pilot_states_dry \
              --out artifacts/sa3/aeco_dry.json
GPU:      _external/stable-audio-3/.venv/bin/python scripts/sa3/rq2_aeco_driver.py --device cuda \
              --lora data/sa3/adapters/L_6.safetensors --state-store artifacts/sa3/pilot_states \
              --expect-commit <sha> --out artifacts/sa3/aeco_L6.json
"""
from __future__ import annotations
import argparse, gc, glob, json, os, subprocess, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, fields as F, probes as P, adapters as AD, metrics as M
from research_sa3 import aeco_predict as APRED
from research_sa3.blockskip import block_mask

SECONDS = 10


def load_post(device, half):
    d = "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device=device)
    if half:
        model = model.half()
    return model, cfg


def field_aeco(post, state_files, blocks, dev, dtype, eta):
    """Compute A_eco(g;L) over blocks from the persisted/captured states. Adapter already attached."""
    def batched(sts):
        xs = torch.cat([x for _, x in sts], dim=0).to(dev, dtype)
        ts = torch.tensor([tau for tau, _ in sts], device=dev, dtype=dtype)
        return xs, ts

    def rep_cc(cc, B):
        return {k: (v.repeat(B, *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v) for k, v in cc.items()}

    num = {g: 0.0 for g in blocks}
    den = 0.0
    fp_sq = 0.0
    per_prompt = {}
    for sf in state_files:
        aid = os.path.basename(sf).split("_")[1].split(".")[0]
        d = torch.load(sf); sts = d["states"]; cap = d["caption"]; Sn = len(sts)
        cc0 = F.prepare_conditioning(post, cap, SECONDS, dev, latent_len=sts[0][1].shape[-1], dtype=dtype)
        cc = rep_cc(cc0, Sn)
        bx, bt = batched(sts)
        # baselines: adapter OFF (strength 0)
        P.set_strength(post, 0.0)
        FP = F.raw_field(post, bx, bt, cc).detach()
        FPmg = {}
        for g in blocks:
            with block_mask(post, [g]):
                FPmg[g] = F.raw_field(post, bx, bt, cc).detach()
        # adapter ON (strength 1): dF(L)
        P.set_strength(post, 1.0)
        dFL = (F.raw_field(post, bx, bt, cc).float() - FP.float())
        den_p = float(F.state_sq_norm(dFL).sum().item())
        fp_p = float(F.state_sq_norm(FP.float()).sum().item())
        den += den_p; fp_sq += fp_p
        for g in blocks:
            P.restrict_to_surviving(post, removed_block=g)   # strength 1 except block g -> 0
            with block_mask(post, [g]):
                fpu_mg = F.raw_field(post, bx, bt, cc)
            dfl_mg = fpu_mg.float() - FPmg[g].float()
            num[g] += float(F.state_sq_norm(dFL - dfl_mg).sum().item())
            P.set_strength(post, 1.0)
        per_prompt[aid] = {"den": den_p, "fp_sq": fp_p}
        print(f"[aeco] prompt {aid} den(||dF(L)||^2)={den_p:.4e} ||F_P||^2={fp_p:.4e}", flush=True)
        gc.collect()
    aeco = M.a_eco({g: num[g] for g in blocks}, den) if den > 0 else {g: float("nan") for g in blocks}
    guard_ok = M.precision_ok(den, fp_sq, eta)
    return {"A_eco": {str(g): aeco[g] for g in blocks}, "den_dFL_sq": den, "fp_sq": fp_sq,
            "precision_ok": bool(guard_ok), "eta": eta, "per_prompt": per_prompt}


def task_manifest(post, sa_wrap, prompts, seed0, out_dir, blocks_removed=None, steps=8):
    """Generate with L on/off on `prompts`; write a score_e_metrics manifest for ΔT_L. Returns manifest."""
    from research_sa3.e2e import generate_audio, save_wav
    os.makedirs(out_dir, exist_ok=True)
    systems = {"post_noL": {}, "post_L": {}}
    caps = {}
    for i, prompt in enumerate(prompts):
        aid = str(i)
        caps[aid] = prompt
        seed = seed0 + i
        for sysname, strength in (("post_noL", 0.0), ("post_L", 1.0)):
            P.set_strength(post, strength)
            audio = generate_audio(sa_wrap, prompt, SECONDS, seed, steps=steps,
                                   skip_blocks=(blocks_removed or []))
            fp = os.path.join(out_dir, f"{sysname}_{aid}.wav")
            save_wav(audio if audio.ndim == 3 else audio.unsqueeze(0), fp)
            systems[sysname][aid] = fp
    manifest = {"reference_system": "post_noL", "prompts": caps, "systems": systems,
                "note": "ΔT_L = T(post_L) - T(post_noL); score with score_e_metrics.py (.venv-metrics)."}
    json.dump(manifest, open(os.path.join(out_dir, "task_manifest.json"), "w"), indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--lora", required=True, help="trained adapter .safetensors")
    ap.add_argument("--mode", choices=["field", "task", "both"], default="field")
    ap.add_argument("--panel", choices=["generic", "domain"], default="generic")
    ap.add_argument("--state-store", default="artifacts/sa3/pilot_states")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--atan-json", default=None, help="JSON {block: A_tan} for the §6 prediction check")
    ap.add_argument("--dp-json", default=None, help="JSON {block: D_P} for the §6 prediction check")
    ap.add_argument("--floor-k", default="2:0,4:0,6:0", help="k:floor,... §4.1 bootstrap floors")
    ap.add_argument("--prompts", default=None, help="comma-separated prompts for --mode task")
    ap.add_argument("--task-out", default="artifacts/sa3/aeco_task_wavs")
    ap.add_argument("--task-steps", type=int, default=8)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--out", default="artifacts/sa3/aeco_result.json")
    a = ap.parse_args()
    if a.dry_run_cpu:
        a.device = "cpu"; a.n = min(a.n, 2); a.blocks = a.blocks or "5,13"
    dev = a.device; half = (dev == "cuda"); dtype = torch.float16 if half else torch.float32
    blocks = [int(x) for x in a.blocks.split(",")] if a.blocks else list(range(20))

    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    post, cfg = load_post(dev, half)
    rep = AD.apply_trained_lora(post, a.lora)
    print(f"[aeco] adapter {a.lora}: type={rep['adapter_type']} rank={rep['rank']} "
          f"blocks={rep['blocks']} n_layers={rep['n_layers']}", flush=True)

    R = {"phase": "aeco", "device": dev, "dry_run_cpu": a.dry_run_cpu, "lora": a.lora,
         "adapter": rep, "panel": a.panel, "blocks": blocks, "mode": a.mode,
         "git_commit": subprocess.getoutput("git rev-parse HEAD"),
         "upstream_commit": loading.SA3_UPSTREAM_COMMIT}
    t0 = time.time()

    if a.mode in ("field", "both"):
        state_files = sorted(glob.glob(os.path.join(a.state_store, "state_*.pt")),
                             key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))[:a.n]
        assert state_files, f"no persisted states in {a.state_store}"
        fr = field_aeco(post, state_files, blocks, dev, dtype, a.eta)
        R["field"] = fr
        aeco_str = ", ".join("%d:%.4f" % (g, fr["A_eco"][str(g)]) for g in blocks)
        print("[aeco] A_eco={ %s } precision_ok=%s" % (aeco_str, fr["precision_ok"]), flush=True)
        # optional §6 prediction check
        if a.atan_json and a.dp_json:
            a_tan = {int(k): float(v) for k, v in json.load(open(a.atan_json)).items()}
            d_p = {int(k): float(v) for k, v in json.load(open(a.dp_json)).items()}
            a_eco = {int(g): fr["A_eco"][str(g)] for g in blocks}
            floors = {int(kv.split(":")[0]): int(kv.split(":")[1]) for kv in a.floor_k.split(",")}
            R["prediction_check"] = APRED.prediction_check(a_eco, a_tan, d_p, floors)
            print(f"[aeco] §6 verdict (k=6): {R['prediction_check']['verdict']}", flush=True)

    if a.mode in ("task", "both"):
        from research_sa3.e2e import wrap_model
        prompts = (a.prompts.split(",") if a.prompts else ["a metallic impact sound"])
        if a.dry_run_cpu:
            prompts = prompts[:1]
        sa_wrap = wrap_model(post, cfg, dev, half)
        tm = task_manifest(post, sa_wrap, prompts, seed0=20260821, out_dir=a.task_out,
                           steps=(2 if a.dry_run_cpu else a.task_steps))
        R["task_manifest"] = os.path.join(a.task_out, "task_manifest.json")
        R["task_systems"] = {k: len(v) for k, v in tm["systems"].items()}
        print(f"[aeco] task manifest written: {R['task_manifest']} "
              f"({R['task_systems']} clips per system)", flush=True)

    R["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(f"[aeco] wrote {a.out} wall={R['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
