#!/usr/bin/env python3
"""Margins + single-block E adversary (protocol section 6.1, 9.2) -- PILOT-level (directional).

Generation-only on a panel: per prompt generate dense at steps {8,7,6,5,4}, R=5 dense-8 seed
streams, and each single-block-removed variant skip-g @ 8 steps. Save wavs + a manifest; scoring
(CLAP/KL_passt/FD_openl3 vs the dense-8 stream-0 reference) + the non-inferiority verdicts run on
the Studio afterward (analyze_adversary.py). This measures:
  * 8->7 margins m_CLAP/m_KL/m_FD (deterioration) with r_m floor from the R=5 dense-vs-dense spread;
  * single-block E({g}) vs the latency-matched dense comparator (skip-g@8 ~ dense-7.5 latency).
On the PILOT panel this is DIRECTIONAL (sizing / kill-criterion direction), not the frozen main
CASE-E decision (that is main-panel, after freezing N_main/margins). Reports which blocks look
inferior to the dense comparator.

Run (GPU):  _external/stable-audio-3/.venv/bin/python scripts/sa3/e_adversary.py --device cuda \
                --n 16 --expect-commit <sha> --out artifacts/sa3/adversary_manifest.json
CPU dry:   OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/e_adversary.py --dry-run-cpu
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, e2e
from research_sa3 import seeds as S

SECONDS = 10


def sha256_file(p):
    import hashlib; return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_post(device, half):
    d = "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device=device)
    if half:
        model = model.half()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--panel", default="configs/sa3/panel_pilot.json")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--blocks", default=None); ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--dense-steps", default="8,7,6,5,4")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--wavdir", default="artifacts/sa3/adversary_wavs")
    ap.add_argument("--out", default="artifacts/sa3/adversary_manifest.json")
    a = ap.parse_args()
    if a.dry_run_cpu:
        a.device = "cpu"; a.n = min(a.n, 2); a.blocks = a.blocks or "5,13"; a.R = 2; a.dense_steps = "8,7"
    dev = a.device; half = (dev == "cuda")
    blocks = [int(x) for x in a.blocks.split(",")] if a.blocks else list(range(20))
    dsteps = [int(x) for x in a.dense_steps.split(",")]
    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    panel = json.load(open(a.panel))
    prompts = sorted(panel["items"], key=lambda x: int(x["audiocap_id"]))[:a.n]
    os.makedirs(a.wavdir, exist_ok=True)
    post, cfg = load_post(dev, half)
    sa = e2e.wrap_model(post, cfg, dev, model_half=half)

    manifest = {"reference_system": "dense8_s0", "panel": a.panel, "panel_sha256": sha256_file(a.panel),
                "R": a.R, "dense_steps": dsteps, "blocks": blocks, "N": len(prompts),
                "git_commit": subprocess.getoutput("git rev-parse HEAD"),
                "prompts": {}, "systems": {}, "dry_run_cpu": a.dry_run_cpu}
    # systems: dense{step}_s0 for each dense step (stream 0), dense8_s1..s{R-1} extra streams, skip{g}
    for st in dsteps:
        manifest["systems"][f"dense{st}_s0"] = {}
    for r in range(1, a.R):
        manifest["systems"][f"dense8_s{r}"] = {}
    for g in blocks:
        manifest["systems"][f"skip{g}"] = {}

    t0 = time.time(); ngen = 0
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]; manifest["prompts"][aid] = cap
        # dense at each step count, stream 0
        for st in dsteps:
            seed = S.derive_seed(0, aid, "init", 0)
            aud = e2e.generate_audio(sa, cap, SECONDS, seed, steps=st, cfg_scale=1.0, apg_scale=1.0)
            wp = f"{a.wavdir}/dense{st}_s0_{aid}.wav"; e2e.save_wav(aud, wp); manifest["systems"][f"dense{st}_s0"][aid] = wp; ngen += 1
        # extra dense-8 streams (R>=2)
        for r in range(1, a.R):
            seed = S.derive_seed(r, aid, "init", 0)
            aud = e2e.generate_audio(sa, cap, SECONDS, seed, steps=8, cfg_scale=1.0, apg_scale=1.0)
            wp = f"{a.wavdir}/dense8_s{r}_{aid}.wav"; e2e.save_wav(aud, wp); manifest["systems"][f"dense8_s{r}"][aid] = wp; ngen += 1
        # single-block removed @ 8 steps, stream 0
        for g in blocks:
            seed = S.derive_seed(0, aid, "init", 0)
            aud = e2e.generate_audio(sa, cap, SECONDS, seed, steps=8, cfg_scale=1.0, apg_scale=1.0, skip_blocks=[g])
            wp = f"{a.wavdir}/skip{g}_{aid}.wav"; e2e.save_wav(aud, wp); manifest["systems"][f"skip{g}"][aid] = wp; ngen += 1
        print(f"[adv] prompt {aid} done ({ngen} gens, {time.time()-t0:.0f}s)", flush=True)
        json.dump(manifest, open(a.out, "w"), indent=2)  # incremental

    manifest["n_gens"] = ngen; manifest["wall_s"] = round(time.time() - t0, 1)
    json.dump(manifest, open(a.out, "w"), indent=2)
    print(f"[adv] wrote {a.out}  systems={len(manifest['systems'])} gens={ngen} wall={manifest['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
