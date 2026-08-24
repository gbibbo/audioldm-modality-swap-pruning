#!/usr/bin/env python3
"""Generate the task audio for the rc1.4 positive-control ΔT_AA gate. For each of the 8 eval_L clips
(paired with its own held-out prompt), generate y_j under each system config and write a manifest for
`score_taa.py`. FROZEN task generation: 8-step ping-pong, cfg_scale 7.0, apg 1.0, 10 s, per-item seed
= SEED0 + index (SEED0=20260824).

Systems per control L_b (host b, external panel G_ext(b) — pre-frozen from N=32 D_P):
  dense base ±L, dense post ±L, post^{-b} ±L (host collapse/identity), post^{-g} ±L for g∈G_ext(b).
No-L configs are shared across controls. G_ext(6)={11,12,13}, G_ext(13)={11,12,14}.

Run (GPU):  _external/stable-audio-3/.venv/bin/python scripts/sa3/control_task_gen.py --device cuda \
     --manifest configs/sa3/adapters/impact_percussion.manifest.json \
     --l6 data/sa3/adapters/L_6.safetensors --l13 data/sa3/adapters/L_13.safetensors \
     --domain-dir data/sa3/adapters/impact_percussion --out-dir data/sa3/adapters/control_taa \
     --score-manifest artifacts/sa3/control_taa_manifest.json --expect-commit <sha>
CPU dry:  ... --dry-run-cpu  (1 eval prompt, base±L6 only, 2 steps)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, probes as P, adapters as AD
from research_sa3.e2e import wrap_model, generate_audio, save_wav

SECONDS, SEED0, STEPS, CFG, APG = 10, 20260824, 8, 7.0, 1.0
GEXT = {6: [11, 12, 13], 13: [11, 12, 14]}   # frozen (rc1.4) from N=32 D_P lowest-3 excluding host


def load_model(dir_, device, half):
    cfg = loading.load_json(f"{dir_}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{dir_}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{dir_}/model.safetensors", device=device)
    if half:
        model = model.half()
    return model, cfg


def gen_all(sa, evals, out_dir, tag, skip, configs):
    """Generate `tag` for every eval item at skip=skip; store paths in configs[tag]."""
    configs.setdefault(tag, {})
    for (eid, prompt, seed) in evals:
        fp = os.path.join(out_dir, f"{tag}__{eid}.wav")
        audio = generate_audio(sa, prompt, SECONDS, seed, steps=STEPS, cfg_scale=CFG,
                               apg_scale=APG, skip_blocks=list(skip))
        save_wav(audio if audio.ndim == 3 else audio.unsqueeze(0), fp)
        configs[tag][eid] = fp
    print(f"[taskgen] {tag} (skip={skip}) -> {len(evals)} clips", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--manifest", required=True); ap.add_argument("--domain-dir", required=True)
    ap.add_argument("--l6", required=True); ap.add_argument("--l13", required=True)
    ap.add_argument("--base-dir", default="data/sa3/small-sfx-base")
    ap.add_argument("--post-dir", default="data/sa3/small-sfx")
    ap.add_argument("--out-dir", default="data/sa3/adapters/control_taa")
    ap.add_argument("--score-manifest", default="artifacts/sa3/control_taa_manifest.json")
    ap.add_argument("--expect-commit", default=None)
    a = ap.parse_args()
    if a.dry_run_cpu:
        a.device = "cpu"
    dev = a.device; half = (dev == "cuda")
    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    man = json.load(open(a.manifest))
    caps = {c["id"]: c["caption"] for c in man["clips"]}
    eval_ids = sorted(man["split"]["eval_L"])
    evals = [(eid, caps[eid], SEED0 + i) for i, eid in enumerate(eval_ids)]
    if a.dry_run_cpu:
        evals = evals[:1]
    os.makedirs(a.out_dir, exist_ok=True)
    configs = {}
    controls = {6: a.l6} if a.dry_run_cpu else {6: a.l6, 13: a.l13}

    # ---- BASE model: base_noL, then base_L{b} ----
    base, bcfg = load_model(a.base_dir, dev, half)
    sab = wrap_model(base, bcfg, dev, half)
    gen_all(sab, evals, a.out_dir, "base_noL", (), configs)
    for b, path in controls.items():
        AD.apply_trained_lora(base, path); P.set_strength(base, 1.0)
        gen_all(sab, evals, a.out_dir, f"base_L{b}", (), configs)
        AD.remove_adapter(base)
    del base, sab
    import gc; gc.collect()
    if not a.dry_run_cpu:
        torch.cuda.empty_cache()

    if not a.dry_run_cpu:
        # ---- POST model: all no-L skips, then per-control L skips ----
        post, pcfg = load_model(a.post_dir, dev, half)
        sap = wrap_model(post, pcfg, dev, half)
        needed_skips = set([()])
        for b in controls:
            needed_skips.add((b,))
            for g in GEXT[b]:
                needed_skips.add((g,))
        for sk in sorted(needed_skips, key=lambda s: (len(s), s)):
            tag = "post_noL" if sk == () else f"post-{sk[0]}_noL"
            gen_all(sap, evals, a.out_dir, tag, sk, configs)
        for b, path in controls.items():
            AD.apply_trained_lora(post, path); P.set_strength(post, 1.0)
            for sk in [()] + [(b,)] + [(g,) for g in GEXT[b]]:
                tag = f"post_L{b}" if sk == () else f"post-{sk[0]}_L{b}"
                gen_all(sap, evals, a.out_dir, tag, sk, configs)
            AD.remove_adapter(post)
        del post, sap; gc.collect(); torch.cuda.empty_cache()

    # ---- score manifest ----
    eval_pairs = [{"eval_id": eid, "ref_wav": os.path.join(a.domain_dir, f"{eid}.wav")}
                  for (eid, _, _) in evals]
    pairs = []
    for b in controls:
        pairs += [{"name": f"L{b}_base", "with_L": f"base_L{b}", "no_L": "base_noL"}]
        if not a.dry_run_cpu:
            pairs += [{"name": f"L{b}_post", "with_L": f"post_L{b}", "no_L": "post_noL"},
                      {"name": f"L{b}_host", "with_L": f"post-{b}_L{b}", "no_L": f"post-{b}_noL"}]
            pairs += [{"name": f"L{b}_ext{g}", "with_L": f"post-{g}_L{b}", "no_L": f"post-{g}_noL"}
                      for g in GEXT[b]]
    score_man = {"eval_pairs": eval_pairs, "configs": configs, "pairs": pairs,
                 "gen": {"steps": STEPS, "cfg": CFG, "apg": APG, "seconds": SECONDS, "seed0": SEED0},
                 "G_ext": GEXT, "git_commit": subprocess.getoutput("git rev-parse HEAD")}
    os.makedirs(os.path.dirname(os.path.abspath(a.score_manifest)), exist_ok=True)
    json.dump(score_man, open(a.score_manifest, "w"), indent=2)
    print(f"[taskgen] wrote {a.score_manifest}: {len(configs)} configs, {len(pairs)} ΔT pairs, "
          f"{len(evals)} eval items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
