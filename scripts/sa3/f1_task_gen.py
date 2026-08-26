#!/usr/bin/env python3
"""F1 functional-sentinel generation (RQ2b). Generate EXACTLY four preregistered systems over the
frozen 64 eval_L items, each with its own held-out caption and a per-item seed = SEED0 + idx:

    base_noL   base_Lfull   post_noL   post_Lfull

The SAME full-backbone LoRA (trained on base only) is applied UNCHANGED to base and to post — that is
the base->post compatibility test. Within each checkpoint the no-L and +L generations differ ONLY by
adapter strength (0 vs 1); all other generation settings are identical. Frozen gen: 8-step, cfg 7.0,
apg 1.0, 10 s.

*** NO structural analysis in F1 ***: no block masks / skip_blocks, no A_eco / A_tan / D_P, no G_ext,
no block-level inspection anywhere. Scoring is the frozen paired CLAP audio-audio T_AA (score_taa.py);
the verdict is the symmetric SESOI gate (f1_verdict.py). Two pairs only:
    F1_base = (base_Lfull vs base_noL)     F1_post = (post_Lfull vs post_noL)

Run (GPU):  _external/stable-audio-3/.venv/bin/python scripts/sa3/f1_task_gen.py --device cuda \
     --manifest configs/sa3/adapters/mechanical.manifest.json --adapter data/sa3/adapters/F1_full.safetensors \
     --domain-dir data/sa3/adapters/mechanical --out-dir data/sa3/adapters/f1_taa \
     --score-manifest artifacts/sa3/f1_taa_manifest.json --expect-commit <sha>
Plan only (CPU, no model): ... --plan-only   (writes the 4-system x 64-id manifest skeleton)
CPU dry:  ... --dry-run-cpu   (1 eval prompt, base_noL + base_Lfull only, 2 steps)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

SECONDS, SEED0, STEPS, CFG, APG = 10, 20260824, 8, 7.0, 1.0   # frozen (rc1.4 gen settings)
SYSTEMS = ["base_noL", "base_Lfull", "post_noL", "post_Lfull"]
PAIRS = [{"name": "F1_base", "with_L": "base_Lfull", "no_L": "base_noL"},
         {"name": "F1_post", "with_L": "post_Lfull", "no_L": "post_noL"}]


def load_eval(manifest, domain_dir, out_dir, limit=None):
    man = json.load(open(manifest))
    caps = {c["id"]: c["caption"] for c in man["clips"]}
    eval_ids = sorted(man["split"]["eval_L"])
    if limit:
        eval_ids = eval_ids[:limit]
    evals = [(eid, caps[eid], SEED0 + i) for i, eid in enumerate(eval_ids)]
    eval_pairs = [{"eval_id": eid, "ref_wav": os.path.join(domain_dir, f"{eid}.wav")}
                  for (eid, _, _) in evals]
    return man, evals, eval_pairs


def build_plan(evals, eval_pairs, out_dir, systems):
    """Score-manifest skeleton (paths that WOULD be generated); no model, no audio."""
    configs = {tag: {eid: os.path.join(out_dir, f"{tag}__{eid}.wav") for (eid, _, _) in evals}
               for tag in systems}
    seeds = {eid: seed for (eid, _, seed) in evals}
    pairs = [p for p in PAIRS if p["with_L"] in systems and p["no_L"] in systems]
    return {"eval_pairs": eval_pairs, "configs": configs, "pairs": pairs, "seeds": seeds,
            "gen": {"steps": STEPS, "cfg": CFG, "apg": APG, "seconds": SECONDS, "seed0": SEED0},
            "systems": systems, "phase": "F1_functional_sentinel",
            "structural_analysis": False,
            "git_commit": subprocess.getoutput("git rev-parse HEAD")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plan-only", action="store_true"); ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--manifest", required=True); ap.add_argument("--domain-dir", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-dir", default="data/sa3/small-sfx-base")
    ap.add_argument("--post-dir", default="data/sa3/small-sfx")
    ap.add_argument("--out-dir", default="data/sa3/adapters/f1_taa")
    ap.add_argument("--score-manifest", default="artifacts/sa3/f1_taa_manifest.json")
    ap.add_argument("--expect-commit", default=None)
    a = ap.parse_args()

    # ---- plan-only: full 4x64 skeleton, no model, no compute ----
    if a.plan_only:
        _, evals, eval_pairs = load_eval(a.manifest, a.domain_dir, a.out_dir)
        plan = build_plan(evals, eval_pairs, a.out_dir, SYSTEMS)
        os.makedirs(os.path.dirname(os.path.abspath(a.score_manifest)), exist_ok=True)
        json.dump(plan, open(a.score_manifest, "w"), indent=2)
        print(f"[f1-plan] wrote {a.score_manifest}: {len(plan['systems'])} systems, "
              f"{len(eval_pairs)} eval ids, {len(plan['pairs'])} pairs")
        return 0

    if a.dry_run_cpu:
        a.device = "cpu"
    dev = a.device; half = (dev == "cuda")
    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"
    assert a.adapter and os.path.exists(a.adapter), f"--adapter checkpoint required (got {a.adapter})"

    # heavy imports only on a real/dry generation path (never for plan-only)
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from research_sa3 import loading, probes as P, adapters as AD
    from research_sa3.e2e import wrap_model, generate_audio, save_wav

    def load_model(dir_):
        cfg = loading.load_json(f"{dir_}/model_config.json")
        cfgp = loading.patch_text_encoder_path(cfg, f"{dir_}/t5gemma-b-b-ul2")
        model, _ = loading.build_model_strict(cfgp, f"{dir_}/model.safetensors", device=dev)
        if half:
            model = model.half()
        return model, cfg

    steps = 2 if a.dry_run_cpu else STEPS
    _, evals, eval_pairs = load_eval(a.manifest, a.domain_dir, a.out_dir,
                                     limit=(1 if a.dry_run_cpu else None))
    systems = ["base_noL", "base_Lfull"] if a.dry_run_cpu else SYSTEMS
    os.makedirs(a.out_dir, exist_ok=True)
    configs = {}

    def gen_all(sa, tag):
        configs.setdefault(tag, {})
        for (eid, prompt, seed) in evals:
            fp = os.path.join(a.out_dir, f"{tag}__{eid}.wav")
            audio = generate_audio(sa, prompt, SECONDS, seed, steps=steps, cfg_scale=CFG, apg_scale=APG)
            save_wav(audio if audio.ndim == 3 else audio.unsqueeze(0), fp)
            configs[tag][eid] = fp
        print(f"[f1-gen] {tag} -> {len(evals)} clips", flush=True)

    # ---- BASE checkpoint: no-L (strength 0) then +L (same adapter, strength 1) ----
    base, bcfg = load_model(a.base_dir)
    sab = wrap_model(base, bcfg, dev, half)
    gen_all(sab, "base_noL")
    AD.apply_trained_lora(base, a.adapter); P.set_strength(base, 1.0)
    gen_all(sab, "base_Lfull")
    AD.remove_adapter(base)
    del base, sab
    import gc; gc.collect()
    if not a.dry_run_cpu:
        torch.cuda.empty_cache()

    # ---- POST checkpoint: SAME adapter applied unchanged (base->post transfer test) ----
    if not a.dry_run_cpu:
        post, pcfg = load_model(a.post_dir)
        sap = wrap_model(post, pcfg, dev, half)
        gen_all(sap, "post_noL")
        AD.apply_trained_lora(post, a.adapter); P.set_strength(post, 1.0)
        gen_all(sap, "post_Lfull")
        AD.remove_adapter(post)
        del post, sap; gc.collect(); torch.cuda.empty_cache()

    plan = build_plan(evals, eval_pairs, a.out_dir, systems)
    plan["configs"] = configs
    os.makedirs(os.path.dirname(os.path.abspath(a.score_manifest)), exist_ok=True)
    json.dump(plan, open(a.score_manifest, "w"), indent=2)
    print(f"[f1-gen] wrote {a.score_manifest}: {len(configs)} systems, {len(evals)} eval, "
          f"{len(plan['pairs'])} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
