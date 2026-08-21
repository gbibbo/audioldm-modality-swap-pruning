#!/usr/bin/env python3
"""CPU dry-run of the generation -> wav -> metrics path on the REAL post model (protocol 9.1, 10).

For N smoke prompts: capture S_traj (verify 8 states, tau matches the frozen schedule), generate
dense-8 audio + one block-removed (g=5) variant, decode via SAME-S, save wavs, then score with the
`.venv-metrics` scorer (reference = dense). Validates the full engineering pipeline on CPU.
NOT scientific (P_smoke; CPU). Real numbers come from the GPU smoke.

Run: SCRATCH=<dir> OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/dryrun_e2e.py --n 2
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import loading, e2e, states
from research_sa3 import seeds as S

DEV = "cpu"; SECONDS = 10
SCRATCH = os.environ.get("SCRATCH", "/tmp")
WAVDIR = os.path.join(SCRATCH, "sa3_e2e_wavs")


def load_post():
    d = "data/sa3/small-sfx"
    cfg = loading.load_json(f"{d}/model_config.json")
    cfgp = loading.patch_text_encoder_path(cfg, f"{d}/t5gemma-b-b-ul2")
    model, _ = loading.build_model_strict(cfgp, f"{d}/model.safetensors", device=DEV)
    return model, cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2); a = ap.parse_args()
    os.makedirs(WAVDIR, exist_ok=True)
    panel = json.load(open("configs/sa3/panel_smoke.json"))
    prompts = sorted(panel["items"], key=lambda x: int(x["audiocap_id"]))[:a.n]
    sched = json.load(open("configs/sa3/schedule_post_10s.json"))["tau_levels"]

    t0 = time.time(); post, cfg = load_post(); print(f"[e2e] post loaded ({time.time()-t0:.0f}s)")
    sa = e2e.wrap_model(post, cfg, DEV, model_half=False)

    # 1) S_traj capture on prompt 0
    aid0 = prompts[0]["audiocap_id"]; cap0 = prompts[0]["caption"]
    seed0 = S.derive_seed(0, aid0, "init", 0)
    t0 = time.time()
    tr = states.capture_trajectory(sa, cap0, SECONDS, seed0, steps=8, cfg_scale=1.0, apg_scale=1.0)
    dt = time.time() - t0
    taus = [round(s[0], 5) for s in tr["states"]]
    tau_match = all(abs(taus[i] - round(sched[i], 5)) < 1e-3 for i in range(min(8, len(taus))))
    print(f"[e2e] S_traj: {len(tr['states'])} states, tau={taus}")
    print(f"[e2e]   frozen schedule tau_levels={[round(x,5) for x in sched]} match={tau_match} (capture {dt:.0f}s)")

    # 2) generate dense-8 + block-skip(5) for each prompt, decode+save
    manifest = {"reference_system": "dense8", "prompts": {}, "systems": {"dense8": {}, "skip5": {}}}
    for it in prompts:
        aid = it["audiocap_id"]; cap = it["caption"]; seed = S.derive_seed(0, aid, "init", 0)
        manifest["prompts"][aid] = cap
        for sysid, skip in (("dense8", []), ("skip5", [5])):
            t0 = time.time()
            audio = e2e.generate_audio(sa, cap, SECONDS, seed, steps=8, cfg_scale=1.0,
                                       apg_scale=1.0, skip_blocks=skip, return_latents=False)
            wp = os.path.join(WAVDIR, f"{sysid}_{aid}.wav")
            e2e.save_wav(audio, wp, 44100)
            manifest["systems"][sysid][aid] = wp
            print(f"[e2e]   gen {sysid} aid={aid} shape={tuple(audio.shape)} ({time.time()-t0:.0f}s) -> {os.path.basename(wp)}")

    man_path = os.path.join(SCRATCH, "sa3_e2e_manifest.json")
    json.dump(manifest, open(man_path, "w"), indent=2)
    del post, sa
    import gc; gc.collect()

    # 3) score with the isolated metrics venv
    out_path = os.path.join(SCRATCH, "sa3_e2e_scores.json")
    env = dict(os.environ, OPENBLAS_CORETYPE="Haswell", SCRATCH=SCRATCH,
               HF_HOME="/teamspace/studios/this_studio/.cache/huggingface")
    print("[e2e] scoring with .venv-metrics ...")
    r = subprocess.run([".venv-metrics/bin/python", "scripts/sa3/score_e_metrics.py",
                        "--manifest", man_path, "--out", out_path, "--device", "cpu"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("[e2e] SCORER FAILED:\n", r.stderr[-2000:]); return 1
    scores = json.load(open(out_path))
    for sysid in ("dense8", "skip5"):
        s = scores[sysid]
        print(f"[e2e]   {sysid}: CLAP={s['CLAP']:.3f} KL_passt={s['KL_passt']:.4f} FD_openl3={s['FD_openl3']:.3f} n={s['n']}")
    ok = (tau_match and len(tr["states"]) == 8
          and abs(scores["dense8"]["KL_passt"]) < 1e-6 and abs(scores["dense8"]["FD_openl3"]) < 1e-2
          and scores["skip5"]["KL_passt"] > 0)  # a removed block should drift from dense
    print("DRYRUN-E2E", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
