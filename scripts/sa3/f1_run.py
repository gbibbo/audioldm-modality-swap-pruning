#!/usr/bin/env python3
"""F1 functional-sentinel run orchestrator (RQ2b) — ONE T4 job: train the full-backbone LoRA on the
frozen mechanical train_L (96), verify the export, then generate the four preregistered systems over
the 64 eval_L items. Scoring (paired CLAP audio-audio T_AA) and the verdict run LOCALLY afterwards
(.venv-metrics, 0 GPU); this job NEVER scores, inspects structure, or launches F2.

Like train_controls_paired.py, the parent imports nothing from the training stack (stdlib + safetensors
verify only) and runs each stage as a SEPARATE CHILD PROCESS so the training child's os._exit (wandb
idle-billing guard) terminates only that child. If training/export verification fails, generation is
NOT launched.

Run (T4):  _external/stable-audio-3/.venv/bin/python scripts/sa3/f1_run.py \
     --train-dir data/sa3/adapters/mechanical_trainL --manifest configs/sa3/adapters/mechanical.manifest.json \
     --domain-dir data/sa3/adapters/mechanical --save-dir data/sa3/adapters \
     --steps 1000 --duration 10 --base_precision fp16 --expect-commit <sha>
CPU dry:  ... --dry-run-cpu   (1-step synthetic train + 1-prompt gen; validates orchestration)
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_control_loras.py")
GEN = os.path.join(HERE, "f1_task_gen.py")
EXPECTED_BLOCKS = set(range(20))


def verify_adapter(path: str, dry: bool) -> dict:
    if not os.path.exists(path):
        return {"exists": False}
    from safetensors import safe_open
    import numpy as np
    with safe_open(path, framework="numpy") as f:
        keys = list(f.keys())
        max_b = 0.0
        for k in keys:
            if k.endswith(".lora_B"):
                max_b = max(max_b, float(np.linalg.norm(f.get_tensor(k).astype("float64"))))
    blocks = sorted({int(m.group(1)) for k in keys for m in [re.search(r"\.layers\.(\d+)\.", k)] if m})
    return {"exists": True, "n_tensors": len(keys), "max_lora_B": max_b,
            "blocks": blocks, "blocks_ok": set(blocks) == EXPECTED_BLOCKS,
            "nonzero_B": (max_b > 0) or dry}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--domain-dir", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--base_precision", default="fp16")
    ap.add_argument("--score-manifest", default="artifacts/sa3/f1_taa_manifest.json")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--dry-run-cpu", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.save_dir, exist_ok=True)
    dry = a.dry_run_cpu

    if a.expect_commit and not dry:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    save = os.path.join(a.save_dir, "F1_full.safetensors")
    if os.path.exists(save):
        os.remove(save)   # pristine

    # ---- stage 1: train full-backbone LoRA (child) ----
    tcmd = [sys.executable, TRAIN, "--backbone", "--steps", str(a.steps), "--duration", str(a.duration),
            "--base_precision", a.base_precision, "--save", save,
            "--smoke-out", os.path.join(a.save_dir, "F1_full_train.json")]
    if dry:
        tcmd.append("--dry-run-cpu")
    else:
        tcmd += ["--data_dir", a.train_dir]
    print(f"[f1-run] ===== train full-backbone LoRA ({a.steps} steps) =====", flush=True)
    rt = subprocess.run(tcmd)
    vinfo = verify_adapter(save, dry)
    train_ok = (rt.returncode == 0) and vinfo.get("exists") and vinfo.get("blocks_ok") and vinfo.get("nonzero_B")
    print(f"[f1-run] train rc={rt.returncode} verify={json.dumps(vinfo)}", flush=True)
    if not train_ok:
        summary = {"phase": "F1_run", "train_ok": False, "gen_ok": False, "verify": vinfo,
                   "note": "training/export verification FAILED -> generation NOT launched"}
        json.dump(summary, open(os.path.join(a.save_dir, "f1_run_summary.json"), "w"), indent=2)
        print("F1_RUN_SUMMARY_BEGIN"); print(json.dumps(summary)); print("F1_RUN_SUMMARY_END")
        return 1

    # ---- stage 2: generate the 4 systems x 64 eval (child) ----
    gcmd = [sys.executable, GEN, "--device", ("cpu" if dry else "cuda"),
            "--manifest", a.manifest, "--adapter", save, "--domain-dir", a.domain_dir,
            "--out-dir", os.path.join(a.save_dir, "f1_taa"), "--score-manifest", a.score_manifest]
    if dry:
        gcmd.append("--dry-run-cpu")
    print("[f1-run] ===== generate F1 systems (base_noL/base_Lfull/post_noL/post_Lfull) =====", flush=True)
    rg = subprocess.run(gcmd)
    gen_ok = (rg.returncode == 0) and os.path.exists(a.score_manifest)
    summary = {"phase": "F1_run", "train_ok": True, "gen_ok": bool(gen_ok), "verify": vinfo,
               "score_manifest": a.score_manifest, "dry_run": dry,
               "next": "score locally (.venv-metrics score_taa.py) then f1_verdict.py; STOP+report before F2",
               "git_commit": subprocess.getoutput("git rev-parse HEAD")}
    json.dump(summary, open(os.path.join(a.save_dir, "f1_run_summary.json"), "w"), indent=2)
    print("F1_RUN_SUMMARY_BEGIN"); print(json.dumps(summary)); print("F1_RUN_SUMMARY_END")
    print(f"[f1-run] done train_ok={summary['train_ok']} gen_ok={gen_ok}")
    return 0 if gen_ok else 1


if __name__ == "__main__":
    sys.exit(main())
