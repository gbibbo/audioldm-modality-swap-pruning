#!/usr/bin/env python3
"""Parent orchestrator: train the two positive controls L_6 and L_13 as TWO INDEPENDENT trainings
inside ONE Lightning job (option (c), Gabriel 2026-08-23) — to amortise the fixed T4 provisioning
tax, NOT to train them jointly. Each control is trained from a PRISTINE `small-sfx-base` with a fresh
LoRA, fresh optimizer, and the same frozen seed reset; they differ only in the single `--include`
block. 1000 steps each — no hyperparameter change.

Implementation: this parent invokes `train_control_loras.py` as a SEPARATE CHILD PROCESS per block.
The child's `os._exit()` (which prevents wandb-thread idle-billing) then terminates only that child;
the parent keeps orchestrating. After each child: verify the exported LoRA (exists, non-zero lora_B,
correct include block) before moving on. If L_6 fails, L_13 is NOT launched.

This parent imports NOTHING from the training stack (only stdlib + numpy/safetensors for the verify),
so it exits cleanly on its own.

Run (in the T4 job):  _external/stable-audio-3/.venv/bin/python scripts/sa3/train_controls_paired.py \
    --data_dir data/sa3/adapters/impact_percussion_trainL --save-dir data/sa3/adapters \
    --steps 1000 --duration 10 --base_precision fp16 --blocks 6,13 --expect-commit <sha>
CPU dry:  ... --dry-run-cpu   (1 step per child, validates orchestration)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_control_loras.py")


def verify_ckpt(path: str, block: int) -> dict:
    """Light verify without importing the training stack: exists, non-zero lora_B, include==block."""
    if not os.path.exists(path):
        return {"exists": False, "nonzero_B": False}
    from safetensors import safe_open
    import numpy as np
    with safe_open(path, framework="numpy") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
        max_b = 0.0
        for k in keys:
            if k.endswith(".lora_B"):
                max_b = max(max_b, float(np.linalg.norm(f.get_tensor(k).astype("float64"))))
    cfg = json.loads(meta.get("lora_config", "{}"))
    inc = cfg.get("include")
    return {"exists": True, "n_tensors": len(keys), "max_lora_B": max_b, "nonzero_B": max_b > 0,
            "include": inc, "include_ok": inc == [f"transformer.layers.{block}."]}


def run_child(block: int, a) -> tuple:
    save = os.path.join(a.save_dir, f"L_{block}.safetensors")
    if os.path.exists(save):
        os.remove(save)   # pristine: never reuse a prior/incomplete checkpoint
    cmd = [sys.executable, TRAIN, "--block", str(block), "--data_dir", a.data_dir,
           "--steps", str(a.steps), "--duration", str(a.duration),
           "--base_precision", a.base_precision, "--save", save,
           "--smoke-out", os.path.join(a.save_dir, f"L_{block}_train.json")]
    if a.dry_run_cpu:
        cmd.append("--dry-run-cpu")
    print(f"[paired] ===== training L_{block}: pristine base, block {block} only, {a.steps} steps =====",
          flush=True)
    r = subprocess.run(cmd)
    info = verify_ckpt(save, block)
    ok = (r.returncode == 0) and info["exists"] and info["nonzero_B"] and info.get("include_ok", False)
    print(f"[paired] L_{block} child_returncode={r.returncode} verify={json.dumps(info)}", flush=True)
    return ok, save, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--base_precision", default="fp16")
    ap.add_argument("--blocks", default="6,13")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--dry-run-cpu", action="store_true")
    a = ap.parse_args()
    blocks = [int(b) for b in a.blocks.split(",")]
    os.makedirs(a.save_dir, exist_ok=True)

    if a.expect_commit and not a.dry_run_cpu:
        cur = subprocess.getoutput("git rev-parse HEAD")
        assert cur.startswith(a.expect_commit) or a.expect_commit.startswith(cur), f"commit {cur}!={a.expect_commit}"
        assert not subprocess.getoutput("git status --porcelain"), "dirty tree"

    results = {}
    for block in blocks:
        ok, save, info = run_child(block, a)
        results[str(block)] = {"ok": ok, "save": save, "verify": info}
        if not ok:
            print(f"[paired] L_{block} FAILED — STOP; not launching remaining controls", flush=True)
            break

    done = [b for b in results if results[b]["ok"]]
    summary = {"phase": "paired_controls", "blocks_requested": blocks,
               "blocks_completed": [int(b) for b in done], "all_ok": len(done) == len(blocks),
               "results": results, "git_commit": subprocess.getoutput("git rev-parse HEAD")}
    json.dump(summary, open(os.path.join(a.save_dir, "paired_controls_summary.json"), "w"), indent=2)
    print("PAIRED_SUMMARY_BEGIN"); print(json.dumps(summary)); print("PAIRED_SUMMARY_END")
    print(f"[paired] done: completed {summary['blocks_completed']} of {blocks}  all_ok={summary['all_ok']}")
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
