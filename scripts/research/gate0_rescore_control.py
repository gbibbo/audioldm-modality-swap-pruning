#!/usr/bin/env python3
"""Gate-0 scorer-pinning reproducibility control (prereg v5 item 4, PRE-PHENOMENON-DATA).

Re-scores the EXISTING frozen 384 Gate-0 WAVs with the PINNED fused-CLAP revision and verifies
the recomputed verdict reproduces the frozen Gate-0 verdict to a pre-defined tolerance. Does NOT
rewrite the historical verdict (writes a separate *_rescore.json). If reproduction fails, exits
non-zero (STOP) so the discrepancy is reported before any downstream generation.

This is a reproducibility CONTROL only — the frozen Gate-0 result stands regardless.
"""
import argparse, hashlib, json, os, subprocess, sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

REPO = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
FROZEN_VERDICT = "artifacts/icassp_gate0/gate0_verdict.json"
JOB_ROOT = "/teamspace/jobs/gate0-gen-1/artifacts/audioldm-modality-swap-pruning"
GEN_MANIFEST = f"{JOB_ROOT}/artifacts/icassp_gate0/gen_gate0/gen_manifest_dense_both.json"
HF_SNAP = ("/teamspace/studios/this_studio/.cache/huggingface/hub/"
           "models--laion--clap-htsat-fused")
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"

# Tolerances (abs). Same revision + same env => expect near-exact CPU reproduction.
TOL_MEAN = 1e-4      # mean base/adapter cosine, ΔCLAP point
TOL_CI = 1e-3        # bootstrap CI endpoints
TOL_PERITEM = 5e-4   # max per-prompt ΔCLAP abs diff


def model_file_hashes():
    """Resolved model weight blob sha256s for the pinned snapshot (git-lfs blob name == sha256)."""
    snap = os.path.join(HF_SNAP, "snapshots", REVISION)
    out = {}
    for fn in ("model.safetensors", "pytorch_model.bin", "config.json"):
        p = os.path.join(snap, fn)
        if os.path.islink(p) or os.path.exists(p):
            real = os.path.realpath(p)
            # blobs are named by their sha256 for lfs; verify by hashing config.json (small)
            if fn == "config.json":
                out[fn] = hashlib.sha256(open(real, "rb").read()).hexdigest()
            else:
                out[fn] = os.path.basename(real)  # lfs blob filename = sha256
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/icassp_gate0/gate0_verdict_rescore.json")
    ap.add_argument("--report", default="artifacts/icassp_gate0/gate0_rescore_control.json")
    args = ap.parse_args()
    os.chdir(REPO)

    frozen = json.load(open(FROZEN_VERDICT))
    # Re-run the SAME verdict driver on the frozen manifest with the pinned scorer.
    cmd = [".venv/bin/python", "scripts/research/gate0_score_verdict.py",
           "--gen-manifest", GEN_MANIFEST, "--wav-root", JOB_ROOT, "--out", args.out]
    print("re-scoring 384 frozen WAVs with pinned CLAP revision", REVISION[:12], "...")
    r = subprocess.run(cmd, env={**os.environ, "OPENBLAS_CORETYPE": "Haswell"})
    if r.returncode != 0:
        print("RESCORE-CONTROL FAIL: verdict driver errored", file=sys.stderr)
        return 2
    new = json.load(open(args.out))

    fd, nd = frozen["gate0_verdict"]["delta_clap"], new["gate0_verdict"]["delta_clap"]
    fdi, ndi = frozen["diagnostics"], new["diagnostics"]
    diffs = {
        "delta_point": abs(fd["point"] - nd["point"]),
        "delta_lo": abs(fd["lo"] - nd["lo"]),
        "delta_hi": abs(fd["hi"] - nd["hi"]),
        "mean_base_cosine": abs(fdi["mean_base_cosine"] - ndi["mean_base_cosine"]),
        "mean_adapter_cosine": abs(fdi["mean_adapter_cosine"] - ndi["mean_adapter_cosine"]),
        "max_per_prompt_delta": max(
            abs(a - b) for a, b in zip(fdi["per_prompt_delta_clap"], ndi["per_prompt_delta_clap"])),
    }
    checks = {
        "delta_point": diffs["delta_point"] <= TOL_MEAN,
        "delta_lo": diffs["delta_lo"] <= TOL_CI,
        "delta_hi": diffs["delta_hi"] <= TOL_CI,
        "mean_base_cosine": diffs["mean_base_cosine"] <= TOL_MEAN,
        "mean_adapter_cosine": diffs["mean_adapter_cosine"] <= TOL_MEAN,
        "max_per_prompt_delta": diffs["max_per_prompt_delta"] <= TOL_PERITEM,
        "PASS_unchanged": bool(frozen["PASS"]) == bool(new["PASS"]) is True,
    }
    ok = all(checks.values())
    report = {
        "control": "gate0_scorer_pinning_reproducibility",
        "pinned_revision": REVISION,
        "model_file_hashes": model_file_hashes(),
        "frozen_verdict_md5": hashlib.md5(open(FROZEN_VERDICT, "rb").read()).hexdigest(),
        "tolerances": {"mean": TOL_MEAN, "ci": TOL_CI, "per_item": TOL_PERITEM},
        "diffs": diffs, "checks": checks,
        "frozen": {"point": fd["point"], "lo": fd["lo"], "hi": fd["hi"], "PASS": frozen["PASS"]},
        "rescored": {"point": nd["point"], "lo": nd["lo"], "hi": nd["hi"], "PASS": new["PASS"]},
        "REPRODUCES": ok,
    }
    json.dump(report, open(args.report, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("RESCORE-CONTROL", "PASS (reproduces)" if ok else "FAIL (discrepancy -> STOP)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
