#!/usr/bin/env python3
"""FROZEN RECOVERY-REVERSAL-V1 verdict — primary CLAP gate + secondary Human-CLAP, no GPU.

Consumes the three per-system CLAP cosine arrays (dense_ema descriptive, p1_pruned_ema_reconstructed,
p1_recovered), each EXACTLY 192 values in canonical (prompt_index, replicate_index) order, and
computes the frozen primary gate:

    R_AC  = mean_p mean_r (C_recovered - C_pruned)          prompt-cluster percentile bootstrap
    I     = R_AC - R_music                                  joint two-sample bootstrap (music retained)
    PASS  = R_AC.point >= 0.025  AND  lo95(R_AC) > 0  AND  lo95(I) > 0        (B=10000, PCG64 20260827)

Dense is DESCRIPTIVE only and cannot change PASS. R_music comes from the durable tracked baseline
configs/research/reversal_v1_r_music_clap.json (64 frozen per-prompt music contrasts). The secondary
Human-CLAP verdict (R_AC_HC, I_HC vs configs/research/reversal_v1_r_music_humanclap.json) is
CORROBORATIVE, has no SESOI and cannot change PASS.

This is the verdict CODE, frozen BEFORE any AudioCaps WAV exists. Input arrays are supplied as JSON
{"dense": [...192], "pruned": [...192], "recovered": [...192]} (+ optional "recovered_hc"/"pruned_hc").
--self-test runs toy cases (no data).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/reversal_v1_verdict.py \
        --scores <scores.json> --out artifacts/icassp_gate0/reversal_v1_verdict.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import (  # noqa: E402
    BOOTSTRAP_SEED_V1, N_PROMPTS_V1, N_REPLICATES_V1, SESOI,
    primary_verdict, secondary_hc_verdict)

R_MUSIC_CLAP = "configs/research/reversal_v1_r_music_clap.json"
R_MUSIC_HC = "configs/research/reversal_v1_r_music_humanclap.json"
N = N_PROMPTS_V1 * N_REPLICATES_V1  # 192


def _grid(arr, name):
    a = np.asarray(arr, dtype=np.float64)
    if a.size != N:
        raise SystemExit(f"{name}: expected exactly {N} scores (96x2 canonical), got {a.size}")
    return a.reshape(N_PROMPTS_V1, N_REPLICATES_V1)  # (prompt, replicate), canonical order


def _load_music(path, key):
    d = json.load(open(path))
    return np.array([p["prompt_mean_diff"] for p in sorted(d["prompts"], key=lambda x: x["prompt_index"])],
                    dtype=np.float64), d.get("artifact_sha256")


def compute(scores: dict) -> dict:
    rec = _grid(scores["recovered"], "recovered")
    pru = _grid(scores["pruned"], "pruned")
    dense = _grid(scores["dense"], "dense") if "dense" in scores else None
    music, music_sha = _load_music(R_MUSIC_CLAP, "clap")
    primary = primary_verdict(rec, pru, music, dense=dense, sesoi=SESOI, seed=BOOTSTRAP_SEED_V1)
    out = {"artifact": "reversal_v1_verdict", "PRIMARY": primary,
           "R_music_baseline_sha256": music_sha}
    if "recovered_hc" in scores and "pruned_hc" in scores:
        rec_hc = _grid(scores["recovered_hc"], "recovered_hc")
        pru_hc = _grid(scores["pruned_hc"], "pruned_hc")
        music_hc, hc_sha = _load_music(R_MUSIC_HC, "hc")
        out["SECONDARY_humanclap"] = secondary_hc_verdict(rec_hc, pru_hc, music_hc, seed=BOOTSTRAP_SEED_V1)
        out["R_music_HC_baseline_sha256"] = hc_sha
    return out


def _self_test() -> int:
    rng = np.random.default_rng(0)
    music = -0.09 + 0.10 * rng.standard_normal(64)
    ok = []
    # (a) strong positive R_AC -> PASS
    rec = 0.25 + 0.05 * rng.standard_normal((96, 2)); pru = 0.15 + 0.05 * rng.standard_normal((96, 2))
    v = primary_verdict(rec, pru, music); ok.append(("strong_pass", v["PASS"] is True))
    # (b) zero effect -> not PASS (R_AC point < SESOI)
    rec0 = 0.15 + 0.05 * rng.standard_normal((96, 2)); pru0 = 0.15 + 0.05 * rng.standard_normal((96, 2))
    v0 = primary_verdict(rec0, pru0, music); ok.append(("null_fail", v0["PASS"] is False))
    # (c) dense cannot change PASS
    dense = 0.9 + 0.01 * rng.standard_normal((96, 2))
    v1 = primary_verdict(rec, pru, music, dense=dense)
    ok.append(("dense_irrelevant", v1["PASS"] == v["PASS"] and v1["R_AC"] == v["R_AC"]))
    # (d) I never binds when music strongly negative (lo95(I)>0 at a passing R_AC)
    ok.append(("I_positive", v["PASS_conditions"]["lo95_I_gt0"] is True))
    for name, cond in ok:
        print(f"  {name}: {cond}")
    good = all(c for _, c in ok)
    print("V1-VERDICT SELF-TEST", "PASS" if good else "FAIL")
    return 0 if good else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", help="JSON {dense,pruned,recovered[,recovered_hc,pruned_hc]}")
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_v1_verdict.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.scores:
        raise SystemExit("need --scores or --self-test")
    out = compute(json.load(open(args.scores)))
    out["artifact_sha256"] = hashlib.sha256(
        json.dumps(out, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"PRIMARY_PASS": out["PRIMARY"]["PASS"],
                      "R_AC": out["PRIMARY"]["R_AC"], "I": out["PRIMARY"]["I"]}, indent=2))
    print("V1 verdict written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
