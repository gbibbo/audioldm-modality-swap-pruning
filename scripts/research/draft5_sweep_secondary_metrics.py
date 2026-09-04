#!/usr/bin/env python3
"""DRAFT5-OPSWEEP-1 — event-level secondary metrics (KL, PANNs top-10 capture) on the sweep cells
(5.12 s, 7.68 s), severity 2, CPU, 0 cr. POST-HOC / CORROBORATIVE: declared after the sweep's CLAP
verdict was read; no gate; cannot change the pre-specified CLAP shape verdict.

Question. Is the MONOTONE-INCREASING shape of the recovery gain (CLAP, docs/draft5_opsweep.md) also seen
by two non-CLAP event-level metrics across all four durations?

Design (same conventions as scripts/research/xsev_secondary_metrics_short.py):
  systems      recovered2 (P+FT), pruned2_A (P, primary seam), dense
  durations    3.84 / 10.24 s = frozen xsev WAVs (recomputed here as a GUARD against the committed A8
               artifact); 5.12 / 7.68 s = DRAFT5-OPSWEEP-1 WAVs
  reference    the real AudioCaps clip of the SAME prompt, 16 kHz band-limited, truncated to the
               generated length: crop (61 472) / d128 (81 952) / d192 (122 912) / full (<=10 s)
  metrics      KL(softmax(ref) || softmax(gen)) per prompt (lower better); PANNs Cnn14-16k top-10
               capture of the prompt's AudioSet labels (higher better)
  contrasts    R(d) = P+FT - P oriented so + = recovery helps; steps D1..D3; duration responses.
               Paired prompt-level percentile bootstrap, B = 10000.

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_sweep_secondary_metrics.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
import torch
import recovery_metric_audit_1 as A

AC_MANIFEST = "configs/research/xsev_audiocaps_manifest.json"
XSEV_ROOT = ("/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning/"
             "artifacts/icassp_gate0/reversal_xsev_gen")
D192_ROOT = ("/teamspace/jobs/xsev-dense-192-1/artifacts/audioldm-modality-swap-pruning/"
             "artifacts/icassp_gate0/xsev_dense_192_gen")
SWEEP_ROOT = ("/teamspace/jobs/draft5-opsweep-1/artifacts/audioldm-modality-swap-pruning/"
              "artifacts/icassp_gate0/draft5_opsweep_gen")
REAL_DIR = "artifacts/icassp_gate0/real_refs"
FROZEN_A8 = "configs/research/xsev_secondary_metrics_short.json"
REC, PA, DEN = "recovered2", "pruned2_A", "dense"
SYSTEMS = (REC, PA, DEN)
# duration -> (context stem, reference tag, generation root per system)
CELLS = {3.84: ("ac_short", "crop", {REC: XSEV_ROOT, PA: XSEV_ROOT, DEN: D192_ROOT}),
         5.12: ("ac_d128", "d128", {s: SWEEP_ROOT for s in SYSTEMS}),
         7.68: ("ac_d192", "d192", {s: SWEEP_ROOT for s in SYSTEMS}),
         10.24: ("ac_native", "full", {REC: XSEV_ROOT, PA: XSEV_ROOT, DEN: D192_ROOT})}
DUR = sorted(CELLS)
NS = "DRAFT5-OPSWEEP-1|SECONDARY-METRICS|BOOTSTRAP|2026-09-04"
BOOT_SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/research/draft5_sweep_secondary_metrics.json")
    ap.add_argument("--scratch", default="/tmp/claude-1000/draft5_sweep_sec")
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    prompts = sorted(json.load(open(AC_MANIFEST))["prompts"], key=lambda p: p["prompt_index"])
    ytids = [p["ytid"] for p in prompts]; n = len(ytids)
    gt = A.load_gt_indices(); gt_idx = [gt[y] for y in ytids]
    from audioldm_eval import EvaluationHelper
    helper = EvaluationHelper(A.SR, torch.device("cpu"))

    ref_feat = {}
    for tag in ("crop", "d128", "d192", "full"):
        files = []
        for p in prompts:
            src = os.path.join(REAL_DIR, f"real_sev2_192_{tag}_p{p['prompt_index']}.wav")
            if not os.path.exists(src):
                raise SystemExit(f"missing reference {src}")
            files.append((src, f"ref_{tag}_p{p['prompt_index']}.wav"))
        f = A.get_features(helper, A.symlink_dir(a.scratch, f"ref_{tag}", files))
        ref_feat[tag] = [f[f"ref_{tag}_p{p['prompt_index']}.wav"] for p in prompts]
        print(f"[refs] {tag}: {len(ref_feat[tag])}", flush=True)

    gen = {}
    for d, (stem, _tag, roots) in CELLS.items():
        for s in SYSTEMS:
            files = []
            for p in prompts:
                w = os.path.join(roots[s], f"{s}_{stem}_p{p['prompt_index']}_r0.wav")
                if not os.path.exists(w):
                    raise SystemExit(f"missing generation {w}")
                files.append((w, os.path.basename(w)))
            f = A.get_features(helper, A.symlink_dir(a.scratch, f"gen_{s}_{stem}", files))
            gen[(s, d)] = [f[f"{s}_{stem}_p{p['prompt_index']}_r0.wav"] for p in prompts]
            print(f"[gen] {s} @{d}: {len(gen[(s, d)])}", flush=True)

    kl, cap = {}, {}
    for d, (_stem, tag, _r) in CELLS.items():
        rf = ref_feat[tag]
        for s in SYSTEMS:
            k = np.zeros(n); c = np.zeros(n)
            for i in range(n):
                lg = gen[(s, d)][i]["logits"]
                k[i] = A.kl_pair(lg, rf[i]["logits"])
                c[i] = len(set(np.argsort(lg)[::-1][:10].tolist()) & set(gt_idx[i]))
            kl[(s, d)] = k; cap[(s, d)] = c

    rng = np.random.default_rng(BOOT_SEED); bi = rng.integers(0, n, size=(B, n))

    def vec(v):
        return {"point": float(v.mean()), "ci95": ci(v[bi].mean(1)), "frac_pos": float((v > 0).mean())}

    out = {"artifact": "draft5_sweep_secondary_metrics", "severity": 2, "n": n,
           "status": "POST-HOC / CORROBORATIVE — declared after the sweep's CLAP shape verdict; no gate",
           "compute": "CPU only, 0 cr; existing WAVs, no generation",
           "reference_convention": "real AudioCaps clip of the same prompt, 16 kHz, truncated to the generated length",
           "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": BOOT_SEED, "unit": "prompt"},
           "caveat_capture": ("PANNs capture at every duration is scored against the AudioSet labels of the "
                              "FULL reference clip; the paired contrast, not the level, is the quantity read"),
           "means": {m: {f"{s}@{d}": float(v[(s, d)].mean()) for s in SYSTEMS for d in DUR}
                     for m, v in (("KL", kl), ("PANN_top10_capture", cap))}}
    R_kl = {d: kl[(PA, d)] - kl[(REC, d)] for d in DUR}       # + = recovery helps (lower KL)
    R_cap = {d: cap[(REC, d)] - cap[(PA, d)] for d in DUR}
    for name, R in (("KL", R_kl), ("PANN_top10_capture", R_cap)):
        blk = {f"R@{d}": vec(R[d]) for d in DUR}
        for k, (a_, b_) in zip(("D1", "D2", "D3"), zip(DUR[:-1], DUR[1:])):
            blk[k] = vec(R[b_] - R[a_])
        blk["J_native_minus_short"] = vec(R[10.24] - R[3.84])
        pts = [blk[k]["point"] for k in ("D1", "D2", "D3")]
        blk["shape_descriptive"] = ("MONOTONE-INCREASING (all steps > 0, no hi95 < 0)"
                                    if all(p > 0 for p in pts) and all(blk[k]["ci95"][1] >= 0 for k in ("D1", "D2", "D3"))
                                    else "NOT MONOTONE by the CLAP rule (descriptive; no gate here)")
        blk["rho_dense"] = {}
        for d in DUR:
            gap = (kl[(PA, d)] - kl[(DEN, d)]) if name == "KL" else (cap[(DEN, d)] - cap[(PA, d)])
            r = R[d]
            bm = r[bi].mean(1) / gap[bi].mean(1)
            blk["rho_dense"][str(d)] = {"point": float(r.mean() / gap.mean()), "ci95": ci(bm)}
        out[f"recovery_gain_{name}"] = blk
    out["duration_steps_per_system"] = {
        m: {s: {f"{a_}->{b_}": vec((-(v[(s, b_)] - v[(s, a_)])) if m == "KL" else (v[(s, b_)] - v[(s, a_)]))
                for a_, b_ in zip(DUR[:-1], DUR[1:])} for s in SYSTEMS}
        for m, v in (("KL", kl), ("PANN_top10_capture", cap))}
    out["duration_steps_per_system"]["_orientation"] = "+ = better at the longer duration (KL sign flipped)"

    # guard: the 3.84 / 10.24 s cells recomputed here must reproduce the committed A8 artifact
    fro = json.load(open(FROZEN_A8))["means"]
    g = {}
    for s in (REC, PA):
        for d, key in ((3.84, "short"), (10.24, "native")):
            g[f"KL {s}@{d}"] = (float(kl[(s, d)].mean()), fro["KL"][f"{s}@{key}"])
            g[f"cap {s}@{d}"] = (float(cap[(s, d)].mean()), fro["PANN_top10_capture"][f"{s}@{key}"])
    worst = max(abs(x - y) for x, y in g.values())
    out["guard_vs_A8_artifact"] = {"cells": {k: {"here": x, "committed": y} for k, (x, y) in g.items()},
                                   "max_absdiff": worst, "PASS": bool(worst < 1e-9)}
    txt = json.dumps(out, indent=1, sort_keys=True)
    out["artifact_sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("means", "recovery_gain_KL", "recovery_gain_PANN_top10_capture",
                                          "guard_vs_A8_artifact")}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
