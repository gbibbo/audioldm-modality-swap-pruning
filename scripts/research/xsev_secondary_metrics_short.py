#!/usr/bin/env python3
"""A8 — event-level secondary metrics (KL, PANNs top-10 capture) at BOTH durations, severity 2 (CPU, 0 cr).

Why. The Draft-5 abstract claimed the duration dependence of the recovery gain is "corroborated by a
second scorer, event-level metrics and a frame-level grounding model". Only the second scorer
(Human-CLAP) was measured at both durations; KL / PANNs / FAD / FineLAP were run at the NATIVE point
only, so they corroborate the native gain, not its duration dependence (external-reviewer simulation
2026-09-03, weakness 4 / action A8). This script closes that gap with a NON-CLAP metric family, on the
frozen WAVs already on disk. No generation, no GPU, no new selection.

Design (post-hoc, declared as such; it cannot change any gate or the primary CLAP verdict):
  systems      recovered2 (P+FT), pruned2_A (P, primary seam convention), pruned2_B (seam sensitivity)
  durations    ac_native (10.24 s) and ac_short (3.84 s), the frozen xsev AudioCaps-192 generations
  reference    the real AudioCaps clip of the SAME prompt, band-limited to 16 kHz, as written by
               draft5_floor_ceiling.py --emit: full (<=10 s) for the native point, first 3.84 s
               (61472 samples) for the short point. Same convention on both sides of every contrast.
  metrics      KL(softmax(ref) || softmax(gen)) per prompt, audioldm_eval convention (lower better);
               PANNs Cnn14-16k top-10 capture of the prompt's AudioSet ground-truth labels (higher
               better); FD on the 2048-d PANNs embedding (distributional, descriptive).
  contrasts    R = P+FT - P oriented so that + = recovery helps, at each duration; the duration
               interaction J = R(native) - R(short); the per-system duration responses s(.).
               Paired per-prompt percentile bootstrap, B = 10000, unit = prompt.
  guard        the native point is ALSO recomputed against the original (non-band-limited) reference
               files used by scripts/research/xsev_secondary_metrics.py, and the resulting means are
               compared with that frozen artifact.

Caveat recorded in the output: PANNs capture at 3.84 s is scored against the AudioSet labels of the
FULL 10 s reference clip, so an event absent from the first 3.84 s cannot be captured; this depresses
both systems identically and is why the paired contrast, not the absolute level, is the quantity read.

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_secondary_metrics_short.py
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
REAL_DIR = "artifacts/icassp_gate0/real_refs"
FROZEN_NATIVE = "configs/research/xsev_secondary_metrics.json"
SYSTEMS = ("recovered2", "pruned2_A", "pruned2_B")
REC, PA, PB = "recovered2", "pruned2_A", "pruned2_B"
DURATIONS = {"native": "ac_native", "short": "ac_short"}
NS = "DRAFT5-SHORT-SECONDARY|BOOTSTRAP|2026-09-03"
BOOT_SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/research/xsev_secondary_metrics_short.json")
    ap.add_argument("--scratch", default="/tmp/claude-1000/xsev_sec_short")
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)

    prompts = sorted(json.load(open(AC_MANIFEST))["prompts"], key=lambda p: p["prompt_index"])
    ytids = [p["ytid"] for p in prompts]
    n = len(ytids)
    gt = A.load_gt_indices()
    gt_idx = [gt[y] for y in ytids]

    from audioldm_eval import EvaluationHelper
    helper = EvaluationHelper(A.SR, torch.device("cpu"))

    # ---- references -------------------------------------------------------------------------
    ref_feat = {}
    for tag, pat in (("full", "real_sev2_192_full_p{}.wav"), ("crop", "real_sev2_192_crop_p{}.wav")):
        files = []
        for p in prompts:
            src = os.path.join(REAL_DIR, pat.format(p["prompt_index"]))
            if not os.path.exists(src):
                raise SystemExit(f"missing real reference {src} (run draft5_floor_ceiling.py --emit)")
            files.append((src, f"ref_{tag}_p{p['prompt_index']}.wav"))
        d = A.symlink_dir(a.scratch, f"ref_{tag}", files)
        f = A.get_features(helper, d)
        ref_feat[tag] = [f[f"ref_{tag}_p{p['prompt_index']}.wav"] for p in prompts]
        print(f"[refs] {tag}: {len(ref_feat[tag])} clips", flush=True)

    # guard reference: the ORIGINAL files used by the frozen native artifact
    files = [(A.find_ref(y), f"Y{y}.wav") for y in ytids]
    fg = A.get_features(helper, A.symlink_dir(a.scratch, "ref_orig", files))
    ref_feat["orig"] = [fg[f"Y{y}.wav"] for y in ytids]
    print("[refs] orig: guard set extracted", flush=True)

    # ---- generations ------------------------------------------------------------------------
    gen = {}
    for dur, stem in DURATIONS.items():
        for s in SYSTEMS:
            files = []
            for p in prompts:
                w = os.path.join(XSEV_ROOT, f"{s}_{stem}_p{p['prompt_index']}_r0.wav")
                if not os.path.exists(w):
                    raise SystemExit(f"missing generation {w}")
                files.append((w, os.path.basename(w)))
            d = A.symlink_dir(a.scratch, f"gen_{s}_{dur}", files)
            f = A.get_features(helper, d)
            gen[(s, dur)] = [f[f"{s}_{stem}_p{p['prompt_index']}_r0.wav"] for p in prompts]
            print(f"[gen] {s} {dur}: {len(gen[(s, dur)])} clips", flush=True)

    # ---- per-prompt metrics -----------------------------------------------------------------
    REFTAG = {"native": "full", "short": "crop"}
    kl = {}
    cap = {}
    for dur in DURATIONS:
        rf = ref_feat[REFTAG[dur]]
        for s in SYSTEMS:
            k = np.zeros(n)
            c = np.zeros(n)
            for i in range(n):
                lg = gen[(s, dur)][i]["logits"]
                k[i] = A.kl_pair(lg, rf[i]["logits"])
                c[i] = len(set(np.argsort(lg)[::-1][:10].tolist()) & set(gt_idx[i]))
            kl[(s, dur)] = k
            cap[(s, dur)] = c
    # guard: native against the original reference files
    kl_guard = {}
    cap_guard = {}
    for s in SYSTEMS:
        k = np.zeros(n)
        c = np.zeros(n)
        for i in range(n):
            lg = gen[(s, "native")][i]["logits"]
            k[i] = A.kl_pair(lg, ref_feat["orig"][i]["logits"])
            c[i] = len(set(np.argsort(lg)[::-1][:10].tolist()) & set(gt_idx[i]))
        kl_guard[s] = k
        cap_guard[s] = c

    rng = np.random.default_rng(BOOT_SEED)
    bi = rng.integers(0, n, size=(B, n))

    def vec(v):
        return {"point": float(v.mean()), "ci95": ci(v[bi].mean(1)), "frac_pos": float((v > 0).mean())}

    out = {
        "artifact": "xsev_secondary_metrics_short",
        "action": "A8 (docs/review/2026-09-03_manuscript_draft5_icassp_reviewer_simulation.md)",
        "status": ("POST-HOC / CORROBORATIVE — declared after the primary result; no gate, no primary "
                   "role; cannot change any frozen CLAP verdict"),
        "compute": "CPU only, 0 cr; frozen WAVs, no generation",
        "severity": 2,
        "n": n,
        "systems": {"P+FT": REC, "P (primary seam)": PA, "P (seam sensitivity)": PB},
        "reference_convention": {
            "native": "real AudioCaps clip, 16 kHz band-limited, full (<=10 s): real_sev2_192_full_p*.wav",
            "short": "same clip, first 3.84 s (61472 samples): real_sev2_192_crop_p*.wav",
            "note": "identical reference on both sides of every paired contrast at a given duration"},
        "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": BOOT_SEED, "unit": "prompt"},
        "caveat_capture_short": ("PANNs top-10 capture at 3.84 s is scored against the AudioSet labels of "
                                 "the FULL reference clip; events absent from the first 3.84 s cannot be "
                                 "captured. This depresses both systems identically, so the paired "
                                 "contrast, not the absolute level, is the quantity read."),
        "means": {m: {f"{s}@{d}": float(v[(s, d)].mean()) for s in SYSTEMS for d in DURATIONS}
                  for m, v in (("KL", kl), ("PANN_top10_capture", cap))},
    }

    # paired contrasts, oriented + = recovery helps
    for pruned, tag in ((PA, "prunedA"), (PB, "prunedB")):
        blk = {}
        for d in DURATIONS:
            blk[f"R_KL@{d}"] = vec(kl[(pruned, d)] - kl[(REC, d)])          # lower KL better
            blk[f"R_cap@{d}"] = vec(cap[(REC, d)] - cap[(pruned, d)])       # higher capture better
        blk["J_KL"] = vec((kl[(pruned, "native")] - kl[(REC, "native")])
                          - (kl[(pruned, "short")] - kl[(REC, "short")]))
        blk["J_cap"] = vec((cap[(REC, "native")] - cap[(pruned, "native")])
                           - (cap[(REC, "short")] - cap[(pruned, "short")]))
        out[f"contrasts_recovered_vs_{tag}"] = blk

    out["duration_response"] = {
        m: {s: vec(-(v[(s, "native")] - v[(s, "short")])) if m == "KL"
            else vec(v[(s, "native")] - v[(s, "short")]) for s in SYSTEMS}
        for m, v in (("KL", kl), ("PANN_top10_capture", cap))}
    out["duration_response"]["_orientation"] = ("+ = the system is better at 10.24 s than at 3.84 s "
                                                "(KL sign flipped so that + = lower KL)")

    # distributional FD on the 2048-d PANNs embedding (descriptive)
    from audioldm_eval.metrics.fid import calculate_fid

    def fdict(feats):
        return {"2048": torch.tensor(np.stack([f["2048"] for f in feats]))}

    fd = {}
    for d in DURATIONS:
        fd_ref = fdict(ref_feat[REFTAG[d]])
        for s in SYSTEMS:
            fd[f"{s}@{d}"] = float(calculate_fid(fdict(gen[(s, d)]), fd_ref, "2048")["frechet_distance"])
    out["FD_pann2048"] = fd
    out["FD_note"] = "distributional, not paired; descriptive only; reference set differs by duration"

    # guard against the frozen native artifact
    frozen = json.load(open(FROZEN_NATIVE))
    guard = {"frozen_artifact": FROZEN_NATIVE, "note": (
        "the frozen artifact scored the native point against the ORIGINAL reference files; recomputed "
        "here with the same references, the per-system means must reproduce it to numerical precision")}
    for s in SYSTEMS:
        guard[s] = {
            "KL_frozen": frozen["means"]["KL"][s], "KL_recomputed_origref": float(kl_guard[s].mean()),
            "KL_absdiff": abs(frozen["means"]["KL"][s] - float(kl_guard[s].mean())),
            "cap_frozen": frozen["means"]["PANN_top10_capture"][s],
            "cap_recomputed_origref": float(cap_guard[s].mean()),
            "cap_absdiff": abs(frozen["means"]["PANN_top10_capture"][s] - float(cap_guard[s].mean())),
            "KL_bandlimited_ref": float(kl[(s, "native")].mean())}
    guard["max_KL_absdiff"] = max(guard[s]["KL_absdiff"] for s in SYSTEMS)
    guard["max_cap_absdiff"] = max(guard[s]["cap_absdiff"] for s in SYSTEMS)
    guard["PASS"] = bool(guard["max_KL_absdiff"] < 1e-9 and guard["max_cap_absdiff"] < 1e-9)
    out["guard_native_vs_frozen"] = guard

    txt = json.dumps(out, indent=1, sort_keys=True)
    out["artifact_sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in
                      ("means", "contrasts_recovered_vs_prunedA", "duration_response",
                       "guard_native_vs_frozen")}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
