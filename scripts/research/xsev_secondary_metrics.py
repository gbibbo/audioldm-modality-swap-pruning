#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — AudioCaps SECONDARY metrics (KL / PANN top-10 capture / FD / FAD).

Descriptive/corroborative ONLY; no primary role, no gate, no rescue. Reuses the VALIDATED
RECOVERY-METRIC-AUDIT-1 machinery (Cnn14-16k logits + 2048 feat, exact audioldm_eval KL, VGGish FAD,
PANN-2048 FD) on the frozen xsev AudioCaps-192 ytids at the NATIVE 10.24 s operating point, for the
three severity-2 systems {recovered2, pruned2_A, pruned2_B}, vs the 192 real AudioCaps reference clips.
Music KL/PANN/FAD/FD are UNAVAILABLE (0/64 MusicCaps real refs) and are NOT computed here.

Contrasts oriented + = recovered better:  R_KL = KL_pruned - KL_recovered ;  R_cap = cap_rec - cap_pruned.
Paired ytid bootstrap PCG64(20260831), B=10000 (same seed family as the primary xsev verdict).

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_secondary_metrics.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
import torch
import recovery_metric_audit_1 as A

AC_MANIFEST = "configs/research/xsev_audiocaps_manifest.json"
XSEV_ROOT = "/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_xsev_gen"
SYSTEMS = ("recovered2", "pruned2_A", "pruned2_B")
BOOT_SEED = 20260831
B = 10000


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/research/xsev_secondary_metrics.json")
    ap.add_argument("--scratch", default="/tmp/claude-1000/xsev_sec")
    ap.add_argument("--skip-fad", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)

    prompts = sorted(json.load(open(AC_MANIFEST))["prompts"], key=lambda p: p["prompt_index"])
    ytids = [p["ytid"] for p in prompts]; n = len(ytids)
    gt = A.load_gt_indices(); gt_idx = [gt[y] for y in ytids]

    ref_files = [(A.find_ref(y), f"Y{y}.wav") for y in ytids]
    ref_dir = A.symlink_dir(a.scratch, "refs", ref_files)

    from audioldm_eval import EvaluationHelper
    helper = EvaluationHelper(A.SR, torch.device("cpu"))
    fr = A.get_features(helper, ref_dir)
    ref_logits = {y: fr[f"Y{y}.wav"]["logits"] for y in ytids}

    # gen feature maps per system (native OP)
    feats = {}
    for s in SYSTEMS:
        m = {f"{s}_ac_native_p{p['prompt_index']}_r0.wav":
             os.path.join(XSEV_ROOT, f"{s}_ac_native_p{p['prompt_index']}_r0.wav") for p in prompts}
        feats[s] = A.get_features(helper, A.symlink_dir(a.scratch, f"gen_{s}", [(v, k) for k, v in m.items()]))

    # per-ytid KL + PANN top-10 capture, per system
    kl = {s: np.zeros(n) for s in SYSTEMS}
    cap = {s: np.zeros(n) for s in SYSTEMS}
    for i, p in enumerate(prompts):
        for s in SYSTEMS:
            lg = feats[s][f"{s}_ac_native_p{p['prompt_index']}_r0.wav"]["logits"]
            kl[s][i] = A.kl_pair(lg, ref_logits[ytids[i]])
            cap[s][i] = len(set(np.argsort(lg)[::-1][:10].tolist()) & set(gt_idx[i]))

    rng = np.random.default_rng(BOOT_SEED)
    bi = rng.integers(0, n, size=(B, n))

    def contrast(rec, pru, higher_better):
        # R oriented + = recovered better
        R = (rec - pru) if higher_better else (pru - rec)
        return {"point": float(R.mean()), "ci95": ci(R[bi].mean(1)), "frac_pos": float((R > 0).mean())}

    out = {"artifact": "xsev_secondary_metrics",
           "status": "SECONDARY/CORROBORATIVE — no primary role, no gate; cannot change the CASE C verdict",
           "operating_point": "AudioCaps NATIVE 10.24 s", "n": n,
           "bootstrap": {"B": B, "seed_pcg64": BOOT_SEED, "unit": "ytid"},
           "means": {"KL": {s: float(kl[s].mean()) for s in SYSTEMS},
                     "PANN_top10_capture": {s: float(cap[s].mean()) for s in SYSTEMS}},
           "contrasts_recovered_vs_prunedA": {
               "R_KL": contrast(kl["pruned2_A"], kl["recovered2"], higher_better=True),   # KL lower better
               "R_cap": contrast(cap["recovered2"], cap["pruned2_A"], higher_better=True)},
           "contrasts_recovered_vs_prunedB": {
               "R_KL": contrast(kl["pruned2_B"], kl["recovered2"], higher_better=True),
               "R_cap": contrast(cap["recovered2"], cap["pruned2_B"], higher_better=True)},
           "music_secondary": "UNAVAILABLE (0/64 MusicCaps real refs); NOT computed; primary music CLAP unaffected"}

    # FD-2048 (PANN) per system, descriptive
    from audioldm_eval.metrics.fid import calculate_fid
    ref_names = [f"Y{y}.wav" for y in ytids]
    fd_ref = {"file_path_": ref_names, "2048": torch.tensor(np.stack([fr[nm]["2048"] for nm in ref_names]))}
    out["FD_pann2048"] = {}
    for s in SYSTEMS:
        names = list(feats[s].keys())
        g = {"file_path_": names, "2048": torch.tensor(np.stack([feats[s][nm]["2048"] for nm in names]))}
        out["FD_pann2048"][s] = float(calculate_fid(g, fd_ref, "2048")["frechet_distance"])

    if not a.skip_fad:
        try:
            out["FAD_vggish"] = {}
            for s in SYSTEMS:
                m = {f"{s}_ac_native_p{p['prompt_index']}_r0.wav":
                     os.path.join(XSEV_ROOT, f"{s}_ac_native_p{p['prompt_index']}_r0.wav") for p in prompts}
                d = A.symlink_dir(a.scratch, f"gen_{s}", [(v, k) for k, v in m.items()])
                sc = helper.frechet.score(d, ref_dir, recalculate=True)
                out["FAD_vggish"][s] = float(sc["frechet_audio_distance"]) if isinstance(sc, dict) else None
        except Exception as e:
            out["FAD_vggish"] = {"error": str(e)}

    out["artifact_sha256"] = hashlib.sha256(json.dumps(out, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("KL means:", {s: round(out["means"]["KL"][s], 3) for s in SYSTEMS})
    print("PANN cap means:", {s: round(out["means"]["PANN_top10_capture"][s], 3) for s in SYSTEMS})
    print("R_KL(A)=", out["contrasts_recovered_vs_prunedA"]["R_KL"]["point"],
          "R_cap(A)=", out["contrasts_recovered_vs_prunedA"]["R_cap"]["point"])
    print("FD:", {s: round(out["FD_pann2048"][s], 2) for s in SYSTEMS})
    if "FAD_vggish" in out and isinstance(out["FAD_vggish"], dict) and "error" not in out["FAD_vggish"]:
        print("FAD:", {s: round(v, 2) for s, v in out["FAD_vggish"].items()})
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
