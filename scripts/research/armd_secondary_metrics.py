#!/usr/bin/env python3
"""OP-DURATION-DISCRIMINATOR-1 (Arm D) — SECONDARY metrics interaction (KL / PANN capture / FAD / FD).

Descriptive/corroborative only; NO primary role, no gate, no composite. Reuses the VALIDATED
RECOVERY-METRIC-AUDIT-1 machinery (Cnn14-16k, exact audioldm_eval KL, VGGish FAD). On the 80-ytid
Arm-D subset, pruned+recovered, at BOTH operating points:
  control = V1.1 3.84 s r0 WAVs;  alt = new 10.24 s r0 WAVs;  refs = the 80 real AudioCaps clips.
Oriented so + = recovered better:  R_KL = KL_pruned - KL_recovered ;  R_cap = cap_recovered - cap_pruned.
J_metric = R_metric_alt - R_metric_ctrl (paired ytid bootstrap PCG64(20260830)). FAD/FD per system/OP.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/armd_secondary_metrics.py \
        --alt-root <armd gen dir> --ctrl-root <V1.1 gen dir> --out configs/research/op_duration_discriminator_1_secondary.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
import torch
import recovery_metric_audit_1 as A

SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
BOOT_SEED = 20260830
B = 10000
CTRL_PREFIX = {"pruned": "p1_pruned_ema_reconstructed_noadapter", "recovered": "p1_recovered_noadapter"}
ALT_PREFIX = {"pruned": "p1_pruned_ema_reconstructed_noadapter_alt10s", "recovered": "p1_recovered_noadapter_alt10s"}


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def features_for(helper, scratch, tag, wav_map):
    files = [(src, bn) for bn, src in wav_map.items()]
    d = A.symlink_dir(scratch, tag, files)
    return A.get_features(helper, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-root", required=True)
    ap.add_argument("--ctrl-root", required=True)
    ap.add_argument("--out", default="configs/research/op_duration_discriminator_1_secondary.json")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("SCRATCH", "/tmp/claude-1000"), "armd_sec"))
    ap.add_argument("--skip-fad", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)

    prompts = sorted(json.load(open(SUBSET))["prompts"], key=lambda p: p["subset_prompt_index"])
    ytids = [p["ytid"] for p in prompts]; n = len(ytids)
    gt = A.load_gt_indices(); gt_idx = [gt[y] for y in ytids]

    # references (80 real clips)
    ref_files = [(A.find_ref(y), f"Y{y}.wav") for y in ytids]
    ref_dir = A.symlink_dir(args.scratch, "refs", ref_files)

    from audioldm_eval import EvaluationHelper
    helper = EvaluationHelper(A.SR, torch.device("cpu"))
    fr = A.get_features(helper, ref_dir)
    ref_logits = {y: fr[f"Y{y}.wav"]["logits"] for y in ytids}

    # build wav maps: {basename: srcpath}
    def gen_map(op, sysk):
        m = {}
        for p in prompts:
            if op == "ctrl":
                bn = f"{CTRL_PREFIX[sysk]}_p{p['v1_1_prompt_index']}_r0.wav"; root = args.ctrl_root
            else:
                bn = f"{ALT_PREFIX[sysk]}_p{p['subset_prompt_index']}_r0.wav"; root = args.alt_root
            m[bn] = os.path.join(root, bn)
        return m

    feats = {}
    for op in ("ctrl", "alt"):
        for sysk in ("pruned", "recovered"):
            feats[(op, sysk)] = features_for(helper, args.scratch, f"gen_{op}_{sysk}", gen_map(op, sysk))

    # per-ytid KL + capture, oriented + = recovered better
    def per_ytid(op):
        pref = {"pruned": CTRL_PREFIX["pruned"] if op == "ctrl" else ALT_PREFIX["pruned"],
                "recovered": CTRL_PREFIX["recovered"] if op == "ctrl" else ALT_PREFIX["recovered"]}
        idx_key = "v1_1_prompt_index" if op == "ctrl" else "subset_prompt_index"
        kl = {"pruned": np.zeros(n), "recovered": np.zeros(n)}
        cap = {"pruned": np.zeros(n), "recovered": np.zeros(n)}
        for i, p in enumerate(prompts):
            for sysk in ("pruned", "recovered"):
                bn = f"{pref[sysk]}_p{p[idx_key]}_r0.wav"
                lg = feats[(op, sysk)][bn]["logits"]
                kl[sysk][i] = A.kl_pair(lg, ref_logits[ytids[i]])
                cap[sysk][i] = len(set(np.argsort(lg)[::-1][:10].tolist()) & set(gt_idx[i]))
        return (kl["pruned"] - kl["recovered"]), (cap["recovered"] - cap["pruned"])  # R_KL, R_cap

    RKL_ctrl, RCAP_ctrl = per_ytid("ctrl")
    RKL_alt, RCAP_alt = per_ytid("alt")
    rng = np.random.default_rng(BOOT_SEED)
    bi = rng.integers(0, n, size=(B, n))

    def interaction(Rc, Ra, name, direction):
        J = Ra - Rc
        out = {"orientation": "+ = recovered better", "direction_note": direction,
               "R_ctrl": {"point": float(Rc.mean()), "ci95": ci(Rc[bi].mean(1))},
               "R_alt": {"point": float(Ra.mean()), "ci95": ci(Ra[bi].mean(1))},
               "J": {"point": float(J.mean()), "ci95": ci(J[bi].mean(1))},
               "frac_ytid_j_pos": float(np.mean(J > 0))}
        return out

    out = {"artifact": "op_duration_discriminator_1_secondary",
           "status": "SECONDARY/DESCRIPTIVE — no primary role, no gate; cannot change V1.1 or Arm-D primary",
           "n": n, "bootstrap": {"B": B, "seed_pcg64": BOOT_SEED, "unit": "ytid"},
           "KL": interaction(RKL_ctrl, RKL_alt, "kl", "R_KL=KL_pruned-KL_recovered (KL lower better)"),
           "PANN_capture": interaction(RCAP_ctrl, RCAP_alt, "cap", "R_cap=cap_recovered-cap_pruned (capture higher better)")}

    # FAD / FD per system per OP (descriptive)
    from audioldm_eval.metrics.fid import calculate_fid
    ref_names = [f"Y{y}.wav" for y in ytids]
    fd_ref = {"file_path_": ref_names, "2048": torch.tensor(np.stack([fr[nm]["2048"] for nm in ref_names]))}
    out["FD_pann2048"] = {}
    for op in ("ctrl", "alt"):
        out["FD_pann2048"][op] = {}
        for sysk in ("pruned", "recovered"):
            names = list(feats[(op, sysk)].keys())
            g = {"file_path_": names, "2048": torch.tensor(np.stack([feats[(op, sysk)][nm]["2048"] for nm in names]))}
            out["FD_pann2048"][op][sysk] = float(calculate_fid(g, fd_ref, "2048")["frechet_distance"])
    if not args.skip_fad:
        try:
            fr_frechet = helper.frechet
            out["FAD_vggish"] = {}
            for op in ("ctrl", "alt"):
                out["FAD_vggish"][op] = {}
                for sysk in ("pruned", "recovered"):
                    d = A.symlink_dir(args.scratch, f"gen_{op}_{sysk}", [(v, k) for k, v in gen_map(op, sysk).items()])
                    sc = fr_frechet.score(d, ref_dir, recalculate=True)
                    out["FAD_vggish"][op][sysk] = float(sc["frechet_audio_distance"]) if isinstance(sc, dict) else None
        except Exception as e:
            out["FAD_vggish"] = {"error": str(e)}

    payload = {**out, "protocol_doc_sha256": A.sha_file("docs/op_duration_discriminator_1.md")
               if os.path.exists("docs/op_duration_discriminator_1.md") else None}
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    json.dump(payload, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("KL   J =", round(out["KL"]["J"]["point"], 4), out["KL"]["J"]["ci95"])
    print("PANN J =", round(out["PANN_capture"]["J"]["point"], 4), out["PANN_capture"]["J"]["ci95"])
    print("FD:", {op: {s: round(out["FD_pann2048"][op][s], 2) for s in ("pruned", "recovered")} for op in ("ctrl", "alt")})
    if "FAD_vggish" in out and "ctrl" in out.get("FAD_vggish", {}):
        print("FAD:", {op: {s: round(out["FAD_vggish"][op][s], 2) for s in ("pruned", "recovered")} for op in ("ctrl", "alt")})
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
