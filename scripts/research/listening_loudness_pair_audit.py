#!/usr/bin/env python3
"""Private per-pair loudness audit (§8). CPU only. Uses the PRIVATE key + bundle manifest
locally; exposes NO filenames or the key. For every EXPERIMENTAL pair, computes the
recovered vs pruned integrated-loudness difference of the LISTENING COPIES the listener
actually hears, by stratum, and decides whether residual loudness systematically favours
either system. Emits configs/research/listening_loudness_pair_audit.json.

Run after build_listening_bundle.py:
  .venv-loudness/bin/python scripts/research/listening_loudness_pair_audit.py
"""
import json, os, hashlib
import numpy as np

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
PRIV = os.path.join(ROOT, "configs/research/listening_study_assignments_private.json")
BM = os.path.join(ROOT, "configs/research/listening_study_bundle_manifest.json")
OUT = os.path.join(ROOT, "configs/research/listening_loudness_pair_audit.json")
FAVOUR_LIMIT = 0.5   # |signed mean dLUFS| above this = meaningful systematic imbalance -> STOP


def summ(vals):
    a = np.abs(np.array(vals)) if vals else np.array([0.0])
    s = np.array(vals) if vals else np.array([0.0])
    return {"n": len(vals),
            "signed_mean_recovered_minus_pruned": round(float(s.mean()), 4),
            "signed_median": round(float(np.median(s)), 4),
            "abs_mean": round(float(a.mean()), 4), "abs_median": round(float(np.median(a)), 4),
            "abs_p95": round(float(np.percentile(a, 95)), 4), "abs_max": round(float(a.max()), 4),
            "frac_abs_gt_1dB": round(float((a > 1).mean()), 4),
            "frac_abs_gt_2dB": round(float((a > 2).mean()), 4)}


def main():
    os.chdir(ROOT)
    priv = json.load(open(PRIV))
    bm = json.load(open(BM))["files"]   # hash -> {copy_lufs, stim_id, ...}
    strata = {"sev1_short": [], "sev1_native": []}
    outlier_cat = {}
    for code, pv in priv["participants"].items():
        for t in pv["trials"]:
            if t["type"] != "experimental":
                continue
            key = f"sev1_{t['duration']}"
            hA, hB = t["audio_A_hash"], t["audio_B_hash"]
            LA, LB = bm[hA]["copy_lufs"], bm[hB]["copy_lufs"]
            rec_L = LA if t["recovered_side"] == "A" else LB
            pru_L = LB if t["recovered_side"] == "A" else LA
            d = rec_L - pru_L
            strata[key].append(d)
            # categorize outliers privately by (system,duration) not filename
            for sysname, Lv in (("recovered", rec_L), ("pruned", pru_L)):
                if abs(Lv + 36.0) > 1.0:
                    outlier_cat[f"{sysname}|{t['duration']}"] = outlier_cat.get(f"{sysname}|{t['duration']}", 0) + 1

    report = {k: summ(v) for k, v in strata.items()}
    all_signed = strata["sev1_short"] + strata["sev1_native"]
    max_favour = max(abs(report[k]["signed_mean_recovered_minus_pruned"]) for k in report)
    verdict = "NEGLIGIBLE — no systematic loudness advantage" if max_favour <= FAVOUR_LIMIT else "IMBALANCE — STOP"
    out = {"artifact": "listening_loudness_pair_audit",
           "note": "differences are of LISTENING COPIES (post -36 LUFS normalization) as heard.",
           "favour_limit_dB": FAVOUR_LIMIT, "by_stratum": report,
           "overall_signed_mean": round(float(np.mean(all_signed)), 4),
           "max_abs_signed_mean_across_strata": round(max_favour, 4),
           "remeasurement_outliers_by_category": outlier_cat,
           "verdict": verdict}
    payload = json.dumps(out, indent=2, sort_keys=True)
    out["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json.dump(out, open(OUT, "w"), indent=2, sort_keys=True)
    for k, v in report.items():
        print(k, "signed_mean(rec-pru)=%.3f dB abs_p95=%.3f max=%.3f >1dB=%.3f" %
              (v["signed_mean_recovered_minus_pruned"], v["abs_p95"], v["abs_max"], v["frac_abs_gt_1dB"]))
    print("overall signed mean:", out["overall_signed_mean"], "| outliers by cat:", outlier_cat)
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
