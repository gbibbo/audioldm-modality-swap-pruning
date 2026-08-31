#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — structural validation of generated WAVs (CPU, free, read-only).

Operational peeking ONLY (process/file health): existence, load, sample-rate, n_samples vs manifest,
finiteness (NaN/Inf), on-disk sha256 integrity vs manifest, CRN (x_T seed shared across systems within
a (context,ytid,rep)), and matched-group readiness (same ytid multiset across the 3 systems per context).
Does NOT read/score CLAP or any confirmatory metric. No adaptive stopping.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_structural_validate.py \
        --jobroot /teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np, soundfile as sf

SR = 16000
EXPECT = {  # (system, context) -> expected n rows
    (s, c): 192 for s in ("recovered2", "pruned2_A", "pruned2_B") for c in ("ac_native", "ac_short", "music")
}
NSAMP = {"ac_native": 163872, "ac_short": 61472, "music": 61472, "dense_native": 163872}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def validate_manifest(mpath, jobroot):
    d = json.load(open(mpath))
    sysn, ctx = d["system"], d["context"]
    errs, rows_ok = [], 0
    seen = set()
    seed_by_key = {}
    for r in d["rows"]:
        wav = os.path.join(jobroot, r["wav"])
        key = (r["ytid"], r["replicate_index"])
        if key in seen:
            errs.append(f"dup {key}")
        seen.add(key)
        seed_by_key[key] = r["seed"]
        if not os.path.exists(wav):
            errs.append(f"missing {os.path.basename(wav)}"); continue
        try:
            data, sr = sf.read(wav, dtype="float32")
        except Exception as e:
            errs.append(f"unreadable {os.path.basename(wav)}: {e}"); continue
        data = np.asarray(data).squeeze()
        if sr != SR:
            errs.append(f"sr {sr} {os.path.basename(wav)}")
        if data.shape[-1] != r["n_samples"]:
            errs.append(f"nsamp {data.shape[-1]}!={r['n_samples']} {os.path.basename(wav)}")
        if NSAMP.get(ctx) and r["n_samples"] != NSAMP[ctx]:
            errs.append(f"nsamp-manifest {r['n_samples']}!={NSAMP[ctx]} {os.path.basename(wav)}")
        if not np.isfinite(data).all():
            errs.append(f"NONFINITE {os.path.basename(wav)}")
        if sha_file(wav) != r["wav_sha256"]:
            errs.append(f"sha-mismatch {os.path.basename(wav)}")
        if str(r.get("device")) != "cuda":
            errs.append(f"device {r.get('device')} {os.path.basename(wav)}")
        rows_ok += 1
    exp = EXPECT.get((sysn, ctx))
    if exp is not None and d["n"] != exp:
        errs.append(f"n {d['n']}!={exp}")
    return {"system": sysn, "context": ctx, "n": d["n"], "rows_checked": rows_ok,
            "errors": errs, "seed_by_key": seed_by_key, "ytids": sorted({k[0] for k in seen})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobroot", default="/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning")
    ap.add_argument("--out", default="artifacts/icassp_gate0/xsev_structural_validation.json")
    a = ap.parse_args()
    gendir = os.path.join(a.jobroot, "artifacts/icassp_gate0/reversal_xsev_gen")
    manifests = sorted(glob.glob(os.path.join(gendir, "gen_manifest_*.json")))
    per = [validate_manifest(m, a.jobroot) for m in manifests]

    # CRN: within a context, the seed for (ytid,rep) must be identical across the 3 systems.
    crn_errs = []
    by_ctx = {}
    for p in per:
        by_ctx.setdefault(p["context"], []).append(p)
    for ctx, plist in by_ctx.items():
        if ctx == "dense_native":
            continue
        allkeys = set().union(*[set(p["seed_by_key"]) for p in plist])
        for k in allkeys:
            seeds = {p["system"]: p["seed_by_key"].get(k) for p in plist}
            uniq = {v for v in seeds.values() if v is not None}
            if len(uniq) > 1:
                crn_errs.append(f"{ctx} {k} seeds differ across systems: {seeds}")
    # matched groups: same ytid multiset across systems per context
    group_errs = []
    for ctx, plist in by_ctx.items():
        ysets = {p["system"]: tuple(p["ytids"]) for p in plist}
        base = None
        for sysn, ys in ysets.items():
            if base is None:
                base = ys
            elif ys != base:
                group_errs.append(f"{ctx}: {sysn} ytids differ from reference")

    total_rows = sum(p["rows_checked"] for p in per)
    total_err = sum(len(p["errors"]) for p in per) + len(crn_errs) + len(group_errs)
    verdict = "STRUCTURAL VALIDATION PASS" if total_err == 0 and total_rows == 1728 else "STRUCTURAL VALIDATION FAIL"
    out = {"artifact": "xsev_structural_validation", "jobroot": a.jobroot,
           "verdict": verdict, "total_rows_checked": total_rows, "expected_rows": 1728,
           "total_errors": total_err,
           "per_stage": [{"system": p["system"], "context": p["context"], "n": p["n"],
                          "rows_checked": p["rows_checked"], "n_errors": len(p["errors"]),
                          "errors": p["errors"][:20]} for p in per],
           "crn_errors": crn_errs[:20], "matched_group_errors": group_errs,
           "note": "1728 severity-2 rows only; dense_native control (80) validated separately once complete."}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{verdict} | rows {total_rows}/1728 | errors {total_err} -> {a.out}")
    for p in per:
        print(f"  {p['system']:11s} {p['context']:10s} rows={p['rows_checked']:3d} errs={len(p['errors'])}")
    if crn_errs: print("CRN ERRORS:", len(crn_errs))
    if group_errs: print("GROUP ERRORS:", group_errs)
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
