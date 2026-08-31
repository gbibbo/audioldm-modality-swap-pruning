#!/usr/bin/env python3
"""Build the FROZEN listening-study stimulus inventory (CPU, read-only over WAVs).

Resolves every candidate perceptual stimulus from already-frozen generation
manifests, verifies existence + SHA256 (against the generation manifest),
sample-rate, duration, and measures ITU-R BS.1770 integrated loudness + sample
peak. Emits a loudness-feasibility verdict for the frozen normalization target.

No new generation. No GPU. Does not modify any scientific WAV. Reads only.

Outputs:
  configs/research/listening_study_inventory.json

Run:  .venv-loudness/bin/python scripts/research/build_listening_inventory.py
Check (recompute, compare to committed): add --check
"""
import json, os, sys, hashlib, argparse
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
ARMD = "/teamspace/jobs/reversal-armd-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_armd_gen"
V11  = "/teamspace/jobs/reversal-v11-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_v1_1_gen"
XSEV = "/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_xsev_gen"

# Frozen normalization design (BS.1770 listening copies only; see protocol §7).
# -23 LUFS (the initial §7 proposal) is INFEASIBLE here: these AudioLDM clips are
# quiet in integrated loudness but carry high transient peaks (crest up to ~34 dB,
# driven by near-silent failed-pruned generations that must NOT be excluded because
# loudness correlates with the pruned-failure effect). The most binding stimulus
# fixes the feasible target at <= -35.06 LUFS for a -1.0 dBFS sample-peak ceiling.
# Frozen conservative choice: -36 LUFS / -1 dBFS -> 0 unsafe over the full pool.
LUFS_TARGET = -36.0          # ITU-R BS.1770 integrated target (feasible for all stimuli)
PEAK_CEILING_DBFS = -1.0     # sample-peak safety ceiling after gain; no limiting

OUT = os.path.join(ROOT, "configs/research/listening_study_inventory.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(p):
    d = json.load(open(p))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def by_index(manifest_path, key="prompt_index"):
    out = {}
    for r in rows(manifest_path):
        out[r[key]] = r
    return out


def measure(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    peak = float(np.max(np.abs(x))) if n else 0.0
    peak_dbfs = 20.0 * np.log10(peak + 1e-12)
    meter = pyln.Meter(sr)  # BS.1770-4
    with np.errstate(divide="ignore"):
        lufs = float(meter.integrated_loudness(x))
    finite = np.isfinite(lufs)
    gain_db = (LUFS_TARGET - lufs) if finite else None
    new_peak_dbfs = (peak_dbfs + gain_db) if finite else None
    peak_safe = (new_peak_dbfs is not None) and (new_peak_dbfs <= PEAK_CEILING_DBFS)
    return {
        "sr": int(sr), "n_samples": int(n), "duration_s": round(n / sr, 6),
        "sample_peak_dbfs": round(peak_dbfs, 4),
        "integrated_lufs": (round(lufs, 4) if finite else None),
        "lufs_finite": bool(finite),
        "gain_to_target_db": (round(gain_db, 4) if finite else None),
        "post_gain_peak_dbfs": (round(new_peak_dbfs, 4) if finite else None),
        "peak_safe_at_target": bool(peak_safe),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    subset = json.load(open("configs/research/op_duration_discriminator_1_subset.json"))["prompts"]
    elig2 = json.load(open("configs/research/finelap_eligibility_sev2.json"))

    # sev1 manifests
    v11_rec = by_index(os.path.join(V11, "gen_manifest_p1_recovered.json"))
    v11_pru = by_index(os.path.join(V11, "gen_manifest_p1_pruned_ema_reconstructed.json"))
    armd_rec = by_index(os.path.join(ARMD, "gen_manifest_p1_recovered.json"), key="subset_prompt_index")
    armd_pru = by_index(os.path.join(ARMD, "gen_manifest_p1_pruned_ema_reconstructed.json"), key="subset_prompt_index")
    # sev2 native manifests
    x_rec = by_index(os.path.join(XSEV, "gen_manifest_recovered2_ac_native.json"))
    x_pruA = by_index(os.path.join(XSEV, "gen_manifest_pruned2_A_ac_native.json"))

    stimuli = {}   # stim_id -> measurement + provenance
    prompts = {"sev1": [], "sev2": []}
    problems = []

    def add_stim(job_dir, row, expect_dur, sev, ytid, system, duration_tag):
        base = os.path.basename(row["wav"])
        path = os.path.join(job_dir, base)
        stim_id = f"{sev}|{system}|{duration_tag}|{ytid}"
        if not os.path.exists(path):
            problems.append(f"MISSING {stim_id} -> {path}")
            return None, None
        sha = sha256_file(path)
        man_sha = row.get("wav_sha256")
        if man_sha and man_sha != sha:
            problems.append(f"SHA_MISMATCH {stim_id}: disk {sha[:12]} vs manifest {man_sha[:12]}")
        m = measure(path)
        if abs(m["duration_s"] - expect_dur) > 0.05:
            problems.append(f"DUR {stim_id}: {m['duration_s']} != {expect_dur}")
        if not m["peak_safe_at_target"]:
            problems.append(f"PEAK_UNSAFE {stim_id}: post-gain peak {m['post_gain_peak_dbfs']} dBFS")
        stimuli[stim_id] = {
            "stim_id": stim_id, "severity": sev, "system": system,
            "duration_tag": duration_tag, "ytid": ytid,
            "src_path": path, "src_basename": base,
            "sha256": sha, "seed": row.get("seed"),
            **m,
        }
        return stim_id, row.get("seed")

    # ---- Severity 1: 80 Arm-D prompts x {3.84, 10.24} x {pruned, recovered} ----
    for p in subset:
        si = p["subset_prompt_index"]; vi = p["v1_1_prompt_index"]
        ytid = p["ytid"]; cap = p["caption"]
        rec_s, seed_rs = add_stim(V11, v11_rec[vi], 3.84, "sev1", ytid, "recovered", "short")
        pru_s, seed_ps = add_stim(V11, v11_pru[vi], 3.84, "sev1", ytid, "pruned", "short")
        rec_n, seed_rn = add_stim(ARMD, armd_rec[si], 10.24, "sev1", ytid, "recovered", "native")
        pru_n, seed_pn = add_stim(ARMD, armd_pru[si], 10.24, "sev1", ytid, "pruned", "native")
        # within-duration pairing check: recovered & pruned share x_T seed
        if seed_rs is not None and seed_rs != seed_ps:
            problems.append(f"SEED_PAIR sev1 short {ytid}: rec {seed_rs} != pru {seed_ps}")
        if seed_rn is not None and seed_rn != seed_pn:
            problems.append(f"SEED_PAIR sev1 native {ytid}: rec {seed_rn} != pru {seed_pn}")
        prompts["sev1"].append({
            "subset_prompt_index": si, "v1_1_prompt_index": vi, "ytid": ytid, "caption": cap,
            "stim": {"short": {"recovered": rec_s, "pruned": pru_s},
                      "native": {"recovered": rec_n, "pruned": pru_n}},
            "seed_short": seed_rs, "seed_native": seed_rn,
        })

    # ---- Severity 2: 110 eligible xsev prompts, native only, recovered2 vs pruned2_A ----
    for q in elig2["prompts"]:
        if not q.get("eligible"):
            continue
        ytid = q["ytid"]; pi = q["prompt_index"]; cap = q["caption"]
        rec_s, seed_r = add_stim(XSEV, x_rec[pi], 10.24, "sev2", ytid, "recovered", "native")
        pru_s, seed_p = add_stim(XSEV, x_pruA[pi], 10.24, "sev2", ytid, "pruned", "native")
        if seed_r is not None and seed_r != seed_p:
            problems.append(f"SEED_PAIR sev2 native {ytid}: rec {seed_r} != pru {seed_p}")
        prompts["sev2"].append({
            "prompt_index": pi, "ytid": ytid, "caption": cap,
            "n_eligible_events": q.get("n_eligible_events"),
            "stim": {"native": {"recovered": rec_s, "pruned": pru_s}},
            "seed_native": seed_r,
        })

    # loudness feasibility summary
    lufs_vals = [s["integrated_lufs"] for s in stimuli.values() if s["integrated_lufs"] is not None]
    unsafe = [s["stim_id"] for s in stimuli.values() if not s["peak_safe_at_target"]]
    nonfinite = [s["stim_id"] for s in stimuli.values() if not s["lufs_finite"]]

    out = {
        "artifact": "listening_study_inventory",
        "status": "FROZEN candidate stimulus inventory + BS.1770 loudness feasibility",
        "sources": {"armd_10s": ARMD, "v11_3p84s": V11, "xsev_native": XSEV},
        "normalization_design": {
            "target_lufs": LUFS_TARGET, "standard": "ITU-R BS.1770-4 (pyloudnorm Meter)",
            "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
            "rule": "listening copies only; single fixed gain per stimulus to reach target; "
                    "NO limiting, NO compression, NO condition-specific target; "
                    "if any study stimulus cannot reach target with post-gain sample peak <= ceiling -> STOP",
        },
        "counts": {
            "sev1_prompts": len(prompts["sev1"]),
            "sev1_stimuli": sum(1 for s in stimuli if s.startswith("sev1|")),
            "sev2_eligible_prompts": len(prompts["sev2"]),
            "sev2_stimuli": sum(1 for s in stimuli if s.startswith("sev2|")),
            "total_unique_stimuli": len(stimuli),
        },
        "loudness_feasibility": {
            "target_lufs": LUFS_TARGET, "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
            "lufs_min": (round(min(lufs_vals), 3) if lufs_vals else None),
            "lufs_max": (round(max(lufs_vals), 3) if lufs_vals else None),
            "n_nonfinite_lufs": len(nonfinite), "nonfinite_ids": nonfinite[:20],
            "n_peak_unsafe": len(unsafe), "peak_unsafe_ids": unsafe[:20],
            "feasible_all": (len(unsafe) == 0 and len(nonfinite) == 0),
        },
        "problems": problems,
        "prompts": prompts,
        "stimuli": stimuli,
    }
    payload = json.dumps(out, indent=2, sort_keys=True, default=str)
    out["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()

    if args.check:
        if not os.path.exists(OUT):
            print("CHECK FAIL: no committed inventory"); sys.exit(2)
        old = json.load(open(OUT))
        same = old.get("self_sha256") == out["self_sha256"]
        print("CHECK", "PASS" if same else "FAIL",
              "self_sha256", out["self_sha256"][:16], "vs", str(old.get("self_sha256"))[:16])
        sys.exit(0 if same else 2)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    print("WROTE", OUT)
    print("counts:", json.dumps(out["counts"]))
    print("loudness_feasibility:", json.dumps(out["loudness_feasibility"], indent=2))
    print("N problems:", len(problems))
    for pr in problems[:30]:
        print("  ", pr)
    print("self_sha256:", out["self_sha256"])


if __name__ == "__main__":
    main()
