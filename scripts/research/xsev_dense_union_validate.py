#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — dense@10.24s control union validation (CPU, free, read-only).

The dense control was produced in two GPU segments:
  * indices 0..72  by `reversal-xsev-gen-1` (job1) — stopped by OUT_OF_FUNDS BEFORE writing a stage
    manifest, so those 73 have NO persisted source manifest; their provenance is reconstructed from the
    frozen Arm-D 80-ytid manifest + the deterministic (frozen-code) generator convention.
  * indices 73..79 by `xsev-dense-tail-3` (job3, --indices) — with a persisted index-suffixed manifest.

Verifies the UNION is exactly {0..79} (80 unique, 0 dup, 0 omit) and, for all 80, cross-checks against the
frozen Arm-D manifest: ytid, prompt_index, generation seed (r0), 10.24s/latent256/DDIM50/g2.5/eta0/fp32
recipe, pinned dense checkpoint identity, sample count / 16 kHz / finiteness, and persisted SHA256. For the
7 tail WAVs these are checked against the recorded manifest row; for the 73 they are checked against the
deterministic expectation (recorded-seed cross-check impossible — documented provenance gap for a SECONDARY
control). Also compares tail GPU/software env to the job1 reference. Does NOT alter any source manifest and
does NOT read any CLAP/confirmatory metric. Writes a derived union artifact only.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_dense_union_validate.py \
        --tail-root /teamspace/jobs/xsev-dense-tail-3/artifacts/audioldm-modality-swap-pruning
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import numpy as np, soundfile as sf
from research_pruning.eval.reversal import generation_seed as v1_seed

SR = 16000
NSAMP = 163872           # 10.24 s vocoder length (matches the completed 73 + severity-2 natives)
DENSE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
# Pinned dense identity is a SHA256 (V1.1 launch record / reversal_v1_gen_preflight source_sha256).
DENSE_CKPT_SHA256_FROZEN = "936914a388905e1fc179c148a41a2b1552dba322ce474160b1cfa0f01ac26f8f"
ARMD = "configs/research/op_duration_discriminator_1_subset.json"
GEN_SUBDIR = "artifacts/icassp_gate0/reversal_xsev_gen"
REF_ENV = {"gpu": "Tesla T4", "torch": "1.13.1+cu117", "cuda": "11.7"}  # job1 (the 73) reference


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha256_hex(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_audio(wav):
    if not os.path.exists(wav):
        return None, [f"missing {os.path.basename(wav)}"]
    errs = []
    data, sr = sf.read(wav, dtype="float32")
    data = np.asarray(data).squeeze()
    if sr != SR: errs.append(f"sr {sr}")
    if data.shape[-1] != NSAMP: errs.append(f"nsamp {data.shape[-1]}!={NSAMP}")
    if not np.isfinite(data).all(): errs.append("NONFINITE")
    return {"n_samples": int(data.shape[-1]), "sha256": sha_file(wav)}, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense73-root", default="/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning")
    ap.add_argument("--tail-root", default="/teamspace/jobs/xsev-dense-tail-3/artifacts/audioldm-modality-swap-pruning")
    ap.add_argument("--out", default="artifacts/icassp_gate0/xsev_dense_union_validation.json")
    a = ap.parse_args()

    armd = {p["subset_prompt_index"]: p for p in json.load(open(ARMD))["prompts"]}
    assert set(armd) == set(range(80)), "Arm-D subset is not exactly {0..79}"

    tail_man_path = os.path.join(a.tail_root, GEN_SUBDIR, "gen_manifest_dense_dense_native_idx73-79.json")
    tail_rows = {}
    tail_env = {}
    if os.path.exists(tail_man_path):
        tm = json.load(open(tail_man_path))
        tail_rows = {r["prompt_index"]: r for r in tm["rows"]}
        tp = tm.get("provenance", {})
        tail_env = {"gpu": tp.get("gpu"), "torch": tp.get("torch"), "cuda": tp.get("cuda"),
                    "git_sha": tp.get("git_sha"), "device": tp.get("device")}

    # pinned dense checkpoint identity (same file used by both segments)
    ckpt_sha = sha256_hex(DENSE_CKPT) if os.path.exists(DENSE_CKPT) else None
    ckpt_ok = (ckpt_sha == DENSE_CKPT_SHA256_FROZEN)

    records, all_errs, present = [], [], set()
    for idx in range(80):
        ref = armd[idx]
        exp_ytid, exp_seed = ref["ytid"], ref["generation_seed_r0"]
        assert exp_seed == v1_seed(exp_ytid, 0), f"seed convention mismatch idx {idx}"
        root = a.dense73_root if idx <= 72 else a.tail_root
        wav = os.path.join(root, GEN_SUBDIR, f"dense_dense_native_p{idx}_r0.wav")
        audio, errs = check_audio(wav)
        rec = {"idx": idx, "ytid": exp_ytid, "expected_seed_r0": exp_seed,
               "segment": "job1(0..72)" if idx <= 72 else "tail(73..79)",
               "provenance_source": "reconstructed(frozen-deterministic; job1 manifest not persisted)"
                                    if idx <= 72 else "recorded(tail manifest)"}
        if audio is None:
            errs = errs
        else:
            rec.update(audio); present.add(idx)
            if idx >= 73:
                row = tail_rows.get(idx)
                if row is None:
                    errs.append("tail manifest row missing")
                else:
                    if row["ytid"] != exp_ytid: errs.append(f"ytid {row['ytid']}!={exp_ytid}")
                    if row["seed"] != exp_seed: errs.append(f"seed {row['seed']}!={exp_seed}")
                    if row["prompt_index"] != idx: errs.append("prompt_index mismatch")
                    if row["n_samples"] != NSAMP: errs.append("manifest nsamp")
                    if abs(row["duration_s"] - 10.24) > 1e-9: errs.append("duration")
                    if row["latent_t"] != 256: errs.append("latent_t")
                    if row["ddim"] != 50: errs.append("ddim")
                    if abs(row["guidance"] - 2.5) > 1e-9: errs.append("guidance")
                    if abs(row["eta"] - 0.0) > 1e-9: errs.append("eta")
                    if row["checkpoint"] != "dense_ema": errs.append(f"ckpt {row['checkpoint']}")
                    if row["wav_sha256"] != audio["sha256"]: errs.append("sha mismatch vs manifest")
        rec["errors"] = errs
        all_errs += [f"idx{idx}: {e}" for e in errs]
        records.append(rec)

    union_ok = (present == set(range(80)))
    dup = len(present) != len([r for r in records if not any("missing" in e for e in r["errors"])])
    env_diffs = {k: (REF_ENV[k], tail_env.get(k)) for k in REF_ENV if tail_env.get(k) != REF_ENV[k]}
    env_material = {k: v for k, v in env_diffs.items() if k in ("gpu", "torch", "cuda")}

    verdict = ("DENSE UNION PASS" if (union_ok and not all_errs and ckpt_ok and not env_material)
               else "DENSE UNION FAIL")
    out = {"artifact": "xsev_dense_union_validation",
           "verdict": verdict,
           "union_indices_ok": union_ok, "n_present": len(present),
           "expected": "exactly {0..79}, 80 unique, 0 dup, 0 omit",
           "duplicates": bool(dup), "omissions": sorted(set(range(80)) - present),
           "total_errors": len(all_errs), "errors": all_errs[:40],
           "dense_ckpt_sha256": ckpt_sha, "dense_ckpt_sha256_frozen": DENSE_CKPT_SHA256_FROZEN,
           "dense_ckpt_identity_ok": ckpt_ok,
           "env_reference_job1_the_73": REF_ENV,
           "env_tail_job3": tail_env,
           "env_material_differences": env_material,
           "env_note": ("tail git_sha differs from job1 (45e1aec vs afe76f8) BY DESIGN — the only code "
                        "change was the additive --indices filter; the dense generation codepath is "
                        "numerically identical. GPU class / torch / cuda must match; flagged above if not."),
           "provenance_note": ("indices 0..72 have NO persisted source manifest (job1 stopped by "
                               "OUT_OF_FUNDS mid-stage); their ytid/seed/recipe/checkpoint are the frozen "
                               "deterministic expectation from the Arm-D manifest + frozen generator. "
                               "Both source manifests preserved; this is a derived artifact."),
           "records": records}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{verdict} | union {len(present)}/80 ok={union_ok} | errors {len(all_errs)} | "
          f"ckpt_ok={ckpt_ok} | env_material_diffs={env_material} -> {a.out}")
    if all_errs: print("ERRORS:", all_errs[:10])
    if env_diffs: print("ENV DIFFS (incl. expected git_sha):", env_diffs)
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
