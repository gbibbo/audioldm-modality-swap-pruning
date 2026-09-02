#!/usr/bin/env python3
"""NATIVE-CROP-ANALYSIS — generation length vs scoring window (CPU, 0 cr, no new generation).

Authorised by Gabriel 2026-09-02 (review item B3). POST-HOC DIAGNOSTIC; changes no frozen verdict.

Question: the short operating point differs from the native one in two ways at once — the model
GENERATES 3.84 s, and the scorer SEES 3.84 s (repeat-padded to 10 s by fused CLAP). Which one carries
the interaction? Test: take the EXISTING native (10.24 s) generations, keep only their first 3.84 s
(exactly the short clip's 61472 samples @16 kHz), score them with the identical frozen scorer
convention, and compare the post-FT - pruned contrast on crops (R_crop) with the frozen R_short
(separately generated 3.84 s clips, same prompts) and R_native.

  R_crop ~ R_native >> R_short  => generation-length effect (what is generated differs)
  R_crop ~ R_short              => scoring-window effect (what is scored differs)

Two stages (so the scorer runs in .venv-metrics exactly as for every frozen score):
  --emit   : write crops + scorer group files (sev-1: 2 x 80 items; sev-2: 3 x 192 items; canonical order)
  --verdict: read the scorer output, compute R_crop, R_crop - R_short (paired), R_native - R_crop (paired)
             with the prompt bootstrap (B=10000, seed namespace "NATIVE-CROP-ANALYSIS|BOOTSTRAP|2026-09-02").

Scoring (between the two stages, CPU):
  OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/gate0_clap_scorer.py \
      --score-groups artifacts/icassp_gate0/_score_tmp/native_crop_groups_in.json \
                     artifacts/icassp_gate0/_score_tmp/native_crop_groups_out.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np

NS = "NATIVE-CROP-ANALYSIS|BOOTSTRAP|2026-09-02"
SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000
SR = 16000
CROP = 61472            # = n_samples of every frozen 3.84 s generation (short + music), exact match
TMP = "artifacts/icassp_gate0/_score_tmp"
CROP_DIR = "artifacts/icassp_gate0/native_crops"
XSEV_ROOT = "/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_xsev_gen"
ARMD_ROOT = "/teamspace/jobs/reversal-armd-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_armd_gen"
AC_MANIFEST = "configs/research/xsev_audiocaps_manifest.json"
ARMD = "configs/research/op_duration_discriminator_1_subset.json"
OPD = "configs/research/op_duration_discriminator_1_result.json"
SEV2 = f"{TMP}/xsev_sev2_groups_out.json"
GROUPS_IN = f"{TMP}/native_crop_groups_in.json"
GROUPS_OUT = f"{TMP}/native_crop_groups_out.json"
OUT = "configs/research/native_crop_analysis_result.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def crop_wav(src, dst):
    import soundfile as sf
    w, sr = sf.read(src, dtype="float32")
    if sr != SR:
        raise SystemExit(f"{src}: sr {sr} != {SR}")
    if w.ndim != 1 or len(w) < CROP:
        raise SystemExit(f"{src}: shape {w.shape} too short for a {CROP}-sample crop")
    sf.write(dst, w[:CROP], SR, subtype="PCM_16")


def emit():
    os.makedirs(CROP_DIR, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    groups = []
    # severity 1 (Arm-D 80, native WAVs, canonical subset_prompt_index order)
    armd = {p["subset_prompt_index"]: p for p in json.load(open(ARMD))["prompts"]}
    assert set(armd) == set(range(80))
    for name, prefix in (("crop_sev1__pruned", "p1_pruned_ema_reconstructed_noadapter_alt10s"),
                         ("crop_sev1__recovered", "p1_recovered_noadapter_alt10s")):
        items = []
        for i in range(80):
            src = os.path.join(ARMD_ROOT, f"{prefix}_p{i}_r0.wav"); dst = os.path.join(CROP_DIR, f"{name}_p{i}.wav")
            crop_wav(src, dst); items.append({"caption": armd[i]["caption"], "wav": dst, "src": src, "src_sha256": sha(src)})
        groups.append({"name": name, "items": items})
    # severity 2 (xsev 192, native WAVs, canonical prompt_index order)
    ac = {p["prompt_index"]: p for p in json.load(open(AC_MANIFEST))["prompts"]}
    assert set(ac) == set(range(192))
    for sysn in ("recovered2", "pruned2_A", "pruned2_B"):
        name = f"crop_sev2__{sysn}"; items = []
        for i in range(192):
            src = os.path.join(XSEV_ROOT, f"{sysn}_ac_native_p{i}_r0.wav"); dst = os.path.join(CROP_DIR, f"{name}_p{i}.wav")
            crop_wav(src, dst); items.append({"caption": ac[i]["caption"], "wav": dst, "src": src, "src_sha256": sha(src)})
        groups.append({"name": name, "items": items})
    json.dump({"groups": groups, "crop_samples": CROP, "sr": SR,
               "convention": "one seed-once scorer call per group (80 items sev-1 / 192 items sev-2), canonical order — identical to the frozen scoring of the short and native groups"},
              open(GROUPS_IN, "w"), indent=1, ensure_ascii=False)
    print("emitted", [(g["name"], len(g["items"])) for g in groups], "->", GROUPS_IN)


def pct(vals, stat, rng):
    n = len(next(iter(vals.values()))); boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n); boots[i] = stat({k: v[idx] for k, v in vals.items()})
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(stat(vals)), "lo": float(lo), "hi": float(hi), "n": int(n)}


def verdict():
    rng = np.random.default_rng(np.random.PCG64(SEED))
    out = json.load(open(GROUPS_OUT)); cr = {r["name"]: np.asarray(r["cosines"], float) for r in out["results"]}
    res = {"artifact": "native_crop_analysis_result", "class": "POST-HOC DIAGNOSTIC (authorised 2026-09-02); changes no frozen verdict",
           "crop": {"samples": CROP, "seconds": CROP / SR, "definition": "first 3.84 s of each frozen 10.24 s native generation"},
           "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": SEED, "unit": "prompt", "ci": "percentile 95%"},
           "inputs": {GROUPS_OUT: sha(GROUPS_OUT), OPD: sha(OPD), SEV2: sha(SEV2)}, "severities": {}}
    # severity 1
    opd = json.load(open(OPD))["raw_cosines"]; o = {k: np.asarray(v, float) for k, v in opd.items()}
    r_short = o["recovered_ctrl"] - o["pruned_ctrl"]; r_nat = o["recovered_alt"] - o["pruned_alt"]
    r_crop = cr["crop_sev1__recovered"] - cr["crop_sev1__pruned"]
    v = {"s": r_short, "n": r_nat, "c": r_crop}
    res["severities"]["sev1_armd80"] = {
        "means": {"crop_pruned": float(cr["crop_sev1__pruned"].mean()), "crop_postft": float(cr["crop_sev1__recovered"].mean()),
                  "short_pruned": float(o["pruned_ctrl"].mean()), "short_postft": float(o["recovered_ctrl"].mean()),
                  "native_pruned": float(o["pruned_alt"].mean()), "native_postft": float(o["recovered_alt"].mean())},
        "R_short_frozen": pct(v, lambda x: x["s"].mean(), rng), "R_native_frozen": pct(v, lambda x: x["n"].mean(), rng),
        "R_crop": pct(v, lambda x: x["c"].mean(), rng),
        "R_crop_minus_R_short": pct(v, lambda x: (x["c"] - x["s"]).mean(), rng),
        "R_native_minus_R_crop": pct(v, lambda x: (x["n"] - x["c"]).mean(), rng),
        "win_rate_crop": float((r_crop > 0).mean())}
    # severity 2 (A' primary, B' sensitivity)
    g = json.load(open(SEV2)); c = {r["name"]: np.asarray(r["cosines"], float) for r in g["results"]}
    for prune in ("pruned2_A", "pruned2_B"):
        r_short = c["recovered2__ac_short"] - c[f"{prune}__ac_short"]; r_nat = c["recovered2__ac_native"] - c[f"{prune}__ac_native"]
        r_crop = cr["crop_sev2__recovered2"] - cr[f"crop_sev2__{prune}"]
        v = {"s": r_short, "n": r_nat, "c": r_crop}
        res["severities"][f"sev2_xsev192_{prune}"] = {
            "means": {"crop_pruned": float(cr[f"crop_sev2__{prune}"].mean()), "crop_postft": float(cr["crop_sev2__recovered2"].mean()),
                      "short_pruned": float(c[f"{prune}__ac_short"].mean()), "short_postft": float(c["recovered2__ac_short"].mean()),
                      "native_pruned": float(c[f"{prune}__ac_native"].mean()), "native_postft": float(c["recovered2__ac_native"].mean())},
            "R_short_frozen": pct(v, lambda x: x["s"].mean(), rng), "R_native_frozen": pct(v, lambda x: x["n"].mean(), rng),
            "R_crop": pct(v, lambda x: x["c"].mean(), rng),
            "R_crop_minus_R_short": pct(v, lambda x: (x["c"] - x["s"]).mean(), rng),
            "R_native_minus_R_crop": pct(v, lambda x: (x["n"] - x["c"]).mean(), rng),
            "win_rate_crop": float((r_crop > 0).mean())}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1)); print("NATIVE-CROP-ANALYSIS PASS ->", OUT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--emit", action="store_true"); ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.emit: emit()
    if a.verdict: verdict()
    if not (a.emit or a.verdict): print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
