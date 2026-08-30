#!/usr/bin/env python3
"""OP-DURATION-DISCRIMINATOR-1 (Arm D) — matched CLAP rescoring + primary interaction (.venv-metrics).

D1: score FOUR matched 80-item groups, each ONE seed-once (np.random.seed(20260826)) FusedClapScorer
call, identical ytid (subset) ordering, pinned rev 365dea6e, truncation='fusion' (the ~10 s crop, D2):
  CONTROL 3.84 s: pruned V1.1 r0 (80) | recovered V1.1 r0 (80)   [existing WAVs]
  ALT    10.24 s: pruned  new  r0 (80) | recovered new  r0 (80)   [new WAVs]

G primary interaction (paired at ytid, bootstrap PCG64(20260830), B=10000):
  r_ctrl_i = CLAP(rec_ctrl_i)-CLAP(pru_ctrl_i);  r_alt_i = CLAP(rec_alt_i)-CLAP(pru_alt_i)
  j_i = r_alt_i - r_ctrl_i;  J_CLAP = mean_i j_i
Secondary (I): Human-CLAP interaction (same matched design), corroborative. NOTHING here changes V1.1.

Run: OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/armd_clap_analysis.py \
        --alt-root <armd gen dir> --ctrl-root <V1.1 gen dir> --out configs/research/op_duration_discriminator_1_result.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np

SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
PROTOCOL = "docs/op_duration_discriminator_1.md"
BOOT_SEED = 20260830
B = 10000
SESOI = 0.025
CTRL_PREFIX = {"pruned": "p1_pruned_ema_reconstructed_noadapter",
               "recovered": "p1_recovered_noadapter"}
ALT_PREFIX = {"pruned": "p1_pruned_ema_reconstructed_noadapter_alt10s",
              "recovered": "p1_recovered_noadapter_alt10s"}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score_group(scorer, caps, wavs):
    for w in wavs:
        if not os.path.exists(w):
            raise SystemExit(f"missing WAV: {w}")
    cos = np.asarray(scorer.cosine(caps, wavs), dtype=np.float64)   # seeds 20260826 once per call
    if cos.size != len(wavs):
        raise SystemExit(f"{cos.size} scores != {len(wavs)}")
    return cos


def analyse(scorer_name, cos, idx):
    """cos: dict with pruned_ctrl/recovered_ctrl/pruned_alt/recovered_alt (each [80]). idx: bootstrap indices."""
    r_ctrl = cos["recovered_ctrl"] - cos["pruned_ctrl"]
    r_alt = cos["recovered_alt"] - cos["pruned_alt"]
    j = r_alt - r_ctrl
    Rc, Ra, J = r_ctrl[idx].mean(1), r_alt[idx].mean(1), j[idx].mean(1)
    out = {
        "R_ctrl_80": {"point": float(r_ctrl.mean()), "ci95": ci(Rc)},
        "R_alt": {"point": float(r_alt.mean()), "ci95": ci(Ra)},
        "J": {"point": float(j.mean()), "ci95": ci(J)},
        "frac_ytid_j_pos": float(np.mean(j > 0)),
        "means": {k: float(cos[k].mean()) for k in cos},
    }
    lo_J = out["J"]["ci95"][0]; hi_J = out["J"]["ci95"][1]
    out["duration_interaction_supported"] = bool(out["J"]["point"] > 0 and lo_J > 0)
    out["interaction_negative"] = bool(hi_J < 0)
    out["material_recovered_advantage_10s"] = bool(out["R_alt"]["point"] >= SESOI and out["R_alt"]["ci95"][0] > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-root", required=True)
    ap.add_argument("--ctrl-root", required=True)
    ap.add_argument("--out", default="configs/research/op_duration_discriminator_1_result.json")
    ap.add_argument("--with-humanclap", action="store_true", default=True)
    args = ap.parse_args()

    prompts = sorted(json.load(open(SUBSET))["prompts"], key=lambda p: p["subset_prompt_index"])
    caps = [p["caption"] for p in prompts]
    ctrl_wav = lambda sysk: [os.path.join(args.ctrl_root, f"{CTRL_PREFIX[sysk]}_p{p['v1_1_prompt_index']}_r0.wav") for p in prompts]
    alt_wav = lambda sysk: [os.path.join(args.alt_root, f"{ALT_PREFIX[sysk]}_p{p['subset_prompt_index']}_r0.wav") for p in prompts]

    from gate0_clap_scorer import FusedClapScorer, REVISION, MODEL_ID
    sc = FusedClapScorer(device="cpu")
    clap = {
        "pruned_ctrl": score_group(sc, caps, ctrl_wav("pruned")),
        "recovered_ctrl": score_group(sc, caps, ctrl_wav("recovered")),
        "pruned_alt": score_group(sc, caps, alt_wav("pruned")),
        "recovered_alt": score_group(sc, caps, alt_wav("recovered")),
    }
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(prompts), size=(B, len(prompts)))
    primary = analyse("clap", clap, idx)

    payload = {
        "artifact": "op_duration_discriminator_1_result",
        "status": "PROSPECTIVELY-SPECIFIED POST-V1.1 FOLLOW-UP — cannot change V1.1 PASS=FALSE",
        "protocol_doc_sha256": sha_file(PROTOCOL),
        "subset_sha256": json.load(open(SUBSET)).get("subset_sha256"),
        "scorer": {"model": MODEL_ID, "revision": REVISION, "seed_per_group": 20260826,
                   "truncation": "fusion (~10 s crop; D2 documented)", "groups": "4 x 80 matched"},
        "bootstrap": {"unit": "ytid", "n": len(prompts), "B": B, "seed_pcg64": BOOT_SEED},
        "PRIMARY_clap": primary,
        "raw_cosines": {k: [float(x) for x in clap[k]] for k in clap},
    }

    if args.with_humanclap:
        try:
            from reversal_humanclap import HumanClapScorer
            hc = HumanClapScorer()
            hcv = {
                "pruned_ctrl": np.asarray(hc.cosine(caps, ctrl_wav("pruned")), dtype=np.float64),
                "recovered_ctrl": np.asarray(hc.cosine(caps, ctrl_wav("recovered")), dtype=np.float64),
                "pruned_alt": np.asarray(hc.cosine(caps, alt_wav("pruned")), dtype=np.float64),
                "recovered_alt": np.asarray(hc.cosine(caps, alt_wav("recovered")), dtype=np.float64),
            }
            payload["SECONDARY_humanclap"] = analyse("hc", hcv, idx)
            payload["SECONDARY_humanclap"]["status"] = "corroborative, CLAP-family, NOT human eval; no primary role"
        except Exception as e:
            payload["SECONDARY_humanclap"] = {"error": str(e)}

    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "raw_cosines"},
                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    json.dump(payload, open(args.out, "w"), indent=2, ensure_ascii=False)

    p = primary
    print("=== ARM D PRIMARY (CLAP) ===")
    print("means:", {k: round(v, 4) for k, v in p["means"].items()})
    print(f"R_ctrl_80 = {p['R_ctrl_80']['point']:+.4f} {p['R_ctrl_80']['ci95']}")
    print(f"R_alt     = {p['R_alt']['point']:+.4f} {p['R_alt']['ci95']}")
    print(f"J_CLAP    = {p['J']['point']:+.4f} {p['J']['ci95']}  frac j>0 = {p['frac_ytid_j_pos']:.3f}")
    print(f"duration_interaction_supported = {p['duration_interaction_supported']}")
    print(f"material_recovered_advantage_10s = {p['material_recovered_advantage_10s']}")
    if "SECONDARY_humanclap" in payload and "J" in payload["SECONDARY_humanclap"]:
        h = payload["SECONDARY_humanclap"]
        print(f"[HC] J = {h['J']['point']:+.4f} {h['J']['ci95']}  R_alt {h['R_alt']['point']:+.4f}")
    print("wrote", args.out, "sha", payload["artifact_sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
