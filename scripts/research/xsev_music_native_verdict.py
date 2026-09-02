#!/usr/bin/env python3
"""XSEV-MUSIC-NATIVE-1 — emit frozen 64-item scorer groups and compute the frozen verdict (CPU, 0 cr).

Protocol: docs/xsev_music_native_1.md (frozen before any music@10.24 s output). Estimands (§6):
  PRIMARY   R_music_nat = C_rec(music,10.24) - C_prunedA(music,10.24)   paired per prompt (64), cluster CI
  secondary J_music     = R_music_nat - R_music_short  (frozen short = per-prompt mean of 3 replicates)
            D_nat       = R_nat(AudioCaps) - R_music_nat  (independent two-sample bootstrap)
            absolute means, win-rate; SESOI 0.025; NO gate (§7 branches are descriptive labels).

  --emit    : validate the 128 generated WAVs (count, sha256 vs generation manifests, n_samples, seeds)
              and write artifacts/icassp_gate0/_score_tmp/music_native_groups_in.json (2 x 64 items)
  --verdict : read the frozen-scorer output and write configs/research/xsev_music_native_1_result.json

Scoring between the stages (CPU):
  OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/gate0_clap_scorer.py --score-groups \
     artifacts/icassp_gate0/_score_tmp/music_native_groups_in.json artifacts/icassp_gate0/_score_tmp/music_native_groups_out.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np
from research_pruning.eval.reversal import derive_paired_seed

NS = "XSEV-MUSIC-NATIVE-1|BOOTSTRAP|2026-09-02"
SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000
SESOI = 0.025
GEN_SALT = "RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30"
MUSIC_MANIFEST = "configs/research/xsev_music_manifest.json"
TMP = "artifacts/icassp_gate0/_score_tmp"
GEN_ROOT = os.environ.get("MUSIC_NATIVE_ROOT", "artifacts/icassp_gate0/xsev_music_native_gen")
SEV2 = f"{TMP}/xsev_sev2_groups_out.json"
GROUPS_IN = f"{TMP}/music_native_groups_in.json"
GROUPS_OUT = f"{TMP}/music_native_groups_out.json"
OUT = "configs/research/xsev_music_native_1_result.json"
PROTO = "docs/xsev_music_native_1.md"
SYSTEMS = ("recovered2", "pruned2_A")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def emit():
    import soundfile as sf
    man = {p["prompt_index"]: p for p in json.load(open(MUSIC_MANIFEST))["prompts"]}
    assert set(man) == set(range(64))
    groups, prov = [], {}
    for sysn in SYSTEMS:
        gm = json.load(open(os.path.join(GEN_ROOT, f"gen_manifest_{sysn}_music_native.json")))
        rows = {r["prompt_index"]: r for r in gm["rows"]}
        if set(rows) != set(range(64)) or gm["n"] != 64:
            raise SystemExit(f"{sysn}: expected 64 rows, got {gm['n']}")
        items = []
        for i in range(64):
            r = rows[i]; wav = os.path.join(GEN_ROOT, os.path.basename(r["wav"]))
            if not os.path.exists(wav):
                raise SystemExit(f"missing {wav}")
            if sha(wav) != r["wav_sha256"]:
                raise SystemExit(f"sha mismatch {wav}")
            if r["n_samples"] != 163872 or r["latent_t"] != 256 or r["ddim"] != 50 or r["replicate_index"] != 0:
                raise SystemExit(f"recipe mismatch in row {i} of {sysn}: {r}")
            if r["ytid"] != man[i]["ytid"] or r["seed"] != man[i]["generation_seeds"][0] \
                    or r["seed"] != derive_paired_seed(GEN_SALT, r["ytid"], 0):
                raise SystemExit(f"seed/ytid mismatch in row {i} of {sysn}")
            w, sr = sf.read(wav, dtype="float32")
            if sr != 16000 or len(w) != 163872 or not np.isfinite(w).all():
                raise SystemExit(f"waveform check failed {wav}")
            items.append({"caption": man[i]["caption"], "wav": wav})
        groups.append({"name": f"{sysn}__music_native", "items": items})
        prov[sysn] = {"gen_manifest_sha256": sha(os.path.join(GEN_ROOT, f"gen_manifest_{sysn}_music_native.json")),
                      "device": gm["rows"][0]["device"], "checkpoint": gm["rows"][0]["checkpoint"],
                      "git_sha": gm["provenance"].get("git_sha")}
    json.dump({"groups": groups, "convention": "one 64-item seed-once call per system, prompt_index 0..63",
               "generation_provenance": prov}, open(GROUPS_IN, "w"), indent=1, ensure_ascii=False)
    print("STRUCTURAL 128/128 PASS; emitted", GROUPS_IN, json.dumps(prov, indent=1))


def pct(vals, stat, rng):
    n = len(next(iter(vals.values()))); boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n); boots[i] = stat({k: v[idx] for k, v in vals.items()})
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(stat(vals)), "lo": float(lo), "hi": float(hi), "n": int(n)}


def two_sample(a, b_, rng):
    boots = np.empty(B)
    for i in range(B):
        boots[i] = a[rng.integers(0, len(a), len(a))].mean() - b_[rng.integers(0, len(b_), len(b_))].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(a.mean() - b_.mean()), "lo": float(lo), "hi": float(hi), "n_a": int(len(a)), "n_b": int(len(b_))}


def verdict():
    rng = np.random.default_rng(np.random.PCG64(SEED))
    out = json.load(open(GROUPS_OUT)); c = {r["name"]: np.asarray(r["cosines"], float) for r in out["results"]}
    rec, pru = c["recovered2__music_native"], c["pruned2_A__music_native"]
    if len(rec) != 64 or len(pru) != 64:
        raise SystemExit("expected 64-item groups")
    g = json.load(open(SEV2)); s = {r["name"]: np.asarray(r["cosines"], float) for r in g["results"]}
    r_short = (s["recovered2__music"] - s["pruned2_A__music"]).reshape(64, 3).mean(1)
    r_nat_ac = s["recovered2__ac_native"] - s["pruned2_A__ac_native"]
    r_nat = rec - pru
    v = {"n": r_nat, "s": r_short}
    R = pct(v, lambda x: x["n"].mean(), rng)
    J = pct(v, lambda x: (x["n"] - x["s"]).mean(), rng)
    D = two_sample(r_nat_ac, r_nat, rng)
    if R["lo"] <= 0 <= R["hi"] and abs(R["point"]) < SESOI:
        branch = "a_domain_specific_native_gain"
    elif R["lo"] > 0 and R["point"] >= SESOI:
        branch = "b_generic_duration_gain_component"
    elif R["hi"] < 0:
        branch = "c_native_music_penalty"
    else:
        branch = "unresolved_at_n64"
    res = {"artifact": "xsev_music_native_1_result", "protocol_doc": PROTO, "protocol_doc_sha256": sha(PROTO),
           "class": "prospective follow-up (design completion); NO gate; changes no frozen verdict",
           "scorer": out.get("scorer_provenance"), "generation_provenance": json.load(open(GROUPS_IN))["generation_provenance"],
           "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": SEED, "unit": "prompt", "ci": "percentile 95%"},
           "inputs": {GROUPS_OUT: sha(GROUPS_OUT), SEV2: sha(SEV2)},
           "means": {"rec_music_native": float(rec.mean()), "pruned_music_native": float(pru.mean()),
                     "rec_music_short_frozen": float(s["recovered2__music"].mean()), "pruned_music_short_frozen": float(s["pruned2_A__music"].mean())},
           "PRIMARY_R_music_native": R, "win_rate_music_native": float((r_nat > 0).mean()),
           "secondary_J_music": J, "secondary_D_native_domain_contrast_AC_minus_music": D,
           "R_music_short_frozen_point": float(r_short.mean()),
           "branch": branch}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1)); print("XSEV-MUSIC-NATIVE-1 VERDICT ->", OUT, "branch", branch)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--emit", action="store_true"); ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.emit: emit()
    if a.verdict: verdict()
    if not (a.emit or a.verdict): print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
