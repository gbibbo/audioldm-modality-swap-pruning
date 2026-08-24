#!/usr/bin/env python3
"""Primary scalar T_AA — paired held-out CLAP AUDIO-AUDIO similarity (rc1.4). RUN WITH
`.venv-metrics/bin/python` (CLAP lives there). NOT the text-audio drift scorer.

For each of the 8 eval_L clips j (paired with its own held-out prompt): generate y_j^S from that
prompt, and s_j(S) = cosine(CLAP_audio(y_j^S), CLAP_audio(x_j^eval)); T_AA(S) = mean_j s_j(S).
No all-pairs / nearest / centroid — the natural prompt↔eval-audio pairing (rc1.4). The generic domain
prompt has no audio reference and is EXCLUDED from T_AA.

Manifest JSON:
  {"eval_pairs": [{"eval_id":.., "ref_wav":..}, ... 8 ...],
   "configs": {config_id: {eval_id: gen_wav_path, ...}, ...},
   "pairs": [{"name":.., "with_L": config_id, "no_L": config_id}, ...]}
For each pair, Δ_j = s_j(with_L) - s_j(no_L); ΔT_AA = mean_j Δ_j; paired bootstrap 95% CI
(aeco_predict.paired_bootstrap_ci, frozen seed 20260824 / B=10000). Self-test: --selftest.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research_sa3.aeco_predict import paired_bootstrap_ci


def _clap():
    import laion_clap
    m = laion_clap.CLAP_Module(enable_fusion=False)
    m.load_ckpt(); m.eval()
    return m


def _l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def audio_emb(clap, files):
    e = clap.get_audio_embedding_from_filelist(x=list(files), use_tensor=False)
    return _l2(np.asarray(e))


def score(manifest: dict, clap=None) -> dict:
    clap = clap or _clap()
    pairs = manifest["eval_pairs"]
    eval_ids = [p["eval_id"] for p in pairs]
    ref_emb = audio_emb(clap, [p["ref_wav"] for p in pairs])          # (8, D)
    # per-config audio embeddings in the SAME eval order
    s = {}   # config -> array of s_j
    T = {}
    for cid, m in manifest["configs"].items():
        gen = audio_emb(clap, [m[e] for e in eval_ids])               # (8, D)
        s_j = np.sum(gen * ref_emb, axis=-1)                          # paired cosine (both L2)
        s[cid] = s_j
        T[cid] = float(np.mean(s_j))
    out = {"T_AA": T, "n_eval": len(eval_ids), "eval_ids": eval_ids, "deltas": {}}
    for pr in manifest["pairs"]:
        dj = (s[pr["with_L"]] - s[pr["no_L"]]).tolist()
        ci = paired_bootstrap_ci(dj)
        out["deltas"][pr["name"]] = {"dT_AA": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                                     "per_j": dj, "with_L": pr["with_L"], "no_L": pr["no_L"]}
    return out


def selftest() -> int:
    import soundfile as sf
    sc = os.environ.get("SCRATCH", "/tmp")
    d = os.path.join(sc, "sa3_taa_selftest"); os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0); srate = 44100
    n = 8
    man = {"eval_pairs": [], "configs": {"with_L": {}, "no_L": {}}, "pairs": [
        {"name": "uplift", "with_L": "with_L", "no_L": "no_L"}]}
    for j in range(n):
        t = np.linspace(0, 3, 3 * srate, endpoint=False)
        ref = 0.4 * np.sin(2 * np.pi * (300 + 40 * j) * t)                       # reference tone
        withL = ref.copy()                                                      # identical audio -> cosine ~1
        noL = 0.4 * rng.standard_normal(len(t))                                 # white noise -> dissimilar to a tone
        for name, x in (("ref", ref), ("with_L", withL), ("no_L", noL)):
            fp = os.path.join(d, f"{name}_{j}.wav")
            sf.write(fp, (x / (np.abs(x).max() + 1e-9) * 0.5).astype("float32"), srate)
            if name == "ref":
                man["eval_pairs"].append({"eval_id": str(j), "ref_wav": fp})
            else:
                man["configs"][name][str(j)] = fp
    r = score(man)
    d_up = r["deltas"]["uplift"]
    ok = (r["T_AA"]["with_L"] > r["T_AA"]["no_L"]) and (d_up["dT_AA"] > 0) and (d_up["lo"] > 0)
    print(json.dumps({"T_with_L": round(r["T_AA"]["with_L"], 4), "T_no_L": round(r["T_AA"]["no_L"], 4),
                      "dT_AA": round(d_up["dT_AA"], 4), "ci": [round(d_up["lo"], 4), round(d_up["hi"], 4)],
                      "n": r["n_eval"]}, indent=2))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    man = json.load(open(a.manifest))
    r = score(man)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=2)
    print(json.dumps({"T_AA": {k: round(v, 4) for k, v in r["T_AA"].items()},
                      "deltas": {k: {"dT_AA": round(v["dT_AA"], 4), "lo": round(v["lo"], 4),
                                     "hi": round(v["hi"], 4)} for k, v in r["deltas"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
