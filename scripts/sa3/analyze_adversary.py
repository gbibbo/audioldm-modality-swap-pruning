#!/usr/bin/env python3
"""Analyze the single-block E adversary (protocol section 6.1, 9.2). RUN WITH `.venv-metrics`.

Reads the e_adversary manifest, scores every wav (CLAP / KL_passt / FD_openl3 vs the dense-8
stream-0 reference), computes the 8->7 margins with the R-stream resolution floor, and gives a
DIRECTIONAL single-block verdict per block g: is skip-g inferior to the latency-matched dense
comparator (dense-7, the nearest dense step below skip-g@8 latency)? Counts inferior blocks.
Point-estimate / pilot-directional -- the CI-based CASE-E decision is main-panel.

Run: OPENBLAS_CORETYPE=Haswell HF_HOME=... .venv-metrics/bin/python scripts/sa3/analyze_adversary.py \
        --manifest artifacts/sa3/adversary_manifest.json --out artifacts/sa3/adversary_analysis.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torchvision  # noqa: F401 -- torchvision-first (timm circular-import guard)


def pct(v, q=95): 
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-fd", action="store_true", help="skip OpenL3/FD (slow on CPU); CLAP+KL verdict only")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_e_metrics import Backends, frechet, _load
    B = Backends(a.device)
    prompts = man["prompts"]; systems = man["systems"]; ref = man["reference_system"]

    cache = {}
    def feats(fp):
        if fp not in cache:
            d = {"post": B.passt_post(_load(fp, 32000))}
            if not a.no_fd:
                d["ol3"] = B.ol3_embed(_load(fp, 48000))
            cache[fp] = d
        return cache[fp]

    # CLAP per system+prompt, KL per system+prompt (vs ref), FD per system (set vs ref set)
    aids = sorted(prompts, key=lambda x: int(x))
    ref_feat = {aid: feats(systems[ref][aid]) for aid in aids if aid in systems[ref]}
    ref_ol3 = None if a.no_fd else np.stack([ref_feat[aid]["ol3"] for aid in aids if aid in ref_feat])
    out = {"systems": {}, "margins": {}, "verdicts": {}, "manifest": os.path.basename(a.manifest)}
    for sid, files in systems.items():
        sa = [aid for aid in aids if aid in files]
        fps = [files[aid] for aid in sa]; caps = [prompts[aid] for aid in sa]
        ae = B.clap.get_audio_embedding_from_filelist(x=fps, use_tensor=False)
        te = B.clap.get_text_embedding(caps, use_tensor=False)
        l2 = lambda z: z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)
        clap_per = {aid: float(c) for aid, c in zip(sa, np.sum(l2(ae) * l2(te), axis=-1))}
        kl_per = {}
        for aid in sa:
            if aid in ref_feat:
                p = feats(files[aid])["post"] + 1e-12; q = ref_feat[aid]["post"] + 1e-12
                kl_per[aid] = float(np.sum(p * np.log(p / q)))
        if a.no_fd:
            fd = float("nan")
        else:
            emb = np.stack([feats(files[aid])["ol3"] for aid in sa])
            fd = 0.0 if sid == ref else frechet(emb, ref_ol3)
        out["systems"][sid] = {"CLAP_mean": float(np.mean(list(clap_per.values()))),
                               "KL_mean": float(np.mean(list(kl_per.values()))) if kl_per else float("nan"),
                               "FD": float(fd), "CLAP_per": clap_per, "KL_per": kl_per}

    S = out["systems"]
    R = man["R"]
    # resolution floors from dense-8 streams s1..s{R-1} vs s0
    r_clap, r_kl, r_fd = [], [], []
    for r in range(1, R):
        sid = f"dense8_s{r}"
        if sid in S:
            for aid in S[sid]["CLAP_per"]:
                if aid in S[ref]["CLAP_per"]:
                    r_clap.append(abs(S[sid]["CLAP_per"][aid] - S[ref]["CLAP_per"][aid]))
            r_kl += [abs(v) for v in S[sid]["KL_per"].values()]
            r_fd.append(abs(S[sid]["FD"]))
    r_CLAP, r_KL, r_FD = pct(r_clap), pct(r_kl), pct(r_fd)
    # 8->7 deterioration
    d8, d7 = S.get("dense8_s0"), S.get("dense7_s0")
    if d8 and d7:
        m_CLAP = max(0.0, d8["CLAP_mean"] - d7["CLAP_mean"]); m_KL = max(0.0, d7["KL_mean"]); m_FD = (float("nan") if a.no_fd else max(0.0, d7["FD"]))
        margins = {"m_CLAP": max(m_CLAP, r_CLAP), "m_KL": max(m_KL, r_KL), "m_FD": max(m_FD, r_FD),
                   "delta_CLAP": m_CLAP, "delta_KL": m_KL, "delta_FD": m_FD,
                   "r_CLAP": r_CLAP, "r_KL": r_KL, "r_FD": r_FD}
    else:
        margins = {"note": "need dense8_s0 and dense7_s0"}
    out["margins"] = margins

    # directional per-block verdict vs dense-7 comparator (nearest dense step below skip-g@8 latency)
    comp = "dense7_s0"
    if comp in S and "m_CLAP" in margins:
        cC, cK, cF = S[comp]["CLAP_mean"], S[comp]["KL_mean"], S[comp]["FD"]
        inferior = []
        for sid in S:
            if not sid.startswith("skip"):
                continue
            g = int(sid.replace("skip", ""))
            clap_def = cC - S[sid]["CLAP_mean"]            # >0 => skip worse than comparator on CLAP
            kl_def = S[sid]["KL_mean"] - cK; fd_def = S[sid]["FD"] - cF
            if a.no_fd:
                verdict = "inferior" if clap_def > margins["m_CLAP"] else "not_inferior_or_indeterminate"
            else:
                verdict = "inferior" if (clap_def > margins["m_CLAP"] or (kl_def > margins["m_KL"] and fd_def > margins["m_FD"])) else "not_inferior_or_indeterminate"
            out["verdicts"][sid] = {"block": g, "CLAP": S[sid]["CLAP_mean"], "KL": S[sid]["KL_mean"],
                                    "FD": S[sid]["FD"], "clap_deficit": clap_def, "kl_deficit": kl_def,
                                    "fd_deficit": fd_def, "directional_verdict": verdict}
            if verdict == "inferior":
                inferior.append(g)
        out["n_blocks"] = sum(1 for s in S if s.startswith("skip"))
        out["n_inferior_directional"] = len(inferior)
        out["inferior_blocks"] = sorted(inferior)
        out["case_e_direction"] = ("STRONG (all/most single-block removals inferior to dense-7)"
                                   if len(inferior) >= 0.8 * out["n_blocks"] else "MIXED")
    json.dump(out, open(a.out, "w"), indent=2)
    print("ADVERSARY_ANALYSIS_JSON_BEGIN"); print(json.dumps({k: v for k, v in out.items() if k != "systems"})); print("ADVERSARY_ANALYSIS_JSON_END")
    print(f"margins={out['margins']}")
    if "n_inferior_directional" in out:
        print(f"n_inferior/{out['n_blocks']}={out['n_inferior_directional']} inferior_blocks={out['inferior_blocks']} -> {out['case_e_direction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
