#!/usr/bin/env python3
"""DRAFT5-OPSWEEP-1 / DRAFT5-PUBRECIPE-1 — validation, scoring and verdict (CPU only, 0 cr).

Protocol: docs/draft5_opsweep.md (frozen + sha256 sidecar BEFORE any generation). This script does not
generate anything; it validates the WAVs a T4 job produced, scores them under the frozen fused-CLAP
convention and applies the PRE-SPECIFIED decision rules of that protocol. It cannot change any frozen
verdict.

  --emit    validate the WAVs against their generation manifest (sha256, ytid, sample count, one
            device) and write the scorer groups, including the device-consistency pair.
  --score   frozen fused-CLAP convention (rev 365dea6e), one seed-once call per group, matched
            cosines + shuffled-caption floor from the same embeddings.
  --verdict paired quantities, unit = prompt, percentile bootstrap B = 10000, seed namespace from the
            protocol; applies the pre-specified shape rule (sweep) or gate (pubrecipe).

Run (CPU):
  OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_opsweep_verdict.py --exp sweep --emit --root <job artifact dir>
  ... --exp sweep --score ; ... --exp sweep --verdict
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np

AC192 = "configs/research/xsev_audiocaps_manifest.json"
PROTOCOL = "docs/draft5_opsweep.md"
TMP = "artifacts/icassp_gate0/_score_tmp"
FROZEN_IN = f"{TMP}/xsev_sev2_groups_in.json"
FROZEN_OUT = f"{TMP}/xsev_sev2_groups_out.json"
B = 10000
SESOI = 0.025

# experiment -> (systems, [(context, tag, expected_frames, duration_s)], out json, seed namespace, n_prompts)
EXP = {
    "sweep": (["dense", "pruned2_A", "recovered2"],
              [("ac_d128", "", 81952, 5.12), ("ac_d192", "", 122912, 7.68)],
              "configs/research/draft5_opsweep_result.json",
              "DRAFT5-OPSWEEP-1|BOOTSTRAP|2026-09-03", 192),
    "pubrecipe": (["pruned2_A", "recovered2"],
                  [("ac_short", "_pub", 61472, 3.84), ("ac_native", "_pub", 163872, 10.24)],
                  "configs/research/draft5_pubrecipe_result.json",
                  "DRAFT5-PUBRECIPE-1|BOOTSTRAP|2026-09-03", 64),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def prompts(n):
    ps = sorted(json.load(open(AC192))["prompts"], key=lambda p: p["prompt_index"])
    return ps[:n]


def gin(exp):
    return f"{TMP}/draft5_{exp}_groups_in.json"


def gout(exp):
    return f"{TMP}/draft5_{exp}_groups_out.json"


# ----------------------------------------------------------------------------------------------- emit
def emit(exp, root):
    import soundfile as sf
    systems, cells, _out, _ns, npr = EXP[exp]
    ps = prompts(npr)
    groups, prov = [], {}
    for ctx, tag, frames, dur in cells:
        for sysname in systems:
            man_p = f"{root}/gen_manifest_{sysname}_{ctx}{tag}.json"
            if not os.path.exists(man_p):
                raise SystemExit(f"missing generation manifest {man_p}")
            man = json.load(open(man_p))
            rows = {r["prompt_index"]: r for r in man["rows"]}
            if len(rows) != npr:
                raise SystemExit(f"{man_p}: {len(rows)} rows, expected {npr}")
            items = []
            for p in ps:
                w = f"{root}/{sysname}_{ctx}{tag}_p{p['prompt_index']}_r0.wav"
                if not os.path.exists(w):
                    raise SystemExit(f"missing WAV {w}")
                info = sf.info(w)
                if info.frames != frames or info.samplerate != 16000:
                    raise SystemExit(f"bad WAV {w}: frames {info.frames} (expected {frames}) sr {info.samplerate}")
                r = rows[p["prompt_index"]]
                if r["ytid"] != p["ytid"] or sha(w) != r["wav_sha256"]:
                    raise SystemExit(f"manifest mismatch for {w}")
                items.append({"caption": p["caption"], "wav": w})
            devices = sorted({r["device"] for r in man["rows"]})
            if len(devices) != 1 or not devices[0].startswith("cuda"):
                raise SystemExit(f"device rule violated for {sysname}/{ctx}: {devices}")
            prov[f"{sysname}__{ctx}{tag}"] = {
                "gen_manifest_sha256": sha(man_p), "n": man["n"], "devices": devices,
                "git_sha": man["provenance"].get("git_sha") or man["provenance"].get("commit"),
                "checkpoint": man["provenance"].get("checkpoint_convention"), "recipe": man["recipe"]}
            groups.append({"name": f"{sysname}__{ctx}{tag}", "items": items})
    # device-consistency pair: 4 frozen-recipe pruned2_A native clips regenerated in THIS job
    dc_root = f"{root}/device_check"
    fro = {g["name"]: g for g in json.load(open(FROZEN_IN))["groups"]}["pruned2_A__ac_native"]["items"]
    dc = [(f"{dc_root}/pruned2_A_ac_native_p{i}_r0.wav", fro[i]["wav"], prompts(4)[i]["caption"])
          for i in range(4) if os.path.exists(f"{dc_root}/pruned2_A_ac_native_p{i}_r0.wav")]
    if dc:
        groups.append({"name": "device_check__new", "items": [{"caption": c, "wav": w} for w, _f, c in dc]})
        groups.append({"name": "device_check__frozen", "items": [{"caption": c, "wav": f} for _w, f, c in dc]})
        prov["device_check_sha_equal"] = [sha(w) == sha(f) for w, f, _c in dc]
    json.dump({"groups": groups, "generation_provenance": prov, "protocol_doc_sha256": sha(PROTOCOL),
               "experiment": exp,
               "convention": "each group = ONE seed-once fused-CLAP call in prompt_index order "
                             "(rev 365dea6e); floors are shuffled captions from the same embeddings"},
              open(gin(exp), "w"), indent=1)
    print(f"emitted {len(groups)} groups -> {gin(exp)}")
    if dc:
        print("device check sha256 equal to frozen:", prov["device_check_sha_equal"])


# ---------------------------------------------------------------------------------------------- score
def score(exp):
    from gate0_clap_scorer import FusedClapScorer, _prov
    din = json.load(open(gin(exp)))
    sc = FusedClapScorer()
    results = []
    for g in din["groups"]:
        t0 = time.time()
        caps = [it["caption"] for it in g["items"]]
        wavs = [it["wav"] for it in g["items"]]
        te = sc._l2(sc.text_embed(caps)); ae = sc._l2(sc.audio_embed(wavs))
        M = te @ ae.T
        diag = np.diag(M).astype(float)
        ca = np.asarray(caps); other = ca[:, None] != ca[None, :]
        floor = np.array([M[other[:, j], j].mean() if other[:, j].any() else np.nan
                          for j in range(len(caps))], dtype=float)
        results.append({"name": g["name"], "n": len(caps), "cosines": diag.tolist(),
                        "floor_item": floor.tolist(), "matched_mean": float(diag.mean()),
                        "floor_mean": float(np.nanmean(floor))})
        print(f"{g['name']:34s} n={len(caps):3d} matched {diag.mean():.4f} floor {np.nanmean(floor):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    json.dump({"results": results, "scorer_provenance": _prov(), "groups_in_sha256": sha(gin(exp))},
              open(gout(exp), "w"), indent=1)
    print("scored ->", gout(exp))


# -------------------------------------------------------------------------------------------- verdict
class Boot:
    def __init__(self, ns, n):
        seed = int(hashlib.sha256(ns.encode()).hexdigest()[:8], 16) % (2 ** 31)
        self.idx = np.random.default_rng(seed).integers(0, n, (B, n))
        self.n, self.seed = n, seed

    def ci(self, v):
        v = np.asarray(v, float); bm = v[self.idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(self.n)}


def verdict(exp):
    systems, cells, outp, ns, npr = EXP[exp]
    dout = json.load(open(gout(exp)))
    din = json.load(open(gin(exp)))
    new = {r["name"]: r for r in dout["results"]}
    fro = {r["name"]: r for r in json.load(open(FROZEN_OUT))["results"]}
    bt = Boot(ns, npr)

    def cos(name, frozen=False, limit=None):
        r = (fro if frozen else new)[name]
        v = np.asarray(r["cosines"], float)
        return v[:limit] if limit else v

    out = {"artifact": f"draft5_{exp}_result", "experiment": exp,
           "protocol_doc_sha256": sha(PROTOCOL), "status": "PRE-SPECIFIED (docs/draft5_opsweep.md)",
           "bootstrap": {"B": B, "seed_namespace": ns, "seed_pcg64": bt.seed, "unit": "prompt", "n": npr},
           "scorer": dout["scorer_provenance"], "generation_provenance": din["generation_provenance"]}

    # device-consistency check
    if "device_check__new" in new:
        d = np.asarray(new["device_check__new"]["cosines"]) - np.asarray(new["device_check__frozen"]["cosines"])
        out["device_check"] = {"max_abs_delta_clap": float(np.abs(d).max()),
                               "sha256_equal": din["generation_provenance"].get("device_check_sha_equal"),
                               "PASS": bool(np.abs(d).max() < 1e-9)}

    lim = npr if exp == "pubrecipe" else None
    if exp == "sweep":
        R = {3.84: cos("recovered2__ac_short", True) - cos("pruned2_A__ac_short", True),
             5.12: cos("recovered2__ac_d128") - cos("pruned2_A__ac_d128"),
             7.68: cos("recovered2__ac_d192") - cos("pruned2_A__ac_d192"),
             10.24: cos("recovered2__ac_native", True) - cos("pruned2_A__ac_native", True)}
        out["R_by_duration"] = {str(d): bt.ci(v) for d, v in R.items()}
        out["levels"] = {f"{s}@{d}": float(np.mean(
            cos(f"{s}__{'ac_short' if d == 3.84 else 'ac_native' if d == 10.24 else 'ac_d128' if d == 5.12 else 'ac_d192'}",
                frozen=d in (3.84, 10.24)))) for d in R for s in ("pruned2_A", "recovered2")}
        steps = {"D1": R[5.12] - R[3.84], "D2": R[7.68] - R[5.12], "D3": R[10.24] - R[7.68]}
        out["steps"] = {k: bt.ci(v) for k, v in steps.items()}
        S = out["steps"]
        pts = [S[k]["point"] for k in ("D1", "D2", "D3")]
        if all(p > 0 for p in pts) and all(S[k]["hi"] >= 0 for k in ("D1", "D2", "D3")):
            shape = "MONOTONE-INCREASING"
        elif S["D3"]["hi"] < 0:
            shape = "PEAKED-BEFORE-NATIVE"
        elif (S["D1"]["lo"] > 0 and S["D2"]["lo"] > 0 and S["D3"]["lo"] <= 0 <= S["D3"]["hi"]
              and abs(S["D3"]["point"]) < SESOI):
            shape = "SATURATING"
        else:
            shape = "UNRESOLVED"
        out["SHAPE_VERDICT"] = shape
        out["shape_rule"] = ("docs/draft5_opsweep.md section 2, declared before scoring; SESOI "
                             f"{SESOI}")
    else:
        Rs = cos("recovered2__ac_short_pub") - cos("pruned2_A__ac_short_pub")
        Rn = cos("recovered2__ac_native_pub") - cos("pruned2_A__ac_native_pub")
        J = Rn - Rs
        out["R_pub_short"], out["R_pub_native"], out["J_pub"] = bt.ci(Rs), bt.ci(Rn), bt.ci(J)
        Jf = ((cos("recovered2__ac_native", True, lim) - cos("pruned2_A__ac_native", True, lim))
              - (cos("recovered2__ac_short", True, lim) - cos("pruned2_A__ac_short", True, lim)))
        out["J_frozen_same64"] = bt.ci(Jf)
        out["J_pub_minus_J_frozen"] = bt.ci(J - Jf)
        out["J_pub_minus_J_frozen_note"] = ("descriptive: the two recipes are prompt-paired but not "
                                            "noise-paired")
        out["GATE_lo95_J_pub_gt_0"] = bool(out["J_pub"]["lo"] > 0)
        out["gate_rule"] = "docs/draft5_opsweep.md section 3, declared before scoring"

    txt = json.dumps(out, indent=1, sort_keys=True)
    out["artifact_sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in
                      ("scorer", "generation_provenance", "levels")}, indent=1))
    print("wrote", outp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXP))
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--root", default="")
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    if a.emit:
        if not a.root:
            raise SystemExit("--emit needs --root <job artifact dir>")
        emit(a.exp, a.root.rstrip("/"))
    if a.score:
        score(a.exp)
    if a.verdict:
        verdict(a.exp)
    if not (a.emit or a.score or a.verdict):
        raise SystemExit("pick --emit / --score / --verdict")


if __name__ == "__main__":
    main()
