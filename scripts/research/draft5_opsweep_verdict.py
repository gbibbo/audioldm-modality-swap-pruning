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
  --secondary (sweep only) the protocol's section-2 secondaries: real-audio ceiling of the same prompts
            truncated to the generated lengths (81 952 / 122 912 samples), scored under the frozen
            convention, so that --verdict can report rho_real(d) and rho_dense(d) at all four durations
            (3.84 / 10.24 s from the committed floor-ceiling and dense-192 artifacts, guarded).

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
REAL_DIR = "artifacts/icassp_gate0/real_refs"
FLOOR_OUT = f"{TMP}/draft5_floor_groups_out.json"             # real_full__sev2_192 / real_crop__sev2_192
D192_OUT = f"{TMP}/xsev_dense192_groups_out.json"             # dense__ac_short / dense__ac_native
FC_RESULT = "configs/research/draft5_floor_ceiling_result.json"
D192_RESULT = "configs/research/xsev_dense_192_control_result.json"

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


# ------------------------------------------------------------------------------------------ secondary
def rin():
    return f"{TMP}/draft5_sweep_real_groups_in.json"


def rout():
    return f"{TMP}/draft5_sweep_real_groups_out.json"


def secondary():
    """Real-audio ceiling at the two new generated lengths (protocol section 2, secondary).

    The real AudioCaps clips were already band-limited to 16 kHz and written as PCM_16 by
    draft5_floor_ceiling.py --emit (real_sev2_192_full_p*.wav). Truncating those files to 81 952 and
    122 912 samples reproduces sample-for-sample what writing w[:N] from the same float array would
    give (PCM_16 quantisation is per sample), and matches the crop convention of the 3.84 s ceiling.
    """
    import soundfile as sf
    from gate0_clap_scorer import FusedClapScorer, _prov
    ps = prompts(192)
    groups, meta = [], {}
    for tag, n in (("d128", 81952), ("d192", 122912)):
        items, short = [], 0
        for p in ps:
            src = f"{REAL_DIR}/real_sev2_192_full_p{p['prompt_index']}.wav"
            x, sr = sf.read(src, dtype="int16")
            if sr != 16000:
                raise SystemExit(f"{src}: sr {sr}")
            if len(x) < n:
                short += 1                      # kept at its full length (the native cell does the same)
            dst = f"{REAL_DIR}/real_sev2_192_{tag}_p{p['prompt_index']}.wav"
            sf.write(dst, x[:n], 16000, subtype="PCM_16")
            items.append({"caption": p["caption"], "wav": dst, "src": src})
        groups.append({"name": f"real_{tag}__sev2_192", "items": items})
        meta[tag] = {"samples": n, "n": len(items), "n_shorter_than_target": short}
    sc = FusedClapScorer()
    results = []
    for g in groups:
        t0 = time.time()
        caps = [it["caption"] for it in g["items"]]; wavs = [it["wav"] for it in g["items"]]
        te = sc._l2(sc.text_embed(caps)); ae = sc._l2(sc.audio_embed(wavs))
        M = te @ ae.T; diag = np.diag(M).astype(float)
        ca = np.asarray(caps); other = ca[:, None] != ca[None, :]
        floor = np.array([M[other[:, j], j].mean() for j in range(len(caps))], dtype=float)
        results.append({"name": g["name"], "n": len(caps), "cosines": diag.tolist(), "floor_item": floor.tolist(),
                        "matched_mean": float(diag.mean()), "floor_mean": float(floor.mean())})
        print(f"{g['name']:34s} n={len(caps):3d} matched {diag.mean():.4f} floor {floor.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"groups": groups, "real_refs": meta, "convention": "as --score (rev 365dea6e, one seed-once call per group)"},
              open(rin(), "w"), indent=1)
    json.dump({"results": results, "scorer_provenance": _prov(), "groups_in_sha256": sha(rin())}, open(rout(), "w"), indent=1)
    print("secondary scored ->", rout(), meta)


def ci_ratio(bt, num, den):
    """ratio of means with paired prompt resampling (same convention as draft5_floor_ceiling.py)."""
    num = np.asarray(num, float); den = np.asarray(den, float)
    bm = num[bt.idx].mean(1) / den[bt.idx].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {"point": float(num.mean() / den.mean()), "lo": float(lo), "hi": float(hi), "n": int(bt.n)}


def device_check_samples(din):
    """Descriptive sample-level comparison of the device-check pair (does not touch the PASS rule)."""
    import soundfile as sf
    g = {x["name"]: x for x in din["groups"]}
    if "device_check__new" not in g:
        return None
    rows = []
    for a, b in zip(g["device_check__new"]["items"], g["device_check__frozen"]["items"]):
        x = sf.read(a["wav"], dtype="int16")[0].astype(np.int64); y = sf.read(b["wav"], dtype="int16")[0].astype(np.int64)
        d = x - y
        rows.append({"new": a["wav"], "frozen": b["wav"], "n": int(len(x)), "max_abs_lsb": int(np.abs(d).max()),
                     "frac_samples_differ": float(np.mean(d != 0)),
                     "corr": float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])})
    return {"per_clip": rows, "max_abs_lsb": max(r["max_abs_lsb"] for r in rows),
            "note": "int16 units; 1 LSB = 3.05e-5 full scale"}


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
                               "delta_clap_per_clip": [float(v) for v in d],
                               "sha256_equal": din["generation_provenance"].get("device_check_sha_equal"),
                               "PASS": bool(np.abs(d).max() < 1e-9),
                               "pass_rule": "max |dCLAP| < 1e-9 (bit-identical expectation written in the frozen "
                                            "protocol section 4; kept as written, not moved after the result)",
                               "samples": device_check_samples(din)}

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
        out["s_response"] = {f"{s_}_{a}_to_{b}": bt.ci(cos(f"{s_}__{cb}", frozen=fb) - cos(f"{s_}__{ca}", frozen=fa))
                             for s_ in ("pruned2_A", "recovered2")
                             for (a, ca, fa), (b, cb, fb) in zip(
                                 [(3.84, "ac_short", True), (5.12, "ac_d128", False), (7.68, "ac_d192", False)],
                                 [(5.12, "ac_d128", False), (7.68, "ac_d192", False), (10.24, "ac_native", True)])}
        if os.path.exists(rout()):
            out["secondary"] = secondary_block(bt, new, fro, R)
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


def secondary_block(bt, new, fro, R):
    """Protocol section 2 secondaries: floors, real ceiling, rho_real(d), rho_dense(d) at all four durations."""
    real = {r["name"]: r for r in json.load(open(rout()))["results"]}
    flo = {r["name"]: r for r in json.load(open(FLOOR_OUT))["results"]}
    d192 = {r["name"]: r for r in json.load(open(D192_OUT))["results"]}
    V = lambda r: np.asarray(r["cosines"], float)
    F = lambda r: np.asarray(r["floor_item"], float)
    cell = {  # duration -> (P, P+FT, dense, real, floorP, floorPFT, floor_dense, floor_real)
        3.84: (V(fro["pruned2_A__ac_short"]), V(fro["recovered2__ac_short"]), V(d192["dense__ac_short"]),
               V(flo["real_crop__sev2_192"]), F(flo["pruned2_A__ac_short"]), F(flo["recovered2__ac_short"]),
               F(d192["dense__ac_short"]), F(flo["real_crop__sev2_192"])),
        5.12: (V(new["pruned2_A__ac_d128"]), V(new["recovered2__ac_d128"]), V(new["dense__ac_d128"]),
               V(real["real_d128__sev2_192"]), F(new["pruned2_A__ac_d128"]), F(new["recovered2__ac_d128"]),
               F(new["dense__ac_d128"]), F(real["real_d128__sev2_192"])),
        7.68: (V(new["pruned2_A__ac_d192"]), V(new["recovered2__ac_d192"]), V(new["dense__ac_d192"]),
               V(real["real_d192__sev2_192"]), F(new["pruned2_A__ac_d192"]), F(new["recovered2__ac_d192"]),
               F(new["dense__ac_d192"]), F(real["real_d192__sev2_192"])),
        10.24: (V(fro["pruned2_A__ac_native"]), V(fro["recovered2__ac_native"]), V(d192["dense__ac_native"]),
                V(flo["real_full__sev2_192"]), F(flo["pruned2_A__ac_native"]), F(flo["recovered2__ac_native"]),
                F(d192["dense__ac_native"]), F(flo["real_full__sev2_192"])),
    }
    # guards: the frozen 3.84 / 10.24 s cells must reproduce the committed floor-ceiling and dense-192 points
    fc = json.load(open(FC_RESULT))["sev2_xsev192"]; dd = json.load(open(D192_RESULT))["PRIMARY"]
    g = {}
    for d, kf, kd in ((3.84, "rho_real_short", "rho_short"), (10.24, "rho_real_native", "rho_native")):
        P, Q, D, Rl = cell[d][:4]
        g[f"rho_real@{d}"] = (float((Q - P).mean() / (Rl - P).mean()), fc[kf]["point"])
        g[f"rho_dense@{d}"] = (float((Q - P).mean() / (D - P).mean()), dd[kd]["point"])
    worst = max(abs(a - b) for a, b in g.values())
    # tolerance 1e-6 as in draft5_floor_ceiling.py: the real-ceiling cells there are float32 re-embeddings of
    # the frozen groups (per-item cosine diffs ~1e-8); the dense-192 ratios reproduce to 0.0
    if worst > 1e-6:
        raise SystemExit(f"guard failed: frozen-cell recovery ratios do not reproduce the committed artifacts {g}")
    out = {"guards_vs_committed": {k: {"here": a, "committed": b} for k, (a, b) in g.items()}, "worst_guard": worst,
           "real_refs": json.load(open(rin()))["real_refs"], "by_duration": {}}
    for d, (P, Q, D, Rl, fP, fQ, fD, fR) in cell.items():
        Rd = Q - P
        out["by_duration"][str(d)] = {
            "levels": {"P": float(P.mean()), "PFT": float(Q.mean()), "dense": float(D.mean()), "real": float(Rl.mean())},
            "floors": {"P": float(fP.mean()), "PFT": float(fQ.mean()), "dense": float(fD.mean()), "real": float(fR.mean())},
            "R": bt.ci(Rd), "R_c": bt.ci((Q - fQ) - (P - fP)),
            "dense_minus_P": bt.ci(D - P), "dense_minus_PFT": bt.ci(D - Q), "real_minus_PFT": bt.ci(Rl - Q),
            "rho_dense": ci_ratio(bt, Rd, D - P), "rho_real": ci_ratio(bt, Rd, Rl - P)}
    # duration responses of dense and real across the four points, paired per prompt
    ds = sorted(cell)
    out["s_dense_steps"] = {f"{a}_to_{b}": bt.ci(cell[b][2] - cell[a][2]) for a, b in zip(ds[:-1], ds[1:])}
    out["s_real_steps"] = {f"{a}_to_{b}": bt.ci(cell[b][3] - cell[a][3]) for a, b in zip(ds[:-1], ds[1:])}
    out["s_dense_3.84_to_10.24"] = bt.ci(cell[10.24][2] - cell[3.84][2])
    out["R_c_steps"] = {k: bt.ci(v) for k, v in {
        "D1": ((cell[5.12][1] - cell[5.12][5]) - (cell[5.12][0] - cell[5.12][4])) - ((cell[3.84][1] - cell[3.84][5]) - (cell[3.84][0] - cell[3.84][4])),
        "D2": ((cell[7.68][1] - cell[7.68][5]) - (cell[7.68][0] - cell[7.68][4])) - ((cell[5.12][1] - cell[5.12][5]) - (cell[5.12][0] - cell[5.12][4])),
        "D3": ((cell[10.24][1] - cell[10.24][5]) - (cell[10.24][0] - cell[10.24][4])) - ((cell[7.68][1] - cell[7.68][5]) - (cell[7.68][0] - cell[7.68][4]))}.items()}
    out["note"] = ("secondary, descriptive (protocol section 2): no gate. rho = ratio of means, paired prompt "
                   "resampling. 3.84 / 10.24 s cells are the committed frozen / dense-192 / floor-ceiling scores.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXP))
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--secondary", action="store_true", help="sweep only: real ceiling at 5.12 / 7.68 s")
    ap.add_argument("--root", default="")
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    if a.emit:
        if not a.root:
            raise SystemExit("--emit needs --root <job artifact dir>")
        emit(a.exp, a.root.rstrip("/"))
    if a.score:
        score(a.exp)
    if a.secondary:
        if a.exp != "sweep":
            raise SystemExit("--secondary is defined for --exp sweep only")
        secondary()
    if a.verdict:
        verdict(a.exp)
    if not (a.emit or a.score or a.verdict or a.secondary):
        raise SystemExit("pick --emit / --score / --secondary / --verdict")


if __name__ == "__main__":
    main()
