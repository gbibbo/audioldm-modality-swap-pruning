#!/usr/bin/env python3
"""REVIEW3-DENSERAW — CPU scoring + verdict of the RAW-weight dense baseline (third review item 1b; 0 cr).

Protocol: docs/review3_denseraw.md (frozen + sha256 before launch). Reads the `dense_raw` WAVs of job r3-denseraw straight
from the settled job dir, validates them against the generation manifests (count, 61 472 / 163 872 samples, sha256, ytid,
cuda device rule; a partial set is allowed per protocol §4 if the job was cut, never below 96 prompts), scores each cell as
ONE seed-once fused-CLAP call with the shuffled-caption floor (r2_verdict.score_groups convention), then computes the
pre-specified estimands on the common prompt set of each contrast:
  O(d)  = CLAP(dense_raw) - CLAP(dense_EMA)            raw-vs-EMA offset at step 0 (frozen XSEV-DENSE-192-CONTROL clips)
  G'(d) = CLAP(denseft_*) - CLAP(dense_raw)            fine-tuning effect net of the weight convention (both dense arms)
  J', dJ'_dense (guard vs r2_EXT2x2_result: baseline cancels), d_O = O(10.24) - O(3.84)
and applies the protocol's readings CONVENTION / DEGRADATION / MIXED.
Run (scoring needs the metrics venv):  OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/r3_denseraw_score.py [--score-only|--verdict-only]
"""
import glob, hashlib, json, os, re, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
import r2_verdict as V

PROTOCOL = "docs/review3_denseraw.md"
NS3 = "REVIEW3|BOOTSTRAP|2026-09-06"
JOB = "/teamspace/jobs/r3-denseraw/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/r3_denseraw"
GEN, DC = f"{JOB}/gen", f"{JOB}/device_check"
GIN, GOUT = f"{V.TMP}/r3_denseraw_groups_in.json", f"{V.TMP}/r3_denseraw_groups_out.json"
OUT = "configs/research/r3_denseraw_result.json"
FR = {"ac_short": 61472, "ac_native": 163872}
AC = json.load(open(V.AC192))["prompts"]
CAP = {p["prompt_index"]: p["caption"] for p in AC}; YT = {p["prompt_index"]: p["ytid"] for p in AC}
ALLPI = sorted(CAP)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def collect(ctx):
    """dense_raw items of one context, validated; returns (items in prompt_index order, provenance)."""
    import soundfile as sf
    man_p = f"{GEN}/gen_manifest_dense_raw_{ctx}.json"
    man = json.load(open(man_p)) if os.path.exists(man_p) else None       # the manifest is written after the LAST wav
    rows = {r["prompt_index"]: r for r in man["rows"]} if man else {}
    if man:
        rec, prv = man["recipe"], man["provenance"]
        devices = sorted({r["device"] for r in man["rows"]})
        if not devices or not all(d.startswith("cuda") for d in devices):
            raise SystemExit(f"device rule violated for dense_raw/{ctx}: {devices}")
        if not (rec["weight_convention"] == "raw" and rec["ddim"] == 50 and rec["guidance"] == 2.5 and rec["name"] == "frozen"
                and prv["checkpoint_convention"].startswith("dense_raw:")):
            raise SystemExit(f"{man_p}: recipe/checkpoint not the frozen dense_raw recipe: {rec} {prv.get('checkpoint_convention')}")
    picked = {}
    for w in sorted(glob.glob(f"{GEN}/dense_raw_{ctx}_p*_r0.wav")):
        pi = int(re.search(rf"dense_raw_{ctx}_p(\d+)_r0", w).group(1))
        info = sf.info(w)
        if info.frames != FR[ctx] or info.samplerate != 16000:
            raise SystemExit(f"bad WAV {w}: frames {info.frames} (expect {FR[ctx]}) sr {info.samplerate}")
        if rows:
            r = rows.get(pi)
            if r is None or r["ytid"] != YT[pi] or sha(w) != r["wav_sha256"]:
                raise SystemExit(f"manifest mismatch for {w}")
        picked[pi] = w
    order = [pi for pi in ALLPI if pi in picked]
    if len(order) < 96:
        raise SystemExit(f"dense_raw/{ctx}: only {len(order)} WAVs — below the protocol's partial-set floor of 96")
    if len(order) < 192:
        print(f"  NOTE dense_raw/{ctx}: {len(order)}/192 prompts (job cut); scoring the available set (protocol §4)")
    prov = {"n": len(order), "manifest_present": bool(man), "gen_manifest_sha256": sha(man_p) if man else None,
            "checkpoint": man["provenance"]["checkpoint_convention"] if man else "dense_raw (manifest absent: job cut before it was written)",
            "git_sha": (man["provenance"].get("git_sha") if man else None), "devices": (devices if man else None), "frames": FR[ctx]}
    return [{"caption": CAP[pi], "wav": picked[pi], "prompt_index": pi, "ytid": YT[pi]} for pi in order], prov


def score():
    groups, prov = [], {}
    for ctx in ("ac_short", "ac_native"):
        items, pv = collect(ctx); prov[f"dense_raw__{ctx}"] = pv
        groups.append({"name": f"dense_raw__{ctx}", "items": items, "reps": 1})
    # device check: dense-EMA ac_native prompts 0-3 regenerated on the job machine vs the frozen XSEV-DENSE-192 clips
    fro = {g["name"]: g for g in json.load(open(f"{V.TMP}/xsev_dense192_groups_in.json"))["groups"]}["dense__ac_native"]["items"]
    dc = [(f"{DC}/dense_ac_native_p{i}_r0.wav", fro[i]["wav"], CAP[i]) for i in range(4) if os.path.exists(f"{DC}/dense_ac_native_p{i}_r0.wav")]
    if dc:
        groups.append({"name": "device_check__new", "items": [{"caption": c, "wav": w, "prompt_index": i} for i, (w, _f, c) in enumerate(dc)], "reps": 1})
        groups.append({"name": "device_check__frozen", "items": [{"caption": c, "wav": f, "prompt_index": i} for i, (_w, f, c) in enumerate(dc)], "reps": 1})
        prov["device_check_sha_equal"] = [sha(w) == sha(f) for w, f, _c in dc]
    json.dump({"groups": groups, "generation_provenance": prov, "protocol_doc_sha256": sha(PROTOCOL), "job": "r3-denseraw",
               "convention": "each group = ONE seed-once fused-CLAP call in prompt_index order (rev 365dea6e); floors are shuffled captions"},
              open(GIN, "w"), indent=1)
    V.score_groups(groups, GOUT, {"groups_in_sha256": sha(GIN), "note": "REVIEW3-DENSERAW CPU scoring (r3_denseraw_score.py)"})


def verdict():
    S = V.SESOI
    R = V.load_results(GOUT); DS = V.load_results(V.gout("denseft_s")); DN = V.load_results(V.gout("denseft_n"))
    X = json.load(open("configs/research/r2_EXT2x2_result.json"))
    def pmap(res):
        c, f, order = V.per_prompt(res); return dict(zip(order, map(float, c))), dict(zip(order, map(float, f)))
    def fpmap(name):                                                 # frozen cells: sorted-prompt_index order, no field
        arr = V.frozen_pp(name); return dict(zip(ALLPI[:192], map(float, arr)))
    R3, fR3 = pmap(R["dense_raw__ac_short"]); R10, fR10 = pmap(R["dense_raw__ac_native"])
    E3, E10 = fpmap("dense__ac_short"), fpmap("dense__ac_native")
    Ds3, _ = pmap(DS["denseft__ac_short"]); Ds10, _ = pmap(DS["denseft__ac_native"])
    Dn3, _ = pmap(DN["denseft__ac_short"]); Dn10, _ = pmap(DN["denseft__ac_native"])

    def ci(tag, expr, *maps):
        keys = sorted(set(maps[0]).intersection(*[set(m) for m in maps[1:]]))
        arrs = [np.array([m[k] for k in keys]) for m in maps]
        return V.Boot(NS3 + f"|{tag}|n{len(keys)}", len(keys)).ci(expr(*arrs))
    d = lambda a, b: a - b
    O = {"3.84": ci("O3", d, R3, E3), "10.24": ci("O10", d, R10, E10)}
    Gp = {"denseft_short": {"3.84": ci("Gs3", d, Ds3, R3), "10.24": ci("Gs10", d, Ds10, R10)},
          "denseft_native": {"3.84": ci("Gn3", d, Dn3, R3), "10.24": ci("Gn10", d, Dn10, R10)}}
    G_ema = {"denseft_short": {"3.84": ci("GsE3", d, Ds3, E3), "10.24": ci("GsE10", d, Ds10, E10)},
             "denseft_native": {"3.84": ci("GnE3", d, Dn3, E3), "10.24": ci("GnE10", d, Dn10, E10)}}
    Jp = {"denseft_short": ci("Js", lambda a, b, c, e: (a - b) - (c - e), Ds10, R10, Ds3, R3),
          "denseft_native": ci("Jn", lambda a, b, c, e: (a - b) - (c - e), Dn10, R10, Dn3, R3)}
    dJp = ci("dJ", lambda a, b, c, e: (a - b) - (c - e), Dn10, Dn3, Ds10, Ds3)
    dO = ci("dO", lambda a, b, c, e: (a - b) - (c - e), R10, E10, R3, E3)
    inside = lambda c: -S < c["lo"] and c["hi"] < S
    four = [Gp[a][dd] for a in Gp for dd in ("3.84", "10.24")]
    conv = all(inside(c) for c in four)
    degr = all(c["hi"] < 0 for c in four) and all(abs(O[dd]["point"]) < abs(np.mean([G_ema[a][dd]["point"] for a in G_ema])) / 2 for dd in ("3.84", "10.24"))
    reading = "CONVENTION" if conv else ("DEGRADATION" if degr else "MIXED")
    def sign_reading(c):
        return "RESOLVED NEGATIVE" if c["hi"] < 0 else ("RESOLVED POSITIVE" if c["lo"] > 0 else ("WITHIN SESOI" if inside(c) else "UNRESOLVED"))
    out = {"artifact": "r3_denseraw_result", "protocol_doc": PROTOCOL, "protocol_doc_sha256": sha(PROTOCOL),
           "class": "pre-specified follow-up (docs/review3_denseraw.md); cannot change any frozen verdict",
           "bootstrap": {"B": V.B, "seed_namespace": NS3, "unit": "prompt", "ci": "percentile 95%"}, "SESOI": S,
           "levels": {"dense_raw_3.84": float(np.mean(list(R3.values()))), "dense_raw_10.24": float(np.mean(list(R10.values()))),
                      "dense_raw_floor_3.84": float(np.mean(list(fR3.values()))), "dense_raw_floor_10.24": float(np.mean(list(fR10.values()))),
                      "dense_ema_3.84": float(np.mean(list(E3.values()))), "dense_ema_10.24": float(np.mean(list(E10.values()))),
                      "n_dense_raw": {"3.84": len(R3), "10.24": len(R10)}},
           "O": O, "O_reading": {k: sign_reading(v) for k, v in O.items()}, "dO": dO, "dO_reading": sign_reading(dO),
           "G_prime_vs_dense_raw": Gp, "G_vs_dense_ema_recomputed": G_ema, "J_prime": Jp, "dJ_prime_dense": dJp,
           "reading": reading,
           "reading_rule": "CONVENTION: all four G'(d) CIs inside +-SESOI; DEGRADATION: all four hi95(G'(d)) < 0 and |O(d)| < |G(d)|/2 at both d; else MIXED (docs/review3_denseraw.md section 4)",
           "guard_dJ_dense_vs_EXT2x2": {"here": dJp["point"], "EXT2x2": X["dense"]["dJ"]["point"], "abs_diff": abs(dJp["point"] - X["dense"]["dJ"]["point"])},
           "inputs": {p: sha(p) for p in (GIN, GOUT, V.gout("denseft_s"), V.gout("denseft_n"), V.D192_OUT, "configs/research/r2_EXT2x2_result.json") if os.path.exists(p)}}
    gin = json.load(open(GIN)); out["generation_provenance"] = gin["generation_provenance"]
    if "device_check__new" in R and "device_check__frozen" in R:
        a = np.asarray(R["device_check__new"]["cosines"]); b = np.asarray(R["device_check__frozen"]["cosines"])
        out["device_check"] = {"max_abs_delta_clap": float(np.max(np.abs(a - b))), "delta_per_clip": (a - b).tolist(),
                               "sha256_equal": gin["generation_provenance"].get("device_check_sha_equal"),
                               "rule": "descriptive (the frozen bit-identical rule has failed at ~4e-6 on every job; reported as written)"}
    js = json.dumps(out, indent=1, sort_keys=True); out["artifact_sha256"] = hashlib.sha256(js.encode()).hexdigest()
    json.dump(out, open(OUT, "w"), indent=1)
    f = lambda c: f"{c['point']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n={c['n']})"
    print("levels dense_raw 3.84/10.24:", round(out["levels"]["dense_raw_3.84"], 3), round(out["levels"]["dense_raw_10.24"], 3),
          "| dense_EMA:", round(out["levels"]["dense_ema_3.84"], 3), round(out["levels"]["dense_ema_10.24"], 3))
    print("O(3.84):", f(O["3.84"]), "| O(10.24):", f(O["10.24"]), "| dO:", f(dO))
    for a in Gp:
        print(f"G'({a}) 3.84:", f(Gp[a]["3.84"]), "| 10.24:", f(Gp[a]["10.24"]), "| J':", f(Jp[a]))
    print("dJ'_dense:", f(dJp), "| guard abs diff vs EXT2x2:", round(out["guard_dJ_dense_vs_EXT2x2"]["abs_diff"], 6))
    print("READING:", reading, "| wrote", OUT)


if __name__ == "__main__":
    if "--verdict-only" not in sys.argv:
        score()
    if "--score-only" not in sys.argv:
        verdict()
