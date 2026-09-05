#!/usr/bin/env python3
"""REVIEWER2-FOLLOWUP — validation, scoring and pre-specified verdicts (CPU only, 0 cr).

Protocol: docs/reviewer2_followup.md (frozen + sha256 sidecar BEFORE any generation). This script generates
nothing. It validates the WAVs the T4 jobs produced, scores them under the frozen fused-CLAP convention
(rev 365dea6e, one seed-once call per group, shuffled-caption floors from the same embeddings) and applies
the protocol's pre-specified readings. It cannot change any frozen verdict.

  --emit --job {a,b,c,shortft} --root <dir>   validate WAVs vs generation manifests; write scorer groups
  --score --job ...                           score the groups (+ device-check pair where present)
  --clotho-refs                               real Clotho references: 16 kHz, first 10.24 s / 3.84 s, then score
  --gate0-dense                               score the existing gate0-gen-1 dense clips on the sev-1 hip-hop battery
                                              and the Kim-193 real clips (E6, CPU-only part)
  --verdict {E5,E6,E7,E8,E1c,B,E3}            pre-specified estimands and readings -> configs/research/r2_<exp>_result.json

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/r2_verdict.py --emit --job a --root <dir> ; ...
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np

PROTOCOL = "docs/reviewer2_followup.md"
TMP = "artifacts/icassp_gate0/_score_tmp"
AC192 = "configs/research/xsev_audiocaps_manifest.json"
MUSIC64 = "configs/research/xsev_music_manifest.json"
MUSIC_EXT = "configs/research/r2_music_ext_manifest.json"
CLOTHO = "configs/research/r2_clotho_manifest.json"
GATE0_BATTERY = "configs/research/icassp_gate0_battery.json"
ARMD_SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
FROZEN_IN, FROZEN_OUT = f"{TMP}/xsev_sev2_groups_in.json", f"{TMP}/xsev_sev2_groups_out.json"
MUSIC_NATIVE_OUT = f"{TMP}/music_native_groups_out.json"
D192_OUT = f"{TMP}/xsev_dense192_groups_out.json"
FLOOR_OUT = f"{TMP}/draft5_floor_groups_out.json"
PHENOM_OUT = "artifacts/icassp_gate0/_phenom_groups_out.json"
GATE0_DENSE_DIR = "/teamspace/jobs/gate0-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/gen_gate0"
KIM_MANIFEST = "artifacts/icassp_gate0/kim193_train_manifest.json"
CLOTHO_7Z = "artifacts/clotho/clotho_audio_evaluation.7z"
CLOTHO_WAV_DIR = "artifacts/clotho/eval_wavs"
CLOTHO_REF_DIR = "artifacts/icassp_gate0/real_refs_clotho"
B = 10000
SESOI = 0.025
NS = "REVIEWER2-FOLLOWUP|BOOTSTRAP|2026-09-05"
FRAMES = {96: 61472, 128: 81952, 192: 122912, 256: 163872}     # 384 (15.36 s) taken from the generation manifest

# job -> list of (system, context, manifest, id key, n prompts, reps, first_n)
JOBS = {
    "a": [("dense", "music", MUSIC64, "ytid", 64, 3, 0), ("dense", "music_native", MUSIC64, "ytid", 64, 1, 0)]
         + [(s, c, MUSIC_EXT, "ytid", 63, 1, 0) for s in ("pruned2_A", "recovered2", "dense") for c in ("music_ext", "music_ext_native")]
         + [("textft", c, AC192, "ytid", 96, 1, 96) for c in ("ac_short", "ac_native")],
    "b": [(s, c, CLOTHO, "clip_id", 96, 1, 0) for s in ("pruned2_A", "recovered2", "dense") for c in ("clotho_short", "clotho_native")]
         + [(s, "ac_d384", AC192, "ytid", 96, 1, 96) for s in ("pruned2_A", "recovered2")],
    "c": [(s, c, AC192, "ytid", 96, 1, 96) for s in ("p1_pruned", "p1_recovered") for c in ("ac_short", "ac_native")],
    "shortft": [("shortft", c, AC192, "ytid", 192, 1, 0) for c in ("ac_short", "ac_native")],
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def prompts(manifest, n=None):
    ps = json.load(open(manifest))["prompts"]
    if ps and "prompt_index" not in ps[0]:            # gate0 battery: list order IS the prompt index (p{i} in the WAV names)
        ps = [{**p, "prompt_index": i} for i, p in enumerate(ps)]
    ps = sorted(ps, key=lambda p: p["prompt_index"])
    return ps[:n] if n else ps


def gin(job): return f"{TMP}/r2_{job}_groups_in.json"
def gout(job): return f"{TMP}/r2_{job}_groups_out.json"


# ----------------------------------------------------------------------------------------------- emit
def emit(job, root):
    import soundfile as sf
    groups, prov = [], {}
    for sysname, ctx, manifest, key, npr, reps, first_n in JOBS[job]:
        man_p = f"{root}/gen_manifest_{sysname}_{ctx}.json"
        if not os.path.exists(man_p):
            raise SystemExit(f"missing generation manifest {man_p}")
        man = json.load(open(man_p))
        if (man["recipe"].get("first_n") or 0) != first_n:
            raise SystemExit(f"{man_p}: first_n {man['recipe'].get('first_n')} != protocol {first_n}")
        rows = {(r["prompt_index"], r["replicate_index"]): r for r in man["rows"]}
        if len(rows) != npr * reps:
            raise SystemExit(f"{man_p}: {len(rows)} rows, expected {npr * reps}")
        exp_frames = FRAMES.get(man["recipe"]["latent_t"]) or man["rows"][0]["n_samples"]
        ps = prompts(manifest, first_n or None)
        if len(ps) != npr:
            raise SystemExit(f"{manifest}: {len(ps)} prompts, expected {npr}")
        items = []
        for p in ps:
            for r in range(reps):
                w = f"{root}/{sysname}_{ctx}_p{p['prompt_index']}_r{r}.wav"
                if not os.path.exists(w):
                    raise SystemExit(f"missing WAV {w}")
                info = sf.info(w)
                if info.frames != exp_frames or info.samplerate != 16000:
                    raise SystemExit(f"bad WAV {w}: frames {info.frames} (expected {exp_frames}) sr {info.samplerate}")
                row = rows[(p["prompt_index"], r)]
                if row["ytid"] != p[key] or sha(w) != row["wav_sha256"]:
                    raise SystemExit(f"manifest mismatch for {w}")
                items.append({"caption": p["caption"], "wav": w, "prompt_index": p["prompt_index"], "replicate": r})
        devices = sorted({r["device"] for r in man["rows"]})
        if len(devices) != 1 or not devices[0].startswith("cuda"):
            raise SystemExit(f"device rule violated for {sysname}/{ctx}: {devices}")
        prov[f"{sysname}__{ctx}"] = {"gen_manifest_sha256": sha(man_p), "n": man["n"], "devices": devices,
                                     "git_sha": man["provenance"].get("git_sha") or man["provenance"].get("commit"),
                                     "checkpoint": man["provenance"].get("checkpoint_convention"), "recipe": man["recipe"],
                                     "frames": exp_frames}
        groups.append({"name": f"{sysname}__{ctx}", "items": items, "reps": reps})
    dc_root = f"{root}/device_check"
    if os.path.isdir(dc_root):
        fro = {g["name"]: g for g in json.load(open(FROZEN_IN))["groups"]}["pruned2_A__ac_native"]["items"]
        ac4 = prompts(AC192, 4)
        dc = [(f"{dc_root}/pruned2_A_ac_native_p{i}_r0.wav", fro[i]["wav"], ac4[i]["caption"]) for i in range(4)
              if os.path.exists(f"{dc_root}/pruned2_A_ac_native_p{i}_r0.wav")]
        if dc:
            groups.append({"name": "device_check__new", "items": [{"caption": c, "wav": w} for w, _f, c in dc], "reps": 1})
            groups.append({"name": "device_check__frozen", "items": [{"caption": c, "wav": f} for _w, f, c in dc], "reps": 1})
            prov["device_check_sha_equal"] = [sha(w) == sha(f) for w, f, _c in dc]
    json.dump({"groups": groups, "generation_provenance": prov, "protocol_doc_sha256": sha(PROTOCOL), "job": job,
               "convention": "each group = ONE seed-once fused-CLAP call in (prompt_index, replicate) order (rev 365dea6e); "
                             "floors are shuffled captions from the same embeddings"}, open(gin(job), "w"), indent=1)
    print(f"emitted {len(groups)} groups -> {gin(job)}")
    if "device_check_sha_equal" in prov:
        print("device check sha256 equal to frozen:", prov["device_check_sha_equal"])


# ---------------------------------------------------------------------------------------------- score
def score_groups(groups, out_path, extra=None):
    from gate0_clap_scorer import FusedClapScorer, _prov
    sc = FusedClapScorer()
    results = []
    for g in groups:
        t0 = time.time()
        caps = [it["caption"] for it in g["items"]]; wavs = [it["wav"] for it in g["items"]]
        te = sc._l2(sc.text_embed(caps)); ae = sc._l2(sc.audio_embed(wavs))
        M = te @ ae.T; diag = np.diag(M).astype(float)
        ca = np.asarray(caps); other = ca[:, None] != ca[None, :]
        floor = np.array([M[other[:, j], j].mean() if other[:, j].any() else np.nan for j in range(len(caps))], dtype=float)
        results.append({"name": g["name"], "n": len(caps), "reps": g.get("reps", 1), "cosines": diag.tolist(),
                        "floor_item": floor.tolist(), "matched_mean": float(diag.mean()), "floor_mean": float(np.nanmean(floor)),
                        "prompt_index": [it.get("prompt_index") for it in g["items"]]})
        print(f"{g['name']:34s} n={len(caps):3d} matched {diag.mean():.4f} floor {np.nanmean(floor):.4f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"results": results, "scorer_provenance": _prov(), **(extra or {})}, open(out_path, "w"), indent=1)
    print("scored ->", out_path)


def score(job):
    din = json.load(open(gin(job)))
    score_groups(din["groups"], gout(job), {"groups_in_sha256": sha(gin(job))})


# ----------------------------------------------------------------------------------- Clotho real refs
def clotho_refs():
    """Real Clotho evaluation clips of the battery: mono 16 kHz, first 10.24 s (163 872) and first 3.84 s (61 472)."""
    import soundfile as sf, librosa
    ps = prompts(CLOTHO)
    if not os.path.isdir(CLOTHO_WAV_DIR) or len(glob.glob(f"{CLOTHO_WAV_DIR}/**/*.wav", recursive=True)) < 1000:
        import py7zr
        os.makedirs(CLOTHO_WAV_DIR, exist_ok=True)
        with py7zr.SevenZipFile(CLOTHO_7Z, "r") as z:
            z.extractall(CLOTHO_WAV_DIR)
    src_index = {os.path.basename(p): p for p in glob.glob(f"{CLOTHO_WAV_DIR}/**/*.wav", recursive=True)}
    os.makedirs(CLOTHO_REF_DIR, exist_ok=True)
    groups, meta = [], {"n_shorter_than_target": {"native": 0, "short": 0}, "src_sha256": {}}
    items = {"native": [], "short": []}
    for p in ps:
        src = src_index[p["clip_id"]]
        y, sr = sf.read(src, dtype="float32", always_2d=True); y = y.mean(1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        meta["src_sha256"][p["clip_id"]] = sha(src)
        for tag, n in (("native", FRAMES[256]), ("short", FRAMES[96])):
            if len(y) < n:
                meta["n_shorter_than_target"][tag] += 1
            dst = f"{CLOTHO_REF_DIR}/real_clotho_{tag}_p{p['prompt_index']}.wav"
            sf.write(dst, y[:n], 16000, subtype="PCM_16")
            items[tag].append({"caption": p["caption"], "wav": dst, "prompt_index": p["prompt_index"], "replicate": 0})
    for tag in ("native", "short"):
        groups.append({"name": f"real_clotho_{tag}", "items": items[tag], "reps": 1})
    json.dump({"groups": groups, "real_refs": meta, "clotho_7z_sha256": sha(CLOTHO_7Z)}, open(f"{TMP}/r2_clotho_real_groups_in.json", "w"), indent=1)
    score_groups(groups, f"{TMP}/r2_clotho_real_groups_out.json", {"real_refs": meta})


# ------------------------------------------------------------------------- gate0 dense + Kim (E6 CPU)
def gate0_dense():
    """Existing gate0-gen-1 dense clips on the severity-1 hip-hop battery (64 x 3, 3.84 s) + Kim-193 real clips."""
    ps = prompts(GATE0_BATTERY)
    man = json.load(open(f"{GATE0_DENSE_DIR}/gen_manifest_dense_both.json"))
    items = []
    for p in ps:
        for r in range(3):
            w = f"{GATE0_DENSE_DIR}/dense_noadapter_p{p['prompt_index']}_r{r}.wav"
            if not os.path.exists(w):
                raise SystemExit(f"missing {w}")
            items.append({"caption": p["caption"], "wav": w, "prompt_index": p["prompt_index"], "replicate": r})
    kim = json.load(open(KIM_MANIFEST))["data"]
    kim_items = [{"caption": d["caption"], "wav": d["wav"], "prompt_index": i, "replicate": 0} for i, d in enumerate(kim)]
    groups = [{"name": "gate0_dense__music_sev1", "items": items, "reps": 3},
              {"name": "kim193_real__hiphop", "items": kim_items, "reps": 1}]
    json.dump({"groups": groups, "gate0_manifest_sha256": sha(f"{GATE0_DENSE_DIR}/gen_manifest_dense_both.json"),
               "kim_manifest_sha256": sha(KIM_MANIFEST)}, open(f"{TMP}/r2_gate0dense_groups_in.json", "w"), indent=1)
    score_groups(groups, f"{TMP}/r2_gate0dense_groups_out.json")


# -------------------------------------------------------------------------------------------- verdict
class Boot:
    def __init__(self, ns, n):
        seed = int(hashlib.sha256(ns.encode()).hexdigest()[:8], 16) % (2 ** 31)
        self.idx = np.random.default_rng(seed).integers(0, n, (B, n)); self.n, self.seed = n, seed

    def ci(self, v):
        v = np.asarray(v, float); bm = v[self.idx].mean(1); lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(self.n)}

    def ratio(self, num, den):
        num = np.asarray(num, float); den = np.asarray(den, float)
        bm = num[self.idx].mean(1) / den[self.idx].mean(1); lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": float(num.mean() / den.mean()), "lo": float(lo), "hi": float(hi), "n": int(self.n)}


def two_sample(ns, a, b):
    """unpaired difference of means a - b, independent resampling of the two prompt sets."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    seed = int(hashlib.sha256((ns + "|2s").encode()).hexdigest()[:8], 16) % (2 ** 31)
    rng = np.random.default_rng(seed)
    bm = a[rng.integers(0, len(a), (B, len(a)))].mean(1) - b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {"point": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi), "n_a": int(len(a)), "n_b": int(len(b))}


def per_prompt(res):
    """per-prompt mean cosine and floor (replicates averaged), in prompt_index order."""
    pi = np.asarray(res["prompt_index"]); c = np.asarray(res["cosines"], float); f = np.asarray(res["floor_item"], float)
    order = sorted(set(pi.tolist()))
    return (np.array([c[pi == k].mean() for k in order]), np.array([f[pi == k].mean() for k in order]), order)


def load_results(*paths):
    out = {}
    for p in paths:
        if os.path.exists(p):
            for r in json.load(open(p))["results"]:
                out[r["name"]] = r
    return out


def frozen_pp(name, n_first=None):
    """per-prompt cosines of a frozen severity-2 cell (prompt_index order, replicates averaged)."""
    R = load_results(FROZEN_OUT, MUSIC_NATIVE_OUT, D192_OUT)
    r = R[name]
    c = np.asarray(r["cosines"], float)
    if "prompt_index" in r:
        c, _f, _o = per_prompt(r)
    elif name.endswith("__music") and len(c) == 192:      # frozen music cell: 64 prompts x 3 replicates, prompt-major
        c = c.reshape(64, 3).mean(1)
    return c[:n_first] if n_first else c


def reading(rules):
    for label, cond in rules:
        if cond:
            return label
    return "UNRESOLVED"


def verdict(exp):
    out = {"artifact": f"r2_{exp}_result", "protocol_doc": PROTOCOL, "protocol_doc_sha256": sha(PROTOCOL),
           "class": "pre-specified follow-up (docs/reviewer2_followup.md); cannot change any frozen verdict",
           "bootstrap": {"B": B, "seed_namespace": NS, "unit": "prompt", "ci": "percentile 95%"}, "SESOI": SESOI}
    A = load_results(gout("a")); Bj = load_results(gout("b")); C = load_results(gout("c")); SF = load_results(gout("shortft"))
    fro = load_results(FROZEN_OUT, MUSIC_NATIVE_OUT, D192_OUT, FLOOR_OUT)

    def R_of(new_pft, new_p, n):
        bt = Boot(NS + f"|{exp}|R|{n}", n); return bt

    if exp == "E5":
        bt = Boot(NS + "|E5", 96); real = load_results(f"{TMP}/r2_clotho_real_groups_out.json")
        cells = {}
        for d, ctx, rtag in (("3.84", "clotho_short", "short"), ("10.24", "clotho_native", "native")):
            P, fP, _ = per_prompt(Bj[f"pruned2_A__{ctx}"]); Q, fQ, _ = per_prompt(Bj[f"recovered2__{ctx}"]); D, fD, _ = per_prompt(Bj[f"dense__{ctx}"])
            rr = real.get(f"real_clotho_{rtag}"); Rv, fR = (per_prompt(rr)[:2] if rr else (None, None))
            cells[d] = {"levels": {"P": float(P.mean()), "PFT": float(Q.mean()), "dense": float(D.mean()), "real": (float(Rv.mean()) if Rv is not None else None)},
                        "floors": {"P": float(fP.mean()), "PFT": float(fQ.mean()), "dense": float(fD.mean()), "real": (float(fR.mean()) if fR is not None else None)},
                        "R": bt.ci(Q - P), "R_c": bt.ci((Q - fQ) - (P - fP)), "dense_minus_PFT": bt.ci(D - Q),
                        "rho_dense": bt.ratio(Q - P, D - P), "rho_real": (bt.ratio(Q - P, Rv - P) if Rv is not None else None),
                        "A_dense_above_chance": bt.ci(D - fD), "W": float(np.mean(Q > P))}
        P3, _, _ = per_prompt(Bj["pruned2_A__clotho_short"]); Q3, _, _ = per_prompt(Bj["recovered2__clotho_short"])
        P10, _, _ = per_prompt(Bj["pruned2_A__clotho_native"]); Q10, _, _ = per_prompt(Bj["recovered2__clotho_native"])
        out["J_clo"] = bt.ci((Q10 - P10) - (Q3 - P3))
        ac = {d: frozen_pp(f"recovered2__{c}", 96) - frozen_pp(f"pruned2_A__{c}", 96) for d, c in (("3.84", "ac_short"), ("10.24", "ac_native"))}
        out["R_AC_first96"] = {d: Boot(NS + "|E5|AC96", 96).ci(v) for d, v in ac.items()}
        out["D_clo"] = {"3.84": two_sample(NS + "|E5|D3", ac["3.84"], Q3 - P3), "10.24": two_sample(NS + "|E5|D10", ac["10.24"], Q10 - P10)}
        R10 = cells["10.24"]["R"]; D10 = out["D_clo"]["10.24"]
        out["reading"] = reading([("TRANSFERS", R10["lo"] > 0 and abs(D10["point"]) < SESOI),
                                  ("PARTIAL", R10["lo"] > 0 and D10["lo"] > SESOI),
                                  ("NO TRANSFER", R10["lo"] <= 0 <= R10["hi"])])
        out["cells"] = cells
    elif exp == "E6":
        g0 = load_results(f"{TMP}/r2_gate0dense_groups_out.json")
        cells = {}
        # severity 2: new dense on the frozen 64 at both durations vs frozen P / P+FT
        for d, dctx, pctx in (("3.84", "music", "music"), ("10.24", "music_native", "music_native")):
            D, fD, _ = per_prompt(A[f"dense__{dctx}"])
            P = frozen_pp(f"pruned2_A__{pctx}"); Q = frozen_pp(f"recovered2__{pctx}")
            bt = Boot(NS + f"|E6|sev2|{d}", 64)
            cells[f"sev2_{d}"] = {"dense": float(D.mean()), "dense_floor": float(fD.mean()), "P": float(P.mean()), "PFT": float(Q.mean()),
                                  "A_dense": bt.ci(D - fD), "rho_dense": bt.ratio(Q - P, D - P), "dense_minus_PFT": bt.ci(D - Q), "dense_minus_P": bt.ci(D - P)}
        # severity 1 (3.84 s): gate0 dense clips vs the phenomenon P / P+FT groups (LoRA off)
        if "gate0_dense__music_sev1" in g0:
            D, fD, _ = per_prompt(g0["gate0_dense__music_sev1"])
            ph = {r["name"]: r for r in json.load(open(PHENOM_OUT))["results"]}
            # guard: the phenomenon job already scored these very dense clips (`dense__off`, LoRA off) under the
            # frozen convention; the re-scoring (needed for the floor) must reproduce it
            d_fro = np.asarray(ph["dense__off"]["cosines"], float).reshape(64, 3).mean(1)
            cells["guard_gate0_dense_vs_phenom_dense_off_max_absdiff"] = float(np.abs(D - d_fro).max())
            names = list(ph)
            pn = next(n for n in names if "pruned" in n and ("off" in n or "noadapter" in n))
            qn = next(n for n in names if "recovered" in n and ("off" in n or "noadapter" in n))
            P = np.asarray(ph[pn]["cosines"], float).reshape(64, 3).mean(1); Q = np.asarray(ph[qn]["cosines"], float).reshape(64, 3).mean(1)
            bt = Boot(NS + "|E6|sev1", 64)
            cells["sev1_3.84"] = {"dense": float(D.mean()), "dense_floor": float(fD.mean()), "P": float(P.mean()), "PFT": float(Q.mean()),
                                  "P_group": pn, "PFT_group": qn, "A_dense": bt.ci(D - fD), "rho_dense": bt.ratio(Q - P, D - P),
                                  "dense_minus_PFT": bt.ci(D - Q), "dense_minus_P": bt.ci(D - P)}
            K, fK, _ = per_prompt(g0["kim193_real__hiphop"])
            cells["kim193_real_3.84_domain_ceiling"] = {"real": float(K.mean()), "floor": float(fK.mean()), "above_chance": Boot(NS + "|E6|kim", len(K)).ci(K - fK),
                                                        "note": "real 4.0-s hip-hop excerpts with their own MusicCaps captions; NOT the battery prompts (domain-level ceiling)"}
        for k, c in cells.items():
            if "A_dense" in c:
                c["battery_discriminates_for_dense"] = c["A_dense"]["lo"] > SESOI
        out["cells"] = cells
    elif exp == "E7":
        cells = {}
        for d, ctx, fctx in (("3.84", "music_ext", "music"), ("10.24", "music_ext_native", "music_native")):
            P, fP, _ = per_prompt(A[f"pruned2_A__{ctx}"]); Q, fQ, _ = per_prompt(A[f"recovered2__{ctx}"]); D, fD, _ = per_prompt(A[f"dense__{ctx}"])
            Pf = frozen_pp(f"pruned2_A__{fctx}"); Qf = frozen_pp(f"recovered2__{fctx}")
            bt63 = Boot(NS + f"|E7|ext|{d}", 63); bt127 = Boot(NS + f"|E7|pooled|{d}", 127)
            pooled = np.concatenate([Qf - Pf, Q - P])
            cells[d] = {"ext": {"P": float(P.mean()), "PFT": float(Q.mean()), "dense": float(D.mean()), "R": bt63.ci(Q - P), "R_c": bt63.ci((Q - fQ) - (P - fP)),
                                "rho_dense": bt63.ratio(Q - P, D - P), "A_dense": bt63.ci(D - fD), "W": float(np.mean(Q > P))},
                        "frozen64_R": Boot(NS + f"|E7|frozen|{d}", 64).ci(Qf - Pf),
                        "pooled127_R": bt127.ci(pooled), "pooled127_W": float(np.mean(pooled > 0)),
                        "ext_minus_frozen": two_sample(NS + f"|E7|het|{d}", Q - P, Qf - Pf)}
        P3, _, _ = per_prompt(A["pruned2_A__music_ext"]); Q3, _, _ = per_prompt(A["recovered2__music_ext"])
        P10, _, _ = per_prompt(A["pruned2_A__music_ext_native"]); Q10, _, _ = per_prompt(A["recovered2__music_ext_native"])
        pooledJ = np.concatenate([frozen_pp("recovered2__music_native") - frozen_pp("pruned2_A__music_native") - (frozen_pp("recovered2__music") - frozen_pp("pruned2_A__music")),
                                  (Q10 - P10) - (Q3 - P3)])
        out["J_music_127"] = Boot(NS + "|E7|J", 127).ci(pooledJ)
        r10 = cells["10.24"]["pooled127_R"]
        out["reading_10.24"] = reading([("ABSENT", r10["lo"] <= 0 <= r10["hi"] and abs(r10["point"]) < SESOI),
                                        ("POSITIVE GAIN", r10["lo"] > 0), ("NEGATIVE GAIN", r10["hi"] < 0)])
        out["cells"] = cells
    elif exp == "E8":
        P3, fP3, _ = per_prompt(C["p1_pruned__ac_short"]); Q3, fQ3, _ = per_prompt(C["p1_recovered__ac_short"])
        P10, fP10, _ = per_prompt(C["p1_pruned__ac_native"]); Q10, fQ10, _ = per_prompt(C["p1_recovered__ac_native"])
        bt = Boot(NS + "|E8|96", 96)
        j96 = (Q10 - P10) - (Q3 - P3)
        opd = json.load(open("configs/research/op_duration_discriminator_1_result.json"))["raw_cosines"]
        j80 = (np.asarray(opd["recovered_alt"]) - np.asarray(opd["pruned_alt"])) - (np.asarray(opd["recovered_ctrl"]) - np.asarray(opd["pruned_ctrl"]))
        out["new96"] = {"levels": {"P_3.84": float(P3.mean()), "PFT_3.84": float(Q3.mean()), "P_10.24": float(P10.mean()), "PFT_10.24": float(Q10.mean())},
                        "R_short": bt.ci(Q3 - P3), "R_native": bt.ci(Q10 - P10), "J": bt.ci(j96),
                        "R_c_short": bt.ci((Q3 - fQ3) - (P3 - fP3)), "R_c_native": bt.ci((Q10 - fQ10) - (P10 - fP10)),
                        "W_short": float(np.mean(Q3 > P3)), "W_native": float(np.mean(Q10 > P10))}
        out["armd80_J_frozen"] = {"point": float(j80.mean()), "n": 80}
        out["pooled176_J"] = Boot(NS + "|E8|176", 176).ci(np.concatenate([j80, j96]))
        pj = out["pooled176_J"]
        out["reading"] = reading([("RESOLVED POSITIVE", pj["lo"] > 0), ("REVERSAL", pj["hi"] < 0),
                                  ("DIRECTIONAL, UNDERPOWERED", pj["lo"] <= 0 <= pj["hi"])])
    elif exp == "E1c":
        P15, fP15, _ = per_prompt(Bj["pruned2_A__ac_d384"]); Q15, fQ15, _ = per_prompt(Bj["recovered2__ac_d384"])
        P10 = frozen_pp("pruned2_A__ac_native", 96); Q10 = frozen_pp("recovered2__ac_native", 96)
        fP10 = np.asarray(fro["pruned2_A__ac_native"]["floor_item"], float)[:96]; fQ10 = np.asarray(fro["recovered2__ac_native"]["floor_item"], float)[:96]
        bt = Boot(NS + "|E1c", 96)
        out["cells"] = {"15.36": {"P": float(P15.mean()), "PFT": float(Q15.mean()), "floors": {"P": float(fP15.mean()), "PFT": float(fQ15.mean())},
                                  "R": bt.ci(Q15 - P15), "R_c": bt.ci((Q15 - fQ15) - (P15 - fP15)), "W": float(np.mean(Q15 > P15))},
                        "10.24_first96": {"P": float(P10.mean()), "PFT": float(Q10.mean()), "R": bt.ci(Q10 - P10)}}
        out["D4"] = bt.ci((Q15 - P15) - (Q10 - P10)); out["D4_c"] = bt.ci(((Q15 - fQ15) - (P15 - fP15)) - ((Q10 - fQ10) - (P10 - fP10)))
        d4 = out["D4"]
        out["reading"] = reading([("PEAKED AT THE FINE-TUNING DURATION", d4["hi"] < 0), ("STILL INCREASING", d4["lo"] > 0),
                                  ("PLATEAU", d4["lo"] <= 0 <= d4["hi"] and abs(d4["point"]) < SESOI)])
    elif exp == "B":
        T3, fT3, _ = per_prompt(A["textft__ac_short"]); T10, fT10, _ = per_prompt(A["textft__ac_native"])
        D3 = frozen_pp("dense__ac_short", 96); D10 = frozen_pp("dense__ac_native", 96)
        P3 = frozen_pp("pruned2_A__ac_short", 96); P10 = frozen_pp("pruned2_A__ac_native", 96)
        bt = Boot(NS + "|B", 96)
        out["levels"] = {"textft_3.84": float(T3.mean()), "textft_10.24": float(T10.mean()), "dense_3.84": float(D3.mean()), "dense_10.24": float(D10.mean()),
                         "floors_textft": {"3.84": float(fT3.mean()), "10.24": float(fT10.mean())}}
        out["G_tf"] = {"3.84": bt.ci(T3 - D3), "10.24": bt.ci(T10 - D10)}
        out["J_tf"] = bt.ci((T10 - D10) - (T3 - D3))
        out["s_textft"] = bt.ci(T10 - T3); out["s_dense_first96"] = bt.ci(D10 - D3)
        out["textft_minus_P"] = {"3.84": bt.ci(T3 - P3), "10.24": bt.ci(T10 - P10)}
        j = out["J_tf"]
        out["reading"] = reading([("TEXT-FT GAINS MORE AT 10.24 s", j["lo"] > 0), ("TEXT-FT GAINS MORE AT 3.84 s", j["hi"] < 0),
                                  ("DURATION-NEUTRAL", j["lo"] <= 0 <= j["hi"])])
        out["role"] = "public dense text-FT REFERENCE; NOT Singh's deleted dense-FT, NOT recipe-matched, NOT a causal control"
    elif exp == "E3":
        S3, fS3, _ = per_prompt(SF["shortft__ac_short"]); S10, fS10, _ = per_prompt(SF["shortft__ac_native"])
        P3 = frozen_pp("pruned2_A__ac_short"); P10 = frozen_pp("pruned2_A__ac_native")
        Q3 = frozen_pp("recovered2__ac_short"); Q10 = frozen_pp("recovered2__ac_native")
        bt = Boot(NS + "|E3", 192)
        out["levels"] = {"shortft_3.84": float(S3.mean()), "shortft_10.24": float(S10.mean()), "P_3.84": float(P3.mean()), "P_10.24": float(P10.mean()),
                         "floors_shortft": {"3.84": float(fS3.mean()), "10.24": float(fS10.mean())}}
        out["R_sf"] = {"3.84": bt.ci(S3 - P3), "10.24": bt.ci(S10 - P10)}
        out["J_sf"] = bt.ci((S10 - P10) - (S3 - P3))
        out["released_R"] = {"3.84": bt.ci(Q3 - P3), "10.24": bt.ci(Q10 - P10)}; out["released_J"] = bt.ci((Q10 - P10) - (Q3 - P3))
        out["shortft_minus_released"] = {"3.84": bt.ci(S3 - Q3), "10.24": bt.ci(S10 - Q10)}
        j = out["J_sf"]; r3 = out["R_sf"]["3.84"]; r10 = out["R_sf"]["10.24"]
        out["reading"] = reading([("UNINFORMATIVE", r3["lo"] <= 0 <= r3["hi"] and r10["lo"] <= 0 <= r10["hi"]),
                                  ("SPECIALISATION-CONSISTENT (gains more at 3.84 s)", j["hi"] < 0),
                                  ("LONGER-IS-EASIER (still gains more at 10.24 s)", j["lo"] > 0),
                                  ("DURATION-NEUTRAL GAIN", j["lo"] <= 0 <= j["hi"] and r3["lo"] > 0)])
        tr = f"artifacts/icassp_gate0/r2_shortft/trainer_report.json"
        if os.path.exists(tr):
            out["trainer_report"] = json.load(open(tr))["report"]
    else:
        raise SystemExit(f"unknown experiment {exp}")
    for job in ("a", "b", "c", "shortft"):
        if os.path.exists(gin(job)):
            out.setdefault("inputs", {})[gin(job)] = sha(gin(job)); out["inputs"][gout(job)] = sha(gout(job)) if os.path.exists(gout(job)) else None
    outp = f"configs/research/r2_{exp}_result.json"
    js = json.dumps(out, indent=1, sort_keys=True); out["artifact_sha256"] = hashlib.sha256(js.encode()).hexdigest()
    json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k in ("reading", "reading_10.24", "J_clo", "J_sf", "J_tf", "D4", "pooled176_J", "J_music_127")}, indent=1))
    print("wrote", outp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true"); ap.add_argument("--score", action="store_true")
    ap.add_argument("--clotho-refs", action="store_true"); ap.add_argument("--gate0-dense", action="store_true")
    ap.add_argument("--verdict", default=None); ap.add_argument("--job", default=None); ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.emit:
        emit(a.job, a.root)
    if a.score:
        score(a.job)
    if a.clotho_refs:
        clotho_refs()
    if a.gate0_dense:
        gate0_dense()
    if a.verdict:
        verdict(a.verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
