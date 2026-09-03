#!/usr/bin/env python3
"""XSEV-DENSE-192-CONTROL — structural validation, scoring and verdict for the paired dense control on the
severity-2 192-prompt set (CPU, 0 cr). Protocol: docs/xsev_dense_192_control.md (frozen before generation).

  --emit    : validate the 384 dense WAVs (+ 4 device-check clips) and write the scorer groups
              (two 192-item groups in prompt_index order; the frozen P / P+FT cells are NOT re-scored)
  --score   : frozen fused-CLAP convention (rev 365dea6e; one seed-once call per group), text + audio
              embeddings -> matched cosines AND the Draft-5 shuffled-caption chance floor per clip
  --verdict : paired quantities (unit = prompt, n = 192, percentile bootstrap B = 10000, seed namespace
              "XSEV-DENSE-192-CONTROL|BOOTSTRAP|2026-09-03"); guards reproduce the frozen severity-2
              points (R_short, R_native, J) from the committed per-item cosines; device check reported.

Estimands (no gate; every outcome reportable):
  s(dense); s(P) - s(dense); s(P+FT) - s(dense)   [paired per prompt]
  rho_short, rho_native = R_op / (dense_op - P_op)  [ratio of means, paired bootstrap]
  G_short(P+FT), G_native(P+FT) = dense - P+FT     [paired]; TOST at +-0.025 reported for the native gap
  floor-corrected versions of all of the above (a = matched - own floor)
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np

NS = "XSEV-DENSE-192-CONTROL|BOOTSTRAP|2026-09-03"
SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000
TMP = "artifacts/icassp_gate0/_score_tmp"
GROUPS_IN = f"{TMP}/xsev_dense192_groups_in.json"
GROUPS_OUT = f"{TMP}/xsev_dense192_groups_out.json"
FROZEN_OUT = f"{TMP}/xsev_sev2_groups_out.json"          # frozen per-item cosines of the sev-2 cells
FROZEN_IN = f"{TMP}/xsev_sev2_groups_in.json"
FLOOR_OUT = f"{TMP}/draft5_floor_groups_out.json"        # Draft-5 floors of the frozen cells
OUT = "configs/research/xsev_dense_192_control_result.json"
AC192 = "configs/research/xsev_audiocaps_manifest.json"
XSEV = "configs/research/xsev_result.json"
DDC = "configs/research/draft4_dense_duration_control_result.json"
PROTOCOL = "docs/xsev_dense_192_control.md"
GEN_ROOT_DEFAULT = "artifacts/icassp_gate0/xsev_dense_192_gen"
N_NATIVE, N_SHORT = 163872, 61472


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def prompts():
    return sorted(json.load(open(AC192))["prompts"], key=lambda p: p["prompt_index"])


# ----------------------------------------------------------------------------------------------- emit
def emit(root):
    import soundfile as sf
    ps = prompts()
    groups, prov = [], {}
    for ctx, n_exp in [("ac_short", N_SHORT), ("ac_native", N_NATIVE)]:
        man_p = f"{root}/gen_manifest_dense_{ctx}.json"
        if not os.path.exists(man_p):
            raise SystemExit(f"missing generation manifest {man_p}")
        man = json.load(open(man_p))
        rows = {r["prompt_index"]: r for r in man["rows"]}
        items = []
        for p in ps:
            w = f"{root}/dense_{ctx}_p{p['prompt_index']}_r0.wav"
            if not os.path.exists(w):
                raise SystemExit(f"missing WAV {w}")
            info = sf.info(w)
            if info.frames != n_exp or info.samplerate != 16000:
                raise SystemExit(f"bad WAV {w}: frames {info.frames} sr {info.samplerate}")
            r = rows[p["prompt_index"]]
            if r["ytid"] != p["ytid"] or sha(w) != r["wav_sha256"]:
                raise SystemExit(f"manifest mismatch for {w}")
            items.append({"caption": p["caption"], "wav": w})
        devices = sorted({r["device"] for r in man["rows"]})
        prov[ctx] = {"gen_manifest_sha256": sha(man_p), "n": man["n"], "devices": devices,
                     "git_sha": man["provenance"].get("git_sha") or man["provenance"].get("commit"),
                     "checkpoint": man["provenance"].get("checkpoint_convention"), "recipe": man["recipe"]}
        if len(devices) != 1 or not devices[0].startswith("cuda"):
            raise SystemExit(f"device rule violated for {ctx}: {devices}")
        groups.append({"name": f"dense__{ctx}", "items": items})
    # device check: 4 pruned2_A native clips regenerated in the job vs the frozen clips
    dc_root = f"{root}/device_check"
    dc_items = []
    fro = {g["name"]: g for g in json.load(open(FROZEN_IN))["groups"]}["pruned2_A__ac_native"]["items"]
    for i in range(4):
        w = f"{dc_root}/pruned2_A_ac_native_p{i}_r0.wav"
        if os.path.exists(w):
            dc_items.append({"caption": ps[i]["caption"], "wav": w, "frozen_wav": fro[i]["wav"]})
    if dc_items:
        groups.append({"name": "device_check__pruned2_A_native_p0-3", "items": [{"caption": it["caption"], "wav": it["wav"]} for it in dc_items]})
        groups.append({"name": "device_check_frozen__pruned2_A_native_p0-3", "items": [{"caption": it["caption"], "wav": it["frozen_wav"]} for it in dc_items]})
    json.dump({"groups": groups, "generation_provenance": prov, "protocol_doc_sha256": sha(PROTOCOL),
               "convention": "each group = ONE seed-once fused-CLAP call in prompt_index order (rev 365dea6e); the device-check "
                             "groups are scored as 4-item calls (same convention for both; only their difference is used)"},
              open(GROUPS_IN, "w"), indent=1)
    print(f"emitted {len(groups)} groups -> {GROUPS_IN}; devices {prov['ac_short']['devices']} / {prov['ac_native']['devices']}")


# ---------------------------------------------------------------------------------------------- score
def score():
    from gate0_clap_scorer import FusedClapScorer, _prov
    din = json.load(open(GROUPS_IN))
    sc = FusedClapScorer()
    results = []
    for g in din["groups"]:
        t0 = time.time()
        caps = [it["caption"] for it in g["items"]]; wavs = [it["wav"] for it in g["items"]]
        te = sc._l2(sc.text_embed(caps)); ae = sc._l2(sc.audio_embed(wavs))
        M = te @ ae.T
        diag = np.diag(M).astype(float)
        cap_arr = np.asarray(caps); other = cap_arr[:, None] != cap_arr[None, :]
        floor = np.array([M[other[:, j], j].mean() if other[:, j].any() else np.nan for j in range(len(caps))], dtype=float)
        results.append({"name": g["name"], "n": len(caps), "cosines": diag.tolist(), "floor_item": floor.tolist(),
                        "matched_mean": float(diag.mean()), "floor_mean": float(np.nanmean(floor)), "seconds": round(time.time() - t0, 1)})
        print(f"{g['name']:44s} n={len(caps):3d} matched {diag.mean():.4f} floor {np.nanmean(floor):.4f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"results": results, "scorer_provenance": _prov(), "groups_in_sha256": sha(GROUPS_IN)}, open(GROUPS_OUT, "w"), indent=1)
    print("scored ->", GROUPS_OUT)


# -------------------------------------------------------------------------------------------- verdict
class Boot:
    def __init__(self, rng, n):
        self.idx = rng.integers(0, n, (B, n)); self.n = n

    def ci(self, v):
        v = np.asarray(v, float); bm = v[self.idx].mean(1); lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(self.n), "boot_frac_le0": float(np.mean(bm <= 0))}

    def ratio(self, num, den):
        num = np.asarray(num, float); den = np.asarray(den, float)
        bm = num[self.idx].mean(1) / den[self.idx].mean(1); lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": float(num.mean() / den.mean()), "lo": float(lo), "hi": float(hi), "n": int(self.n)}


def verdict():
    din = json.load(open(GROUPS_IN)); dout = json.load(open(GROUPS_OUT))
    res = {r["name"]: r for r in dout["results"]}
    fro = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(FROZEN_OUT))["results"]}
    flo = {r["name"]: r for r in json.load(open(FLOOR_OUT))["results"]}
    d_sh, d_na = np.asarray(res["dense__ac_short"]["cosines"]), np.asarray(res["dense__ac_native"]["cosines"])
    f_dsh, f_dna = np.asarray(res["dense__ac_short"]["floor_item"]), np.asarray(res["dense__ac_native"]["floor_item"])
    p_sh, p_na = fro["pruned2_A__ac_short"], fro["pruned2_A__ac_native"]
    q_sh, q_na = fro["recovered2__ac_short"], fro["recovered2__ac_native"]
    f_psh, f_pna = np.asarray(flo["pruned2_A__ac_short"]["floor_item"]), np.asarray(flo["pruned2_A__ac_native"]["floor_item"])
    f_qsh, f_qna = np.asarray(flo["recovered2__ac_short"]["floor_item"]), np.asarray(flo["recovered2__ac_native"]["floor_item"])
    for a in (d_sh, d_na, p_sh, p_na, q_sh, q_na):
        assert a.shape == (192,), a.shape
    xs = json.load(open(XSEV))["PRIMARY_A"]
    guards = {"R_short": (float((q_sh - p_sh).mean()), xs["R_short"]["point"]),
              "R_native": (float((q_na - p_na).mean()), xs["R_native"]["point"]),
              "J": (float(((q_na - p_na) - (q_sh - p_sh)).mean()), xs["J"]["point"])}
    worst = max(abs(a - b) for a, b in guards.values())
    if worst > 1e-9:
        raise SystemExit(f"guard FAILED (frozen sev-2 points not reproduced): {guards}")
    rng = np.random.default_rng(np.random.PCG64(SEED)); bt = Boot(rng, 192)
    a = lambda m, f: m - f
    out = {"artifact": "xsev_dense_192_control_result", "protocol_doc": PROTOCOL, "protocol_doc_sha256": sha(PROTOCOL),
           "class": "prospective design completion (frozen before generation; NO gate); paired dense control on the severity-2 192 prompts",
           "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": SEED, "unit": "prompt", "ci": "percentile 95%"},
           "scorer": dout["scorer_provenance"], "generation_provenance": din["generation_provenance"],
           "inputs": {GROUPS_IN: sha(GROUPS_IN), GROUPS_OUT: sha(GROUPS_OUT), FROZEN_OUT: sha(FROZEN_OUT), FLOOR_OUT: sha(FLOOR_OUT), XSEV: sha(XSEV), DDC: sha(DDC)},
           "consistency_guards": {k: {"here": x, "frozen": y} for k, (x, y) in guards.items()}, "consistency_guards_max_abs_diff": worst,
           "means": {"dense_short": float(d_sh.mean()), "dense_native": float(d_na.mean()), "pruned_short": float(p_sh.mean()),
                     "pruned_native": float(p_na.mean()), "postft_short": float(q_sh.mean()), "postft_native": float(q_na.mean()),
                     "dense_floor_short": float(f_dsh.mean()), "dense_floor_native": float(f_dna.mean())},
           "PRIMARY": {
               "s_dense": bt.ci(d_na - d_sh), "s_pruned": bt.ci(p_na - p_sh), "s_postft": bt.ci(q_na - q_sh),
               "s_pruned_minus_s_dense": bt.ci((p_na - p_sh) - (d_na - d_sh)),
               "s_postft_minus_s_dense": bt.ci((q_na - q_sh) - (d_na - d_sh)),
               "rho_short": bt.ratio(q_sh - p_sh, d_sh - p_sh), "rho_native": bt.ratio(q_na - p_na, d_na - p_na),
               "G_short_pruned": bt.ci(d_sh - p_sh), "G_native_pruned": bt.ci(d_na - p_na),
               "G_short_postft": bt.ci(d_sh - q_sh), "G_native_postft": bt.ci(d_na - q_na),
           },
           "FLOOR_CORRECTED": {
               "s_c_dense": bt.ci(a(d_na, f_dna) - a(d_sh, f_dsh)), "s_c_pruned": bt.ci(a(p_na, f_pna) - a(p_sh, f_psh)),
               "s_c_postft": bt.ci(a(q_na, f_qna) - a(q_sh, f_qsh)),
               "rho_c_short": bt.ratio(a(q_sh, f_qsh) - a(p_sh, f_psh), a(d_sh, f_dsh) - a(p_sh, f_psh)),
               "rho_c_native": bt.ratio(a(q_na, f_qna) - a(p_na, f_pna), a(d_na, f_dna) - a(p_na, f_pna)),
               "G_c_native_postft": bt.ci(a(d_na, f_dna) - a(q_na, f_qna)),
               "dense_floor_short": bt.ci(f_dsh), "dense_floor_native": bt.ci(f_dna),
           }}
    gN = out["PRIMARY"]["G_native_postft"]
    out["PRIMARY"]["tost_native_postft_pm0.025"] = {"equivalent": bool(gN["lo"] > -0.025 and gN["hi"] < 0.025),
                                                     "note": "restored-to-dense wording allowed only if the 95% CI of dense - P+FT lies within +-0.025"}
    # cross-set comparison with the severity-1 dense control (descriptive)
    ddc = json.load(open(DDC))
    out["crossset_vs_sev1_dense"] = {"s_dense_sev1_80": ddc["slopes"]["dense"]["point"], "dense_short_sev1": ddc["means"]["dense_short"],
                                     "dense_native_sev1": ddc["means"]["dense_native"], "note": "descriptive; different prompt sets"}
    # device check
    if "device_check__pruned2_A_native_p0-3" in res:
        new = np.asarray(res["device_check__pruned2_A_native_p0-3"]["cosines"]); old = np.asarray(res["device_check_frozen__pruned2_A_native_p0-3"]["cosines"])
        out["device_check"] = {"n": int(len(new)), "abs_diff_per_clip": [float(x) for x in np.abs(new - old)], "max_abs_diff": float(np.max(np.abs(new - old))),
                               "note": "4 pruned2_A native clips regenerated in the dense job vs the frozen clips, both scored as 4-item calls; descriptive (expected < 0.01)"}
    json.dump(out, open(OUT, "w"), indent=1)
    f = lambda c: f"{c['point']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]"
    print("means:", {k: round(v, 4) for k, v in out["means"].items()})
    for k, v in out["PRIMARY"].items():
        if isinstance(v, dict) and "point" in v: print(f"  {k:26s} {f(v)}")
    print("  TOST native:", out["PRIMARY"]["tost_native_postft_pm0.025"]["equivalent"])
    for k, v in out["FLOOR_CORRECTED"].items():
        print(f"  {k:26s} {f(v)}")
    if "device_check" in out: print("  device check max|dCLAP| =", round(out["device_check"]["max_abs_diff"], 4))
    print(f"guards max |diff| {worst:.2e}; wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true"); ap.add_argument("--score", action="store_true"); ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--root", default=GEN_ROOT_DEFAULT, help="directory holding the dense WAVs + gen manifests (job artifact dir)")
    a_ = ap.parse_args()
    if a_.emit: emit(a_.root)
    if a_.score: score()
    if a_.verdict: verdict()
    if not (a_.emit or a_.score or a_.verdict): ap.print_help()
