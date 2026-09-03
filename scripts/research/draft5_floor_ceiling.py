#!/usr/bin/env python3
"""DRAFT5-FLOOR-CEILING — chance floor (shuffled captions) and real-audio ceiling for every reported
CLAP cell, under the FROZEN scoring convention (CPU, 0 cr, no generation).

POST-HOC SENSITIVITY / ANCHOR ANALYSIS (Gabriel review request 2026-09-02, manuscript Draft-5 pass).
Changes no frozen verdict, gate, or raw score.

Why. Every absolute CLAP level in the paper (e.g. P at 83 % pruning: 0.015 at 3.84 s, 0.055 at 10.24 s)
is read against an implicit zero, and every duration response s(.) bundles a possible SCORER effect (fused
CLAP repeat-pads 3.84 s audio and centre-crops 10.24 s audio) with a generation effect. Two anchors settle
both, at zero cost:
  * CHANCE FLOOR: the mean cosine between a clip and the OTHER captions of the same battery (shuffled-caption
    baseline). A cell whose matched cosine equals its floor is unaligned; a floor that moves with duration
    is a scorer/level effect and must be subtracted before s(.), R, J are interpreted.
  * REAL-AUDIO CEILING: the real AudioCaps reference clip of each prompt (band-limited to the generators'
    16 kHz), scored at its full 10 s and as its first 3.84 s under the identical convention; s(real) is the
    scorer + content-window duration response with NO generation involved.

How the floor is obtained WITHOUT changing a single frozen score. Text embeddings are batch-invariant and
audio embeddings are reproduced exactly by the frozen convention (one seed-once fixed-order call per
group, rev 365dea6e, transformers 4.30.2). For each frozen group we recompute BOTH embedding sets in the
frozen arrangement, form the full text x audio cosine matrix, and (guard) require its diagonal to equal the
frozen per-item cosines (max |diff| < 1e-6, else abort). The off-diagonal entries are the floor: for audio
j, floor_j = mean_i cos(text_i, audio_j) over captions i of a DIFFERENT prompt (replicates of the same
prompt are excluded). Floor-corrected alignment a_j = matched_j - floor_j.

Cells (all existing WAVs; groups re-emitted item-for-item from the frozen groups_in files):
  sev-2 (n=192): recovered2 / pruned2_A / pruned2_B x {ac_native, ac_short, music(64x3)}; music_native (64).
  sev-1 (n=80, Arm-D subset order): dense / P / P+FT at 10.24 s (xsev dense groups) and at 3.84 s (V1.1 r0
         WAVs; dense group = Draft-4 control, P and P+FT groups re-emitted from the same V1.1 WAVs — their
         diagonals must reproduce the frozen Arm-D raw_cosines); sev-1 music 3.84 s (64x3, phenomenon job).
  crops: the 5 native_crop groups (first 3.84 s of the 10.24 s generations).
  real:  real AudioCaps clips of the sev-2 192 and sev-1 80 prompts, full (<=10 s) and first 3.84 s
         (61472 samples at 16 kHz, the generated short-clip length), written under artifacts/ (gitignored).

Quantities (unit = prompt; music replicates averaged first; percentile bootstrap B = 10000, 95 %):
  per cell: matched mean (guard = frozen), floor mean, above-chance mean a = matched - floor.
  sev-1 (paired, n=80): s_raw(sys) (guard = frozen), s_c(sys) floor-corrected, floor shift per system,
         R_c(short/native), J_c, recovery ratio rho = R/(dense - P) raw (guard = Draft-4) and corrected,
         real ceiling at both durations, s(real).
  sev-2 (paired, n=192): s_c(P), s_c(P+FT), R_c, J_c, floors, real ceiling on the 192, s(real);
         cross-set descriptive rho using the sev-1 dense means (point only; labelled cross-set).
  music (n=64, both severities): floors, above-chance, R_c.
  caption tokens: fraction of music captions exceeding the 77-token limit of AudioLDM's CLAP text
         conditioner (RoBERTa tokenizer) vs AudioCaps.

Seed namespace "DRAFT5-FLOOR-CEILING|BOOTSTRAP|2026-09-02" -> PCG64(int(sha256(ns)[:8],16) % 2**31).

Run:
  OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/draft5_floor_ceiling.py --emit
  OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/draft5_floor_ceiling.py --score
  OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/draft5_floor_ceiling.py --verdict
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np

NS = "DRAFT5-FLOOR-CEILING|BOOTSTRAP|2026-09-02"
SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000
TMP = "artifacts/icassp_gate0/_score_tmp"
REAL_DIR = "artifacts/icassp_gate0/real_refs"
GROUPS_IN = f"{TMP}/draft5_floor_groups_in.json"
GROUPS_OUT = f"{TMP}/draft5_floor_groups_out.json"
OUT = "configs/research/draft5_floor_ceiling_result.json"
ZIP = "data/dataset/audioset/zip_audios"
V11_GEN = "/teamspace/jobs/reversal-v11-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_v1_1_gen"
CROP_SAMPLES = 61472          # = generated 3.84 s clip length at 16 kHz (native_crop_analysis convention)
SR_GEN = 16000

SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
OPD = "configs/research/op_duration_discriminator_1_result.json"
XSEV = "configs/research/xsev_result.json"
AC192 = "configs/research/xsev_audiocaps_manifest.json"
MUS64 = "configs/research/xsev_music_manifest.json"
DDC = "configs/research/draft4_dense_duration_control_result.json"
CROP = "configs/research/native_crop_analysis_result.json"
MUSN = "configs/research/xsev_music_native_1_result.json"
RMUS1 = "configs/research/reversal_v1_r_music_clap.json"

# frozen groups reused item-for-item (file, group name, key under which frozen cosines live)
FROZEN = [
    (f"{TMP}/xsev_sev2_groups_in.json", f"{TMP}/xsev_sev2_groups_out.json",
     ["recovered2__ac_native", "recovered2__ac_short", "recovered2__music",
      "pruned2_A__ac_native", "pruned2_A__ac_short", "pruned2_A__music",
      "pruned2_B__ac_native", "pruned2_B__ac_short", "pruned2_B__music"]),
    (f"{TMP}/music_native_groups_in.json", f"{TMP}/music_native_groups_out.json",
     ["recovered2__music_native", "pruned2_A__music_native"]),
    (f"{TMP}/xsev_dense_groups_in.json", f"{TMP}/xsev_dense_groups_out.json",
     ["dense10s__dense", "dense10s__pruned_sev1", "dense10s__recovered_sev1"]),
    (f"{TMP}/draft4_dense_short_groups_in.json", f"{TMP}/draft4_dense_short_groups_out.json",
     ["dense_short_sev1__armd80"]),
    (f"{TMP}/native_crop_groups_in.json", f"{TMP}/native_crop_groups_out.json",
     ["crop_sev1__pruned", "crop_sev1__recovered", "crop_sev2__recovered2", "crop_sev2__pruned2_A",
      "crop_sev2__pruned2_B"]),
    ("artifacts/icassp_gate0/_phenom_groups_in.json", "artifacts/icassp_gate0/_phenom_groups_out.json",
     ["p1_recovered__off", "p1_pruned_ema_reconstructed__off"]),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def subset():
    return sorted(json.load(open(SUBSET))["prompts"], key=lambda p: p["subset_prompt_index"])


def find_ref(ytid):
    for pat in (f"{ZIP}/**/Y{ytid}.wav", f"{ZIP}/**/{ytid}.wav"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    raise SystemExit(f"reference wav not found for {ytid}")


# ----------------------------------------------------------------------------------------------- emit
def emit():
    import librosa, soundfile as sf
    groups = []
    frozen_ref = {}
    for gin, gout, names in FROZEN:
        din = json.load(open(gin)); dout = json.load(open(gout))
        byname = {g["name"]: g for g in din["groups"]}
        cos = {r["name"]: r["cosines"] for r in dout["results"]}
        for nm in names:
            items = [{"caption": it["caption"], "wav": it["wav"]} for it in byname[nm]["items"]]
            for it in items:
                if not os.path.exists(it["wav"]):
                    raise SystemExit(f"missing WAV {it['wav']}")
            groups.append({"name": nm, "items": items, "frozen_from": gout, "n_frozen": len(cos[nm])})
            frozen_ref[nm] = cos[nm]
    # sev-1 P and P+FT at 3.84 s on the Arm-D 80 in subset order (V1.1 r0 WAVs) -> guard vs raw_cosines
    prompts = subset()
    for nm, stem, key in [("pruned_short_sev1__armd80", "p1_pruned_ema_reconstructed_noadapter", "pruned_ctrl"),
                          ("postft_short_sev1__armd80", "p1_recovered_noadapter", "recovered_ctrl")]:
        items = []
        for p in prompts:
            w = f"{V11_GEN}/{stem}_p{p['v1_1_prompt_index']}_r0.wav"
            if not os.path.exists(w):
                raise SystemExit(f"missing V1.1 WAV {w}")
            items.append({"caption": p["caption"], "wav": w})
        groups.append({"name": nm, "items": items, "frozen_from": OPD, "n_frozen": 80})
        frozen_ref[nm] = json.load(open(OPD))["raw_cosines"][key]
    # real AudioCaps references (band-limited to 16 kHz), full and first 3.84 s
    os.makedirs(REAL_DIR, exist_ok=True)
    real_meta = {}
    for setname, plist, ikey in [("sev2_192", json.load(open(AC192))["prompts"], "prompt_index"),
                                 ("sev1_80", prompts, "subset_prompt_index")]:
        full_items, crop_items, durs = [], [], []
        for p in plist:
            src = find_ref(p["ytid"])
            w, _ = librosa.load(src, sr=SR_GEN, mono=True)
            w = w.astype(np.float32)
            durs.append(len(w) / SR_GEN)
            f_full = f"{REAL_DIR}/real_{setname}_full_p{p[ikey]}.wav"
            f_crop = f"{REAL_DIR}/real_{setname}_crop_p{p[ikey]}.wav"
            sf.write(f_full, w, SR_GEN, subtype="PCM_16")
            sf.write(f_crop, w[:CROP_SAMPLES], SR_GEN, subtype="PCM_16")
            full_items.append({"caption": p["caption"], "wav": f_full, "src": src})
            crop_items.append({"caption": p["caption"], "wav": f_crop, "src": src})
        groups.append({"name": f"real_full__{setname}", "items": full_items, "frozen_from": None, "n_frozen": 0})
        groups.append({"name": f"real_crop__{setname}", "items": crop_items, "frozen_from": None, "n_frozen": 0})
        real_meta[setname] = {"n": len(plist), "duration_s_min": float(min(durs)), "duration_s_median": float(np.median(durs)),
                              "n_shorter_than_9s": int(sum(d < 9.0 for d in durs)), "n_shorter_than_crop": int(sum(d < CROP_SAMPLES / SR_GEN for d in durs))}
    json.dump({"groups": groups,
               "convention": "each group = ONE seed-once fused-CLAP call in the frozen item order (rev 365dea6e); "
                             "text x audio cosine matrix; diagonal must reproduce the frozen cosines",
               "real_refs": {"sr": SR_GEN, "crop_samples": CROP_SAMPLES, "meta": real_meta},
               "frozen_reference_cosines": frozen_ref},
              open(GROUPS_IN, "w"), indent=1)
    print(f"emitted {len(groups)} groups, {sum(len(g['items']) for g in groups)} items -> {GROUPS_IN}")
    print(json.dumps(real_meta, indent=1))


# ---------------------------------------------------------------------------------------------- score
def score():
    from gate0_clap_scorer import FusedClapScorer, _prov
    din = json.load(open(GROUPS_IN))
    frozen = din["frozen_reference_cosines"]
    sc = FusedClapScorer()
    results = []
    worst_all = 0.0
    for g in din["groups"]:
        t0 = time.time()
        caps = [it["caption"] for it in g["items"]]; wavs = [it["wav"] for it in g["items"]]
        te = sc._l2(sc.text_embed(caps))
        ae = sc._l2(sc.audio_embed(wavs))                 # seed-once inside (frozen convention)
        M = te @ ae.T                                      # M[i, j] = cos(text_i, audio_j)
        diag = np.diag(M).astype(float)
        # prompt identity = caption string (replicates of one prompt share the caption)
        cap_arr = np.asarray(caps)
        other = cap_arr[:, None] != cap_arr[None, :]       # other[i, j]: caption i belongs to a different prompt than j
        floor = np.array([M[other[:, j], j].mean() for j in range(len(caps))], dtype=float)
        guard = None
        if g["name"] in frozen:
            ref = np.asarray(frozen[g["name"]], float)
            guard = float(np.max(np.abs(ref - diag)))
            worst_all = max(worst_all, guard)
            if guard > 1e-6:
                raise SystemExit(f"GUARD FAILED for {g['name']}: max|diag - frozen| = {guard:.3e}")
        results.append({"name": g["name"], "n": len(caps), "cosines": diag.tolist(), "floor_item": floor.tolist(),
                        "matched_mean": float(diag.mean()), "floor_mean": float(floor.mean()),
                        "n_unique_prompts": int(len(set(caps))), "guard_max_abs_diff_vs_frozen": guard,
                        "seconds": round(time.time() - t0, 1)})
        print(f"{g['name']:34s} n={len(caps):3d} matched {diag.mean():.4f} floor {floor.mean():.4f} "
              f"guard {'-' if guard is None else f'{guard:.1e}'}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"results": results, "scorer_provenance": _prov(), "worst_guard": worst_all,
               "groups_in_sha256": sha(GROUPS_IN)}, open(GROUPS_OUT, "w"), indent=1)
    print(f"scored {len(results)} groups -> {GROUPS_OUT}; worst guard {worst_all:.2e}")


# -------------------------------------------------------------------------------------------- verdict
class Boot:
    def __init__(self, rng):
        self.rng = rng; self.idx = {}

    def ci(self, vec_fn, n):
        if n not in self.idx:
            self.idx[n] = self.rng.integers(0, n, (B, n))
        idx = self.idx[n]
        point = float(vec_fn(np.arange(n)).mean())
        boots = np.array([vec_fn(i).mean() for i in idx]) if False else None
        # vectorised: vec_fn must accept an index array and return per-prompt values; use full vector then index
        v = np.asarray(vec_fn(np.arange(n)), float)
        bm = v[idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": point, "lo": float(lo), "hi": float(hi), "n": int(n), "boot_frac_le0": float(np.mean(bm <= 0))}

    def ci_vec(self, v):
        v = np.asarray(v, float)
        return self.ci(lambda i: v, len(v))

    def ci_ratio(self, num, den):
        """ratio of means with bootstrap CI (paired resampling of prompts)."""
        num = np.asarray(num, float); den = np.asarray(den, float); n = len(num)
        if n not in self.idx:
            self.idx[n] = self.rng.integers(0, n, (B, n))
        idx = self.idx[n]
        bm = num[idx].mean(1) / den[idx].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        return {"point": float(num.mean() / den.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


def per_prompt(res, name, reps=1):
    """(matched, floor) per prompt; music groups are (prompt, replicate) ordered -> average replicates."""
    r = res[name]
    m = np.asarray(r["cosines"], float); f = np.asarray(r["floor_item"], float)
    if reps > 1:
        assert len(m) % reps == 0
        m = m.reshape(-1, reps).mean(1); f = f.reshape(-1, reps).mean(1)
    return m, f


def verdict():
    dout = json.load(open(GROUPS_OUT)); din = json.load(open(GROUPS_IN))
    res = {r["name"]: r for r in dout["results"]}
    if dout["worst_guard"] > 1e-6:
        raise SystemExit("guard failure recorded in groups_out")
    rng = np.random.default_rng(np.random.PCG64(SEED)); bt = Boot(rng)
    out = {"artifact": "draft5_floor_ceiling_result",
           "class": "POST-HOC SENSITIVITY / ANCHOR (Draft-5 review pass, 2026-09-02); no new generation; changes no frozen verdict",
           "definitions": {
               "floor": "mean cosine between the clip and the captions of the OTHER prompts of the same battery (shuffled-caption chance level); per clip, then averaged",
               "above_chance": "matched cosine minus the clip's own floor",
               "s_c": "floor-corrected duration response: mean over prompts of a(10.24 s) - a(3.84 s)",
               "R_c": "floor-corrected recovery gain a(P+FT) - a(P)", "J_c": "R_c(native) - R_c(short)",
               "rho": "recovery ratio R / (dense - P) = fraction of the pruned checkpoint's gap to dense closed by fine-tuning (ratio of means)",
               "real": "real AudioCaps reference clip of the prompt, band-limited to 16 kHz; full (<=10 s) and first 3.84 s (61472 samples)"},
           "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": SEED, "unit": "prompt", "ci": "percentile 95%"},
           "scorer": dout["scorer_provenance"], "worst_guard_max_abs_diff_vs_frozen": dout["worst_guard"],
           "inputs": {GROUPS_IN: sha(GROUPS_IN), GROUPS_OUT: sha(GROUPS_OUT), OPD: sha(OPD), XSEV: sha(XSEV), DDC: sha(DDC),
                      CROP: sha(CROP), MUSN: sha(MUSN), RMUS1: sha(RMUS1)},
           "real_refs": din["real_refs"]}

    # ---- per-cell table --------------------------------------------------------------------------
    cells = {}
    spec = [  # name, reps
        ("pruned2_A__ac_short", 1), ("recovered2__ac_short", 1), ("pruned2_A__ac_native", 1), ("recovered2__ac_native", 1),
        ("pruned2_B__ac_short", 1), ("pruned2_B__ac_native", 1),
        ("pruned2_A__music", 3), ("recovered2__music", 3), ("pruned2_B__music", 3),
        ("pruned2_A__music_native", 1), ("recovered2__music_native", 1),
        ("pruned_short_sev1__armd80", 1), ("postft_short_sev1__armd80", 1), ("dense_short_sev1__armd80", 1),
        ("dense10s__pruned_sev1", 1), ("dense10s__recovered_sev1", 1), ("dense10s__dense", 1),
        ("p1_pruned_ema_reconstructed__off", 3), ("p1_recovered__off", 3),
        ("crop_sev1__pruned", 1), ("crop_sev1__recovered", 1), ("crop_sev2__pruned2_A", 1), ("crop_sev2__recovered2", 1), ("crop_sev2__pruned2_B", 1),
        ("real_full__sev2_192", 1), ("real_crop__sev2_192", 1), ("real_full__sev1_80", 1), ("real_crop__sev1_80", 1),
    ]
    PP = {}
    for nm, reps in spec:
        m, f = per_prompt(res, nm, reps); PP[nm] = (m, f)
        cells[nm] = {"n_prompts": int(len(m)), "matched_mean": float(m.mean()), "floor": bt.ci_vec(f),
                     "above_chance": bt.ci_vec(m - f), "guard_vs_frozen": res[nm]["guard_max_abs_diff_vs_frozen"]}
    out["cells"] = cells

    def a(nm):
        m, f = PP[nm]; return m - f

    def m_(nm):
        return PP[nm][0]

    def f_(nm):
        return PP[nm][1]

    # ---- severity 1 (n=80, paired across all cells) ----------------------------------------------
    s1 = {}
    for lab, sh, na in [("dense", "dense_short_sev1__armd80", "dense10s__dense"),
                        ("pruned", "pruned_short_sev1__armd80", "dense10s__pruned_sev1"),
                        ("postft", "postft_short_sev1__armd80", "dense10s__recovered_sev1"),
                        ("real", "real_crop__sev1_80", "real_full__sev1_80")]:
        s1[f"s_raw_{lab}"] = bt.ci_vec(m_(na) - m_(sh))
        s1[f"s_c_{lab}"] = bt.ci_vec(a(na) - a(sh))
        s1[f"floor_shift_{lab}"] = bt.ci_vec(f_(na) - f_(sh))
    s1["R_c_short"] = bt.ci_vec(a("postft_short_sev1__armd80") - a("pruned_short_sev1__armd80"))
    s1["R_c_native"] = bt.ci_vec(a("dense10s__recovered_sev1") - a("dense10s__pruned_sev1"))
    s1["J_c"] = bt.ci_vec((a("dense10s__recovered_sev1") - a("dense10s__pruned_sev1")) -
                          (a("postft_short_sev1__armd80") - a("pruned_short_sev1__armd80")))
    s1["R_raw_short_guard"] = float((m_("postft_short_sev1__armd80") - m_("pruned_short_sev1__armd80")).mean())
    s1["R_raw_native_guard"] = float((m_("dense10s__recovered_sev1") - m_("dense10s__pruned_sev1")).mean())
    s1["rho_raw_short"] = bt.ci_ratio(m_("postft_short_sev1__armd80") - m_("pruned_short_sev1__armd80"),
                                      m_("dense_short_sev1__armd80") - m_("pruned_short_sev1__armd80"))
    s1["rho_raw_native"] = bt.ci_ratio(m_("dense10s__recovered_sev1") - m_("dense10s__pruned_sev1"),
                                       m_("dense10s__dense") - m_("dense10s__pruned_sev1"))
    s1["rho_c_short"] = bt.ci_ratio(a("postft_short_sev1__armd80") - a("pruned_short_sev1__armd80"),
                                    a("dense_short_sev1__armd80") - a("pruned_short_sev1__armd80"))
    s1["rho_c_native"] = bt.ci_ratio(a("dense10s__recovered_sev1") - a("dense10s__pruned_sev1"),
                                     a("dense10s__dense") - a("dense10s__pruned_sev1"))
    s1["dense_minus_postft_native_c"] = bt.ci_vec(a("dense10s__dense") - a("dense10s__recovered_sev1"))
    # fraction of the pruned checkpoint's gap to REAL AUDIO closed by fine-tuning (paired; real = same prompts)
    s1["rho_real_short"] = bt.ci_ratio(m_("postft_short_sev1__armd80") - m_("pruned_short_sev1__armd80"),
                                       m_("real_crop__sev1_80") - m_("pruned_short_sev1__armd80"))
    s1["rho_real_native"] = bt.ci_ratio(m_("dense10s__recovered_sev1") - m_("dense10s__pruned_sev1"),
                                        m_("real_full__sev1_80") - m_("dense10s__pruned_sev1"))
    s1["dense_frac_of_real_short"] = bt.ci_ratio(m_("dense_short_sev1__armd80") - m_("pruned_short_sev1__armd80"),
                                                 m_("real_crop__sev1_80") - m_("pruned_short_sev1__armd80"))
    s1["dense_frac_of_real_native"] = bt.ci_ratio(m_("dense10s__dense") - m_("dense10s__pruned_sev1"),
                                                  m_("real_full__sev1_80") - m_("dense10s__pruned_sev1"))
    s1["real_minus_dense_native"] = bt.ci_vec(m_("real_full__sev1_80") - m_("dense10s__dense"))
    s1["real_minus_dense_short"] = bt.ci_vec(m_("real_crop__sev1_80") - m_("dense_short_sev1__armd80"))
    # music sev-1 (n=64)
    s1["music_R_c"] = bt.ci_vec(a("p1_recovered__off") - a("p1_pruned_ema_reconstructed__off"))
    s1["music_R_raw_guard"] = float((m_("p1_recovered__off") - m_("p1_pruned_ema_reconstructed__off")).mean())
    out["sev1_armd80"] = s1

    # ---- severity 2 (n=192 paired; music n=64) ---------------------------------------------------
    s2 = {}
    for lab, sh, na in [("pruned", "pruned2_A__ac_short", "pruned2_A__ac_native"),
                        ("postft", "recovered2__ac_short", "recovered2__ac_native"),
                        ("prunedB", "pruned2_B__ac_short", "pruned2_B__ac_native"),
                        ("real", "real_crop__sev2_192", "real_full__sev2_192")]:
        s2[f"s_raw_{lab}"] = bt.ci_vec(m_(na) - m_(sh))
        s2[f"s_c_{lab}"] = bt.ci_vec(a(na) - a(sh))
        s2[f"floor_shift_{lab}"] = bt.ci_vec(f_(na) - f_(sh))
    s2["R_c_short"] = bt.ci_vec(a("recovered2__ac_short") - a("pruned2_A__ac_short"))
    s2["R_c_native"] = bt.ci_vec(a("recovered2__ac_native") - a("pruned2_A__ac_native"))
    s2["J_c"] = bt.ci_vec((a("recovered2__ac_native") - a("pruned2_A__ac_native")) - (a("recovered2__ac_short") - a("pruned2_A__ac_short")))
    s2["J_c_B"] = bt.ci_vec((a("recovered2__ac_native") - a("pruned2_B__ac_native")) - (a("recovered2__ac_short") - a("pruned2_B__ac_short")))
    s2["R_raw_short_guard"] = float((m_("recovered2__ac_short") - m_("pruned2_A__ac_short")).mean())
    s2["R_raw_native_guard"] = float((m_("recovered2__ac_native") - m_("pruned2_A__ac_native")).mean())
    s2["J_raw_guard"] = float(((m_("recovered2__ac_native") - m_("pruned2_A__ac_native")) - (m_("recovered2__ac_short") - m_("pruned2_A__ac_short"))).mean())
    s2["real_minus_postft_native"] = bt.ci_vec(m_("real_full__sev2_192") - m_("recovered2__ac_native"))
    s2["rho_real_short"] = bt.ci_ratio(m_("recovered2__ac_short") - m_("pruned2_A__ac_short"),
                                       m_("real_crop__sev2_192") - m_("pruned2_A__ac_short"))
    s2["rho_real_native"] = bt.ci_ratio(m_("recovered2__ac_native") - m_("pruned2_A__ac_native"),
                                        m_("real_full__sev2_192") - m_("pruned2_A__ac_native"))
    s2["rho_real_c_short"] = bt.ci_ratio(a("recovered2__ac_short") - a("pruned2_A__ac_short"),
                                         a("real_crop__sev2_192") - a("pruned2_A__ac_short"))
    s2["rho_real_c_native"] = bt.ci_ratio(a("recovered2__ac_native") - a("pruned2_A__ac_native"),
                                          a("real_full__sev2_192") - a("pruned2_A__ac_native"))
    s2["real_minus_postft_short"] = bt.ci_vec(m_("real_crop__sev2_192") - m_("recovered2__ac_short"))
    # crop decomposition, floor-corrected: a(native) - a(crop) = scoring-window part; a(crop) - a(short) = generation-length part
    for lab, sh, cr, na in [("pruned", "pruned2_A__ac_short", "crop_sev2__pruned2_A", "pruned2_A__ac_native"),
                            ("postft", "recovered2__ac_short", "crop_sev2__recovered2", "recovered2__ac_native")]:
        s2[f"window_part_c_{lab}"] = bt.ci_vec(a(na) - a(cr))
        s2[f"length_part_c_{lab}"] = bt.ci_vec(a(cr) - a(sh))
    s2["R_c_crop"] = bt.ci_vec(a("crop_sev2__recovered2") - a("crop_sev2__pruned2_A"))
    # music (n=64), both durations
    s2["music_R_c_short"] = bt.ci_vec(a("recovered2__music") - a("pruned2_A__music"))
    s2["music_R_c_native"] = bt.ci_vec(a("recovered2__music_native") - a("pruned2_A__music_native"))
    s2["music_R_raw_short_guard"] = float((m_("recovered2__music") - m_("pruned2_A__music")).mean())
    s2["music_R_raw_native_guard"] = float((m_("recovered2__music_native") - m_("pruned2_A__music_native")).mean())
    # cross-set descriptive recovery ratio using the sev-1 dense means (point only; different prompt sets)
    dsh = float(m_("dense_short_sev1__armd80").mean()); dna = float(m_("dense10s__dense").mean())
    s2["rho_crossset_descriptive"] = {
        "dense_short_sev1_mean": dsh, "dense_native_sev1_mean": dna,
        "rho_short_point": float((m_("recovered2__ac_short") - m_("pruned2_A__ac_short")).mean() / (dsh - m_("pruned2_A__ac_short").mean())),
        "rho_native_point": float((m_("recovered2__ac_native") - m_("pruned2_A__ac_native")).mean() / (dna - m_("pruned2_A__ac_native").mean())),
        "note": "DESCRIPTIVE ONLY: dense means come from the severity-1 80-prompt set; no CI; replaced by a paired estimate if dense is generated on the 192 set"}
    out["sev2_xsev192"] = s2

    # crop decomposition sev-1 too
    c1 = {}
    for lab, sh, cr, na in [("pruned", "pruned_short_sev1__armd80", "crop_sev1__pruned", "dense10s__pruned_sev1"),
                            ("postft", "postft_short_sev1__armd80", "crop_sev1__recovered", "dense10s__recovered_sev1")]:
        c1[f"window_part_c_{lab}"] = bt.ci_vec(a(na) - a(cr))
        c1[f"length_part_c_{lab}"] = bt.ci_vec(a(cr) - a(sh))
    c1["R_c_crop"] = bt.ci_vec(a("crop_sev1__recovered") - a("crop_sev1__pruned"))
    out["sev1_crop_decomposition"] = c1

    # ---- consistency guards vs frozen points -------------------------------------------------------
    opd = json.load(open(OPD))["PRIMARY_clap"]; xs = json.load(open(XSEV))["PRIMARY_A"]
    mn = json.load(open(MUSN)); rm1 = json.load(open(RMUS1)); dd = json.load(open(DDC))
    guards = {
        "sev1 R_short": (s1["R_raw_short_guard"], opd["R_ctrl_80"]["point"]),
        "sev1 R_native": (s1["R_raw_native_guard"], opd["R_alt"]["point"]),
        "sev1 s_raw dense": (s1["s_raw_dense"]["point"], dd["slopes"]["dense"]["point"]),
        "sev1 rho_raw short": (s1["rho_raw_short"]["point"], dd["gap_closed_fraction"]["short"]["point"]),
        "sev1 rho_raw native": (s1["rho_raw_native"]["point"], dd["gap_closed_fraction"]["native"]["point"]),
        "sev2 R_short": (s2["R_raw_short_guard"], xs["R_short"]["point"]),
        "sev2 R_native": (s2["R_raw_native_guard"], xs["R_native"]["point"]),
        "sev2 J": (s2["J_raw_guard"], xs["J"]["point"]),
        "sev2 R_music short": (s2["music_R_raw_short_guard"], xs["R_music"]["point"]),
        "sev2 R_music native": (s2["music_R_raw_native_guard"], mn["PRIMARY_R_music_native"]["point"]),
        "sev1 R_music": (s1["music_R_raw_guard"], rm1["R_music"]["point"]),
    }
    worst = max(abs(x - y) for x, y in guards.values())
    out["consistency_guards"] = {k: {"here": x, "frozen": y} for k, (x, y) in guards.items()}
    out["consistency_guards_max_abs_diff"] = worst
    # tolerance 1e-6 (not 1e-9 as in the Draft-4 script): the per-item cosines here are RE-COMPUTED float32
    # embeddings (guarded per item at 1e-6 in --score), so means differ from the frozen float64 points by
    # float32 rounding (~1e-8), not by any change in data or convention.
    if worst > 1e-6:
        raise SystemExit(f"consistency guard FAILED (max |diff| {worst:.3e})")

    # ---- caption tokens vs the 77-token conditioner limit ------------------------------------------
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("roberta-base")
        def toklen(caps):
            return np.array([len(tok(c)["input_ids"]) for c in caps])
        mus = [p["caption"] for p in json.load(open(MUS64))["prompts"]]
        ac2 = [p["caption"] for p in json.load(open(AC192))["prompts"]]
        ac1 = [p["caption"] for p in subset()]
        lm, l2, l1 = toklen(mus), toklen(ac2), toklen(ac1)
        out["caption_tokens_vs_conditioner_limit"] = {
            "reference_length_tokens": 77,
            "tokenizer": "roberta-base (CLAP text branch)",
            "music64": {"median": float(np.median(lm)), "max": int(lm.max()), "frac_over_77": float((lm > 77).mean())},
            "audiocaps192": {"median": float(np.median(l2)), "max": int(l2.max()), "frac_over_77": float((l2 > 77).mean())},
            "audiocaps80_sev1": {"median": float(np.median(l1)), "max": int(l1.max()), "frac_over_77": float((l1 > 77).mean())},
            "note": "VERIFIED in code: AudioLDM-M-Full's conditioner (CLAPAudioEmbeddingClassifierFreev2.tokenizer, "
                    "audioldm_train/conditional_models.py) tokenizes with max_length=512, so NO caption here is truncated at "
                    "generation; 77 is the caption length the CLAP text tower was PRE-TRAINED with "
                    "(audioldm_train/modules/clap/training/data.py max_length=77). The scorer (ClapProcessor, padding=True) also "
                    "sees full captions. Music captions are therefore near/above the pre-training caption length for BOTH the "
                    "conditioner and the scorer — a caption-style covariate of the domain axis, not a truncation artefact."}
    except Exception as e:  # pragma: no cover
        out["caption_tokens_vs_conditioner_limit"] = {"error": str(e)}

    json.dump(out, open(OUT, "w"), indent=1)
    # ---- human-readable summary
    def fmt(c):
        return f"{c['point']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]"
    print(f"\n== cells (matched | floor | above chance) ==")
    for nm, c in cells.items():
        print(f"{nm:34s} n={c['n_prompts']:3d}  {c['matched_mean']:.3f} | {c['floor']['point']:.3f} [{c['floor']['lo']:.3f},{c['floor']['hi']:.3f}] | {fmt(c['above_chance'])}")
    print("\n== severity 1 (n=80) ==")
    for k in ["s_raw_dense", "s_c_dense", "s_raw_pruned", "s_c_pruned", "s_raw_postft", "s_c_postft", "s_raw_real", "s_c_real",
              "floor_shift_dense", "floor_shift_pruned", "floor_shift_postft", "floor_shift_real",
              "R_c_short", "R_c_native", "J_c", "rho_raw_short", "rho_raw_native", "rho_c_short", "rho_c_native",
              "dense_minus_postft_native_c", "real_minus_dense_native", "real_minus_dense_short", "rho_real_short", "rho_real_native",
              "dense_frac_of_real_short", "dense_frac_of_real_native", "music_R_c"]:
        print(f"  {k:28s} {fmt(s1[k])}")
    print("\n== severity 2 (n=192; music 64) ==")
    for k in ["s_raw_pruned", "s_c_pruned", "s_raw_postft", "s_c_postft", "s_raw_real", "s_c_real",
              "floor_shift_pruned", "floor_shift_postft", "floor_shift_real", "R_c_short", "R_c_native", "J_c", "J_c_B",
              "real_minus_postft_native", "real_minus_postft_short", "window_part_c_pruned", "length_part_c_pruned",
              "window_part_c_postft", "length_part_c_postft", "R_c_crop", "music_R_c_short", "music_R_c_native",
              "rho_real_short", "rho_real_native", "rho_real_c_short", "rho_real_c_native"]:
        print(f"  {k:28s} {fmt(s2[k])}")
    print("  rho cross-set (descriptive):", {k: round(v, 3) for k, v in s2["rho_crossset_descriptive"].items() if isinstance(v, float)})
    print("\n== sev-1 crop decomposition ==")
    for k, v in c1.items():
        print(f"  {k:28s} {fmt(v)}")
    print("\ncaption tokens:", json.dumps(out["caption_tokens_vs_conditioner_limit"], indent=None)[:600])
    print(f"\nconsistency guards max |diff| = {worst:.2e}; worst scorer guard = {dout['worst_guard']:.2e}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true"); ap.add_argument("--score", action="store_true"); ap.add_argument("--verdict", action="store_true")
    a_ = ap.parse_args()
    if a_.emit: emit()
    if a_.score: score()
    if a_.verdict: verdict()
    if not (a_.emit or a_.score or a_.verdict): ap.print_help()
