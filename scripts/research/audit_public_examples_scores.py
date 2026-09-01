#!/usr/bin/env python3
"""INTERNAL per-example score audit of the public audio examples (CPU, read-only).

Reads ONLY existing frozen scorer outputs (no recompute, no selection change) for the
10 already-selected public examples, to relate the displayed comparisons to the automatic
metrics. NOT deployed to gh-pages. Descriptive only.

Metrics: CLAP (primary, laion/clap-htsat-fused rev 365dea6e), Human-CLAP (where per-example
raw exists), FineLAP frame-level grounding (native, FineLAP-eligible prompts only).

Outputs: configs/research/public_audio_examples_score_audit.json
         docs/public_audio_examples_score_audit.md
"""
import json, os, hashlib

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
os.chdir(ROOT)
SESOI = 0.025  # descriptive near-zero band for |ΔCLAP|, project SESOI

man = json.load(open("configs/research/public_audio_examples_manifest.json"))
sel = man["selected"]

# ---- id maps ----
subset = json.load(open("configs/research/op_duration_discriminator_1_subset.json"))["prompts"]
sub_idx = {p["ytid"]: p["subset_prompt_index"] for p in subset}
xac = json.load(open("configs/research/xsev_audiocaps_manifest.json"))["prompts"]
xac_idx = {p["ytid"]: p["prompt_index"] for p in xac}
xmus = json.load(open("configs/research/xsev_music_manifest.json"))["prompts"]
xmus_idx = {p["ytid"]: p["prompt_index"] for p in xmus}
cap = {p["ytid"]: p["caption"] for p in subset}
cap.update({p["ytid"]: p["caption"] for p in xac})
cap.update({p["ytid"]: p["caption"] for p in xmus})

# ---- CLAP raw ----
opd = json.load(open("configs/research/op_duration_discriminator_1_result.json"))["raw_cosines"]
def sev2_groups(path):
    r = json.load(open(path))["results"]
    return {g["name"]: g["cosines"] for g in r}
G = sev2_groups("artifacts/icassp_gate0/_score_tmp/xsev_sev2_groups_out.json")
HC = sev2_groups("artifacts/icassp_gate0/_score_tmp/xsev_sev2_hc_groups_out.json")

def fl_load(sev):
    d = json.load(open(f"artifacts/finelap_temporal/scores_{sev}.json"))
    return {p["ytid"]: p for p in d["prompts"]}, d["boundary_frame"]
FL1, B1 = fl_load("sev1"); FL2, B2 = fl_load("sev2")

def fl_stats(entry, system, boundary):
    if entry is None or system not in entry["scores"]:
        return None
    # mean over eligible events of the 64-frame array; mean / early / late
    import statistics as st
    frames = None
    for mid, arr in entry["scores"][system].items():
        frames = arr if frames is None else [a + b for a, b in zip(frames, arr)]
    ne = len(entry["scores"][system])
    frames = [x / ne for x in frames]
    mean = sum(frames) / len(frames)
    early = sum(frames[:boundary]) / boundary
    late = sum(frames[boundary:]) / (len(frames) - boundary)
    return {"mean": round(mean, 4), "early": round(early, 4), "late": round(late, 4)}

rows = []       # flat per (example, duration) displayed comparison
examples_out = []

def add(section, exno, ytid, cells, fl_map, boundary):
    ex = {"section": section, "example": exno, "ytid": ytid, "caption": cap.get(ytid, ""), "durations": {}}
    for dur, cP, cR in cells:
        clap_p = round(cP, 4); clap_r = round(cR, 4); dC = round(cR - cP, 4)
        band = "near_zero" if abs(dC) < SESOI else ("post_ft_higher" if dC > 0 else "pruned_higher")
        ex["durations"][dur] = {"CLAP_pruned": clap_p, "CLAP_post_ft": clap_r, "dCLAP": dC, "CLAP_band": band}
        rows.append({"section": section, "example": exno, "ytid": ytid, "duration": dur, "dCLAP": dC, "band": band})
    # FineLAP (native only, if eligible)
    fe = fl_map.get(ytid)
    ex["finelap_native"] = {"post_ft": fl_stats(fe, "recovered", boundary),
                            "pruned": fl_stats(fe, "pruned_A", boundary),
                            "eligible": fe is not None}
    examples_out.append(ex)
    return ex

# ---- Section A (sev1) : CLAP from op_duration raw ; HC per-example UNAVAILABLE ----
for i, y in enumerate(sel["sev1_audiocaps"], 1):
    k = sub_idx[y]
    add("sev1_audiocaps", i, y, [
        ("3.84s", opd["pruned_ctrl"][k], opd["recovered_ctrl"][k]),
        ("10.24s", opd["pruned_alt"][k], opd["recovered_alt"][k]),
    ], FL1, B1)
    examples_out[-1]["human_clap"] = "UNAVAILABLE per-example (only aggregate op_duration Human-CLAP exists)"

# ---- Section B (sev2 AC) : CLAP + HC per-example ----
for i, y in enumerate(sel["sev2_audiocaps"], 1):
    k = xac_idx[y]
    ex = add("sev2_audiocaps", i, y, [
        ("3.84s", G["pruned2_A__ac_short"][k], G["recovered2__ac_short"][k]),
        ("10.24s", G["pruned2_A__ac_native"][k], G["recovered2__ac_native"][k]),
    ], FL2, B2)
    ex["human_clap"] = {
        "3.84s": {"HC_pruned": round(HC["pruned2_A__ac_short"][k], 4), "HC_post_ft": round(HC["recovered2__ac_short"][k], 4),
                   "dHC": round(HC["recovered2__ac_short"][k] - HC["pruned2_A__ac_short"][k], 4)},
        "10.24s": {"HC_pruned": round(HC["pruned2_A__ac_native"][k], 4), "HC_post_ft": round(HC["recovered2__ac_native"][k], 4),
                    "dHC": round(HC["recovered2__ac_native"][k] - HC["pruned2_A__ac_native"][k], 4)},
    }

# ---- Section C (sev2 music) : r0 clip (matches displayed) ; 64x3 prompt-major ----
for i, y in enumerate(sel["sev2_music"], 1):
    k = xmus_idx[y]; r0 = 3 * k
    ex = add("sev2_music", i, y, [
        ("3.84s", G["pruned2_A__music"][r0], G["recovered2__music"][r0]),
    ], {}, B2)  # FineLAP is AudioCaps-eligibility only; music not covered
    ex["finelap_native"] = {"eligible": False, "note": "FineLAP eligibility is AudioCaps-only; music not covered"}
    ex["human_clap"] = {"3.84s": {"HC_pruned": round(HC["pruned2_A__music"][r0], 4),
                                   "HC_post_ft": round(HC["recovered2__music"][r0], 4),
                                   "dHC": round(HC["recovered2__music"][r0] - HC["pruned2_A__music"][r0], 4)}}
    ex["music_note"] = "CLAP/HC are the r0 clip (the displayed audio); per-prompt frozen analysis used the 3-rep mean."

# ---- aggregate over displayed comparisons ----
npos = sum(1 for r in rows if r["band"] == "post_ft_higher")
nneg = sum(1 for r in rows if r["band"] == "pruned_higher")
nz = sum(1 for r in rows if r["band"] == "near_zero")

out = {
    "artifact": "public_audio_examples_score_audit",
    "status": "INTERNAL descriptive audit — NOT deployed; selection unchanged; no scores on public page",
    "clap_scorer": {"model": "laion/clap-htsat-fused", "revision": "365dea6ef167def6676140ed93bbc43f84dabb28"},
    "near_zero_band": {"metric": "|dCLAP|", "threshold": SESOI, "basis": "project SESOI (frozen before inspection)"},
    "public_manifest_self_sha256": man["self_sha256"],
    "n_displayed_clap_comparisons": len(rows),
    "aggregate": {"dCLAP_post_ft_higher": npos, "near_zero": nz, "dCLAP_pruned_higher": nneg},
    "examples": examples_out,
}
payload = json.dumps(out, indent=2, sort_keys=True)
out["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
json.dump(out, open("configs/research/public_audio_examples_score_audit.json", "w"), indent=2, sort_keys=True)

# quick console summary
print("displayed CLAP comparisons:", len(rows),
      "| post-FT higher:", npos, "| near-zero:", nz, "| pruned higher:", nneg)
for ex in examples_out:
    if ex["section"] == "sev1_audiocaps":
        d = ex["durations"]
        print(f"  sev1 Ex{ex['example']} {ex['ytid']}: "
              f"3.84 dCLAP={d['3.84s']['dCLAP']:+.3f}({d['3.84s']['CLAP_band']}) "
              f"10.24 dCLAP={d['10.24s']['dCLAP']:+.3f}({d['10.24s']['CLAP_band']})"
              f" | FineLAP native post_ft={ex['finelap_native']['post_ft']} pruned={ex['finelap_native']['pruned']}")
print("self_sha256:", out["self_sha256"][:16])
