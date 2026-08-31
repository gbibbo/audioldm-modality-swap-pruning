#!/usr/bin/env python3
"""A1 — OUTCOME-BLIND FineLAP requested-event eligibility manifests (Part A).

Builds, from ONLY information fixed independently of any generated output, the per-severity
eligible requested-event sets used by the Part-A temporal-semantic profile. No generated
waveform, CLAP/FineLAP/HC score, or any outcome may influence inclusion.

FROZEN RULE (identical to scripts/research/build_v4_manifests.py:186-208):
  requested_events(prompt) = { m in ground_truth_labels(ytid)
                               : some strict alias of m matches the caption, \\b word-boundary,
                                 case-insensitive } .
  eligible(prompt) <=> len(requested_events) >= 1 .
Inputs (all frozen, hashed):
  * caption            : the exact text conditioned on at generation (selection manifest).
  * ground-truth labels: AudioSet mids for the ytid (audiocaps_test_label.json; identical
                         across caption rows).
  * caption->event alias: configs/research/event_synonyms_strict.json (527 classes; comma
                         aliases + minimal plural morphology).
FineLAP scoring phrase (frozen) = display_name.split(",")[0].strip() (primary event phrase).
Independent unit = prompt; event-level quantities are averaged WITHIN prompt before inference.

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/build_finelap_eligibility.py
           [--check]   # verify determinism against on-disk manifest_sha256
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = os.path.join(ROOT, "data/dataset/metadata/audiocaps")
CSV_PATH = os.path.join(META, "class_labels_indices.csv")
TEST_LABEL = os.path.join(META, "datafiles/audiocaps_test_label.json")
STRICT = os.path.join(ROOT, "configs/research/event_synonyms_strict.json")
SUBSET_SEV1 = os.path.join(ROOT, "configs/research/op_duration_discriminator_1_subset.json")
XSEV_SEV2 = os.path.join(ROOT, "configs/research/xsev_audiocaps_manifest.json")

XSEV_ROOT = ("/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning/"
             "artifacts/icassp_gate0/reversal_xsev_gen")
ARMD_ROOT = ("/teamspace/jobs/reversal-armd-gen-1/artifacts/audioldm-modality-swap-pruning/"
             "artifacts/icassp_gate0/reversal_armd_gen")

# WAV filename templates per severity/system, keyed by prompt_index (native 10.24 s only).
WAV_TEMPLATES = {
    "sev1": {"root": ARMD_ROOT,
             "recovered": "p1_recovered_noadapter_alt10s_p{p}_r0.wav",
             "pruned_A": "p1_pruned_ema_reconstructed_noadapter_alt10s_p{p}_r0.wav"},
    "sev2": {"root": XSEV_ROOT,
             "recovered": "recovered2_ac_native_p{p}_r0.wav",
             "pruned_A": "pruned2_A_ac_native_p{p}_r0.wav",
             "pruned_B": "pruned2_B_ac_native_p{p}_r0.wav"},
}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_gt():
    """ytid -> sorted list of AudioSet mids (from the test label string)."""
    m = {}
    for d in json.load(open(TEST_LABEL))["data"]:
        bn = os.path.basename(d["wav"])          # Y<ytid>.wav
        ytid = bn[1:-4]
        m.setdefault(ytid, sorted(x for x in d["labels"].split(",") if x))
    return m


def build_patterns():
    ev = json.load(open(STRICT))["events"]        # {mid: {index, display_name, aliases}}
    pats = {m: [re.compile(r"\b" + re.escape(a) + r"\b") for a in ev[m]["aliases"]] for m in ev}
    return ev, pats


def sev1_prompts():
    pr = json.load(open(SUBSET_SEV1))["prompts"]
    return [{"prompt_index": p["subset_prompt_index"], "ytid": p["ytid"], "caption": p["caption"]}
            for p in sorted(pr, key=lambda x: x["subset_prompt_index"])]


def sev2_prompts():
    pr = json.load(open(XSEV_SEV2))["prompts"]
    return [{"prompt_index": p["prompt_index"], "ytid": p["ytid"], "caption": p["caption"]}
            for p in sorted(pr, key=lambda x: x["prompt_index"])]


def phrase_of(display_name):
    return display_name.split(",")[0].strip()


def build(sev, prompts, ev, pats, gt):
    rows, n_occ, n_excl = [], 0, 0
    tmpl = WAV_TEMPLATES[sev]
    for p in prompts:
        cap_l = p["caption"].lower()
        gt_mids = gt.get(p["ytid"], [])
        req = []
        for m in gt_mids:
            if m in ev and any(rx.search(cap_l) for rx in pats[m]):
                matched = [a for a, rx in zip(ev[m]["aliases"], pats[m]) if rx.search(cap_l)]
                req.append({"mid": m, "display_name": ev[m]["display_name"],
                            "phrase": phrase_of(ev[m]["display_name"]),
                            "audioset_index": ev[m]["index"], "matched_aliases": matched})
        eligible = len(req) > 0
        if not eligible:
            n_excl += 1
        n_occ += len(req)
        wavs = {sysname: t.format(p=p["prompt_index"]) for sysname, t in tmpl.items() if sysname != "root"}
        rows.append({
            "prompt_id": f"{sev}_p{p['prompt_index']}", "prompt_index": p["prompt_index"],
            "ytid": p["ytid"], "caption": p["caption"], "ground_truth_mids": gt_mids,
            "eligible": eligible,
            "exclusion_reason": None if eligible else "no ground-truth AudioSet label matched by any strict caption alias",
            "eligible_events": req, "n_eligible_events": len(req), "wav_files": wavs})
    eligible_rows = [r for r in rows if r["eligible"]]
    obj = {
        "artifact": f"finelap_eligibility_{sev}",
        "status": "OUTCOME-BLIND requested-event eligibility (frozen before any FineLAP scoring)",
        "severity": sev, "wav_root": tmpl["root"],
        "rule": ("requested_events = {m in gt_labels(ytid) : strict alias of m matches caption "
                 "(\\b word-boundary, case-insensitive)}; eligible <=> >=1; unit=prompt; "
                 "event quantities averaged within prompt before inference; "
                 "scoring phrase = display_name.split(',')[0].strip()"),
        "zero_outcome_dependent_filtering": True,
        "inputs_sha256": {
            "class_labels_indices_csv": sha_file(CSV_PATH),
            "audiocaps_test_label_json": sha_file(TEST_LABEL),
            "event_synonyms_strict_json": sha_file(STRICT),
            "selection_manifest": sha_file(SUBSET_SEV1 if sev == "sev1" else XSEV_SEV2)},
        "funnel": {"n_prompts_considered": len(rows), "n_excluded": n_excl,
                   "exclusion_reasons": {"no_requested_event_in_ground_truth": n_excl},
                   "n_eligible_prompts": len(eligible_rows),
                   "n_requested_event_occurrences": n_occ},
        "prompts": rows}
    obj["prompts_sha256"] = hashlib.sha256(
        json.dumps([{k: r[k] for k in ("prompt_id", "ytid", "eligible",
                                       "eligible_events")} for r in rows],
                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    body = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    obj["manifest_sha256"] = hashlib.sha256(body).hexdigest()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ev, pats = build_patterns()
    gt = load_gt()
    outs = {}
    for sev, prompts in [("sev1", sev1_prompts()), ("sev2", sev2_prompts())]:
        obj = build(sev, prompts, ev, pats, gt)
        out = os.path.join(ROOT, f"configs/research/finelap_eligibility_{sev}.json")
        f = obj["funnel"]
        print(f"{sev}: considered={f['n_prompts_considered']} eligible={f['n_eligible_prompts']} "
              f"excluded={f['n_excluded']} occurrences={f['n_requested_event_occurrences']} "
              f"manifest_sha={obj['manifest_sha256'][:8]}")
        if a.check:
            if not os.path.exists(out):
                print(f"  MISSING {out}"); return 1
            disk = json.load(open(out))
            same = disk.get("manifest_sha256") == obj["manifest_sha256"]
            print(f"  --check {sev}: {'MATCH' if same else 'MISMATCH'}")
            if not same:
                return 1
        else:
            json.dump(obj, open(out, "w"), indent=2, ensure_ascii=False)
            print(f"  wrote {out}")
        outs[sev] = obj
    return 0


if __name__ == "__main__":
    sys.exit(main())
