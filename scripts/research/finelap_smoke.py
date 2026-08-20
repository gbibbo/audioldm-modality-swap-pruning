#!/usr/bin/env python3
"""FineLAP CPU validity smoke under torch 1.13.1 (CPU queue Q3 -> resolves DECISION-V4-06).

Loads `AndreasXi/FineLAP` (MIT) from a pinned local snapshot with trust_remote_code, runs
`get_frame_level_score` on 5 known single-label AudioCaps clips, applies the FROZEN
score->duration (temporal-occupancy) rule, and decides PASS/FAIL:

  PASS requires ALL of:
    (1) the model loads and runs on CPU under the frozen env (torch 1.13.1, transformers
        4.30.2) with no code change;
    (2) get_frame_level_score returns finite sigmoid scores of shape (B, N, T);
    (3) grounding is meaningful: for each single-label clip, the mean frame score of the
        clip's OWN event phrase exceeds the mean over distractor phrases (>= 4 of 5 clips),
        i.e. the model is not returning noise.

If PASS -> H-acoustic (temporal occupancy) stays in the primary Gate M. If FAIL ->
H-acoustic leaves the primary Gate M; the single-AudioSet-label subset is sensitivity only
(plan §3 / DECISION-V4-06).

FROZEN score->duration rule (occupancy):
    occupancy(e) = mean_t [ sigmoid_score(e, t) >= TAU ]   with TAU = 0.5
    duration_s(e) = occupancy(e) * CLIP_SECONDS            with CLIP_SECONDS = 10.24
This is independent of the outcome detector (PANNs), as the plan requires.

    .venv/bin/python scripts/research/finelap_smoke.py \
        --model _external/FineLAP --out artifacts/finelap_smoke/result.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

TAU = 0.5
CLIP_SECONDS = 10.24
# Curated CLEAR single-label events (concrete temporal signature) — representative of the
# E* events whose FineLAP-mask occupancy Gate M actually uses. Not abstract events like
# "Vibration"; the smoke tests validity of the tool for its real use, not FineLAP accuracy
# on hard/abstract labels.
PREFERRED_EVENTS = ["Gunshot", "Applause", "Whistling", "Sneeze", "Bell", "Meow", "Telephone"]
# Distractors chosen to be clearly disjoint from every preferred event.
DISTRACTORS = ["Dog", "Piano", "Rain", "Helicopter", "Laughter"]

HF_REPO = "AndreasXi/FineLAP"
HF_FILES = ["config.json", "configuration_finelap.py", "configuration_eat.py",
            "modeling_finelap.py", "modeling_eat.py", "eat_model.py", "eat_model_core.py",
            "model.safetensors"]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = os.path.join(ROOT, "data/dataset/metadata/audiocaps")
ROOT_JSON = os.path.join(ROOT, "data/dataset/metadata/dataset_root.json")
TRAIN_JSON = os.path.join(META, "datafiles/audiocaps_train_label.json")
CSV_PATH = os.path.join(META, "class_labels_indices.csv")


def ensure_finelap(model_dir):
    """Download the FineLAP snapshot (code + config + weights) if missing. Reproducible."""
    need = [f for f in HF_FILES if not os.path.exists(os.path.join(model_dir, f))]
    if not need:
        return
    from huggingface_hub import hf_hub_download
    os.makedirs(model_dir, exist_ok=True)
    for f in need:
        hf_hub_download(HF_REPO, f, local_dir=model_dir)
        print(f"fetched {f}")


def pick_clips(n=5):
    mid2name = {r["mid"]: r["display_name"].split(",")[0].strip()
                for r in csv.DictReader(open(CSV_PATH))}
    audio_root = json.load(open(ROOT_JSON))["audiocaps"]
    data = json.load(open(TRAIN_JSON))["data"]
    # deterministic: first single-label clip for each curated clear event, in preferred order
    by_event = {}
    for e in data:
        mids = [m for m in e["labels"].split(",") if m in mid2name]
        if len(mids) != 1:
            continue
        name = mid2name[mids[0]]
        if name in PREFERRED_EVENTS and name not in by_event:
            path = os.path.join(audio_root, e["wav"])
            if os.path.exists(path):
                by_event[name] = {"wav": path, "event": name, "mid": mids[0],
                                  "caption": e["caption"]}
    picked = [by_event[ev] for ev in PREFERRED_EVENTS if ev in by_event]
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "_external/FineLAP"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/finelap_smoke/result.json"))
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    result = {"tau": TAU, "clip_seconds": CLIP_SECONDS, "model": args.model,
              "checks": {}, "clips": []}

    import torch
    import transformers
    result["env"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    print("env:", result["env"])

    ensure_finelap(args.model)
    result["hf_repo"] = HF_REPO

    clips = pick_clips(args.n)
    if len(clips) < args.n:
        print(f"FATAL: only found {len(clips)} single-label clips", file=sys.stderr)
        return 2
    audio_paths = [c["wav"] for c in clips]
    event_phrases = [c["event"] for c in clips]
    phrases = list(dict.fromkeys(event_phrases + DISTRACTORS))  # unique, order-preserving
    print("clips:", [(c["event"], os.path.basename(c["wav"])) for c in clips])
    print("phrases:", phrases)

    # ---- version-gap shim (NOT a numerical change, NOT a FineLAP code change):
    # FineLAP's config was serialized by transformers 4.51 and nests an EATConfig object
    # under `audio_config`. transformers 4.30.2's `to_json_string` (called from a
    # `logger.info(f"Model config {config}")` during from_pretrained) does not recurse into
    # nested PretrainedConfig attributes, so json.dumps raises "EATConfig is not JSON
    # serializable". Newer transformers recurse; we backport exactly that behaviour here so
    # the (purely cosmetic) config repr works. Affects logging/serialization only.
    from transformers import PretrainedConfig
    if not getattr(PretrainedConfig, "_finelap_nested_shim", False):
        _orig_to_dict = PretrainedConfig.to_dict

        def _to_dict_recursive(self):
            d = _orig_to_dict(self)
            for k, v in list(d.items()):
                if isinstance(v, PretrainedConfig):
                    d[k] = v.to_dict()
            return d

        PretrainedConfig.to_dict = _to_dict_recursive
        PretrainedConfig._finelap_nested_shim = True
        result["checks"]["nested_config_shim_applied"] = True

    # ---- (1) load as a LOCAL PACKAGE (bypass trust_remote_code's dynamic-module copying,
    # which in transformers 4.30.2 fails to copy FineLAP's auxiliary .py files ->
    # ModuleNotFoundError transformers_modules.FineLAP.eat_model). Build the model from
    # config.json and load model.safetensors directly. Reproducible and env-controlled.
    import importlib
    try:
        model_dir = os.path.abspath(args.model)
        ext_parent, pkg_name = os.path.dirname(model_dir), os.path.basename(model_dir)
        init_py = os.path.join(model_dir, "__init__.py")
        if not os.path.exists(init_py):
            open(init_py, "w").close()
        if ext_parent not in sys.path:
            sys.path.insert(0, ext_parent)
        cfgmod = importlib.import_module(f"{pkg_name}.configuration_finelap")
        modmod = importlib.import_module(f"{pkg_name}.modeling_finelap")
        cfg_dict = json.load(open(os.path.join(model_dir, "config.json")))
        config = cfgmod.FineLAPConfig(**cfg_dict)
        model = modmod.FineLAPModel(config)
        from safetensors.torch import load_file
        sd = load_file(os.path.join(model_dir, "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval()
        result["checks"]["loads"] = True
        result["checks"]["missing_keys"] = len(missing)
        result["checks"]["unexpected_keys"] = len(unexpected)
        print(f"loaded FineLAP (missing={len(missing)} unexpected={len(unexpected)})")
    except Exception as e:
        result["checks"]["loads"] = False
        result["load_error"] = repr(e)[:1000]
        print("LOAD FAILED:", repr(e)[:500])
        _write(result, args.out)
        print("\nRESULT: FAIL (load)")
        return 1

    # ---- (2) run + shape
    try:
        with torch.no_grad():
            scores = model.get_frame_level_score(audio_paths, phrases)  # (B, N, T)
        scores = scores.detach().cpu().float()
        B, N, T = scores.shape
        finite = bool(torch.isfinite(scores).all())
        result["checks"]["shape_BNT"] = [int(B), int(N), int(T)]
        result["checks"]["runs_finite"] = finite and B == len(clips) and N == len(phrases)
        print(f"frame scores shape (B,N,T) = ({B},{N},{T}); finite={finite}")
    except Exception as e:
        result["checks"]["runs_finite"] = False
        result["run_error"] = repr(e)[:1000]
        print("RUN FAILED:", repr(e)[:500])
        _write(result, args.out)
        print("\nRESULT: FAIL (run)")
        return 1

    # ---- (3) grounding + frozen occupancy rule
    pidx = {p: i for i, p in enumerate(phrases)}
    correct_beats_distractor = 0
    for b, c in enumerate(clips):
        own = pidx[c["event"]]
        distr = [pidx[d] for d in DISTRACTORS if d in pidx]
        own_mean = float(scores[b, own].mean())
        distr_mean = float(scores[b, distr].mean())
        occ = float((scores[b, own] >= TAU).float().mean())
        dur = occ * CLIP_SECONDS
        beat = own_mean > distr_mean
        correct_beats_distractor += int(beat)
        result["clips"].append({
            "event": c["event"], "wav": os.path.basename(c["wav"]),
            "own_mean_score": own_mean, "distractor_mean_score": distr_mean,
            "own_beats_distractor": beat, "occupancy": occ, "duration_s": dur,
        })
        print(f"  {c['event']:16s} own={own_mean:.3f} distr={distr_mean:.3f} "
              f"beat={beat}  occupancy={occ:.2f} dur={dur:.2f}s")

    result["checks"]["grounding_meaningful"] = correct_beats_distractor >= 4
    result["grounding_correct_of_5"] = correct_beats_distractor

    ok = (result["checks"].get("loads") and result["checks"].get("runs_finite")
          and result["checks"].get("grounding_meaningful"))
    result["verdict"] = "PASS" if ok else "FAIL"
    result["decision_v4_06"] = ("H-acoustic stays in the primary Gate M"
                                if ok else "H-acoustic leaves primary Gate M; single-label subset sensitivity only")
    _write(result, args.out)
    print(f"\nRESULT: {result['verdict']}  (grounding {correct_beats_distractor}/5)")
    return 0 if ok else 1


def _write(result, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
