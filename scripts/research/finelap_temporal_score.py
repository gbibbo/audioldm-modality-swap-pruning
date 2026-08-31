#!/usr/bin/env python3
"""Part A — FineLAP frame-level scoring of EXISTING native 10.24-s WAVs (CPU, 0 GPU).

Implements the FROZEN protocol `docs/finelap_temporal_protocol.md`. Scores each eligible prompt's
recovered / pruned_A (/ pruned_B for sev-2) native generation against its eligible requested-event
canonical phrases, producing 64-frame FineLAP grounding scores. Deterministic (eval mode; kaldi
dither=0; no sampling). Persists raw frame scores for the separate verdict step.

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/finelap_temporal_score.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/research"))
from finelap_geometry_audit import load_finelap   # identical pinned-model loader

N_FRAMES = 64
BOUNDARY_FRAME = 24   # 3.84 / 0.16 (frozen, A0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "_external/FineLAP"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts/finelap_temporal"))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model, miss, unexp = load_finelap(a.model)
    print(f"loaded FineLAP (missing={miss} unexpected={unexp})")

    for sev in ("sev1", "sev2"):
        man = json.load(open(os.path.join(ROOT, f"configs/research/finelap_eligibility_{sev}.json")))
        root = man["wav_root"]
        systems = [s for s in ("recovered", "pruned_A", "pruned_B")
                   if any(s in r["wav_files"] for r in man["prompts"])]
        elig = [r for r in man["prompts"] if r["eligible"]]
        out = {"artifact": f"finelap_temporal_scores_{sev}", "severity": sev,
               "eligibility_manifest_sha256": man["manifest_sha256"],
               "systems": systems, "n_frames": N_FRAMES, "boundary_frame": BOUNDARY_FRAME,
               "phrase_convention": "display_name.split(',')[0].strip()",
               "model_weights_sha256": "13b9646c9f9d48513c0145bed75e654179e83f0fd8d49ed4ffc5d6b8f3353fb4",
               "prompts": []}
        for r in elig:
            events = r["eligible_events"]
            phrases = [e["phrase"] for e in events]
            wavs = [os.path.join(root, r["wav_files"][s]) for s in systems]
            with torch.no_grad():
                sc = model.get_frame_level_score(wavs, phrases)   # (S, N_events, 64)
            sc = sc.detach().cpu().float().numpy()
            assert sc.shape == (len(systems), len(phrases), N_FRAMES), (sc.shape, r["prompt_id"])
            pr = {"prompt_id": r["prompt_id"], "ytid": r["ytid"],
                  "events": [{"mid": e["mid"], "phrase": e["phrase"]} for e in events],
                  "scores": {}}
            for si, s in enumerate(systems):
                pr["scores"][s] = {events[ei]["mid"]: sc[si, ei].tolist()
                                   for ei in range(len(events))}
            out["prompts"].append(pr)
        out["scores_sha256"] = hashlib.sha256(
            json.dumps([p["scores"] for p in out["prompts"]], sort_keys=True).encode()).hexdigest()
        os.makedirs(a.out_dir, exist_ok=True)
        outp = os.path.join(a.out_dir, f"scores_{sev}.json")
        json.dump(out, open(outp, "w"), indent=1)
        print(f"{sev}: scored {len(elig)} eligible prompts x {systems} "
              f"(events total {sum(len(p['events']) for p in out['prompts'])}); "
              f"scores_sha={out['scores_sha256'][:8]} -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
