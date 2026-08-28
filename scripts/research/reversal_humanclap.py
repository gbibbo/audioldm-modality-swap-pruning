#!/usr/bin/env python3
"""D2 Human-CLAP corroboration of the historical music contrast (Recovery-Reversal robustness).

CORROBORATION, NOT independent human evaluation. Human-CLAP (Takano et al., APSIPA ASC 2025,
arXiv:2506.23553; weights sarulab-speech/human-clap-wsce-mae) is a LAION-CLAP-fused model
fine-tuned on human text-audio relevance scores. It remains a CLAP-family model; its published
validation is not music-specific. Agreement here only shows the recovered<pruned music ordering
is not idiosyncratic to the exact frozen CLAP checkpoint — it does NOT prove human preference.

Rescos the ALREADY-PERSISTED historical OFF (no-adapter) WAVs for the two standalone backbones
(p1_recovered, p1_pruned_ema_reconstructed), 64 prompts x 3 replicates each, using the SAME
frozen scoring convention as the primary scorer (SR 48000, truncation="fusion",
get_audio_features / get_text_features cosine, np.random.seed(20260826) reset once per 192-item
system -> reproducible AND paired by position). Reports the paired recovered-minus-pruned contrast
in Human-CLAP's own cosine scale with a prompt-cluster percentile bootstrap CI (seed 20260826).

Does NOT modify the frozen R_music (CLAP) value. Different scale; the 0.025 CLAP SESOI does NOT
apply. CPU only, no GPU, no paid compute (one-time free HF weight download). Runs in .venv-metrics.

Run: OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/reversal_humanclap.py \
        --out artifacts/icassp_gate0/reversal_humanclap.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
from research_pruning.eval.cluster_bootstrap import cluster_percentile_ci  # noqa: E402

MODEL_ID = "sarulab-speech/human-clap-wsce-mae"
PROC_ID = "laion/clap-htsat-fused"
SR = 48000
SEED = 20260826
BATTERY = "configs/research/icassp_gate0_battery.json"
DOWN_ROOT = ("/teamspace/jobs/gate0-phenom-1/artifacts/audioldm-modality-swap-pruning/"
             "artifacts/icassp_gate0/gen_phenomenon")
SYSTEMS = {"p1_recovered": "p1_recovered_noadapter",
           "p1_pruned_ema_reconstructed": "p1_pruned_ema_reconstructed_noadapter"}
N_PROMPTS, N_REPS = 64, 3


class HumanClapScorer:
    def __init__(self):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.model = ClapModel.from_pretrained(MODEL_ID).eval()
        self.proc = ClapProcessor.from_pretrained(PROC_ID)

    def cosine(self, captions, wav_paths):
        """Frozen convention: text batch, then ONE seed-once audio batch (truncation fusion),
        paired per-item cosine of L2-normalized embeddings."""
        import librosa
        torch = self.torch
        inp = self.proc(text=list(captions), return_tensors="pt", padding=True)
        with torch.no_grad():
            te = self.model.get_text_features(**inp)
        wavs = [librosa.load(p, sr=SR, mono=True)[0] for p in wav_paths]
        np.random.seed(SEED)  # seed-once-per-system (paired nuisance), identical to primary
        inp = self.proc(audios=wavs, sampling_rate=SR, return_tensors="pt",
                        padding=True, truncation="fusion")
        with torch.no_grad():
            ae = self.model.get_audio_features(**inp)
        te = torch.nn.functional.normalize(te, dim=-1)
        ae = torch.nn.functional.normalize(ae, dim=-1)
        return (te * ae).sum(dim=-1).cpu().numpy().astype(np.float64)


def _grid(cos):
    arr = np.full((N_PROMPTS, N_REPS), np.nan)
    k = 0
    for p in range(N_PROMPTS):
        for r in range(N_REPS):
            arr[p, r] = cos[k]; k += 1
    return arr


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run(out_path: str) -> dict:
    battery = json.load(open(BATTERY))
    caps = [p["caption"] for p in battery["prompts"]]
    ytids = [p["ytid"] for p in battery["prompts"]]
    # canonical (prompt, replicate) order, mirroring the frozen systems
    order_caps, order_wavs = {}, {}
    for sysname, prefix in SYSTEMS.items():
        c, w = [], []
        for p in range(N_PROMPTS):
            for r in range(N_REPS):
                c.append(caps[p]); w.append(os.path.join(DOWN_ROOT, f"{prefix}_p{p}_r{r}.wav"))
        order_caps[sysname] = c; order_wavs[sysname] = w
    for sysname in SYSTEMS:
        for fp in order_wavs[sysname]:
            if not os.path.exists(fp):
                raise SystemExit(f"missing WAV: {fp}")

    sc = HumanClapScorer()
    grids = {}
    for sysname in SYSTEMS:
        cos = sc.cosine(order_caps[sysname], order_wavs[sysname])
        grids[sysname] = _grid(cos)

    rec, pru = grids["p1_recovered"], grids["p1_pruned_ema_reconstructed"]
    per_prompt = (rec - pru).mean(axis=1)
    ci = cluster_percentile_ci(per_prompt, seed=SEED)
    import transformers
    payload = {
        "artifact": "reversal_humanclap",
        "status": "DIAGNOSTIC / CORROBORATIVE — CLAP-family, NOT human eval; does not redefine R_music",
        "model": MODEL_ID, "processor": PROC_ID, "sampling_rate": SR,
        "convention": "frozen: SR48k, truncation=fusion, get_*_features cosine, seed-once/192 system",
        "lib_versions": {"transformers": transformers.__version__},
        "recovered_off_mean_cosine": float(rec.mean()),
        "pruned_off_mean_cosine": float(pru.mean()),
        "R_music_humanclap": {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "n": ci.n, "b": ci.b,
                              "seed": SEED, "scale": "human-clap cosine (own scale)",
                              "definition": "mean_p mean_r (recovered_off - pruned_off)"},
        "direction": "recovered < pruned" if ci.point < 0 else "recovered >= pruned",
        "prompt_sign_fraction_neg": float((per_prompt < 0).mean()),
        "note_vs_primary_CLAP": "R_music (frozen laion CLAP) = -0.0941 [-0.1241, -0.0646]; the "
                                "0.025 CLAP SESOI does NOT transfer to this scale.",
        "prompts": [{"prompt_index": p, "ytid": ytids[p],
                     "recovered_off": [float(x) for x in rec[p]],
                     "pruned_off": [float(x) for x in pru[p]],
                     "prompt_mean_diff": float(per_prompt[p])} for p in range(N_PROMPTS)],
    }
    payload["artifact_sha256"] = _sha256_obj(payload)
    json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
    open(out_path + ".sha256", "w").write(
        hashlib.sha256(open(out_path, "rb").read()).hexdigest() + "  "
        + os.path.basename(out_path) + "\n")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/icassp_gate0/reversal_humanclap.json")
    args = ap.parse_args()
    payload = run(args.out)
    print(json.dumps({k: payload[k] for k in
                      ("recovered_off_mean_cosine", "pruned_off_mean_cosine",
                       "R_music_humanclap", "direction", "prompt_sign_fraction_neg")}, indent=2))
    print("written to", args.out, "sha256", payload["artifact_sha256"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
