#!/usr/bin/env python3
"""M0 closure: reproduce the PANNs Top-K semantic pipeline (PruningAudioLDM README section 5).

Loads the pretrained PANNs Cnn14 (16 kHz, mAP=0.438) and, for every wav in a
folder, returns the Top-K predicted AudioSet sound events (index, label, prob).
This is the machinery the semantic-quality analysis uses to compare predicted
events before/after pruning and recovery; here it is exercised on real AudioCaps
clips to prove the pipeline is reproduced and ready (no pruned/generated audio
exists yet — that is M4/M5).

Requires (both gitignored, fetched during M0):
  * ckpt/Cnn14_16k_mAP=0.438.pth  (PANNs 16 kHz, CPU-sanitised)
  * artifacts/m0_baseline_reproduction/class_labels_indices.csv  (AudioSet 527 labels)

    .venv/bin/python scripts/research/panns_topk.py --dir DIR --k 10 --out JSON
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import librosa

from audioldm_eval.feature_extractors.panns import Cnn14

LABELS_CSV = "artifacts/m0_baseline_reproduction/class_labels_indices.csv"


def load_labels(path: str):
    labels = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            labels[int(row["index"])] = row["display_name"]
    return labels


def build_cnn14():
    # Same hyperparameters audioldm_eval uses for 16 kHz; __init__ loads ckpt/.
    return Cnn14(
        features_list=["clipwise_output"],
        sample_rate=16000, window_size=512, hop_size=160,
        mel_bins=64, fmin=50, fmax=8000, classes_num=527,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of wavs to classify")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="max files")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = load_labels(LABELS_CSV)
    print(f"loaded {len(labels)} AudioSet labels")

    model = build_cnn14()
    model.eval()
    print("Cnn14 16k loaded")

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(".wav"))
    if args.limit:
        files = files[: args.limit]

    results = {}
    for name in files:
        wav, _ = librosa.load(os.path.join(args.dir, name), sr=16000, mono=True)
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        probs = out["clipwise_output"].squeeze(0).cpu().numpy()
        top = np.argsort(probs)[::-1][: args.k]
        results[name] = [
            {"index": int(i), "label": labels.get(int(i), "?"), "prob": float(probs[i])}
            for i in top
        ]
        top3 = ", ".join(f"{r['label']}({r['prob']:.2f})" for r in results[name][:3])
        print(f"  {name:<24} top-{args.k}[0:3]: {top3}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"dir": args.dir, "k": args.k, "n_files": len(files),
                       "topk": results}, handle, indent=2)
        print(f"\nwrote {args.out}")
    print(f"\nPANNs top-{args.k} pipeline: classified {len(files)} clips, exit 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
