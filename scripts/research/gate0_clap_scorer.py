#!/usr/bin/env python3
"""Gate-0 primary scorer: LAION-CLAP **fused** text-audio cosine (Kim's scorer).

Runs in .venv-metrics (torch 2.2.2, transformers 4.30.2, librosa) — CPU-capable. This is the
frozen primary endpoint tool for ICASSP Gate 0 and the phenomenon falsifier: per (caption, wav)
it returns the cosine between L2-normalized fused-CLAP text and audio embeddings.

Model: laion/clap-htsat-fused (prereg primary_scorer). Audio is resampled to 48 kHz (CLAP's rate).

`--dry-run` self-test (no GPU, local AudioCaps audio): loads the model, checks determinism, and
verifies matched (caption_i, wav_i) cosine beats mismatched (caption_i, wav_{shuffled}) on average
— a sanity gate that the scorer discriminates, mirroring the paired ΔCLAP the gate will use.
"""
import argparse, json, os, sys
import numpy as np

MODEL_ID = "laion/clap-htsat-fused"
SR = 48000
AUDIOCAPS_BASE = "data/dataset/audioset"  # val manifest wav paths are relative to this


class FusedClapScorer:
    def __init__(self, device="cpu"):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device
        self.model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
        self.proc = ClapProcessor.from_pretrained(MODEL_ID)

    @staticmethod
    def _l2(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)

    def _load(self, fp):
        import librosa
        w, _ = librosa.load(fp, sr=SR, mono=True)
        # DETERMINISM: CLAP-fused `rand_trunc` randomly crops audio > 10 s. Deterministically
        # centre-crop to <= 10 s so the fusion path never randomizes. Gate-0 gens are 3.84 s
        # (a no-op here); this only bites the >10 s dry-run clips.
        cap = SR * 10
        if len(w) > cap:
            s = (len(w) - cap) // 2
            w = w[s:s + cap]
        return w.astype(np.float32)

    def text_embed(self, captions):
        inp = self.proc(text=list(captions), return_tensors="pt", padding=True)
        with self.torch.no_grad():
            te = self.model.get_text_features(**{k: v.to(self.device) for k, v in inp.items()})
        return te.cpu().numpy()

    def audio_embed(self, wav_paths):
        wavs = [self._load(fp) for fp in wav_paths]
        # DETERMINISM: CLAP-fused's `_random_mel_fusion` calls np.random for chunk selection when
        # audio > ~10 s. Our Gate-0 gens are 3.84 s (no fusion path) so this is moot in production,
        # but seed np.random so the tool is reproducible on any input.
        np.random.seed(20260826)
        inp = self.proc(audios=wavs, sampling_rate=SR, return_tensors="pt",
                        padding=True, truncation="fusion")
        with self.torch.no_grad():
            ae = self.model.get_audio_features(**{k: v.to(self.device) for k, v in inp.items()})
        return ae.cpu().numpy()

    def cosine(self, captions, wav_paths):
        """Paired per-item cosine between caption_i and wav_i (len must match)."""
        te = self._l2(self.text_embed(captions))
        ae = self._l2(self.audio_embed(wav_paths))
        return np.sum(te * ae, axis=-1)


def _dry_run(n=8):
    items = json.load(open("configs/research/val_split_disjoint.json"))["items"][:n]
    caps = [it["caption"] for it in items]
    wavs = [os.path.join(AUDIOCAPS_BASE, it["wav"]) for it in items]
    missing = [w for w in wavs if not os.path.exists(w)]
    if missing:
        print(f"DRY-RUN FAIL: {len(missing)} local wavs missing, e.g. {missing[0]}", file=sys.stderr)
        return 2
    sc = FusedClapScorer()
    matched = sc.cosine(caps, wavs)
    matched2 = sc.cosine(caps, wavs)                       # determinism
    perm = list(range(1, n)) + [0]                          # fixed derangement
    mismatched = sc.cosine([caps[i] for i in perm], wavs)
    det = float(np.max(np.abs(matched - matched2)))
    res = {
        "model": MODEL_ID, "n": n,
        "matched_mean_cosine": float(matched.mean()),
        "mismatched_mean_cosine": float(mismatched.mean()),
        "margin": float(matched.mean() - mismatched.mean()),
        "determinism_max_abs_diff": det,
        "matched_per_item": [round(float(x), 4) for x in matched],
        "discriminates": bool(matched.mean() > mismatched.mean() + 0.02),
        "deterministic": det < 1e-5,
    }
    print(json.dumps(res, indent=2))
    ok = res["discriminates"] and res["deterministic"]
    print("\nDRY-RUN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    if args.dry_run:
        return _dry_run(args.n)
    print("Import FusedClapScorer and call .cosine(captions, wav_paths). Use --dry-run to self-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
