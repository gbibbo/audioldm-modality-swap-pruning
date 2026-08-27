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
import argparse, hashlib, json, os, sys
# E-BLAS guard: set before importing numpy so this CPU never hits the OpenBLAS wrong-result defect.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np

MODEL_ID = "laion/clap-htsat-fused"
# PINNED HF revision (reproducibility control, prereg v5 item 4). Resolved from the Studio HF
# cache refs/main on 2026-08-27; the exact snapshot that scored Gate 0. Frozen for Gate 0 AND
# every phenomenon scoring run so no scorer-version/session drift enters the dense-vs-downstream
# comparison. Do NOT bump without a new reproducibility control + Gabriel's sign-off.
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
SR = 48000
AUDIOCAPS_BASE = "data/dataset/audioset"  # val manifest wav paths are relative to this


def _lib_versions():
    import torch, transformers
    try:
        import librosa; lv = librosa.__version__
    except Exception:
        lv = None
    return {"torch": torch.__version__, "transformers": transformers.__version__, "librosa": lv}


class FusedClapScorer:
    def __init__(self, device="cpu", revision=REVISION):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device
        self.revision = revision
        # revision-pinned load; the snapshot is present in the HF cache so this resolves offline.
        self.model = ClapModel.from_pretrained(MODEL_ID, revision=revision).to(device).eval()
        self.proc = ClapProcessor.from_pretrained(MODEL_ID, revision=revision)

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
        # DETERMINISM (measured, 2026-08-27): CLAP-fused's mel-fusion DOES consume np.random even for
        # our 3.84-s clips (an earlier "no fusion path at 3.84 s" note was WRONG — audio embeddings are
        # batch/order-sensitive: max|Δemb|~0.08 when re-chunked). We seed np.random ONCE here, so a
        # single fixed-order batch is fully reproducible; the Gate-0 verdict was scored exactly this way
        # (one seed-once batch per system, 192 items). Scoring MUST keep this seed-once-per-batch,
        # fixed-order arrangement (Option B, DECISION-V4-13/PHENOM-SCORING-B) — do NOT re-chunk a batch,
        # or the fusion draws (hence the scores) change.
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


def _git_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# FROZEN scoring convention (Option B, DECISION-V4-13 / PHENOM-SCORING-B, 2026-08-27):
# The pinned fused-CLAP feature extractor (transformers 4.30.2) contains a BATCH-LEVEL stochastic
# is_longer selection for short-audio batches (when all audios are shorter than the fusion window it
# randomly marks ONE batch member is_longer=True). We therefore FREEZE the original Gate-0 evaluation
# unit — exactly 192 samples per system, item order = (prompt_index, replicate_index), np.random.seed
# 20260826 reset ONCE immediately before preprocessing each complete 192-item system — and apply the
# identical convention to every experimental system. Batch size, order and RNG seed are PART OF the
# frozen endpoint. Because the same seed restarts per system, the batch-level nuisance is paired by
# position across all six systems. Do NOT score 1152 as one batch, in 32/64 chunks, or per-item.
SCORING_CONVENTION = {
    "unit": "one 192-item scorer call per system (64 prompts x 3 replicates)",
    "order": "(prompt_index, replicate_index)",
    "rng": "np.random.seed(20260826) reset once before each 192-item system",
    "quirk": "transformers 4.30.2 CLAP-fused has a batch-level stochastic is_longer selection for "
             "short-audio batches; frozen batch/order/seed makes it reproducible AND paired by "
             "position across systems",
    "revision": REVISION,
}


def _prov():
    return {"model": MODEL_ID, "hf_revision": REVISION, "lib_versions": _lib_versions(),
            "scoring_git_sha": _git_sha(), "sr": SR, "convention": SCORING_CONVENTION}


def score_json(in_path, out_path):
    """Read {items:[{caption,wav}]} -> write {cosines:[...]} (per-item fused-CLAP text-audio cosine).
    ONE seed-once batch (the frozen Gate-0 per-system convention). Stamps scorer provenance."""
    items = json.load(open(in_path))["items"]
    sc = FusedClapScorer()
    cos = sc.cosine([it["caption"] for it in items], [it["wav"] for it in items])
    out = {"cosines": [float(x) for x in cos], "n": len(items),
           "model": MODEL_ID, "revision": REVISION, "scorer_provenance": _prov()}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def score_groups(in_path, out_path):
    """Read {groups:[{name, items:[{caption,wav}]}]} -> write {results:[{name, cosines, n}]}.
    Model loaded ONCE; EACH group scored as one independent seed-once batch (the frozen Gate-0
    per-system convention). This is the phenomenon scoring entry: one 192-item call per system,
    never one 1152 batch and never arbitrary chunks."""
    groups = json.load(open(in_path))["groups"]
    sc = FusedClapScorer()                       # model loaded exactly once
    results = []
    for g in groups:
        items = g["items"]
        cos = sc.cosine([it["caption"] for it in items], [it["wav"] for it in items])
        results.append({"name": g["name"], "n": len(items), "cosines": [float(x) for x in cos]})
    out = {"results": results, "model": MODEL_ID, "revision": REVISION, "scorer_provenance": _prov()}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"scored {len(results)} systems, {sum(r['n'] for r in results)} items -> {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--score-json", nargs=2, metavar=("IN", "OUT"),
                    help="score an {items:[{caption,wav}]} manifest -> {cosines:[...]}")
    ap.add_argument("--score-groups", nargs=2, metavar=("IN", "OUT"),
                    help="score {groups:[{name,items}]} -> {results:[{name,cosines}]}; model once, "
                         "one seed-once batch per group (frozen per-system convention)")
    args = ap.parse_args()
    if args.dry_run:
        return _dry_run(args.n)
    if args.score_json:
        return score_json(args.score_json[0], args.score_json[1])
    if args.score_groups:
        return score_groups(args.score_groups[0], args.score_groups[1])
    print("Import FusedClapScorer and call .cosine(captions, wav_paths). Use --dry-run to self-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
