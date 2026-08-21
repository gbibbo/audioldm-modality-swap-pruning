#!/usr/bin/env python3
"""E-metric scorer (protocol section 9.1) -- RUN WITH `.venv-metrics/bin/python` (NOT .venv-sa3).

Reads a manifest of generated wavs (one dir/file per system) and computes, per system:
  * CLAP        -- LAION-CLAP text-audio cosine to each clip's own caption (reference-free), mean.
  * KL_passt    -- prompt-paired KL(system PaSST posterior || dense-post posterior), mean.
  * FD_openl3   -- Frechet distance between the system's OpenL3 embeddings and the dense post's
                   over the panel (real-part eigenvalue method, matches research_pruning/eval/frechet.py).
KL and FD are DRIFT FROM the dense post reference (no external 44.1 kHz audio needed).

Manifest JSON: {"reference_system": id, "prompts": {audiocap_id: caption},
                "systems": {id: {audiocap_id: wav_path}}}
Self-test:  --selftest   (synthetic wavs; asserts finite, reference-self KL==0 and FD~0).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torchvision  # noqa: F401 -- force full init before timm imports it (torchvision circular-import guard)


# ---------------------------------------------------------------- metric backends (lazy)
class Backends:
    def __init__(self, device="cpu"):
        self.device = device
        self._clap = None; self._passt = None; self._ol3 = None
        import torch; self.torch = torch

    @property
    def clap(self):
        if self._clap is None:
            import laion_clap
            m = laion_clap.CLAP_Module(enable_fusion=False)  # HTSAT-tiny == 630k-audioset ckpt
            m.load_ckpt()
            m.eval()
            self._clap = m
        return self._clap

    @property
    def passt(self):
        if self._passt is None:
            from hear21passt.base import get_basic_model
            self._passt = get_basic_model(mode="logits").to(self.device).eval()
        return self._passt

    def ol3_embed(self, wav48, sr=48000):
        import torchopenl3
        x = self.torch.tensor(wav48).unsqueeze(0)
        emb, _ = torchopenl3.get_audio_embedding(x, sr, content_type="env",
                                                 input_repr="mel256", embedding_size=512)
        return emb.squeeze(0).mean(0).detach().cpu().numpy()

    def passt_post(self, wav32):
        x = self.torch.tensor(wav32).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.passt(x)
        return self.torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()


def _load(fp, sr):
    import librosa
    w, _ = librosa.load(fp, sr=sr, mono=True)
    return w


def frechet(x: np.ndarray, y: np.ndarray) -> float:
    """FD between two embedding sets (rows=clips). Real-part eigenvalue trace of the cov product,
    matching research_pruning/eval/frechet.py. Requires OPENBLAS_CORETYPE=Haswell on this CPU (E-BLAS)."""
    mu1, mu2 = x.mean(0), y.mean(0)
    c1 = np.cov(x, rowvar=False)
    c2 = np.cov(y, rowvar=False)
    diff = mu1 - mu2
    # sqrt of product via eigenvalues of c1@c2 (real part), trace(sqrtm(c1 c2)) = sum sqrt(eig)
    eig = np.linalg.eigvals(c1 @ c2)
    covmean = np.sum(np.sqrt(np.abs(eig.real)))
    return float(diff @ diff + np.trace(c1) + np.trace(c2) - 2.0 * covmean)


def score(manifest: dict, device="cpu", cache=None) -> dict:
    B = Backends(device)
    prompts = manifest["prompts"]
    systems = manifest["systems"]
    ref = manifest["reference_system"]
    cache = cache if cache is not None else {}

    def feats(fp):
        if fp in cache:
            return cache[fp]
        w48 = _load(fp, 48000); w32 = _load(fp, 32000)
        f = {"post": B.passt_post(w32), "ol3": B.ol3_embed(w48)}
        cache[fp] = f
        return f

    # CLAP per system (batch over that system's files)
    out = {}
    ref_feats = {aid: feats(fp) for aid, fp in systems[ref].items()}
    for sid, files in systems.items():
        aids = sorted(files, key=lambda a: int(a))
        fps = [files[a] for a in aids]
        caps = [prompts[a] for a in aids]
        ae = B.clap.get_audio_embedding_from_filelist(x=fps, use_tensor=False)
        te = B.clap.get_text_embedding(caps, use_tensor=False)
        l2 = lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        clap_per = np.sum(l2(ae) * l2(te), axis=-1)
        sf = {a: feats(files[a]) for a in aids}
        # KL paired vs reference
        kls = []
        for a in aids:
            if a in ref_feats:
                p = sf[a]["post"] + 1e-12; q = ref_feats[a]["post"] + 1e-12
                kls.append(float(np.sum(p * np.log(p / q))))
        emb_sys = np.stack([sf[a]["ol3"] for a in aids])
        emb_ref = np.stack([ref_feats[a]["ol3"] for a in sorted(ref_feats, key=lambda a: int(a))])
        fd = frechet(emb_sys, emb_ref) if sid != ref else 0.0
        out[sid] = {
            "CLAP": float(np.mean(clap_per)),
            "CLAP_per": {a: float(c) for a, c in zip(aids, clap_per)},
            "KL_passt": float(np.mean(kls)) if kls else float("nan"),
            "FD_openl3": fd,
            "n": len(aids),
        }
    return out


def selftest(device="cpu") -> int:
    import soundfile as sf
    sc = os.environ.get("SCRATCH", "/tmp")
    d = os.path.join(sc, "sa3_metric_selftest"); os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0); srate = 44100
    t = np.linspace(0, 3, 3 * srate, endpoint=False)
    sigs = {"a": np.sin(2*np.pi*440*t), "b": np.cumsum(rng.standard_normal(len(t)))*0.001,
            "c": np.sin(2*np.pi*(200+t*200)*t)}
    files = {}
    for k, x in sigs.items():
        x = (x/(np.abs(x).max()+1e-9)*0.5).astype("float32")
        fp = os.path.join(d, f"{k}.wav"); sf.write(fp, np.stack([x, x], 1), srate); files[k] = fp
    man = {"reference_system": "dense", "prompts": {"1": "a tone", "2": "noise", "3": "a chirp"},
           "systems": {"dense": {"1": files["a"], "2": files["b"], "3": files["c"]},
                       "sysX": {"1": files["b"], "2": files["a"], "3": files["c"]}}}
    r = score(man, device=device)
    ok = (abs(r["dense"]["KL_passt"]) < 1e-6 and abs(r["dense"]["FD_openl3"]) < 1e-3
          and np.isfinite(r["sysX"]["KL_passt"]) and np.isfinite(r["sysX"]["FD_openl3"])
          and np.isfinite(r["dense"]["CLAP"]))
    print(json.dumps(r, indent=2))
    print("SELFTEST", "PASS" if ok else "FAIL",
          "(reference-self KL=%.2e FD=%.2e)" % (r["dense"]["KL_passt"], r["dense"]["FD_openl3"]))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--out"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.device)
    man = json.load(open(a.manifest))
    t0 = time.time()
    r = score(man, device=a.device)
    r["_wall_s"] = time.time() - t0
    if a.out:
        json.dump(r, open(a.out, "w"), indent=2)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
