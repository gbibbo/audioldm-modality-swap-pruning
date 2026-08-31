#!/usr/bin/env python3
"""A0 — FineLAP EXACT temporal-geometry audit (CPU). GATE for the Part-A temporal profile.

Establishes, by code inspection AND empirical verification on the pinned local
`AndreasXi/FineLAP` snapshot, the exact waveform->mel->patch->frame mapping for our
native 10.24-s / 16-kHz generations, so the EARLY/LATE split at 3.84 s is unambiguous.

Answers the seven A0 questions:
  1. exact waveform samples consumed for a 10.24-s 16-kHz WAV;
  2. whether any samples are dropped/padded;
  3. feature (mel) frame count before the encoder;
  4. dense output frame count;
  5. temporal stride / effective frame-center mapping;
  6. 0.15625 vs 0.16 applicability;
  7. whether contextualization makes a frame embedding depend on the whole sequence.

Verdict PASS requires ALL empirical checks to confirm the analytic mapping. If any is
ambiguous -> STOP (do not score generated audio).

Run (CPU):
  OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/finelap_geometry_audit.py \
      --out artifacts/finelap_temporal/geometry_audit.json
"""
from __future__ import annotations
import argparse, importlib, json, math, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMD = ("/teamspace/jobs/reversal-armd-gen-1/artifacts/audioldm-modality-swap-pruning/"
        "artifacts/icassp_gate0/reversal_armd_gen")
META = os.path.join(ROOT, "data/dataset/metadata/audiocaps")
CSV_PATH = os.path.join(META, "class_labels_indices.csv")
TRAIN_JSON = os.path.join(META, "datafiles/audiocaps_train_label.json")
ROOT_JSON = os.path.join(ROOT, "data/dataset/metadata/dataset_root.json")

SR = 16000
CLIP_SECONDS = 10.24
BOUNDARY_S = 3.84


def load_finelap(model_dir):
    """Load FineLAP on CPU via local-package import (same env-gap adapters as the smoke)."""
    import torch  # noqa
    from transformers import PretrainedConfig
    if not getattr(PretrainedConfig, "_finelap_nested_shim", False):
        _orig = PretrainedConfig.to_dict

        def _rec(self):
            d = _orig(self)
            for k, v in list(d.items()):
                if isinstance(v, PretrainedConfig):
                    d[k] = v.to_dict()
            return d
        PretrainedConfig.to_dict = _rec
        PretrainedConfig._finelap_nested_shim = True

    model_dir = os.path.abspath(model_dir)
    ext_parent, pkg = os.path.dirname(model_dir), os.path.basename(model_dir)
    if not os.path.exists(os.path.join(model_dir, "__init__.py")):
        open(os.path.join(model_dir, "__init__.py"), "w").close()
    if ext_parent not in sys.path:
        sys.path.insert(0, ext_parent)
    cfgmod = importlib.import_module(f"{pkg}.configuration_finelap")
    modmod = importlib.import_module(f"{pkg}.modeling_finelap")
    cfg = cfgmod.FineLAPConfig(**json.load(open(os.path.join(model_dir, "config.json"))))
    model = modmod.FineLAPModel(cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, len(missing), len(unexpected)


def kaldi_num_frames(n_samples, sr=SR, frame_ms=25.0, shift_ms=10.0, snip_edges=True):
    """torchaudio.compliance.kaldi frame count (snip_edges=True default)."""
    win = int(round(sr * frame_ms / 1000.0))     # 400
    hop = int(round(sr * shift_ms / 1000.0))      # 160
    if snip_edges:
        return 0 if n_samples < win else 1 + (n_samples - win) // hop
    return int(math.floor(n_samples / hop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "_external/FineLAP"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/finelap_temporal/geometry_audit.json"))
    a = ap.parse_args()
    import numpy as np
    import torch
    import torchaudio

    R = {"model": a.model, "clip_seconds": CLIP_SECONDS, "boundary_s": BOUNDARY_S, "checks": {}}
    R["env"] = {"torch": torch.__version__, "torchaudio": torchaudio.__version__}

    model, miss, unexp = load_finelap(a.model)
    R["checks"]["loads"] = True
    R["checks"]["missing_keys"], R["checks"]["unexpected_keys"] = miss, unexp

    # ---- config-derived geometry
    ac = model.config.audio_config
    img_t, img_f = ac.img_size
    ps = ac.patch_size
    n_time_patch = img_t // ps           # 64
    n_freq_patch = img_f // ps           # 8
    R["config"] = {"img_size": [img_t, img_f], "patch_size": ps, "stride": ac.stride,
                   "n_time_patch": n_time_patch, "n_freq_patch": n_freq_patch,
                   "target_len_mel": 1024, "frame_shift_ms": 10, "depth": ac.depth}

    # ---- (1)(2)(3) Kaldi mel geometry for our native clips
    real_native = 163840                 # 10.24 s * 16 kHz, exact
    gen_native = 163872                  # AudioLDM native sample count (from gen manifests)
    for tag, ns in [("exact_10p24s_16k", real_native), ("audioldm_native", gen_native)]:
        nf = kaldi_num_frames(ns)
        R["checks"][f"mel_frames_{tag}"] = {
            "n_samples": ns, "kaldi_mel_frames": nf, "target_len": 1024,
            "truncated": nf > 1024, "zero_padded_frames": max(0, 1024 - nf),
            "samples_dropped": max(0, ns - (400 + (min(nf, 1024) - 1) * 160)) if nf <= 1024 else "TRUNCATION"}

    # empirical mel count via the model's own load_audio path (pre-pad) — reproduce fbank
    wavs = sorted([f for f in os.listdir(ARMD) if f.endswith(".wav") and "alt10s" in f])
    assert wavs, "no Arm-D native WAVs found"
    demo_wav = os.path.join(ARMD, wavs[0])
    wav, sr = torchaudio.load(demo_wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav1 = wav.squeeze(0)
    wav1 = wav1 - wav1.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        wav1.unsqueeze(0), htk_compat=True, sample_frequency=SR, use_energy=False,
        window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
    R["checks"]["empirical_demo_wav"] = os.path.basename(demo_wav)
    R["checks"]["empirical_demo_samples"] = int(wav1.shape[0])
    R["checks"]["empirical_mel_frames_prepad"] = int(mel.shape[0])

    # ---- (4) dense output frame count on a REAL native WAV
    dense = model.get_dense_audio_embeds([demo_wav])     # (1, T, D)
    R["checks"]["dense_output_frames"] = int(dense.shape[1])
    R["checks"]["dense_frames_eq_64"] = int(dense.shape[1]) == n_time_patch

    # ---- (3b/4b) verify encode_audio internal shapes
    mel_batch = model.load_audio([demo_wav])             # (1,1,1024,128)
    R["checks"]["mel_batch_shape"] = list(mel_batch.shape)
    raw = model.audio_encoder.extract_features(mel_batch)
    raw = raw["x"] if isinstance(raw, dict) else raw
    R["checks"]["encoder_tokens"] = list(raw.shape)      # (1, 1+512, 768)
    n_patches = raw.shape[1] - 1
    R["checks"]["n_patches_eq_512"] = n_patches == n_time_patch * n_freq_patch

    # ---- (3) PATCH ORDERING: prove time-major (p = h*8 + w) by perturbation of one time block
    patch_embed = model.audio_encoder.model.local_encoder   # PatchEmbed_new
    with torch.no_grad():
        base = torch.zeros(1, 1, img_t, img_f)
        out0 = patch_embed(base)                            # (1,512,768) bias-only baseline
        k = 5                                               # perturb time-patch row k
        pert = base.clone()
        pert[:, :, ps * k:ps * (k + 1), :] += 1.0           # all freq at time-block k
        out1 = patch_embed(pert)
        changed = (out1 - out0).abs().sum(-1).squeeze(0)    # (512,) per-patch change magnitude
        changed_idx = torch.nonzero(changed > 1e-6).squeeze(-1).tolist()
    expected = list(range(n_freq_patch * k, n_freq_patch * (k + 1)))   # [40..47]
    R["checks"]["patch_ordering_time_major"] = (changed_idx == expected)
    R["checks"]["patch_probe"] = {"perturbed_time_block": k, "changed_patch_idx": changed_idx,
                                  "expected_contiguous_group": expected}

    # ---- (4c) reshape(B,64,8,D).mean groups by TIME (math identity given p=h*8+w)
    # patch p = h*8 + w  ->  p // 8 == h (time), p % 8 == w (freq); reshape(64,8) groups by h.
    idx = torch.arange(512).reshape(64, 8)
    R["checks"]["reshape_groups_by_time"] = bool((idx // 8 == torch.arange(64)[:, None]).all())

    # ---- (5)(6) TIME MAPPING
    res = CLIP_SECONDS / n_time_patch          # 0.16
    R["time_mapping"] = {
        "seconds_per_frame": res,
        "frame_i_covers_s": "[0.16*i, 0.16*(i+1))",
        "frame_center_s": "0.16*i + 0.08 (window mid); +~0.0125 kaldi offset negligible",
        "eval_doc_0p15625": "= 10.0/64, assumes a 10.0-s clip; NOT applicable to our 10.24-s WAVs",
        "applicable": 0.16, "boundary_frame_index": int(round(BOUNDARY_S / res)),
        "boundary_is_exact_patch_boundary": abs(BOUNDARY_S / res - round(BOUNDARY_S / res)) < 1e-9,
        "EARLY_frames": [0, int(round(BOUNDARY_S / res)) - 1],
        "LATE_frames": [int(round(BOUNDARY_S / res)), n_time_patch - 1]}

    # ---- (5b) EMPIRICAL localization: real event placed early vs late must light the right frames
    loc = localization_test(model, torch, torchaudio, np)
    R["localization"] = loc
    R["checks"]["localization_early_beats_late"] = loc["ok"]

    # ---- (7) CONTEXTUALIZATION: architectural (global self-attention, no causal mask) + demo
    ctx = contextualization_test(model, torch)
    R["contextualization"] = ctx
    R["checks"]["contextualized_global"] = ctx["late_change_affects_early_frames"]

    keys = ["loads", "dense_frames_eq_64", "n_patches_eq_512", "patch_ordering_time_major",
            "reshape_groups_by_time", "localization_early_beats_late"]
    ok = all(bool(R["checks"].get(k)) for k in keys)
    # boundary must be exact and mel not truncated
    ok = ok and R["time_mapping"]["boundary_is_exact_patch_boundary"]
    ok = ok and not R["checks"]["mel_frames_audioldm_native"]["truncated"]
    R["verdict"] = "PASS" if ok else "FAIL"
    R["gate_keys"] = keys

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps({k: R["checks"][k] for k in keys}, indent=1))
    print("time_mapping:", json.dumps(R["time_mapping"]))
    print("localization:", json.dumps({k: loc[k] for k in ("early_ratio", "late_ratio", "ok")}))
    print("contextualization:", json.dumps(ctx))
    print("VERDICT:", R["verdict"])
    print("wrote", a.out)
    return 0 if ok else 1


def _pick_event_clip():
    import csv
    mid2name = {r["mid"]: r["display_name"].split(",")[0].strip()
                for r in csv.DictReader(open(CSV_PATH))}
    audio_root = json.load(open(ROOT_JSON))["audiocaps"]
    data = json.load(open(TRAIN_JSON))["data"]
    # sustained, temporally-fillable single-label events for a clean early/late contrast
    prefer = ["Applause", "Siren", "Rain", "Stream", "Engine", "Whistling"]
    by = {}
    for e in data:
        mids = [m for m in e["labels"].split(",") if m in mid2name]
        if len(mids) != 1:
            continue
        nm = mid2name[mids[0]]
        if nm in prefer and nm not in by:
            p = os.path.join(audio_root, e["wav"])
            if os.path.exists(p):
                by[nm] = {"wav": p, "event": nm}
    for nm in prefer:
        if nm in by:
            return by[nm]
    raise SystemExit("no sustained single-label clip found for localization test")


def localization_test(model, torch, torchaudio, np):
    """Place a real event in [0,3.84) vs [3.84,10.24) of an otherwise-silent 10.24-s clip;
    the event-phrase frame scores must localize to the correct window."""
    import tempfile, soundfile as sf
    clip = _pick_event_clip()
    wav, sr = torchaudio.load(clip["wav"])
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    x = wav.squeeze(0).numpy()
    N = int(round(CLIP_SECONDS * SR))            # 163840
    b = int(round(BOUNDARY_S * SR))              # 61440
    seg_early = x[:b] if len(x) >= b else np.pad(x, (0, b - len(x)))
    late_len = N - b
    seg_late = x[:late_len] if len(x) >= late_len else np.pad(x, (0, late_len - len(x)))
    early_sig = np.concatenate([seg_early, np.zeros(N - b)]).astype(np.float32)
    late_sig = np.concatenate([np.zeros(b), seg_late]).astype(np.float32)

    d = tempfile.mkdtemp(prefix="finelap_loc_")
    pe, pl = os.path.join(d, "early.wav"), os.path.join(d, "late.wav")
    sf.write(pe, early_sig, SR); sf.write(pl, late_sig, SR)
    with torch.no_grad():
        s = model.get_frame_level_score([pe, pl], [clip["event"]]).squeeze(1)   # (2, 64)
    s = s.cpu().float().numpy()
    bf = int(round(BOUNDARY_S / (CLIP_SECONDS / s.shape[1])))   # 24
    early_scores, late_scores = s[0], s[1]
    er = float(early_scores[:bf].mean() / (early_scores[bf:].mean() + 1e-9))
    lr = float(late_scores[bf:].mean() / (late_scores[:bf].mean() + 1e-9))
    ok = (early_scores[:bf].mean() > early_scores[bf:].mean()) and \
         (late_scores[bf:].mean() > late_scores[:bf].mean())
    return {"event": clip["event"], "boundary_frame": bf,
            "early_sig_early_mean": float(early_scores[:bf].mean()),
            "early_sig_late_mean": float(early_scores[bf:].mean()),
            "late_sig_early_mean": float(late_scores[:bf].mean()),
            "late_sig_late_mean": float(late_scores[bf:].mean()),
            "early_ratio": er, "late_ratio": lr, "ok": bool(ok)}


def contextualization_test(model, torch):
    """Perturb ONLY the late mel region; if early-frame embeddings change, frames are
    globally contextualized (not local)."""
    import numpy as np, tempfile, soundfile as sf
    N = int(round(CLIP_SECONDS * SR)); b = int(round(BOUNDARY_S * SR))
    rng = np.random.default_rng(0)
    base = np.zeros(N, dtype=np.float32)
    pert = base.copy(); pert[b:] = 0.1 * rng.standard_normal(N - b).astype(np.float32)
    d = tempfile.mkdtemp(prefix="finelap_ctx_")
    p0, p1 = os.path.join(d, "z.wav"), os.path.join(d, "latenoise.wav")
    sf.write(p0, base, SR); sf.write(p1, pert, SR)
    with torch.no_grad():
        e0 = model.get_dense_audio_embeds([p0])[0]     # (64, D)
        e1 = model.get_dense_audio_embeds([p1])[0]
    bf = 24
    early_delta = float((e1[:bf] - e0[:bf]).abs().mean())
    late_delta = float((e1[bf:] - e0[bf:]).abs().mean())
    return {"early_embed_delta": early_delta, "late_embed_delta": late_delta,
            "late_change_affects_early_frames": early_delta > 1e-6,
            "architectural": "12 AltBlocks, full bidirectional self-attention, no causal/window mask"}


if __name__ == "__main__":
    sys.exit(main())
