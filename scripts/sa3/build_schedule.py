#!/usr/bin/env python3
"""Build the frozen SA3 sampling schedules (protocol section 1.4, 1.5), CPU only.

The 8 post noise levels tau_1..tau_8 (ping-pong, rf_denoiser) and the 50 base levels (Euler,
rectified_flow) are computed with the REAL upstream code (`build_schedule` + the checkpoint's
own `sampling_dist_shift`), never typed by hand. Both configs carry
`sampling_distribution_shift_options: null`, so the wrapper's inference default is
`LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)` (diffusion.py:78-79); with rate=0 the
schedule is sequence-length independent. The `distribution_shift_options.type=full` object is
recorded as `reference_full_shift` for audit only. The smoke will re-capture tau via the sampler
callback and MUST match these values -- this file is the pre-registered expectation.

Run:   .venv-sa3/bin/python scripts/sa3/build_schedule.py --write
Check: .venv-sa3/bin/python scripts/sa3/build_schedule.py --check
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys

OUT_DIR = "configs/sa3"
SECONDS_TOTAL = 10
SAMPLE_RATE = 44100
DOWNSAMPLING = 4096
POST_STEPS = 8
BASE_STEPS = 50


def effective_seq_len(seconds, sr, ds):
    return math.ceil(int(seconds * sr) / ds)


def build(steps, shift):
    import torch
    from stable_audio_3.inference.sampling import build_schedule
    esl = effective_seq_len(SECONDS_TOTAL, SAMPLE_RATE, DOWNSAMPLING)
    sig = build_schedule(steps=steps, sigma_max=1.0, dist_shift=shift,
                         effective_seq_len=esl, fallback_seq_len=esl,
                         include_endpoint=True, device="cpu")
    return [float(x) for x in sig.tolist()], esl


def make_shifts():
    from stable_audio_3.inference.distribution_shift import LogSNRShift, DistributionShift
    # inference default (sampling_distribution_shift_options == null -> this fallback)
    sampling = LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)
    # training/reference shift from distribution_shift_options.type == full
    full = DistributionShift(min_length=256, max_length=4096)
    return sampling, full


def serialize(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (a.write or a.check):
        ap.error("pass --write or --check")

    sampling, full = make_shifts()
    esl = effective_seq_len(SECONDS_TOTAL, SAMPLE_RATE, DOWNSAMPLING)

    post_sig, _ = build(POST_STEPS, sampling)
    base_sig, _ = build(BASE_STEPS, sampling)
    post_full, _ = build(POST_STEPS, full)

    files = {
        "schedule_post_10s.json": {
            "role": "post", "diffusion_objective": "rf_denoiser", "sampler": "pingpong",
            "steps": POST_STEPS, "seconds_total": SECONDS_TOTAL, "sample_rate": SAMPLE_RATE,
            "downsampling_ratio": DOWNSAMPLING, "effective_seq_len": esl,
            "shift": {"type": "LogSNRShift", "rate": 0, "anchor_logsnr": -6.2, "logsnr_end": 2.0,
                      "source": "config sampling_distribution_shift_options=null -> wrapper fallback"},
            "sigmas": post_sig,  # length steps+1 = 9: sigmas[0]=1.0 ... sigmas[8]=0.0
            "tau_levels": post_sig[:-1],  # the 8 states captured (i=0..7)
            "reference_full_shift": {
                "shift": {"type": "DistributionShift(full)", "min_length": 256, "max_length": 4096},
                "sigmas": post_full,
            },
        },
        "schedule_base_10s.json": {
            "role": "base", "diffusion_objective": "rectified_flow", "sampler": "euler",
            "steps": BASE_STEPS, "seconds_total": SECONDS_TOTAL, "sample_rate": SAMPLE_RATE,
            "downsampling_ratio": DOWNSAMPLING, "effective_seq_len": esl,
            "shift": {"type": "LogSNRShift", "rate": 0, "anchor_logsnr": -6.2, "logsnr_end": 2.0,
                      "source": "config sampling_distribution_shift_options=null -> wrapper fallback"},
            "sigmas": base_sig,
        },
    }

    rc = 0
    for fn, obj in files.items():
        path = os.path.join(OUT_DIR, fn)
        data = serialize(obj)
        digest = hashlib.sha256(data).hexdigest()
        if a.check:
            cur = open(path, "rb").read() if os.path.exists(path) else b""
            same = cur == data
            print(f"{'OK  ' if same else 'DIFF'} {path}  sha256={digest}")
            rc |= 0 if same else 1
        else:
            open(path, "w").write(data.decode())
            print(f"WROTE {path}  sha256={digest}")
            print(f"       tau (post first/last of {obj['steps']}): {obj['sigmas'][0]:.6f} .. {obj['sigmas'][-1]:.6f}"
                  if 'post' in fn else f"       base sigmas[0..1]={obj['sigmas'][0]:.4f},{obj['sigmas'][1]:.4f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
