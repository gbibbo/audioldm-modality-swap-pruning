#!/usr/bin/env python3
"""A10 — credit estimate for the GPU runs the external reviewer asked for (CPU, 0 cr, read-only).

Action A10 of docs/review/2026-09-03_manuscript_draft5_icassp_reviewer_simulation.md: duration sweep,
published-recipe spot check, short-duration fine-tune. Gabriel asked for the credit cost BEFORE any
new generation is authorised, so nothing here launches anything.

Everything is derived from SETTLED T4 job costs recorded in docs/compute_budget.md. Two models:

  cost per WAV (DDIM 50)      cr(L) = a + b*L,  L = latent length (96 = 3.84 s, 256 = 10.24 s)
                              a, b fitted to the settled short jobs (mean) and the settled native job.
  per-clip GPU time           t(steps, L) = c + d*steps*L, calibrated on the two measured throughputs
                              (7.14 s/clip at DDIM 50 / L=96; 13.9 s/clip at DDIM 50 / L=256).
                              A different sampler budget enters as the ratio t(steps,L)/t(50,L).
  per-job overhead            F = 0.145 cr, the residual of the model against the settled
                              xsev-dense-192-1 job (provisioning + 2 checkpoint loads + device check).

The training line (E3) is SPECULATIVE, not derived: it assumes fwd+bwd+opt ~ 3x a forward pass and
batch 8, neither of which has been measured in this repository. It must not be used to authorise a run
without a short measured benchmark first.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/a10_gpu_cost_estimate.py
"""
from __future__ import annotations

# ---- MEASURED: settled T4 job costs (docs/compute_budget.md) -------------------------------------
SETTLED = {
    "gate0-phenom-1":      (768, 96, 1.6020),
    "gate0-gen-1":         (384, 96, 0.8844),
    "reversal-v11-gen-1":  (576, 96, 1.2620),
    "xsev-music-native-1": (128, 256, 0.4651),
}
DENSE192_SETTLED = 1.2633          # 192 WAVs at L=96 + 192 at L=256, 2 checkpoint loads
T_CLIP_50_96 = 7.14                # s/clip, measured
T_CLIP_50_256 = 13.9               # s/clip, derived from the settled native job at ~0.94 cr/hr
CR_PER_HOUR = 0.94                 # T4, measured


def main():
    short = sum(c / n for n, L, c in SETTLED.values() if L == 96) / 3
    native = SETTLED["xsev-music-native-1"][2] / SETTLED["xsev-music-native-1"][0]
    b = (native - short) / (256 - 96)
    a = short - 96 * b
    F = DENSE192_SETTLED - (192 * (a + 96 * b) + 192 * (a + 256 * b))

    def w(L):
        return a + b * L

    d = (T_CLIP_50_256 - T_CLIP_50_96) / (50 * (256 - 96))
    c = T_CLIP_50_96 - d * 50 * 96

    def mult(L, steps):
        return (c + d * steps * L) / (c + d * 50 * L)

    print(f"cr/WAV  L=96 {short:.6f}  L=256 {native:.6f}   fit a={a:.6f} b={b:.3e}   F={F:.3f} cr/job")
    print(f"time    c={c:.2f} s  d={d:.3e} s/(step*latent)")
    print(f"DDIM200 multiplier: L=96 {mult(96,200):.2f}x  L=256 {mult(256,200):.2f}x\n")

    JOBS = [
        ("E1  duration sweep 5.12+7.68 s, sev 2, P/P+FT/dense, 192 prompts",
         [(576, w(128)), (576, w(192))]),
        ("E1b same, without the dense control", [(384, w(128)), (384, w(192))]),
        ("E1c one point beyond the training length (15.36 s), 3 systems", [(576, w(384))]),
        ("E2a published recipe (DDIM 200, g 3.5), 192 prompts, P/P+FT, both durations",
         [(384, w(96) * mult(96, 200)), (384, w(256) * mult(256, 200))]),
        ("E2b published recipe, 64-prompt subset, P/P+FT, both durations",
         [(128, w(96) * mult(96, 200)), (128, w(256) * mult(256, 200))]),
        ("E2c E2b plus best-of-3, as in the published recipe",
         [(384, w(96) * mult(96, 200)), (384, w(256) * mult(256, 200))]),
        ("E4  held-out domain, AudioCaps-length captions, P/P+FT/dense, both durations",
         [(576, w(96)), (576, w(256))]),
        ("E4b same, without the dense control", [(384, w(96)), (384, w(256))]),
    ]
    print(f"{'item':76s} {'WAVs':>5s} {'point':>7s} {'cap':>6s}")
    for name, items in JOBS:
        tot = sum(n * cw for n, cw in items) + F
        print(f"{name:76s} {sum(n for n, _ in items):5d} {tot:7.2f} {tot*1.2:6.2f}")

    step_s = 3 * 8 * (d * 96)      # SPECULATIVE: 3x forward, batch 8, L=96
    print(f"\nE3 short-duration fine-tune (SPECULATIVE, needs a measured benchmark first):"
          f" {step_s:.2f} s/step -> {step_s/3600*CR_PER_HOUR:.6f} cr/step")
    for st in (20_000, 50_000, 1_000_000):
        h = st * step_s / 3600
        print(f"   {st:>9,d} steps: {h:7.1f} h   {h*CR_PER_HOUR:8.2f} cr")
    print(f"   + evaluating the resulting checkpoint (384 WAVs, both durations): "
          f"{384*(w(96)+w(256))/2 + F:.2f} cr")


if __name__ == "__main__":
    main()
