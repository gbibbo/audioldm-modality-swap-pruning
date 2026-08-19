#!/usr/bin/env python3
"""M1 state / EMA / optimizer / full-resume tests (CPU-only).

    S1 ADAPTER-ROUNDTRIP  optimizer groups are {lora,bias,groupnorm_affine};
                          adapter-only state saves and reloads bit-identically.
    S2 EMA-TRAINABLE-ONLY TrainableOnlyEMA tracks exactly the trainable tensors
                          and never the frozen base.
    S3 F7-FULL-RESUME     training_state_dict() bundles adapter + optimizer +
                          EMA + global_step, pickles through torch.save, and
                          reloads into a fresh model/optimizer/EMA so that adapter
                          params, optimizer moments, EMA shadows and step all match.
    S4 SNAPSHOT-IMMUTABLE training_state_dict() must return a true point-in-time
                          snapshot: continuing to train after taking it must not
                          change it (F11). S3 could not catch this because it
                          round-trips through torch.save, and serialising silently
                          breaks the aliasing; the bug only bites when the dict is
                          held in memory, which is what a resume test does.

Run directly:

    .venv/bin/python tests/research/test_state_ema_optimizer.py
"""
from __future__ import annotations

import io
import sys

import torch
from torch import nn

from audioldm_peft import (
    PeftConfig, freeze_for_peft, inject_lora, configure_auxiliary_trainables,
    adaptation_state_dict, load_adaptation_state_dict, build_parameter_groups,
    training_state_dict, load_training_state_dict, TrainableOnlyEMA, parameter_report,
)


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.GroupNorm(2, 4), nn.SiLU(), nn.Conv2d(4, 4, 3, padding=1)
        )

    def forward(self, x):
        return self.model.diffusion_model(x)


def configured(seed=0):
    torch.manual_seed(seed)
    m = Dummy()
    freeze_for_peft(m)
    cfg = PeftConfig(root_path="model.diffusion_model", rank=2, alpha=4)
    inject_lora(m, cfg)
    configure_auxiliary_trainables(m, cfg)
    return m


def check_s1_adapter_roundtrip() -> bool:
    m = configured()
    groups = build_parameter_groups(m, 1e-4, 1e-4)
    names = {g["group_name"] for g in groups}
    ok_groups = names == {"lora", "bias", "groupnorm_affine"}
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(0.123)
    state = adaptation_state_dict(m)
    m2 = configured()
    load_adaptation_state_dict(m2, state)
    s2 = adaptation_state_dict(m2)
    ok_state = state.keys() == s2.keys() and all(torch.equal(state[k], s2[k]) for k in state)
    print(f"  groups={sorted(names)}  adapter_keys={len(state)}  roundtrip_equal={ok_state}")
    return bool(ok_groups and ok_state)


def check_s2_ema_trainable_only() -> bool:
    m = configured()
    report = parameter_report(m)
    ema = TrainableOnlyEMA(m, decay=0.9)
    tracked = len(ema.name_to_buffer)
    expected_tensors = sum(1 for p in m.parameters() if p.requires_grad)
    total_tensors = sum(1 for _ in m.parameters())
    ok = (tracked == expected_tensors and tracked < total_tensors
          and report["trainable_total"] < report["total"])
    print(f"  tracked={tracked} trainable_tensors={expected_tensors} total_tensors={total_tensors}")
    return bool(ok)


def check_s3_full_resume() -> bool:
    m = configured(seed=0)
    opt = torch.optim.AdamW(build_parameter_groups(m, 1e-3, 1e-3))
    ema = TrainableOnlyEMA(m, decay=0.9)

    x = torch.randn(2, 3, 8, 8)
    loss = m(x).square().mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    ema.update(m)
    step = 1

    saved = training_state_dict(m, optimizer=opt, ema=ema, global_step=step)
    buf = io.BytesIO()
    torch.save(saved, buf)
    buf.seek(0)
    loaded = torch.load(buf)

    # Fresh model with DIFFERENT base init (seed=1) — only trainable state is restored.
    m2 = configured(seed=1)
    opt2 = torch.optim.AdamW(build_parameter_groups(m2, 1e-3, 1e-3))
    ema2 = TrainableOnlyEMA(m2, decay=0.9)
    gs = load_training_state_dict(m2, loaded, optimizer=opt2, ema=ema2)

    # adapter params match
    a1, a2 = adaptation_state_dict(m), adaptation_state_dict(m2)
    adapter_ok = a1.keys() == a2.keys() and all(torch.equal(a1[k], a2[k]) for k in a1)
    # optimizer moments restored (non-empty state, exp_avg matches for first param)
    st1 = opt.state_dict()["state"]
    st2 = opt2.state_dict()["state"]
    opt_ok = bool(st2) and st1.keys() == st2.keys()
    if opt_ok:
        k0 = next(iter(st1))
        opt_ok = torch.allclose(st1[k0]["exp_avg"], st2[k0]["exp_avg"]) and int(st1[k0]["step"]) == int(st2[k0]["step"])
    # EMA shadows match
    ema_ok = all(torch.equal(getattr(ema, b), getattr(ema2, b)) for b in ema.name_to_buffer.values())
    step_ok = gs == step
    print(f"  adapter_ok={adapter_ok} opt_ok={opt_ok} ema_ok={ema_ok} step={gs}")
    return bool(adapter_ok and opt_ok and ema_ok and step_ok)


def check_s4_snapshot_immutable() -> bool:
    """F11: the snapshot must be immune to training that happens after it is taken."""
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()

    step()
    snap = training_state_dict(model, optimizer=opt, global_step=1)
    adapter_before = {k: v.clone() for k, v in snap["adapter"].items()}
    moments_before = {i: {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
                      for i, st in snap["optimizer"]["state"].items()}

    for _ in range(5):
        step()                      # keep training while holding the snapshot

    adapter_ok = all(torch.equal(adapter_before[k], snap["adapter"][k])
                     for k in adapter_before)
    deltas = [(mb[k] - snap["optimizer"]["state"][i][k]).abs().max().item()
              for i, mb in moments_before.items() for k in mb]
    opt_ok = all(d == 0.0 for d in deltas)
    print(f"  adapter unchanged: {adapter_ok}")
    print(f"  optimizer moments unchanged: {opt_ok} (max drift {max(deltas):.3e})")
    print(f"  global_step preserved: {snap['global_step'] == 1}")
    return adapter_ok and opt_ok and snap["global_step"] == 1


TESTS = [
    ("S1 ADAPTER-ROUNDTRIP", check_s1_adapter_roundtrip),
    ("S2 EMA-TRAINABLE-ONLY", check_s2_ema_trainable_only),
    ("S3 F7-FULL-RESUME", check_s3_full_resume),
    ("S4 SNAPSHOT-IMMUTABLE", check_s4_snapshot_immutable),
]


def main() -> int:
    results = {}
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False
    print("\n==== M1 STATE/EMA/OPTIMIZER TESTS ====")
    for name, _ in TESTS:
        print(f"  {name:<24} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
