#!/usr/bin/env python3
"""CPU smoke for the PEFT stack on a tiny U-Net-like module.

Adopted from the recovered M1 scaffold. Not a scientific test; it exists so the
inject -> freeze -> auxiliary -> report -> forward/backward path can be exercised
in a second on any machine, without checkpoints. For the real pruned U-Net see
tests/research/test_peft_real_unet.py.

    .venv/bin/python scripts/research/cpu_smoke_peft.py
"""
import torch
from torch import nn

from audioldm_peft import (
    PeftConfig, freeze_for_peft, inject_lora, configure_auxiliary_trainables,
    parameter_report, assert_peft_ready,
)


class TinyUNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1),
        )
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.model.diffusion_model(x)


def main():
    m = TinyUNetLike()
    freeze_for_peft(m)
    cfg = PeftConfig(root_path="model.diffusion_model", rank=4, alpha=8)
    print("Injected:", inject_lora(m, cfg))
    print("Aux:", configure_auxiliary_trainables(m, cfg))
    assert_peft_ready(m, cfg)
    print("Report:", parameter_report(m))
    x = torch.randn(2, 4, 16, 16)
    y = m(x)
    loss = y.square().mean()
    loss.backward()
    print("CPU smoke OK", tuple(y.shape), float(loss))


if __name__ == "__main__":
    main()
