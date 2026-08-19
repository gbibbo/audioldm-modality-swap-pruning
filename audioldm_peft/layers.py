"""LoRA layers for parameter-efficient recovery.

Adopted from the recovered M1 scaffold; audited in ``docs/m1_scaffold_audit.md``.

Fix F5: ``LoRAConv2d.forward`` no longer materialises the full ``[out, in, kh, kw]``
delta and runs a second full convolution every step. It factorises the update
into ``conv2d(x, A) -> conv2d(., B_1x1)``, which is mathematically identical
(convolution is linear) but costs roughly ``rank/out_channels`` of the base
convolution instead of a second full one. ``delta_weight()`` is retained
unchanged so merge/unmerge stay bit-for-bit as before.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _LoRABase(nn.Module):
    def __init__(self, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def merge(self) -> None:
        if not self.merged:
            with torch.no_grad():
                self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            with torch.no_grad():
                self.base.weight.sub_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = False


class LoRALinear(_LoRABase):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__(rank, alpha, dropout)
        self.base = base
        # Create the adapters on the base layer's device/dtype (F9). Injection happens
        # AFTER the checkpoint load (integration_notes I4), by which point the model may
        # already live on the GPU; defaulting to CPU would silently produce a
        # cuda/cpu mismatch at the first forward.
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.weight.requires_grad = False

    def delta_weight(self):
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        if self.merged:
            return self.base(x)
        base_out = self.base(x)
        x_lora = self.lora_dropout(x)
        delta = F.linear(F.linear(x_lora, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta


class LoRAConv2d(_LoRABase):
    """Weight-space LoRA for ordinary (groups=1) Conv2d.

    The convolution kernel is flattened to ``[out_channels, in_channels*kh*kw]``
    and receives a rank-r update ``B @ A``. ``delta_weight`` reshapes that back to
    a kernel, keeping merge/unmerge exact. ``forward`` uses the factorised form.
    """

    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        if base.groups != 1:
            raise ValueError("LoRAConv2d v1 supports groups=1 only")
        super().__init__(rank, alpha, dropout)
        self.base = base
        kh, kw = base.kernel_size
        flat_in = base.in_channels * kh * kw
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}   # F9, see LoRALinear
        self.lora_A = nn.Parameter(torch.empty(rank, flat_in, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.weight.requires_grad = False

    def delta_weight(self):
        kh, kw = self.base.kernel_size
        flat = (self.lora_B @ self.lora_A) * self.scaling
        return flat.view(self.base.out_channels, self.base.in_channels, kh, kw)

    def forward(self, x):
        if self.merged:
            return self.base(x)
        base_out = self.base(x)
        x_lora = self.lora_dropout(x)
        kh, kw = self.base.kernel_size
        # Factorised low-rank convolution (F5): first project down to `rank`
        # channels with the base's spatial geometry, then a 1x1 conv mixes the
        # rank channels up to out_channels. Equivalent to conv2d(x, B@A) since
        # convolution is linear.
        a_kernel = self.lora_A.view(self.rank, self.base.in_channels, kh, kw).to(dtype=x.dtype)
        h = F.conv2d(
            x_lora, a_kernel, bias=None,
            stride=self.base.stride, padding=self.base.padding,
            dilation=self.base.dilation, groups=1,
        )
        b_kernel = self.lora_B.view(self.base.out_channels, self.rank, 1, 1).to(dtype=x.dtype)
        delta = F.conv2d(h, b_kernel, bias=None) * self.scaling
        return base_out + delta


def iter_lora_modules(module: nn.Module):
    for child in module.modules():
        if isinstance(child, (LoRALinear, LoRAConv2d)):
            yield child


def lora_param_ids(module: nn.Module) -> set:
    """ids of every LoRA adapter parameter (lora_A / lora_B) under `module`."""
    ids = set()
    for child in iter_lora_modules(module):
        ids.add(id(child.lora_A))
        ids.add(id(child.lora_B))
    return ids


def merge_all_lora(module: nn.Module) -> None:
    for child in iter_lora_modules(module):
        child.merge()


def unmerge_all_lora(module: nn.Module) -> None:
    for child in iter_lora_modules(module):
        child.unmerge()
