"""Mask-induced LoRA transfer (Scenario B): deterministic slicing of a dense-trained
LoRA update onto a structurally pruned backbone.

Setting. A dense layer with weight ``W`` (Linear: ``(d_out, d_in)``; Conv2d:
``(d_out, d_in, kh, kw)``) is pruned by pure index selection with kept-index vectors
``K_out`` / ``K_in`` (this repo's materializer is pure selection — see
``tests/research/test_materialize_channel_mult.py`` P4: the published
``l1_audioldm-m-full_p1.ckpt`` equals ``materialize(dense, L1 ranking)`` bit-exact).
A dense-trained LoRA contributes ``dW = B @ A`` with ``B: (d_out, r)``, ``A: (r, d_in)``.

The mask-induced transfer is defined as slicing the SAME selections into the factors:

    A' = A[:, K_in]        B' = B[K_out, :]

Then ``B' @ A' == (B @ A)[K_out][:, K_in]`` EXACTLY (row/column selection commutes with
the factored product; see ``test_lora_mask_transfer.py``). So the transferred adapter
adds, on the surviving coordinates, exactly the update the dense adapter would have
added there — no learned parameters, no adapter data, no retraining. The transfer is
unambiguous wherever backbone pruning itself is pure row/column selection with recorded
indices. It is NOT an approximation theorem: the scientific question (differential
fragility) is precisely whether the discarded ``dW`` mass mattered functionally.

Nesting: for nested ladders ``K2 ⊆ K1``, slicing to ``K1`` then re-slicing to ``K2``
equals slicing to ``K2`` directly (composition of selections), so per-severity adapters
derive consistently from one dense adapter.
"""
from __future__ import annotations

import torch


def _check_indices(idx: torch.Tensor, dim_size: int, name: str) -> torch.Tensor:
    idx = torch.as_tensor(idx, dtype=torch.long)
    if idx.ndim != 1:
        raise ValueError(f"{name}: kept-index vector must be 1-D, got shape {tuple(idx.shape)}")
    if idx.numel() == 0:
        raise ValueError(f"{name}: empty kept-index vector")
    if idx.min() < 0 or idx.max() >= dim_size:
        raise ValueError(f"{name}: indices out of range for dim {dim_size}")
    if idx.unique().numel() != idx.numel():
        raise ValueError(f"{name}: duplicate indices")
    return idx


def slice_lora_linear(A: torch.Tensor, B: torch.Tensor,
                      keep_in: torch.Tensor | None, keep_out: torch.Tensor | None):
    """Slice a Linear-layer LoRA pair (A: (r, d_in), B: (d_out, r)).

    ``keep_in``/``keep_out`` may be None when that dimension is unpruned.
    Returns (A', B').
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
        raise ValueError(f"incompatible LoRA factors A{tuple(A.shape)} B{tuple(B.shape)}")
    if keep_in is not None:
        A = A.index_select(1, _check_indices(keep_in, A.shape[1], "keep_in"))
    if keep_out is not None:
        B = B.index_select(0, _check_indices(keep_out, B.shape[0], "keep_out"))
    return A, B


def slice_lora_conv2d(A: torch.Tensor, B: torch.Tensor,
                      keep_in: torch.Tensor | None, keep_out: torch.Tensor | None):
    """Slice a Conv2d-layer LoRA pair (A: (r, d_in, kh, kw), B: (d_out, r, 1, 1)).

    Channel pruning selects output channels of the composed kernel via B's dim 0 and
    input channels via A's dim 1; kernel taps are untouched. Returns (A', B').
    """
    if A.ndim != 4 or B.ndim != 4 or A.shape[0] != B.shape[1]:
        raise ValueError(f"incompatible conv LoRA factors A{tuple(A.shape)} B{tuple(B.shape)}")
    if B.shape[2] != 1 or B.shape[3] != 1:
        raise ValueError(f"conv LoRA B must be 1x1, got {tuple(B.shape)}")
    if keep_in is not None:
        A = A.index_select(1, _check_indices(keep_in, A.shape[1], "keep_in"))
    if keep_out is not None:
        B = B.index_select(0, _check_indices(keep_out, B.shape[0], "keep_out"))
    return A, B


def delta_w_linear(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Composed dense update dW = B @ A for a Linear layer."""
    return B @ A


def delta_w_conv2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Composed dense update dW[o,i,h,w] = sum_r B[o,r,0,0] * A[r,i,h,w]."""
    return torch.einsum("orxy,rihw->oihw", B, A)
