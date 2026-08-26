"""Configuration for parameter-efficient recovery.

Adopted from the recovered M1 scaffold and audited in ``docs/m1_scaffold_audit.md``.
Fix F2: ``train_layernorm_affine`` is now an explicit flag. Previously LayerNorm
*biases* were swept into the generic ``train_bias`` bucket while their *gains*
stayed frozen, leaving every LayerNorm half-trained and unreported. LayerNorm
affine is now all-or-nothing under this flag and reported as its own category.
It defaults to ``False`` to match the master-plan recovery description ("biases
and GroupNorm affine parameters"), which does not include LayerNorm gains.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PeftConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_linear: bool = True
    target_conv2d: bool = True
    train_bias: bool = True
    train_groupnorm_affine: bool = True
    train_layernorm_affine: bool = False  # F2: explicit, separately reported
    root_path: str = "model.diffusion_model"
    include_name_substrings: Tuple[str, ...] = field(default_factory=tuple)
    exclude_name_substrings: Tuple[str, ...] = field(default_factory=tuple)
    # LoRA A initialization: "kaiming" (default, our original) or "gaussian" (peft
    # init_lora_weights="gaussian": lora_A ~ N(0, (1/rank)^2), lora_B = 0). Gate 0 uses
    # "gaussian" to match Kim's peft==0.13.2 recipe. lora_B is always zeros either way.
    init_lora_weights: str = "kaiming"

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_lora_weights not in ("kaiming", "gaussian"):
            raise ValueError("init_lora_weights must be 'kaiming' or 'gaussian'")
