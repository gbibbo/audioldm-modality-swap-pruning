"""Modality-swap diagnostics: D_mod / R_mod. Diagnostics, never pruning losses.

M2 conditioning-path validation lives in ``conditioning.py`` (audio/text CLAP
paths + FiLM epsilon prediction). D_mod / R_mod themselves are M3 work and are
NOT implemented here yet.
"""

from research_pruning.diagnostics.conditioning import (  # noqa: F401
    PairedSlots,
    build_clap,
    build_paired_slots,
    build_unet,
    clap_embed,
    eps_pred,
    load_config,
    paired_eps,
    tensor_hash,
)
