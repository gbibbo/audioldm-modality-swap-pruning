"""Modality-swap diagnostics: D_mod / R_mod. Diagnostics, never pruning losses.

M2 conditioning-path validation lives in ``conditioning.py`` (audio/text CLAP
paths + FiLM epsilon prediction). D_mod / R_mod themselves are M3 work and are
NOT implemented here yet.
"""

from research_pruning.diagnostics.conditioning import (  # noqa: F401
    NoiseSchedule,
    PairedSlots,
    build_clap,
    build_paired_slots,
    build_unet,
    build_vae,
    clap_embed,
    eps_pred,
    load_config,
    paired_eps,
    read_scale_factor,
    tensor_hash,
    vae_encode,
)
from research_pruning.diagnostics.modality_diagnostics import (  # noqa: F401
    EPS_DEFAULT,
    aggregate_over_strata,
    modality_diagnostics,
)
