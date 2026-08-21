"""Deterministic seed derivation for SA3 generation (protocol section 2.2).

Per prompt p and stream s: ONE initial-noise seed (shared by every system for that (s,p)) and
a per-step re-noising seed stream for ping-pong. All seeds are a pure function of
(master_seed, stream, audiocap_id, kind, step) via SHA-256, so no giant table is stored: the
seed_table.json records the rule + a hash of the fully materialized seeds for smoke+pilot.

The SAME function is used by states.py at generation time; a mismatch would change the hash.
"""
from __future__ import annotations
import hashlib
from typing import List

MASTER_SEED = 20260818
R_STREAMS = 5
N_PINGPONG = 8


def derive_seed(stream: int, audiocap_id, kind: str, step: int = 0) -> int:
    """A 63-bit deterministic seed. `kind` in {"init","renoise"}; step=0 for init."""
    key = f"{MASTER_SEED}|{int(stream)}|{str(audiocap_id)}|{kind}|{int(step)}"
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def prompt_seeds(stream: int, audiocap_id, n_steps: int = N_PINGPONG) -> dict:
    return {
        "init_seed": derive_seed(stream, audiocap_id, "init", 0),
        "renoise_seeds": [derive_seed(stream, audiocap_id, "renoise", i) for i in range(n_steps)],
    }


def initial_noise(stream: int, audiocap_id, shape, device="cpu", dtype=None):
    """Seed-paired initial latent noise for (stream, prompt). Uses a CPU generator for
    cross-device reproducibility, then moves to `device`."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(derive_seed(stream, audiocap_id, "init", 0))
    x = torch.randn(*shape, generator=g)
    if dtype is not None:
        x = x.to(dtype)
    return x.to(device)


def renoise(stream: int, audiocap_id, step: int, shape, device="cpu", dtype=None):
    """Seed-paired ping-pong re-noising sample at a given step."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(derive_seed(stream, audiocap_id, "renoise", step))
    x = torch.randn(*shape, generator=g)
    if dtype is not None:
        x = x.to(dtype)
    return x.to(device)


def materialized_digest(audiocap_ids: List, streams: int = R_STREAMS, n_steps: int = N_PINGPONG) -> str:
    """SHA-256 over every (stream, audiocap_id, init_seed, renoise_seeds), canonical order."""
    h = hashlib.sha256()
    for s in range(streams):
        for aid in sorted(audiocap_ids, key=lambda x: int(x)):
            ps = prompt_seeds(s, aid, n_steps)
            h.update(f"{s}|{aid}|{ps['init_seed']}|{','.join(map(str, ps['renoise_seeds']))}\n".encode())
    return h.hexdigest()
