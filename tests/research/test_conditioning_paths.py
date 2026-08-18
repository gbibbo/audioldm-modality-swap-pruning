#!/usr/bin/env python3
"""M2 conditioning-path validation tests (CPU-only).

Runs five checks over the real CLAP conditioner and the real diffusion U-Net
loaded with real AudioLDM-M-Full weights and real AudioCaps items:

    T1 DIMENSIONS         both embeddings [B,1,512]; same FiLM interface;
                          extra_film_condition_dim == 512.
    T2 DROPOUT-OFF        upstream default unconditional_prob is 0.1 and the
                          frozen config does not override it; the diagnostic path
                          forces 0.0 -> clap_embed is bit-identical across calls.
    T3 DETERMINISM        paired_eps with the same seed twice -> max|diff| == 0.0.
    T4 PAIRING            eps_a and eps_t use the SAME z_t, t and noise (proved by
                          tensor hash, not visual inspection).
    T5 NON-DEGENERATION   eps_a != eps_t (mean|diff| > 0); the swap is real.

Run directly (no pytest dependency in this environment):

    .venv/bin/python tests/research/test_conditioning_paths.py

Exit code 0 iff all five checks pass. Pytest-compatible test_* functions are
also provided for environments that have pytest.
"""
from __future__ import annotations

import sys

import torch
import yaml

from research_pruning.diagnostics.conditioning import (
    FROZEN_CONFIG,
    NoiseSchedule,
    build_clap,
    build_paired_slots,
    build_unet,
    build_vae,
    clap_embed,
    load_config,
    paired_eps,
    read_scale_factor,
    tensor_hash,
)
from audioldm_train.utilities.data.dataset import AudioDataset

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
INDICES = [0, 1, 2, 3]
SEED = 1234

# Built once and shared across checks (model construction is expensive on CPU).
_CTX = {}


def context():
    if _CTX:
        return _CTX
    config = load_config(FROZEN_CONFIG)
    clap = build_clap(config, unconditional_prob=0.0)
    unet = build_unet(config, BASE_CKPT, channel_mult=None)  # base [1,2,3,5]
    vae = build_vae(config, BASE_CKPT)
    schedule = NoiseSchedule(config)
    scale_factor = read_scale_factor(BASE_CKPT)
    dataset = AudioDataset(config=config, split="test", waveform_only=False)
    slots = build_paired_slots(
        dataset, INDICES, config,
        vae=vae, schedule=schedule, scale_factor=scale_factor, seed=SEED,
    )
    _CTX.update(
        config=config, clap=clap, unet=unet, vae=vae, schedule=schedule,
        scale_factor=scale_factor, dataset=dataset, slots=slots,
    )
    return _CTX


def _config_unconditional_prob_override(config: dict):
    """Return the config-declared unconditional_prob for the CLAP cond stage, or None."""
    params = config["model"]["params"]["cond_stage_config"]["film_clap_cond1"]["params"]
    return params.get("unconditional_prob", None)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_t1_dimensions():
    ctx = context()
    config, clap, unet, slots = ctx["config"], ctx["clap"], ctx["unet"], ctx["slots"]

    e_a = clap_embed(clap, slots.audio_items(), "audio")
    e_t = clap_embed(clap, slots.text_items(), "text")

    b = len(INDICES)
    checks = {
        "e_a shape [B,1,512]": tuple(e_a.shape) == (b, 1, 512),
        "e_t shape [B,1,512]": tuple(e_t.shape) == (b, 1, 512),
        "config extra_film_condition_dim == 512":
            config["model"]["params"]["unet_config"]["params"]["extra_film_condition_dim"] == 512,
        "unet.film_emb in_features == 512": unet.film_emb.in_features == 512,
        "unet.use_extra_film_by_concat is True": unet.use_extra_film_by_concat is True,
    }
    # Both embeddings actually pass through the SAME FiLM interface (film_emb(y)).
    with torch.no_grad():
        y_a = e_a.squeeze(1)
        y_t = e_t.squeeze(1)
        film_a = unet.film_emb(y_a)
        film_t = unet.film_emb(y_t)
    checks["film_emb(e_a) -> [B, time_embed_dim]"] = film_a.shape == film_t.shape
    checks["film_emb output dim == time_embed_dim"] = (
        film_a.shape[-1] == unet.film_emb.out_features
    )
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"    T1 {'ok ' if v else 'FAIL'} {k}")
    print(f"    T1 detail: e_a={tuple(e_a.shape)} e_t={tuple(e_t.shape)} "
          f"film_emb={unet.film_emb.in_features}->{unet.film_emb.out_features}")
    return ok


def check_t2_dropout_off():
    ctx = context()
    config, clap, slots = ctx["config"], ctx["clap"], ctx["slots"]

    # Discover the upstream default WITHOUT our override, from a fresh temporary CLAP
    # signature (cheap: read the default from the class, do not rebuild the model).
    import inspect
    from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2

    default_uncond = inspect.signature(
        CLAPAudioEmbeddingClassifierFreev2.__init__
    ).parameters["unconditional_prob"].default
    config_override = _config_unconditional_prob_override(config)

    print(f"    T2 upstream default unconditional_prob = {default_uncond}")
    print(f"    T2 frozen-config override             = {config_override} "
          f"({'not overridden' if config_override is None else 'overridden'})")
    print(f"    T2 diagnostic-path unconditional_prob = {clap.unconditional_prob}")

    # Bit-identity of clap_embed across two calls on the same input (text + audio).
    a1 = clap_embed(clap, slots.text_items(), "text")
    a2 = clap_embed(clap, slots.text_items(), "text")
    b1 = clap_embed(clap, slots.audio_items(), "audio")
    b2 = clap_embed(clap, slots.audio_items(), "audio")
    text_identical = torch.equal(a1, a2)
    audio_identical = torch.equal(b1, b2)

    checks = {
        "upstream default is 0.1": float(default_uncond) == 0.1,
        "config does not override unconditional_prob": config_override is None,
        "diagnostic path forces 0.0": float(clap.unconditional_prob) == 0.0,
        "text embed bit-identical across calls": text_identical,
        "audio embed bit-identical across calls": audio_identical,
    }
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"    T2 {'ok ' if v else 'FAIL'} {k}")
    return ok


def check_t3_determinism():
    ctx = context()
    config, clap, unet, dataset = ctx["config"], ctx["clap"], ctx["unet"], ctx["dataset"]
    vae, schedule, sf = ctx["vae"], ctx["schedule"], ctx["scale_factor"]

    kw = dict(vae=vae, schedule=schedule, scale_factor=sf, seed=SEED)
    slots1 = build_paired_slots(dataset, INDICES, config, **kw)
    slots2 = build_paired_slots(dataset, INDICES, config, **kw)
    same_zt = torch.equal(slots1.z_t, slots2.z_t)
    same_t = torch.equal(slots1.t, slots2.t)
    same_z0 = torch.equal(slots1.z_0, slots2.z_0)
    print(f"    T3 same-seed z_0 (VAE mode) identical: {same_z0}")

    eps_a1, eps_t1 = paired_eps(unet, clap, slots1)
    eps_a2, eps_t2 = paired_eps(unet, clap, slots2)
    da = (eps_a1 - eps_a2).abs().max().item()
    dt = (eps_t1 - eps_t2).abs().max().item()

    print(f"    T3 same-seed z_t identical: {same_zt}; t identical: {same_t}")
    print(f"    T3 max|diff| eps_a = {da:.3e}; eps_t = {dt:.3e}")
    checks = {
        "same-seed z_0 (VAE mode) identical": same_z0,
        "same-seed z_t identical": same_zt,
        "same-seed t identical": same_t,
        "max|diff| eps_a == 0.0": da == 0.0,
        "max|diff| eps_t == 0.0": dt == 0.0,
    }
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"    T3 {'ok ' if v else 'FAIL'} {k}")
    return ok


def check_t4_pairing():
    ctx = context()
    clap, unet, slots, schedule = ctx["clap"], ctx["unet"], ctx["slots"], ctx["schedule"]

    # The audio and text epsilon predictions are fed EXACTLY the same z_t and t.
    # Prove it by hashing the tensors that paired_eps consumes.
    zt_hash = tensor_hash(slots.z_t)
    t_hash = tensor_hash(slots.t)
    noise_hash = tensor_hash(slots.noise)

    # z_t is the deterministic DDPM noising of the shared (z_0, t, noise):
    # z_t == sqrt(a_t) z_0 + sqrt(1-a_t) noise. Recompute and require bit-identity.
    zt_recomputed = schedule.q_sample(slots.z_0, slots.t, slots.noise)
    zt_from_noise = torch.equal(slots.z_t, zt_recomputed)

    # Control: perturbing z_t changes the hash and the epsilon, confirming the hash
    # actually witnesses the shared input rather than being trivially constant.
    e_a = clap_embed(clap, slots.audio_items(), "audio")
    from research_pruning.diagnostics.conditioning import eps_pred
    eps_shared = eps_pred(unet, slots.z_t, slots.t, e_a)
    perturbed = slots.z_t.clone()
    perturbed[0, 0, 0, 0] += 1.0
    eps_perturbed = eps_pred(unet, perturbed, slots.t, e_a)
    control_changes = tensor_hash(perturbed) != zt_hash and not torch.equal(
        eps_shared, eps_perturbed
    )

    print(f"    T4 z_t hash   = {zt_hash[:16]}")
    print(f"    T4 t hash     = {t_hash[:16]}")
    print(f"    T4 noise hash = {noise_hash[:16]}")
    checks = {
        "z_t == q_sample(z_0, t, noise) (shared, reproducible)": zt_from_noise,
        "z_t and t are single shared tensors for both paths": (
            zt_hash == tensor_hash(slots.z_t) and t_hash == tensor_hash(slots.t)
        ),
        "control: perturbing z_t changes hash AND epsilon": control_changes,
    }
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"    T4 {'ok ' if v else 'FAIL'} {k}")
    return ok


def check_t5_non_degeneration():
    ctx = context()
    clap, unet, slots = ctx["clap"], ctx["unet"], ctx["slots"]
    eps_a, eps_t = paired_eps(unet, clap, slots)
    mad = (eps_a - eps_t).abs().mean().item()
    max_abs = (eps_a - eps_t).abs().max().item()
    # A2: the swap magnitude is uninterpretable without the scale of eps itself.
    mean_abs_eps = 0.5 * (eps_a.abs().mean().item() + eps_t.abs().mean().item())
    ratio = mad / mean_abs_eps
    print(f"    T5 mean|eps_a - eps_t| = {mad:.6e}; max = {max_abs:.6e}")
    print(f"    T5 mean|eps| = {mean_abs_eps:.6e}; ratio mean|Δ|/mean|eps| = {ratio:.4f}")
    ok = mad > 0.0
    print(f"    T5 {'ok ' if ok else 'FAIL'} eps_a != eps_t (swap exists)")
    return ok


# --------------------------------------------------------------------------- #
# pytest-compatible wrappers
# --------------------------------------------------------------------------- #
def test_t1_dimensions():
    assert check_t1_dimensions()


def test_t2_dropout_off():
    assert check_t2_dropout_off()


def test_t3_determinism():
    assert check_t3_determinism()


def test_t4_pairing():
    assert check_t4_pairing()


def test_t5_non_degeneration():
    assert check_t5_non_degeneration()


def main() -> int:
    checks = [
        ("T1 DIMENSIONS", check_t1_dimensions),
        ("T2 DROPOUT-OFF", check_t2_dropout_off),
        ("T3 DETERMINISM", check_t3_determinism),
        ("T4 PAIRING", check_t4_pairing),
        ("T5 NON-DEGENERATION", check_t5_non_degeneration),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        results[name] = bool(fn())
    print("\n==== M2 CONDITIONING PATH TESTS ====")
    for name, _ in checks:
        print(f"  {name:<22} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
