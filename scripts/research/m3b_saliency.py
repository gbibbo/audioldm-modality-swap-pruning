#!/usr/bin/env python3
"""M3B — compute P0/P1/P2/P3 channel saliency on the base model + evaluate Gate B.

THIS IS THE RQ2 SCIENTIFIC RUN. It computes, for the first time, real per-channel
saliency on the base `(1,2,3,5)` AudioLDM-M-Full U-Net using the paired audio/text
CLAP conditioning, over exactly the 28 conv layers the published L1 artifact ranks
(structure-matched, finding 9.4). It consumes the FROZEN pre-registered calibration
slot manifest (`configs/research/calibration_manifest.json`, sha256 pinned below) and
must not re-draw any slot.

Criteria (master plan §4-5; pilot protocol FROZEN 2026-08-20):
  * P0-published : data-free, keep LOWEST-L1 (Arshdeep's inverted convention, DECISION-M3B-002)
  * P0-L1        : data-free, keep HIGHEST-L1 (secondary reference; DECISION-M3B-003)
  * P1           : text-only Taylor, 2B text grad-evals   (mandatory RQ2 baseline; load-bearing)
  * P2           : paired mean, B audio + B text grad-evals
  * P3           : swap-robust max, shares P2's S_a, S_t

Faithfulness to the pre-registered statistic: S_c = mean_slots |g_c · ∂L/∂g_c| with the
absolute value taken PER SLOT (one forward+backward per (example, noise, timestep) slot,
one modality). So the gradient is computed at **batch=1 per slot** — batching examples
into one backward would compute |mean| instead of mean|·| and cancel opposite-sign
gradients across examples, which is a different (wrong) statistic. This is slower but is
the frozen definition.

Cost discipline (mirrors gpu_benchmark.py): refuses to write a result without CUDA;
`--dry-run-cpu` validates the whole flow on a tiny slot subset for free and refuses
`--out`; fail-fast preflight (commit, clean tree, checkpoints, CUDA + --expect-gpu, and
the frozen manifest sha256). Records the exact commit and prints JSON to stdout.

The base pruned/L1 checkpoint `l1_audioldm-m-full_p1.ckpt` is NEVER opened here — P0-P3
are computed on the BASE model, and the L1 comparison is via the published ranking pkl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import torch

from research_pruning.diagnostics import (
    load_config, build_unet, build_clap, build_vae, vae_encode,
    read_scale_factor, NoiseSchedule,
)
from research_pruning.diagnostics.random_masks import (
    load_l1_ranking, ranking_driven_layers, kept_counts, ranking_full_lengths,
)
from research_pruning.taylor import (
    attach_gates, conv_modules, p0_importance, p0_l1_magnitude,
    normalize_within_layer, keep_topk,
)
from research_pruning.taylor.layer_set import l1_prunable_layer_names, verify_prunable_layers
from research_pruning.paired_modality.criteria import compute_criteria
from research_pruning.paired_modality.overlap import evaluate_gate_b
from research_pruning.paired_modality.gate_b_prime import per_slot_saliency, save_per_slot

BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
RANKING_PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
MANIFEST = "configs/research/calibration_manifest.json"
FROZEN_MANIFEST_SHA = "8d7de0659554385389d3d71d349037d39c39e5842a7488e85037c060532b2d80"
NORM_MODE = "sum"  # pilot protocol default within-layer normalization


# --------------------------------------------------------------------------- slots
@dataclass
class Slot:
    z_t: torch.Tensor    # [1, C, H, W]
    t: torch.Tensor      # [1] long
    noise: torch.Tensor  # [1, C, H, W]  (the epsilon target)
    cond: torch.Tensor   # [1, 1, 512]   (audio or text CLAP embedding)


def slot_noise(shape, master_seed, rank, stratum, kind, draw):
    """Deterministic per-slot N(0,I), seeded (master_seed, rank, stratum, kind, draw).

    kind: 0 = paired slot (shared by S_a and S_t), 1 = P1 slot. Generated on CPU so the
    realisation is identical on CPU and GPU, then moved to the model device by the caller.
    """
    seed = (master_seed * 1_000_003 + rank * 10_007 + stratum * 211 + kind * 53 + draw) % (2**63 - 1)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


# --------------------------------------------------------------------------- provenance
def git_provenance() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], text=True).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def preflight(args, result):
    prov = git_provenance()
    result["git"] = prov
    if args.expect_commit and prov["commit"] != args.expect_commit:
        raise SystemExit(f"PREFLIGHT FAIL: commit {prov['commit']} != {args.expect_commit}")
    if prov["dirty"] and not args.allow_dirty and not args.dry_run_cpu:
        raise SystemExit("PREFLIGHT FAIL: dirty tree (use --allow-dirty only for CPU dev)")
    for p in (BASE_CKPT, RANKING_PKL, MANIFEST):
        if not os.path.exists(p):
            raise SystemExit(f"PREFLIGHT FAIL: missing {p}")
    sha = sha256_file(MANIFEST)
    result["manifest_sha256"] = sha
    if sha != FROZEN_MANIFEST_SHA:
        raise SystemExit(f"PREFLIGHT FAIL: calibration manifest sha256 {sha} != frozen {FROZEN_MANIFEST_SHA}")
    if not args.dry_run_cpu:
        if not torch.cuda.is_available():
            raise SystemExit("PREFLIGHT FAIL: no CUDA and not --dry-run-cpu (refusing to invent saliency)")
        name = torch.cuda.get_device_name(0)
        result["gpu_name"] = name
        if args.expect_gpu and args.expect_gpu.lower() not in name.lower():
            raise SystemExit(f"PREFLIGHT FAIL: GPU {name} != expected {args.expect_gpu}")


# --------------------------------------------------------------------------- build
def build_dataset(config, items):
    """Build an AudioDataset over the manifest wavs.

    CRITICAL: a custom `dataset_json` bypasses `_relative_path_to_absolute_path`, so
    `feature_extraction` calls `read_audio_file(datum["wav"])` on the RAW relative path
    (`zip_audios/...`), which does not exist relative to cwd — the dataset then silently
    substitutes an EMPTY waveform (dataset.py:422-430). That would make the audio-branch
    CLAP embedding and z_0 come from silence, invalidating S_a / P2 / P3. So we resolve
    each wav to an absolute path with the SAME root the built-in split uses
    (`metadata_root["audiocaps"]`, i.e. `os.path.join(root, wav)`), and assert the files
    exist before building.
    """
    import os
    import json as _json
    from audioldm_train.utilities.data.dataset import AudioDataset
    metadata_root = _json.load(open(config["metadata_root"]))
    root = metadata_root["audiocaps"]
    data = []
    for it in items:
        abs_wav = os.path.join(root, it["wav"])
        if not os.path.exists(abs_wav):
            raise SystemExit(f"BUILD FAIL: calibration wav does not resolve on disk: {abs_wav}")
        data.append({"wav": abs_wav, "caption": it["caption"]})
    return AudioDataset(config=config, split="test", waveform_only=False,
                        dataset_json={"data": data})


@torch.no_grad()
def precompute_per_example(dataset, manifest_slots, clap, vae, scale_factor, device):
    """z_0, e_audio, e_text for each example (frozen paths, no grad)."""
    from research_pruning.diagnostics.conditioning import clap_embed
    z0s, e_as, e_ts = [], [], []
    for i, slot in enumerate(manifest_slots):
        sample = dataset[i]
        wav = sample["waveform"]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # Data comes off the dataloader on CPU; vae/clap are on `device`. Move the inputs
        # to `device` before encoding (a CPU dry-run cannot catch this mismatch — both
        # sides are CPU there — so it must be handled explicitly for the GPU run).
        wav = wav.unsqueeze(0).float().to(device)  # [1,1,T]
        mel = sample["log_mel_spec"].unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,1024,64]
        z0 = vae_encode(vae, mel, scale_factor)  # [1,C,H,W]
        e_a = clap_embed(clap, wav, "audio")  # [1,1,T] -> [1,1,512]
        e_t = clap_embed(clap, [slot["caption"]], "text")
        z0s.append(z0.squeeze(0).cpu())
        e_as.append(e_a.squeeze(0).cpu())
        e_ts.append(e_t.squeeze(0).cpu())
    return z0s, e_as, e_ts


def make_slots(manifest_slots, z0s, e_as, e_ts, schedule, master_seed, strata_limit=None):
    """Build the paired (audio+text share z_t) and P1 (text-only) slot lists.

    Returns (audio_slots, text_slots_p2p3, text_slots_p1) as lists of Slot, each a
    single (example, noise, timestep) unit — batch=1 per slot (see module docstring).
    """
    audio_slots, text_p2p3, text_p1 = [], [], []
    for rank, ms in enumerate(manifest_slots):
        z0 = z0s[rank]  # [C,H,W]
        shape = (1,) + tuple(z0.shape)
        n_strata = len(ms["t_paired"]) if strata_limit is None else strata_limit
        for s in range(n_strata):
            # paired slot: one timestep, shared noise; audio and text differ only in cond
            t_p = torch.tensor([ms["t_paired"][s]], dtype=torch.long)
            eps = slot_noise(shape, master_seed, rank, s, kind=0, draw=0)
            z_t = schedule.q_sample(z0.unsqueeze(0), t_p, eps)
            audio_slots.append(Slot(z_t, t_p, eps, e_as[rank].unsqueeze(0)))
            text_p2p3.append(Slot(z_t, t_p, eps, e_ts[rank].unsqueeze(0)))
            # P1: two text draws per (example, stratum)
            for d in range(len(ms["t_p1"][s])):
                t1 = torch.tensor([ms["t_p1"][s][d]], dtype=torch.long)
                eps1 = slot_noise(shape, master_seed, rank, s, kind=1, draw=d)
                z_t1 = schedule.q_sample(z0.unsqueeze(0), t1, eps1)
                text_p1.append(Slot(z_t1, t1, eps1, e_ts[rank].unsqueeze(0)))
    return audio_slots, text_p2p3, text_p1


def make_loss_fn(unet, device):
    """Grad-enabled diffusion MSE for one slot, routed through the gated U-Net."""
    def loss_fn(slot: Slot):
        z_t = slot.z_t.to(device)
        t = slot.t.to(device)
        noise = slot.noise.to(device)
        y = slot.cond.to(device).squeeze(1)  # [1,512]
        eps = unet(z_t, timesteps=t, y=y, context_list=[], context_attn_mask_list=[])
        return torch.mean((eps - noise) ** 2)
    return loss_fn


def to_weightkey(sal_by_module):
    """accumulate_taylor keys are module paths; Gate B / kept_counts use weight names."""
    return {name + ".weight": v for name, v in sal_by_module.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-seed", type=int, default=20260818)
    ap.add_argument("--limit-examples", type=int, default=None, help="dev only: subset E")
    ap.add_argument("--limit-strata", type=int, default=None, help="dev only: subset K")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-saliency", default=None, help="path to save the saliency tensors (.pt)")
    ap.add_argument("--per-slot-store", default=None,
                    help="path to save P1 PER-SLOT saliency contributions (.pt) for Gate B' "
                         "(plan §7 Tier 0: 'per-slot storage'); enables the null-split "
                         "overlap distribution to be computed later on CPU")
    args = ap.parse_args()

    if args.dry_run_cpu and args.out:
        raise SystemExit("--dry-run-cpu produces no result and refuses --out")

    result = {"script": "m3b_saliency.py", "dry_run_cpu": args.dry_run_cpu}
    preflight(args, result)

    if args.dry_run_cpu:
        args.limit_examples = args.limit_examples or 2
        args.limit_strata = args.limit_strata or 1

    device = torch.device("cpu" if args.dry_run_cpu else "cuda")
    config = load_config()

    manifest = json.load(open(MANIFEST))
    slots_meta = manifest["slots"]
    if args.limit_examples:
        slots_meta = slots_meta[:args.limit_examples]
    result["E"] = len(slots_meta)
    result["K"] = args.limit_strata or manifest["K"]
    result["master_seed"] = args.master_seed

    # --- build the base model + conditioners + gates ---
    t0 = time.perf_counter()
    unet = build_unet(config, BASE_CKPT, channel_mult=None, strict=True).to(device)
    ranking = load_l1_ranking(RANKING_PKL)
    verify_prunable_layers(unet, ranking)  # raises if the 28 layers don't resolve
    layer_names = l1_prunable_layer_names(ranking)
    gates = attach_gates(unet, layer_names)
    # freeze base weights: only the channel gates carry gradient (saves the 415M weight-grad buffers)
    for p in unet.parameters():
        p.requires_grad_(False)
    for g in gates.values():
        g.gate.requires_grad_(True)
    convs = conv_modules(gates)

    clap = build_clap(config, unconditional_prob=0.0).to(device)
    vae = build_vae(config, BASE_CKPT).to(device)
    scale_factor = read_scale_factor(BASE_CKPT)
    schedule = NoiseSchedule(config)
    result["build_s"] = time.perf_counter() - t0
    result["scale_factor"] = scale_factor
    result["n_gates"] = len(gates)

    # --- data + per-example precompute ---
    dataset = build_dataset(config, slots_meta)
    t1 = time.perf_counter()
    z0s, e_as, e_ts = precompute_per_example(dataset, slots_meta, clap, vae, scale_factor, device)
    result["precompute_s"] = time.perf_counter() - t1

    # --- slots (batch=1 each) ---
    audio_slots, text_p2p3, text_p1 = make_slots(
        slots_meta, z0s, e_as, e_ts, schedule, args.master_seed, strata_limit=args.limit_strata)
    result["n_audio_slots"] = len(audio_slots)
    result["n_text_p2p3_slots"] = len(text_p2p3)
    result["n_text_p1_slots"] = len(text_p1)

    # --- saliency (the expensive part) ---
    loss_fn = make_loss_fn(unet, device)
    with torch.enable_grad():
        t2 = time.perf_counter()
        crit = compute_criteria(
            gates, loss_fn, loss_fn,
            audio_slots=audio_slots, text_slots_p2p3=text_p2p3, text_slots_p1=text_p1,
            norm_mode=NORM_MODE,
        )
    result["saliency_s"] = time.perf_counter() - t2
    result["budget_grad_evals"] = crit.budget_grad_evals

    # Move every saliency to CPU before the downstream Gate B / keep_topk / save logic,
    # which was only ever exercised on CPU (topk/set-overlap on device tensors is an
    # untested path). Do this on the raw Criteria fields so everything below is CPU.
    def _cpu(d):
        return {k: v.detach().cpu() for k, v in d.items()}
    p1_c, p2_c, p3_c = _cpu(crit.p1), _cpu(crit.p2), _cpu(crit.p3)
    s_a_c, s_t_c = _cpu(crit.s_audio_norm), _cpu(crit.s_text_norm)

    # --- P0 (data-free) ---
    p0_pub = _cpu(normalize_within_layer(p0_importance(convs, convention="published"), NORM_MODE))
    p0_l1 = _cpu(normalize_within_layer(p0_importance(convs, convention="standard"), NORM_MODE))

    # --- Gate B on S_a vs S_t over the 12 ranking-driven layers ---
    rd_layers = ranking_driven_layers(config, ranking)
    kcounts = kept_counts(config, rd_layers)
    fulllen = {k: ranking_full_lengths(ranking)[k] for k in rd_layers}
    gate_b = evaluate_gate_b(
        to_weightkey(s_a_c), to_weightkey(s_t_c),
        k_per_layer=kcounts, n_per_layer=fulllen, layers=rd_layers,
    )
    result["gate_b"] = {
        "pass": bool(gate_b.passed),
        "weighted_overlap": float(gate_b.weighted_overlap),
        "n_layers_at_or_below_layer_max": len(gate_b.layers_at_or_below_layer_max),
        "layers_at_or_below_layer_max": list(gate_b.layers_at_or_below_layer_max),
        "per_layer": {r.name: {"overlap": float(r.overlap), "chance": float(r.chance),
                               "adjusted": float(r.adjusted), "intersection": int(r.intersection),
                               "k": int(r.k_kept), "n": int(r.n_channels)}
                      for r in gate_b.per_layer},
        "thresholds": {"weighted_max": gate_b.weighted_max,
                       "layer_max": gate_b.layer_max, "min_layers": gate_b.min_layers},
    }

    # --- persist saliency tensors + kept-sets for M4 materialization ---
    if args.save_saliency:
        kc_all = kept_counts(config, list(ranking))
        ksets = {
            "P0_published": keep_topk(to_weightkey(p0_pub), kc_all),
            "P1": keep_topk(to_weightkey(p1_c), kc_all),
            "P2": keep_topk(to_weightkey(p2_c), kc_all),
            "P3": keep_topk(to_weightkey(p3_c), kc_all),
        }
        torch.save({
            "saliency": {
                "P0_published": to_weightkey(p0_pub), "P0_L1": to_weightkey(p0_l1),
                "P1": to_weightkey(p1_c), "P2": to_weightkey(p2_c), "P3": to_weightkey(p3_c),
                "S_audio_norm": to_weightkey(s_a_c),
                "S_text_norm": to_weightkey(s_t_c),
            },
            "kept_sets": ksets,
            "provenance": {k: result.get(k) for k in ("git", "manifest_sha256", "master_seed", "E", "K")},
        }, args.save_saliency)
        result["saved_saliency"] = args.save_saliency

    # --- per-slot P1 saliency contributions for Gate B' (null-split overlap) ---
    if args.per_slot_store:
        with torch.enable_grad():
            per = per_slot_saliency(gates, loss_fn, text_p1)     # {module: (n_slots, n_ch)}
        per_wk = to_weightkey({k: v.detach().cpu() for k, v in per.items()})
        save_per_slot(per_wk, args.per_slot_store, meta={
            "purpose": "P1 per-slot Taylor contributions for Gate B' null-split",
            "n_slots": len(text_p1),
            "ranking_driven_layers": rd_layers,
            "kept_counts": kcounts, "full_lengths": fulllen,
            "norm_mode": NORM_MODE, "master_seed": args.master_seed,
            "git": result.get("git"), "manifest_sha256": result.get("manifest_sha256"),
        })
        result["per_slot_store"] = args.per_slot_store
        result["per_slot_n_slots"] = len(text_p1)

    result["measured"] = not args.dry_run_cpu
    out = json.dumps(result, indent=2)
    print(out)
    if args.dry_run_cpu:
        print("\nDRY RUN — NO RESULT WRITTEN. Flow validated on a slot subset.")
    elif args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"\nRESULT written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
