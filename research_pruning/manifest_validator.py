"""ONE parametric generation-manifest validator, shared by Gate 0 and the phenomenon falsifier.

There must be exactly one set of strict rules; a second bespoke checker would drift. Both the
generator (self-check on write) and the verdict/scoring drivers call `validate_manifest`. The
rules are configured by a `ManifestSpec` (expected battery, backbone/system set, adapter-SHA set,
prompt count, replicates, recipe). Every row is checked for:

  prompt_index, ytid, replicate_index, deterministic paired seed, backbone_id, adapter_state,
  adapter_id/SHA, ddim_steps, guidance, eta, latent_t, source checkpoint SHA, wav_sha256.

`validate_manifest` returns (ok, errors, summary); `assert_valid` raises on the first failure.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


def derive_paired_seed(salt: str, ytid: str, replicate: int) -> int:
    """The frozen common-random-number seed: int.from_bytes(sha256(salt|ytid|r)[:8], 'big').
    Identical to scripts/research/gate0_generator.paired_seed (asserted by a test)."""
    h = hashlib.sha256(f"{salt}|{ytid}|{replicate}".encode()).digest()
    return int.from_bytes(h[:8], "big")


@dataclass
class ManifestSpec:
    n_prompts: int
    replicates: int
    battery_ytids: Dict[int, str]            # {prompt_index -> ytid} (from the frozen battery)
    backbones: Set[str]                      # allowed backbone_id values
    # allowed adapter_id per adapter_state, e.g. {"off": {"none"}, "on": {"<sha>"}}
    adapter_state_ids: Dict[str, Set[str]]
    recipe: Dict[str, float]                 # {ddim_steps, guidance, eta, latent_t}
    seed_salt: str
    require_wav_sha: bool = True
    require_checkpoint: bool = True
    expected_checkpoints: Optional[Set[str]] = None   # if set, each row.checkpoint must be in it
    prompts_sha256: Optional[str] = None              # optional battery integrity cross-ref
    require_paired_off_on: bool = True                # OFF/ON share ytid+seed per (prompt,rep)


def _recipe_field(row, key):
    # rows use ddim_steps/guidance/eta/latent_t
    return row.get(key)


def validate_manifest(manifest: dict, spec: ManifestSpec):
    errors = []
    rows = manifest.get("rows", [])
    n_expected = spec.n_prompts * spec.replicates * len(spec.adapter_state_ids)
    if len(rows) != n_expected:
        errors.append(f"row count {len(rows)} != expected {n_expected} "
                      f"({spec.n_prompts}x{spec.replicates}x{len(spec.adapter_state_ids)})")
    if manifest.get("n") is not None and manifest.get("n") != len(rows):
        errors.append(f"manifest n={manifest.get('n')} != len(rows)={len(rows)}")

    states = set(spec.adapter_state_ids)
    cells = {}                                  # (pi, ri, state) -> count
    by_pr = {}                                  # (pi, ri) -> {state: row}
    seen_states = set()
    for i, r in enumerate(rows):
        pi, ri, st = r.get("prompt_index"), r.get("replicate_index"), r.get("adapter_state")
        seen_states.add(st)
        # index ranges
        if not (isinstance(pi, int) and 0 <= pi < spec.n_prompts):
            errors.append(f"row {i}: prompt_index {pi} out of 0..{spec.n_prompts-1}"); continue
        if not (isinstance(ri, int) and 0 <= ri < spec.replicates):
            errors.append(f"row {i}: replicate_index {ri} out of 0..{spec.replicates-1}"); continue
        if st not in states:
            errors.append(f"row {i}: adapter_state {st!r} not in {states}"); continue
        cells[(pi, ri, st)] = cells.get((pi, ri, st), 0) + 1
        by_pr.setdefault((pi, ri), {})[st] = r
        # ytid vs battery
        exp_yt = spec.battery_ytids.get(pi)
        if r.get("ytid") != exp_yt:
            errors.append(f"row {i} p{pi}: ytid {r.get('ytid')} != battery {exp_yt}")
        # deterministic paired seed
        exp_seed = derive_paired_seed(spec.seed_salt, r.get("ytid"), ri)
        if r.get("seed") != exp_seed:
            errors.append(f"row {i} p{pi}r{ri}: seed {r.get('seed')} != derived {exp_seed}")
        # backbone
        if r.get("backbone_id") not in spec.backbones:
            errors.append(f"row {i}: backbone_id {r.get('backbone_id')} not in {spec.backbones}")
        # adapter id per state
        allowed = spec.adapter_state_ids.get(st, set())
        if r.get("adapter_id") not in allowed:
            errors.append(f"row {i}: adapter_id {r.get('adapter_id')} not allowed for state {st} "
                          f"(allowed {allowed})")
        # recipe
        for key, exp in (("ddim_steps", spec.recipe.get("ddim_steps")),
                         ("guidance", spec.recipe.get("guidance")),
                         ("eta", spec.recipe.get("eta")),
                         ("latent_t", spec.recipe.get("latent_t"))):
            if exp is not None and _recipe_field(r, key) != exp:
                errors.append(f"row {i}: {key}={_recipe_field(r, key)} != {exp}")
        # checkpoint + wav sha presence
        if spec.require_checkpoint and not r.get("checkpoint"):
            errors.append(f"row {i}: missing source checkpoint field")
        if spec.expected_checkpoints is not None and r.get("checkpoint") not in spec.expected_checkpoints:
            errors.append(f"row {i}: checkpoint {r.get('checkpoint')} not in {spec.expected_checkpoints}")
        if spec.require_wav_sha and not (isinstance(r.get("wav_sha256"), str) and len(r["wav_sha256"]) == 64):
            errors.append(f"row {i}: missing/invalid wav_sha256")

    # exactly one row per (prompt, replicate, state)
    if not errors or True:
        for pi in range(spec.n_prompts):
            for ri in range(spec.replicates):
                for st in states:
                    c = cells.get((pi, ri, st), 0)
                    if c != 1:
                        errors.append(f"cell (p{pi},r{ri},{st}) has {c} rows (expected 1)")
    # no stray states
    if seen_states - states:
        errors.append(f"unexpected adapter_state(s): {seen_states - states}")

    # OFF/ON pairing (shared ytid + seed) when both states present
    if spec.require_paired_off_on and states == {"off", "on"}:
        for (pi, ri), pair in by_pr.items():
            if "off" in pair and "on" in pair:
                o, n = pair["off"], pair["on"]
                if o.get("ytid") != n.get("ytid"):
                    errors.append(f"(p{pi},r{ri}): OFF/ON ytid mismatch")
                if o.get("seed") != n.get("seed"):
                    errors.append(f"(p{pi},r{ri}): OFF/ON seed mismatch")

    summary = {
        "n_rows": len(rows), "n_expected": n_expected, "states": sorted(seen_states),
        "backbones": sorted({r.get("backbone_id") for r in rows}),
        "adapter_ids": sorted({r.get("adapter_id") for r in rows}),
        "checkpoints": sorted({r.get("checkpoint") for r in rows}),
        "n_errors": len(errors),
    }
    return (len(errors) == 0), errors, summary


def assert_valid(manifest: dict, spec: ManifestSpec) -> dict:
    ok, errors, summary = validate_manifest(manifest, spec)
    if not ok:
        raise ValueError("manifest validation FAILED:\n  " + "\n  ".join(errors[:40]))
    return summary


def battery_ytids(battery: dict) -> Dict[int, str]:
    return {i: p["ytid"] for i, p in enumerate(battery["prompts"])}
