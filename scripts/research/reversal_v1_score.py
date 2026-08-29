#!/usr/bin/env python3
"""Build the frozen 192-item scoring groups for RECOVERY-REVERSAL-V1 (primary CLAP + Human-CLAP).

From the frozen AudioCaps manifest + a WAV root per system, assembles one group per standalone
backbone with EXACTLY 192 items in canonical (prompt_index, replicate_index) order — deliberately
preserving the frozen Option-B batch cardinality (one 192-item seed-once scorer call per system).
REFUSES any group whose cardinality is not 192 or whose order is non-canonical (guards against
chunking / per-item scoring / wrong batch size).

The emitted groups JSON is scored by the UNCHANGED frozen scorers:
  * primary CLAP:  .venv-metrics/bin/python scripts/research/gate0_clap_scorer.py --score-groups IN OUT
                   (laion/clap-htsat-fused rev 365dea6e, np.random.seed(20260826) once per system)
  * secondary HC:  scripts/research/reversal_humanclap.py machinery (sarulab-speech/human-clap-wsce-mae)
Both consume the SAME 192-item groups / WAVs. No GPU here; WAVs must already exist to score.

Run (emit groups, no scoring): OPENBLAS_CORETYPE=Haswell .venv/bin/python \
        scripts/research/reversal_v1_score.py --manifest <manifest> --wav-root <root> --emit <groups.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import BACKBONES_V1, N_PROMPTS_V1, N_REPLICATES_V1  # noqa: E402

N = N_PROMPTS_V1 * N_REPLICATES_V1  # 192
# WAV filename convention for the future run (mirrors the historical phenom naming)
PREFIX = {"dense_ema": "dense_noadapter",
          "p1_pruned_ema_reconstructed": "p1_pruned_ema_reconstructed_noadapter",
          "p1_recovered": "p1_recovered_noadapter"}


def build_groups(manifest: dict, wav_roots: dict) -> dict:
    """One group/backbone, 192 items, canonical (prompt, replicate) order; cardinality-guarded."""
    prompts = sorted(manifest["prompts"], key=lambda p: p["prompt_index"])
    groups = []
    for bk in BACKBONES_V1:
        root = wav_roots[bk]
        items = []
        for p in prompts:
            for r in range(N_REPLICATES_V1):
                items.append({"caption": p["caption"],
                              "wav": os.path.join(root, f"{PREFIX[bk]}_p{p['prompt_index']}_r{r}.wav")})
        if len(items) != N:
            raise SystemExit(f"system {bk}: {len(items)} items != {N} (frozen Option-B cardinality)")
        # verify canonical order
        order = [(pp["prompt_index"], r) for pp in prompts for r in range(N_REPLICATES_V1)]
        if order != [(pi, r) for pi in range(N_PROMPTS_V1) for r in range(N_REPLICATES_V1)]:
            raise SystemExit(f"system {bk}: non-canonical (prompt, replicate) order")
        groups.append({"name": bk, "items": items})
    return {"groups": groups, "convention": "one 192-item seed-once call per system (Option-B)"}


def assert_scored_cardinality(scored: dict) -> None:
    """Guard a scorer OUTPUT: each result must carry exactly 192 cosines."""
    for r in scored.get("results", []):
        if len(r.get("cosines", [])) != N:
            raise SystemExit(f"scored system {r.get('name')}: {len(r.get('cosines', []))} cosines != {N}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/research/reversal_v1_audiocaps_manifest.json")
    ap.add_argument("--wav-root", help="single root holding all systems' WAVs")
    ap.add_argument("--emit", required=True, help="write the 192x3 groups JSON here")
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    roots = {bk: (args.wav_root or "") for bk in BACKBONES_V1}
    groups = build_groups(manifest, roots)
    json.dump(groups, open(args.emit, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"systems": [g["name"] for g in groups["groups"]],
                      "items_per_system": [len(g["items"]) for g in groups["groups"]]}, indent=2))
    print("V1 scoring groups written to", args.emit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
