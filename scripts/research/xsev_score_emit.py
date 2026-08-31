#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — emit frozen CLAP scoring groups (CPU, free). No scoring here.

Builds matched seed-once groups for the UNCHANGED frozen scorer
(gate0_clap_scorer.py --score-groups, laion/clap-htsat-fused rev 365dea6e, np.random.seed(20260826)
once per group), from the frozen manifests + existing WAV roots. Two group files:

  (A) severity-2 primary/sensitivity: 9 groups = {recovered2,pruned2_A,pruned2_B} x
      {ac_native,ac_short,music}, EXACTLY 192 items each, canonical order
      (AudioCaps: prompt_index 0..191; music: (prompt_index 0..63, replicate 0..2)).
  (B) dense@10.24s control (severity-1 completion, §4): 3 groups = {dense,pruned_sev1,recovered_sev1}
      x 80 Arm-D ytids, order subset_prompt_index 0..79, all 10.24s / paired x_T.

REFUSES any group of wrong cardinality or non-canonical order. Verifies every WAV exists.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_score_emit.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

AC_MANIFEST = "configs/research/xsev_audiocaps_manifest.json"
MUSIC_MANIFEST = "configs/research/xsev_music_manifest.json"
ARMD = "configs/research/op_duration_discriminator_1_subset.json"

XSEV_ROOT = "/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_xsev_gen"
TAIL_ROOT = "/teamspace/jobs/xsev-dense-tail-3/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_xsev_gen"
ARMD_ROOT = "/teamspace/jobs/reversal-armd-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_armd_gen"

SEV2_SYSTEMS = ("recovered2", "pruned2_A", "pruned2_B")
AC_CONTEXTS = ("ac_native", "ac_short")


def _check(items, n, ctx):
    if len(items) != n:
        raise SystemExit(f"group {ctx}: {len(items)} != {n} (frozen cardinality)")
    for it in items:
        if not os.path.exists(it["wav"]):
            raise SystemExit(f"group {ctx}: missing WAV {it['wav']}")


def build_sev2():
    ac = {p["prompt_index"]: p for p in json.load(open(AC_MANIFEST))["prompts"]}
    mu = {p["prompt_index"]: p for p in json.load(open(MUSIC_MANIFEST))["prompts"]}
    assert set(ac) == set(range(192)) and set(mu) == set(range(64)), "manifest index set"
    groups = []
    for sysn in SEV2_SYSTEMS:
        for ctx in AC_CONTEXTS:
            items = [{"caption": ac[pi]["caption"],
                      "wav": os.path.join(XSEV_ROOT, f"{sysn}_{ctx}_p{pi}_r0.wav")}
                     for pi in range(192)]
            _check(items, 192, f"{sysn}/{ctx}")
            groups.append({"name": f"{sysn}__{ctx}", "items": items})
        # music: 64 x 3 canonical (prompt_index, replicate)
        items = [{"caption": mu[pi]["caption"],
                  "wav": os.path.join(XSEV_ROOT, f"{sysn}_music_p{pi}_r{r}.wav")}
                 for pi in range(64) for r in range(3)]
        _check(items, 192, f"{sysn}/music")
        groups.append({"name": f"{sysn}__music", "items": items})
    return {"groups": groups,
            "convention": "one 192-item seed-once call per (system,context); AudioCaps 192x1, music 64x3",
            "order": "AudioCaps prompt_index 0..191; music (prompt_index 0..63, replicate 0..2)"}


def build_dense_control():
    armd = {p["subset_prompt_index"]: p for p in json.load(open(ARMD))["prompts"]}
    assert set(armd) == set(range(80)), "Arm-D subset index set"
    groups = []
    # dense: 0..72 from XSEV_ROOT, 73..79 from TAIL_ROOT
    dense_items = []
    for i in range(80):
        root = XSEV_ROOT if i <= 72 else TAIL_ROOT
        dense_items.append({"caption": armd[i]["caption"],
                            "wav": os.path.join(root, f"dense_dense_native_p{i}_r0.wav")})
    _check(dense_items, 80, "dense10s/dense")
    groups.append({"name": "dense10s__dense", "items": dense_items})
    # pruned_sev1 / recovered_sev1 from Arm-D (alt10s)
    for sysn, prefix in (("pruned_sev1", "p1_pruned_ema_reconstructed_noadapter_alt10s"),
                         ("recovered_sev1", "p1_recovered_noadapter_alt10s")):
        items = [{"caption": armd[i]["caption"], "wav": os.path.join(ARMD_ROOT, f"{prefix}_p{i}_r0.wav")}
                 for i in range(80)]
        _check(items, 80, f"dense10s/{sysn}")
        groups.append({"name": f"dense10s__{sysn}", "items": items})
    return {"groups": groups, "convention": "one 80-item seed-once call per system (Arm-D dense control)",
            "order": "subset_prompt_index 0..79; all 10.24s / paired x_T"}


def main():
    os.makedirs("artifacts/icassp_gate0/_score_tmp", exist_ok=True)
    sev2 = build_sev2()
    dense = build_dense_control()
    p2 = "artifacts/icassp_gate0/_score_tmp/xsev_sev2_groups_in.json"
    pd = "artifacts/icassp_gate0/_score_tmp/xsev_dense_groups_in.json"
    json.dump(sev2, open(p2, "w"), indent=1, ensure_ascii=False)
    json.dump(dense, open(pd, "w"), indent=1, ensure_ascii=False)
    print("SEV2 groups:", [g["name"] for g in sev2["groups"]])
    print("DENSE groups:", [g["name"] for g in dense["groups"]])
    print("wrote", p2, "and", pd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
