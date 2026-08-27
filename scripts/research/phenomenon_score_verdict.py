#!/usr/bin/env python3
"""Phenomenon-falsifier scoring + verdict (prereg v5, decision statistic D).

Consumes the FROZEN dense Gate-0 manifest plus the two downstream manifests
(p1_pruned_ema_reconstructed, p1_recovered), validates all three with the ONE shared parametric
validator, then scores the 6 systems with the FROZEN Option-B convention (PHENOM-SCORING-B): ONE
192-item scorer call PER SYSTEM (order (prompt_index, replicate_index), np.random.seed 20260826 reset
once per system), model loaded once, pinned fused-CLAP revision 365dea6e. NOT one 1152 batch, NOT
arbitrary 32/64 chunks, NOT per-item seeding — the CLAP feature extractor has a batch-level stochastic
is_longer selection for short-audio batches, so batch/order/seed are part of the frozen endpoint and,
because the same seed restarts per system, the nuisance is PAIRED by position across all six systems.
Then applies the pre-registered dual gate PER severity:

  standalone_preserved   iff upper_CI95[E(s)] <= 0.025
  differential_fragility  iff point D(s) >= 0.025 AND lower_CI95[D(s)] > 0     # D, never F
  phenomenon             iff BOTH

PRIMARY confirmatory endpoint = p1_recovered (single endpoint, no MCC).
SECONDARY / mechanistic = p1_pruned_ema_reconstructed (context, not a second route to the claim).

--self-test runs the assembly+verdict path on synthetic cosines (no audio, no model). The real run
requires the 768 downstream WAVs (GPU) which are NOT generated yet.
"""
import argparse, json, os, subprocess, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np

sys.path.insert(0, "scripts/research")
BATTERY = "configs/research/icassp_gate0_battery.json"
METRICS_PY = ".venv-metrics/bin/python"
SCORER = "scripts/research/gate0_clap_scorer.py"
SLICED_META = "artifacts/icassp_gate0/sliced_adapter/gate0_sliced_adapter_1_2_3_1_meta.json"
DENSE_ADAPTER_SHA = "84a24a38fd95856dd9a5de58c4a0885ca42a03503c88ad66919132f3a57b7c6e"
SALT = "icassp-gate0-noise-20260826"


def _spec(battery, backbones, on_id):
    from research_pruning.manifest_validator import ManifestSpec, battery_ytids
    return ManifestSpec(
        n_prompts=64, replicates=3, battery_ytids=battery_ytids(battery),
        backbones=set(backbones), adapter_state_ids={"off": {"none"}, "on": {on_id}},
        recipe={"ddim_steps": 50, "guidance": 2.5, "eta": 0.0, "latent_t": 96}, seed_salt=SALT)


def to_grid(rows, cos):
    ps = sorted({r["prompt_index"] for r in rows}); rs = sorted({r["replicate_index"] for r in rows})
    pi = {p: i for i, p in enumerate(ps)}; ri = {r: j for j, r in enumerate(rs)}
    arr = np.full((len(ps), len(rs)), np.nan)
    for row, c in zip(rows, cos):
        arr[pi[row["prompt_index"]], ri[row["replicate_index"]]] = c
    if np.isnan(arr).any():
        raise SystemExit("grid has missing cells")
    return arr


def score_per_system(groups, out_dir):
    """Score the 6 systems via ONE score_groups call (model loaded once; EACH system = one 192-item
    seed-once batch, the frozen Gate-0 convention — Option B / PHENOM-SCORING-B). NOT one 1152 batch,
    NOT arbitrary chunks, NOT per-item seeding. Returns {name: cosines np.array}, scorer_provenance."""
    os.makedirs(out_dir, exist_ok=True)
    inp = os.path.join(out_dir, "_phenom_groups_in.json"); out = os.path.join(out_dir, "_phenom_groups_out.json")
    json.dump({"groups": groups}, open(inp, "w"))
    subprocess.run([METRICS_PY, SCORER, "--score-groups", inp, out], check=True,
                   env={**os.environ, "OPENBLAS_CORETYPE": "Haswell"}, stdout=subprocess.DEVNULL)
    d = json.load(open(out))
    return {r["name"]: np.array(r["cosines"], np.float64) for r in d["results"]}, d.get("scorer_provenance")


def severity(name, dense_off, dense_on, s_off, s_on):
    from research_pruning.eval.cluster_bootstrap import severity_verdict
    return severity_verdict(name, dense_off, s_off, dense_on, dense_off, s_on, s_off).as_dict()


def _self_test():
    """Synthetic wiring check: build 3 manifests + cosines, run assembly + verdict, sanity gates."""
    from research_pruning.manifest_validator import derive_paired_seed, battery_ytids
    battery = json.load(open(BATTERY)); yt = battery_ytids(battery)
    rng = np.random.default_rng(0)
    # synthetic per-prompt truth: dense uplift ~0.05; recovered preserves standalone (E~0.01),
    # loses adapter uplift (D~0.04); pruned loses standalone a lot (E~0.06) -> descriptive only.
    def mk(backbone, on_id, base_shift, uplift):
        rows = []
        for st, aid in (("off", "none"), ("on", on_id)):
            for p in range(64):
                for r in range(3):
                    rows.append({"ytid": yt[p], "prompt_index": p, "replicate_index": r,
                                 "seed": derive_paired_seed(SALT, yt[p], r), "backbone_id": backbone,
                                 "adapter_state": st, "adapter_id": aid, "checkpoint": "x",
                                 "ddim_steps": 50, "eta": 0.0, "guidance": 2.5, "latent_t": 96,
                                 "wav_sha256": "a" * 64})
        return {"n": len(rows), "rows": rows}
    dense = mk("dense", DENSE_ADAPTER_SHA, 0.0, 0.05)
    recov = mk("p1_recovered", "sliced", 0.01, 0.01)
    prun = mk("p1_pruned_ema_reconstructed", "sliced", 0.06, 0.0)
    # synthetic cosines consistent with the truth (per prompt/seed)
    base = rng.normal(0.20, 0.03, (64, 3))
    cos = {
        ("dense", "off"): base, ("dense", "on"): base + 0.05 + rng.normal(0, 0.004, (64, 3)),
        ("p1_recovered", "off"): base - 0.01, ("p1_recovered", "on"): base - 0.01 + 0.01,
        ("p1_pruned_ema_reconstructed", "off"): base - 0.06,
        ("p1_pruned_ema_reconstructed", "on"): base - 0.06 + 0.0,
    }
    def arr(man):
        b = man["rows"][0]["backbone_id"]
        off = to_grid([r for r in man["rows"] if r["adapter_state"] == "off"],
                      [cos[(b, "off")][r["prompt_index"], r["replicate_index"]]
                       for r in man["rows"] if r["adapter_state"] == "off"])
        on = to_grid([r for r in man["rows"] if r["adapter_state"] == "on"],
                     [cos[(b, "on")][r["prompt_index"], r["replicate_index"]]
                      for r in man["rows"] if r["adapter_state"] == "on"])
        return off, on
    d_off, d_on = arr(dense); r_off, r_on = arr(recov); p_off, p_on = arr(prun)
    prim = severity("p1_recovered", d_off, d_on, r_off, r_on)
    sec = severity("p1_pruned_ema_reconstructed", d_off, d_on, p_off, p_on)
    ok = (prim["decision_statistic"] == "D" and prim["differential_fragility"]
          and prim["standalone_preserved"] and prim["phenomenon"]
          and (not sec["standalone_preserved"]) and (not sec["phenomenon"]))
    print(json.dumps({"primary_recovered": {k: prim[k] for k in
                      ("standalone_preserved", "differential_fragility", "phenomenon", "decision_statistic")},
                      "secondary_pruned": {k: sec[k] for k in
                      ("standalone_preserved", "differential_fragility", "phenomenon")}}, indent=2))
    print("PHENOM-VERDICT SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-manifest", help="frozen Gate-0 both-manifest (dense)")
    ap.add_argument("--pruned-manifest")
    ap.add_argument("--recovered-manifest")
    ap.add_argument("--wav-root", default="", help="applied to ALL manifests if the per-root flags are unset")
    ap.add_argument("--dense-wav-root", default=None, help="root for the dense WAVs (gate0-gen-1 job dir)")
    ap.add_argument("--downstream-wav-root", default=None,
                    help="root for the pruned+recovered WAVs (gate0-phenom-1 job dir)")
    ap.add_argument("--out", default="artifacts/icassp_gate0/phenomenon_verdict.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not (args.dense_manifest and args.pruned_manifest and args.recovered_manifest):
        raise SystemExit("need --dense-manifest, --pruned-manifest, --recovered-manifest (or --self-test)")

    from research_pruning.manifest_validator import assert_valid
    battery = json.load(open(BATTERY))
    caps = {i: p["caption"] for i, p in enumerate(battery["prompts"])}
    on_id = json.load(open(SLICED_META))["sliced_adapter_sha256"]

    mans = {"dense": json.load(open(args.dense_manifest)),
            "p1_pruned_ema_reconstructed": json.load(open(args.pruned_manifest)),
            "p1_recovered": json.load(open(args.recovered_manifest))}
    assert_valid(mans["dense"], _spec(battery, {"dense"}, DENSE_ADAPTER_SHA))
    assert_valid(mans["p1_pruned_ema_reconstructed"], _spec(battery, {"p1_pruned_ema_reconstructed"}, on_id))
    assert_valid(mans["p1_recovered"], _spec(battery, {"p1_recovered"}, on_id))

    # Frozen convention (Option B): ONE 192-item scorer call PER SYSTEM, order (prompt_index,
    # replicate_index), seed-once per system. Build 6 systems in a fixed order (primary first).
    # dense WAVs live in the gate0-gen-1 job dir; downstream in the gate0-phenom-1 job dir.
    dense_root = args.dense_wav_root if args.dense_wav_root is not None else args.wav_root
    down_root = args.downstream_wav_root if args.downstream_wav_root is not None else args.wav_root
    def wav(r, backbone):
        root = dense_root if backbone == "dense" else down_root
        return os.path.join(root, r["wav"]) if root else r["wav"]
    SYSTEMS = [("dense", "off"), ("dense", "on"),
               ("p1_recovered", "off"), ("p1_recovered", "on"),
               ("p1_pruned_ema_reconstructed", "off"), ("p1_pruned_ema_reconstructed", "on")]
    groups = []; sysrows = {}
    for bk, st in SYSTEMS:
        rows = sorted([r for r in mans[bk]["rows"] if r["adapter_state"] == st],
                      key=lambda r: (r["prompt_index"], r["replicate_index"]))
        if len(rows) != 192:
            raise SystemExit(f"system {bk}/{st} has {len(rows)} rows (expected 192)")
        sysrows[(bk, st)] = rows
        groups.append({"name": f"{bk}__{st}",
                       "items": [{"caption": caps[r["prompt_index"]], "wav": wav(r, bk)} for r in rows]})
    cos_by_name, scorer_prov = score_per_system(groups, os.path.dirname(args.out) or ".")
    grids = {(bk, st): to_grid(sysrows[(bk, st)], cos_by_name[f"{bk}__{st}"]) for bk, st in SYSTEMS}

    d_off, d_on = grids[("dense", "off")], grids[("dense", "on")]
    primary = severity("p1_recovered", d_off, d_on,
                       grids[("p1_recovered", "off")], grids[("p1_recovered", "on")])
    secondary = severity("p1_pruned_ema_reconstructed", d_off, d_on,
                         grids[("p1_pruned_ema_reconstructed", "off")],
                         grids[("p1_pruned_ema_reconstructed", "on")])
    res = {
        "decision_statistic": "D", "prereg": "v5",
        "PRIMARY_p1_recovered": primary, "SECONDARY_p1_pruned_ema_reconstructed": secondary,
        "PRIMARY_PASS": bool(primary["phenomenon"]),
        "scorer_provenance": scorer_prov,
        "provenance": {"dense_manifest": args.dense_manifest, "pruned_manifest": args.pruned_manifest,
                       "recovered_manifest": args.recovered_manifest,
                       "dense_provenance": mans["dense"].get("provenance"),
                       "pruned_provenance": mans["p1_pruned_ema_reconstructed"].get("provenance"),
                       "recovered_provenance": mans["p1_recovered"].get("provenance")},
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({"PRIMARY_PASS": res["PRIMARY_PASS"],
                      "primary": {k: primary[k] for k in ("E", "D", "phenomenon")}}, indent=2))
    print("PHENOMENON-VERDICT written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
