#!/usr/bin/env python3
"""Gate-0 verdict: fused-CLAP score the dense +/- LoRA WAVs, compute the prompt-clustered paired
ΔCLAP with the frozen cluster bootstrap, and emit the pre-registered Gate-0 verdict.

Reads a `--adapter-mode both` generation manifest (rows tagged adapter_state=off|on), scores each
system's (caption_i, wav_i) with the frozen fused-CLAP scorer (run in the metrics venv), arranges
the cosines into (n_prompts, n_replicates) paired arrays in a fixed (prompt_index, replicate_index)
order, and calls research_pruning.eval.cluster_bootstrap.gate0_verdict.

PASS iff point ΔCLAP >= SESOI (0.025) AND lower-CI95 > 0 (unit = prompt; B, seed from the frozen code).
Diagnostics (per-prompt ΔCLAP, per-replicate means, raw cosines) are reported to detect implementation
pathology WITHOUT changing the gate. No science parameter is set here — all thresholds come from the
frozen prereg/code.
"""
import argparse, json, os, subprocess, sys
import numpy as np

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")

BATTERY = "configs/research/icassp_gate0_battery.json"
METRICS_PY = ".venv-metrics/bin/python"
SCORER = "scripts/research/gate0_clap_scorer.py"


def score_items(items, metrics_python, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    inp = os.path.join(tmp_dir, "_score_in.json")
    out = os.path.join(tmp_dir, "_score_out.json")
    json.dump({"items": items}, open(inp, "w"))
    subprocess.run([metrics_python, SCORER, "--score-json", inp, out], check=True,
                   env={**os.environ, "OPENBLAS_CORETYPE": "Haswell"}, stdout=subprocess.DEVNULL)
    return np.array(json.load(open(out))["cosines"], dtype=np.float64)


def to_grid(rows, cosines, prompts_sorted, reps_sorted):
    pi_idx = {pi: i for i, pi in enumerate(prompts_sorted)}
    r_idx = {r: j for j, r in enumerate(reps_sorted)}
    arr = np.full((len(prompts_sorted), len(reps_sorted)), np.nan)
    for row, cos in zip(rows, cosines):
        arr[pi_idx[row["prompt_index"]], r_idx[row["replicate_index"]]] = cos
    if np.isnan(arr).any():
        raise SystemExit("score grid has missing (prompt,replicate) cells — manifest incomplete")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-manifest", required=True, help="a --adapter-mode both generation manifest")
    ap.add_argument("--battery", default=BATTERY)
    ap.add_argument("--metrics-python", default=METRICS_PY)
    ap.add_argument("--wav-root", default="", help="prepend to each manifest wav path if WAVs moved")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from research_pruning.eval.cluster_bootstrap import gate0_verdict, SESOI

    man = json.load(open(args.gen_manifest))
    rows = man["rows"]
    captions = {p_i: p["caption"] for p_i, p in enumerate(json.load(open(args.battery))["prompts"])}

    base_rows = sorted([r for r in rows if r["adapter_state"] == "off"],
                       key=lambda r: (r["prompt_index"], r["replicate_index"]))
    adap_rows = sorted([r for r in rows if r["adapter_state"] == "on"],
                       key=lambda r: (r["prompt_index"], r["replicate_index"]))
    if not base_rows or not adap_rows:
        raise SystemExit("manifest must contain BOTH adapter_state=off and adapter_state=on rows "
                         "(run the generator with --adapter-mode both)")
    key = lambda rs: [(r["prompt_index"], r["replicate_index"]) for r in rs]
    if key(base_rows) != key(adap_rows):
        raise SystemExit("off/on rows are not paired on (prompt_index, replicate_index)")

    prompts_sorted = sorted({r["prompt_index"] for r in base_rows})
    reps_sorted = sorted({r["replicate_index"] for r in base_rows})

    def wav(r):
        return os.path.join(args.wav_root, r["wav"]) if args.wav_root else r["wav"]

    tmp = os.path.join(os.path.dirname(args.out) or ".", "_score_tmp")
    base_items = [{"caption": captions[r["prompt_index"]], "wav": wav(r)} for r in base_rows]
    adap_items = [{"caption": captions[r["prompt_index"]], "wav": wav(r)} for r in adap_rows]
    base_cos = score_items(base_items, args.metrics_python, tmp)
    adap_cos = score_items(adap_items, args.metrics_python, tmp)

    base_arr = to_grid(base_rows, base_cos, prompts_sorted, reps_sorted)
    adap_arr = to_grid(adap_rows, adap_cos, prompts_sorted, reps_sorted)

    verdict = gate0_verdict(adap_arr, base_arr)
    per_prompt_delta = (adap_arr - base_arr).mean(axis=1)     # ΔCLAP per prompt (unit of the bootstrap)
    per_rep_delta = (adap_arr - base_arr).mean(axis=0)        # mean ΔCLAP per replicate (diagnostic)

    res = {
        "gate0_verdict": verdict.as_dict(),
        "SESOI": SESOI,
        "PASS": bool(verdict.passed),
        "n_prompts": len(prompts_sorted), "n_replicates": len(reps_sorted),
        "diagnostics": {
            "mean_base_cosine": float(base_arr.mean()),
            "mean_adapter_cosine": float(adap_arr.mean()),
            "mean_delta_clap": float((adap_arr - base_arr).mean()),
            "per_prompt_delta_clap": [round(float(x), 6) for x in per_prompt_delta],
            "per_replicate_mean_delta": [round(float(x), 6) for x in per_rep_delta],
            "frac_prompts_positive": float((per_prompt_delta > 0).mean()),
        },
        "provenance": {"gen_manifest": args.gen_manifest, "battery": args.battery,
                       "adapter": man.get("adapter"), "backbone": man.get("backbone"),
                       "recipe": man.get("recipe")},
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("PASS", "gate0_verdict", "n_prompts", "n_replicates")}, indent=2))
    print("GATE-0", "PASS" if res["PASS"] else "FAIL", "-> verdict written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
