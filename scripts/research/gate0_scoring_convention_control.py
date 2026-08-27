#!/usr/bin/env python3
"""Option-B scoring-convention bit-exact control (PHENOM-SCORING-B, 2026-08-27).

Freezes the phenomenon scoring convention = ONE 192-item scorer call PER SYSTEM (64 prompts x 3
replicates), order (prompt_index, replicate_index), np.random.seed(20260826) reset once per system,
pinned fused-CLAP revision. Re-scores the EXISTING frozen dense OFF (192) + dense ON (192) via the
new `--score-groups` path and REQUIRES bit-exact reproduction of the frozen Gate-0 per-item cosines
(not merely the PASS verdict). If not bit-exact -> STOP. Historical Gate-0 verdict is NOT rewritten.
"""
import argparse, json, os, subprocess, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np

REPO = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
JOB_ROOT = "/teamspace/jobs/gate0-gen-1/artifacts/audioldm-modality-swap-pruning"
GEN_MANIFEST = f"{JOB_ROOT}/artifacts/icassp_gate0/gen_gate0/gen_manifest_dense_both.json"
FROZEN_VERDICT = "artifacts/icassp_gate0/gate0_verdict.json"
BATTERY = "configs/research/icassp_gate0_battery.json"
METRICS_PY = ".venv-metrics/bin/python"
SCORER = "scripts/research/gate0_clap_scorer.py"


def sorted_items(rows, state, caps):
    rs = sorted([r for r in rows if r["adapter_state"] == state],
                key=lambda r: (r["prompt_index"], r["replicate_index"]))
    return rs, [{"caption": caps[r["prompt_index"]], "wav": os.path.join(JOB_ROOT, r["wav"])} for r in rs]


def run_scorer(args_json):
    env = {**os.environ, "OPENBLAS_CORETYPE": "Haswell"}
    subprocess.run([METRICS_PY, SCORER] + args_json, check=True, env=env, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default="artifacts/icassp_gate0/_convctl")
    ap.add_argument("--report", default="artifacts/icassp_gate0/gate0_scoring_convention_control.json")
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.tmp, exist_ok=True)
    from research_pruning.eval.cluster_bootstrap import gate0_verdict

    man = json.load(open(GEN_MANIFEST))
    caps = {i: p["caption"] for i, p in enumerate(json.load(open(BATTERY))["prompts"])}
    off_rows, off_items = sorted_items(man["rows"], "off", caps)
    on_rows, on_items = sorted_items(man["rows"], "on", caps)

    # (1) score both systems via the NEW per-system score_groups (one 192-batch seed-once each)
    gin = os.path.join(a.tmp, "groups_in.json"); gout = os.path.join(a.tmp, "groups_out.json")
    json.dump({"groups": [{"name": "dense_off", "items": off_items},
                          {"name": "dense_on", "items": on_items}]}, open(gin, "w"))
    run_scorer(["--score-groups", gin, gout])
    gres = {r["name"]: np.array(r["cosines"], np.float64) for r in json.load(open(gout))["results"]}
    g_off, g_on = gres["dense_off"], gres["dense_on"]

    # (2) score each system via the existing score_json convention (frozen Gate-0 path)
    def sj(items, tag):
        i = os.path.join(a.tmp, f"{tag}_in.json"); o = os.path.join(a.tmp, f"{tag}_out.json")
        json.dump({"items": items}, open(i, "w")); run_scorer(["--score-json", i, o])
        return np.array(json.load(open(o))["cosines"], np.float64)
    j_off = sj(off_items, "off"); j_on = sj(on_items, "on")

    # (3) per-item bit-exactness: score_groups == score_json (same frozen convention)
    peritem_off = float(np.abs(g_off - j_off).max())
    peritem_on = float(np.abs(g_on - j_on).max())

    # (4) verdict from score_groups grids vs frozen verdict (full precision)
    def grid(rows, cos):
        ps = sorted({r["prompt_index"] for r in rows}); rr = sorted({r["replicate_index"] for r in rows})
        pi = {p: k for k, p in enumerate(ps)}; ri = {x: k for k, x in enumerate(rr)}
        arr = np.full((len(ps), len(rr)), np.nan)
        for r, c in zip(rows, cos): arr[pi[r["prompt_index"]], ri[r["replicate_index"]]] = c
        return arr
    base_arr = grid(off_rows, g_off); adap_arr = grid(on_rows, g_on)
    v = gate0_verdict(adap_arr, base_arr)
    fr = json.load(open(FROZEN_VERDICT))
    fd = fr["gate0_verdict"]["delta_clap"]; fdi = fr["diagnostics"]
    d = {
        "peritem_max_abs_diff_off": peritem_off, "peritem_max_abs_diff_on": peritem_on,
        "mean_base_cosine_vs_frozen": abs(float(base_arr.mean()) - fdi["mean_base_cosine"]),
        "mean_adapter_cosine_vs_frozen": abs(float(adap_arr.mean()) - fdi["mean_adapter_cosine"]),
        "delta_point_vs_frozen": abs(v.delta_clap.point - fd["point"]),
        "delta_lo_vs_frozen": abs(v.delta_clap.lo - fd["lo"]),
        "delta_hi_vs_frozen": abs(v.delta_clap.hi - fd["hi"]),
    }
    bit_exact = (peritem_off == 0.0 and peritem_on == 0.0
                 and d["mean_base_cosine_vs_frozen"] == 0.0 and d["mean_adapter_cosine_vs_frozen"] == 0.0)
    verdict_exact = (d["delta_point_vs_frozen"] == 0.0 and d["delta_lo_vs_frozen"] == 0.0
                     and d["delta_hi_vs_frozen"] == 0.0 and bool(v.passed) == bool(fr["PASS"]))
    ok = bit_exact and verdict_exact
    report = {"control": "phenom_scoring_convention_B_bitexact",
              "convention": "one 192-item seed-once call per system, order (prompt,rep), rev 365dea6e",
              "diffs": d, "bit_exact_per_item": bit_exact, "verdict_exact": verdict_exact,
              "score_groups_matches_score_json": peritem_off == 0.0 and peritem_on == 0.0,
              "rescored_point": v.delta_clap.point, "frozen_point": fd["point"],
              "PASS_unchanged": bool(v.passed) == bool(fr["PASS"]), "REPRODUCES": ok}
    json.dump(report, open(a.report, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("SCORING-CONVENTION-CONTROL", "PASS (bit-exact)" if ok else "FAIL -> STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
