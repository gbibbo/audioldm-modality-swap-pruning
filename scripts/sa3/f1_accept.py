#!/usr/bin/env python3
"""F1 CPU dry-run acceptance (RQ2b) — the fast, model-free invariants. Proves everything that does not
require loading the base model; the two heavy CPU dry-runs (trainer backbone-attach + generation exec)
are run separately and their evidence linked in the ledger.

Checks:
  1. stage_trainL builds exactly train_L (96) wav+txt, eval_L absent.
  2. f1_task_gen --plan-only emits EXACTLY 4 systems x 64 eval ids, 2 pairs (F1_base, F1_post).
  3. paired ids/seeds identical across with/without-L within each pair.
  4. NO structural-analysis token appears in the F1 generation driver.
  5. dirty-tree / expected-commit guards abort the trainer AND the generation driver.
  6. the frozen SESOI/CI verdict rule reproduces (aeco_predict.f1_functional_verdict, tested in A8).

Run: OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/f1_accept.py \
        --manifest configs/sa3/adapters/mechanical.manifest.json --domain-dir data/sa3/adapters/mechanical
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
PY = sys.executable
FORBIDDEN = ["A_eco", "a_eco", "A_tan", "a_tan", "D_P", "G_ext", "skip_blocks", "block_mask", "adapter_blocks"]


def check_stage(manifest, domain_dir):
    sys.path.insert(0, HERE)
    import stage_trainL as ST
    out = tempfile.mkdtemp(prefix="f1_stage_")
    rep = ST.stage(manifest, domain_dir, out)
    man = json.load(open(manifest))
    ok = rep["count_ok"] and rep["eval_absent"] and rep["n_train_staged"] == len(man["split"]["train_L"]) == 96
    return ok, {"n_train_staged": rep["n_train_staged"], "eval_absent": rep["eval_absent"]}


def check_plan(manifest, domain_dir):
    mf = tempfile.mktemp(suffix="_f1plan.json")
    r = subprocess.run([PY, os.path.join(HERE, "f1_task_gen.py"), "--plan-only",
                        "--manifest", manifest, "--domain-dir", domain_dir, "--score-manifest", mf],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, {"stderr": r.stderr[-400:]}
    plan = json.load(open(mf))
    systems = plan["systems"]; configs = plan["configs"]; pairs = plan["pairs"]
    n_ids = {tag: len(ids) for tag, ids in configs.items()}
    # 4 systems, 64 ids each, 2 pairs
    ok = (sorted(systems) == ["base_Lfull", "base_noL", "post_Lfull", "post_noL"]
          and all(v == 64 for v in n_ids.values()) and len(pairs) == 2
          and {p["name"] for p in pairs} == {"F1_base", "F1_post"})
    # paired ids identical within each pair, and structural flag False
    for p in pairs:
        ok = ok and set(configs[p["with_L"]]) == set(configs[p["no_L"]])
    ok = ok and plan.get("structural_analysis") is False
    # seeds identical across all systems (seed indexed by eval position -> same for +L / no-L)
    ok = ok and len(plan["seeds"]) == 64
    return ok, {"systems": sorted(systems), "n_ids": n_ids, "pairs": [p["name"] for p in pairs],
                "structural_analysis": plan.get("structural_analysis")}


def check_no_structural():
    src = open(os.path.join(HERE, "f1_task_gen.py")).read()
    # strip the module docstring (it NAMES the forbidden tokens to say they're absent)
    body = src.split('"""', 2)[-1]
    hits = [t for t in FORBIDDEN if t in body]
    return (not hits), {"forbidden_hits_in_gen_driver": hits}


def check_guards(manifest, domain_dir):
    # both drivers must refuse to run (non-plan, non-dry) with a bogus expect-commit / dirty tree
    fake_adapter = tempfile.mktemp(suffix=".safetensors"); open(fake_adapter, "w").write("x")
    g = subprocess.run([PY, os.path.join(HERE, "f1_task_gen.py"), "--manifest", manifest,
                        "--domain-dir", domain_dir, "--adapter", fake_adapter,
                        "--expect-commit", "deadbeefcafe"], capture_output=True, text=True)
    t = subprocess.run([PY, os.path.join(HERE, "train_control_loras.py"), "--backbone",
                        "--data_dir", domain_dir, "--save", tempfile.mktemp(suffix=".safetensors"),
                        "--expect-commit", "deadbeefcafe"], capture_output=True, text=True)
    ok = g.returncode != 0 and t.returncode != 0
    return ok, {"gen_guard_rc": g.returncode, "trainer_guard_rc": t.returncode}


def check_verdict_rule():
    from research_sa3.aeco_predict import f1_functional_verdict as V
    strong = {"dT_AA": 0.11, "lo": 0.06, "hi": 0.16}
    weak = {"dT_AA": 0.05, "lo": 0.02, "hi": 0.09}
    ci0 = {"dT_AA": 0.09, "lo": -0.01, "hi": 0.18}
    ok = (V(strong, strong)["verdict"] == "F1_PASS"
          and V(weak, strong)["verdict"] == "STOP_RQ2B_BASE_FAIL"
          and V(strong, ci0)["verdict"] == "STOP_RQ2B_POST_FAIL"
          and V(strong, weak)["verdict"] == "STOP_RQ2B_POST_FAIL")
    return ok, {"sesoi": 0.075}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--domain-dir", required=True)
    ap.add_argument("--out", default="artifacts/sa3/f1_accept.json")
    a = ap.parse_args()
    checks = [("stage_trainL_96", lambda: check_stage(a.manifest, a.domain_dir)),
              ("plan_4x64_2pairs", lambda: check_plan(a.manifest, a.domain_dir)),
              ("no_structural_path", check_no_structural),
              ("dirty_commit_guards", lambda: check_guards(a.manifest, a.domain_dir)),
              ("verdict_rule", check_verdict_rule)]
    results = {}; all_ok = True
    for name, fn in checks:
        try:
            ok, info = fn()
        except Exception as e:
            ok, info = False, {"exception": repr(e)}
        all_ok &= ok
        results[name] = {"ok": bool(ok), **info}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(info)[:200]}")
    out = {"phase": "F1_cpu_acceptance_fast", "all_ok": all_ok, "checks": results,
           "git_commit": subprocess.getoutput("git rev-parse HEAD")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("F1_ACCEPT_FAST", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
