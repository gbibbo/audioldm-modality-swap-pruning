#!/usr/bin/env python3
"""Freeze the SA3 seed table (protocol section 2.2, 9.1), CPU only.

R=5 dense seed streams; per (stream,prompt) one init-noise seed + 8 ping-pong re-noise seeds,
derived deterministically (research_sa3/seeds.py). Records the derivation rule and a hash over
the materialized seeds for the smoke + pilot panels (main is added when N_main is frozen).

Run:   .venv-sa3/bin/python scripts/sa3/build_seed_table.py --write
Check: .venv-sa3/bin/python scripts/sa3/build_seed_table.py --check
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import seeds as S

OUT = "configs/sa3/seed_table.json"


def panel_ids(name):
    d = json.load(open(f"configs/sa3/panel_{name}.json"))
    return [it["audiocap_id"] for it in d["items"]], d["items"][0]["audiocap_id"]


def serialize(obj):
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (a.write or a.check): ap.error("pass --write or --check")

    smoke_ids, _ = panel_ids("smoke")
    pilot_ids, _ = panel_ids("pilot")
    # example seeds for the first smoke prompt (eyeball only)
    ex_aid = sorted(smoke_ids, key=int)[0]
    example = {str(s): S.prompt_seeds(s, ex_aid) for s in range(S.R_STREAMS)}
    obj = {
        "master_seed": S.MASTER_SEED,
        "R_streams": S.R_STREAMS,
        "n_pingpong": S.N_PINGPONG,
        "derivation": "seed = int(sha256('MASTER|stream|audiocap_id|kind|step')[:8]) & (2**63-1); "
                      "kind in {init(step0), renoise(step0..7)}",
        "shared_semantics": "init_seed shared by every system for (stream,prompt); renoise stream "
                            "shared across systems; comparisons are seed-paired on stream 0.",
        "materialized_sha256": {
            "smoke": S.materialized_digest(smoke_ids),
            "pilot": S.materialized_digest(pilot_ids),
        },
        "example_stream_seeds_first_smoke_prompt": {"audiocap_id": ex_aid, "streams": example},
        "note": "main-panel seeds added to this file when N_main is frozen after the pilot.",
    }
    data = serialize(obj)
    digest = hashlib.sha256(data).hexdigest()
    if a.check:
        cur = open(OUT, "rb").read() if os.path.exists(OUT) else b""
        same = cur == data
        print(f"{'OK  ' if same else 'DIFF'} {OUT}  sha256={digest}")
        return 0 if same else 1
    open(OUT, "w").write(data.decode())
    print(f"WROTE {OUT}  file_sha256={digest}")
    print(f"  materialized smoke={obj['materialized_sha256']['smoke'][:16]}  pilot={obj['materialized_sha256']['pilot'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
