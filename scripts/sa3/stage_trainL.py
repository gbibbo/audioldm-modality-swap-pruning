#!/usr/bin/env python3
"""Stage the FROZEN train_L split of a domain into a training directory the upstream SampleDataset can
read: one `<id>.wav` + one `<id>.txt` (caption) per train_L clip, and NOTHING else. The trainer scans
its --data_dir for wav+txt pairs, so the staging dir MUST contain train_L only — this is what keeps
eval_L held out during F1/F2 training.

Captions come from the FROZEN manifest (`build_domain_manifest.derive_caption`), the single source of
truth — the per-clip `.meta.json` has no caption. Deterministic + idempotent: the out dir is rebuilt
from scratch each run. Asserts: exactly len(train_L) wav+txt pairs, every train_L id present, and NO
eval_L id present.

Run:  OPENBLAS_CORETYPE=Haswell .venv-sa3/bin/python scripts/sa3/stage_trainL.py \
          --manifest configs/sa3/adapters/mechanical.manifest.json \
          --domain-dir data/sa3/adapters/mechanical --out data/sa3/adapters/mechanical_trainL
"""
from __future__ import annotations
import argparse, json, os, shutil, sys


def stage(manifest_path: str, domain_dir: str, out_dir: str) -> dict:
    man = json.load(open(manifest_path))
    caps = {c["id"]: c["caption"] for c in man["clips"]}
    train_ids = sorted(man["split"]["train_L"])
    eval_ids = set(man["split"]["eval_L"])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    staged = []
    for cid in train_ids:
        assert cid not in eval_ids, f"{cid} is in eval_L -- must never be staged for training"
        src = os.path.join(domain_dir, f"{cid}.wav")
        assert os.path.exists(src), f"missing source wav {src}"
        cap = caps[cid]
        assert cap and cap.strip(), f"empty caption for {cid}"
        dst_wav = os.path.join(out_dir, f"{cid}.wav")
        try:
            os.link(src, dst_wav)           # hardlink (data is gitignored; saves space)
        except OSError:
            shutil.copy2(src, dst_wav)
        with open(os.path.join(out_dir, f"{cid}.txt"), "w") as fh:
            fh.write(cap.strip())
        staged.append(cid)
    # invariants
    wavs = sorted(f[:-4] for f in os.listdir(out_dir) if f.endswith(".wav"))
    txts = sorted(f[:-4] for f in os.listdir(out_dir) if f.endswith(".txt"))
    assert wavs == train_ids, f"staged wavs != train_L ({len(wavs)} vs {len(train_ids)})"
    assert txts == train_ids, f"staged txts != train_L ({len(txts)} vs {len(train_ids)})"
    assert not (set(wavs) & eval_ids), "eval_L leaked into staging dir"
    rep = {"domain": man["domain"], "manifest_sha256": man.get("manifest_sha256"),
           "out_dir": out_dir, "n_train_staged": len(staged), "n_train_L": len(train_ids),
           "n_eval_L": len(eval_ids), "eval_absent": not (set(wavs) & eval_ids),
           "count_ok": len(staged) == len(train_ids)}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--domain-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = stage(a.manifest, a.domain_dir, a.out)
    print(json.dumps(rep, indent=2))
    ok = rep["count_ok"] and rep["eval_absent"]
    print("STAGE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
