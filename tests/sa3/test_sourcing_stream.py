#!/usr/bin/env python3
"""CPU tests for the streaming sourcer: backfill-after-rejection, automatic prompts_L, provenance
(no network). Run: .venv-sa3/bin/python tests/sa3/test_sourcing_stream.py  (needs numpy)."""
import os, sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, "scripts", "sa3"))
import numpy as np
import fetch_freesound_domain as FS
import build_domain_manifest as BDM


def _tone(freq, sr=44100, dur=0.5):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype("float32")


def t1_backfill_past_rejections():
    good1, good2, good3 = _tone(200), _tone(300), _tone(400)
    dup = good1.copy()
    silent = np.zeros(22050, "float32")
    audio = {0: silent, 1: good1, 2: dup, 3: good2, 4: None, 5: good3}   # rejects at 0,2,4
    cands = [{"rank": i, "id": i, "source_original_duration": 1.0} for i in range(6)]
    accepted, decisions = FS.stream_accept(iter(cands), lambda c: audio[c["id"]], want=3)
    ok = [c["id"] for c in accepted] == [1, 3, 5]                        # backfilled past 0,2,4
    ok = ok and len(decisions) == 6
    rej = {d["id"]: d["reason"] for d in decisions if not d["accepted"]}
    ok = ok and "silent" in rej[0] and "duplicate" in rej[2] and "undecodable" in rej[4]
    ok = ok and all("_audio_sha" in c for c in accepted)
    print(f"    T1 accepted={[c['id'] for c in accepted]} rejects={sorted(rej)} ({len(decisions)} decisions)")
    return ok


def t2_duration_reject_and_exhaustion():
    # a long clip is rejected on duration even with good audio; stream can end below `want`
    cands = [{"rank": 0, "id": 0, "source_original_duration": 20.0},   # > DUR_MAX -> reject
             {"rank": 1, "id": 1, "source_original_duration": 0.1}]    # < DUR_MIN -> reject
    accepted, decisions = FS.stream_accept(iter(cands), lambda c: _tone(200), want=5)
    ok = accepted == [] and len(decisions) == 2 and all(not d["accepted"] for d in decisions)
    ok = ok and all("duration" in d["reason"] for d in decisions)
    print(f"    T2 duration rejects both, accepted={accepted} (exhausted below want)")
    return ok


def t3_auto_prompts_from_eval_no_leak():
    recs = [{"id": f"c{i}", "audio_sha256_44k_mono": f"{i}", "caption": f"a sound of thing {i}"} for i in range(8)]
    train, ev = BDM.deterministic_split([r["id"] for r in recs], seed=20260821)
    prompts = BDM.derive_prompts_L(recs, train, ev, "impact_percussion")
    by_id = {r["id"]: r for r in recs}
    train_caps = {by_id[i]["caption"] for i in train}
    eval_caps = {by_id[i]["caption"] for i in ev}
    ok = len(prompts) >= 1 and "a impact sound" in prompts                 # generic domain prompt appended
    ok = ok and all(p in eval_caps or p == "a impact sound" for p in prompts)  # only eval captions + generic
    ok = ok and not (set(prompts) & train_caps)                            # no train leakage
    # determinism
    ok = ok and BDM.derive_prompts_L(recs, train, ev, "impact_percussion") == prompts
    print(f"    T3 auto prompts_L (n={len(prompts)}) from eval + generic, no train leak")
    return ok


def t4_provenance_shape():
    # the pieces main() persists: filter/query/sort frozen; decisions carry rank/id/reason/sha
    ok = FS.build_filter("impact_percussion") == 'license:"Creative Commons 0" tag:(impact OR hit OR percussion OR clap OR knock)'
    ok = ok and FS.ACQ_REPR == "preview-hq-mp3" and FS.SORT == "downloads_desc"
    _, decisions = FS.stream_accept(iter([{"rank": 0, "id": 9, "source_original_duration": 1.0}]),
                                    lambda c: _tone(200), want=1)
    d = decisions[0]
    ok = ok and set(("rank", "id", "accepted", "reason", "audio_sha256_44k_mono")) <= set(d)
    ok = ok and d["accepted"] and bool(d["audio_sha256_44k_mono"])
    print(f"    T4 provenance frozen filter/sort/acq + decision keys present")
    return bool(ok)


def main():
    checks = [("T1", t1_backfill_past_rejections), ("T2", t2_duration_reject_and_exhaustion),
              ("T3", t3_auto_prompts_from_eval_no_leak), ("T4", t4_provenance_shape)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
