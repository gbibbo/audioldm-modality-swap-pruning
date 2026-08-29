#!/usr/bin/env python3
"""Regression tests for the RECOVERY-REVERSAL-V1.1 pre-data hashing correction (Convention B).

Guards the corrected deterministic rules and makes accidental regression to V1.0 hard:

A1 YTID-TOKEN        selection_key_v11 contains literal |YTID|; removing it changes the hash.
A2 CAPTION-TOKEN     caption_hash uses literal |CAPTION|; caption TEXT never enters the hash.
A3 MULTISET-KEEP     duplicate caption rows are preserved (NOT deduplicated); 5 rows -> n==5.
A4 SORT-DET          sorting duplicate UTF-8 strings is deterministic.
A5 PERM-INVARIANT    permuting source row order does not change the canonical multiset.
A6 MULTIPLICITY      duplicate multiplicity CAN change the modulo pick vs a deduplicated set.
A7 DET-INDEX         same ytid + same n -> deterministic caption index; n drives only the modulo.
A8 NO-DEDUP-CODE     canonical_caption_rows_v11 does not call set()/dict.fromkeys()/np.unique.
A9 NOT-V10-YTID      V1.1 ytid key != V1.0 selection_order_key (sha256(SALT|ytid)).
A10 NOT-V10-CAPTION  V1.1 caption pick is modulo-over-rows, NOT the V1.0 argmin(sha256(SALT|ytid|cap)).
A11 BUILDER-STOP     the V1.1 builder STOPs if a selected ytid has a caption-row count != 5.
A12 PROD-USES-V11    the V1.1 builder imports choose_caption_v11 (multiset), not V1.0 choose_caption.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python tests/research/test_reversal_v1_1.py
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/research")
from research_pruning.eval.reversal import (  # noqa: E402
    SELECTION_SALT_V1, canonical_caption_rows_v11, choose_caption, choose_caption_v11,
    canonical_caption_rows_v11 as _canon, selection_key_v11, selection_order_key)

DUP = ["dog barks", "dog barks", "car passes", "speech", "wind"]


def a1_ytid_token():
    exp = hashlib.sha256(f"{SELECTION_SALT_V1}|YTID|abc".encode()).hexdigest()
    no_tok = hashlib.sha256(f"{SELECTION_SALT_V1}|abc".encode()).hexdigest()
    ok = selection_key_v11("abc") == exp and exp != no_tok
    print(f"  A1 |YTID| token present & load-bearing={ok}")
    return ok


def a2_caption_token():
    c1 = choose_caption_v11("abc", DUP)["caption_hash_hex"]
    c2 = choose_caption_v11("abc", ["totally", "different", "five", "caption", "texts"])["caption_hash_hex"]
    exp = hashlib.sha256(f"{SELECTION_SALT_V1}|CAPTION|abc".encode()).hexdigest()
    ok = c1 == exp and c1 == c2  # hash depends only on ytid, not caption text
    print(f"  A2 |CAPTION| token, caption text NOT hashed={ok}")
    return ok


def a3_multiset_keep():
    canon = canonical_caption_rows_v11(DUP)
    ok = len(canon) == 5 and canon.count("dog barks") == 2
    print(f"  A3 multiset preserved n={len(canon)} dup_kept={canon.count('dog barks')==2} ok={ok}")
    return ok


def a4_sort_det():
    ok = canonical_caption_rows_v11(DUP) == canonical_caption_rows_v11(DUP)
    ok = ok and canonical_caption_rows_v11(DUP) == sorted(DUP, key=lambda c: c.encode())
    print(f"  A4 deterministic UTF-8 sort of duplicates={ok}")
    return ok


def a5_perm_invariant():
    import itertools
    base = canonical_caption_rows_v11(DUP)
    perms = [canonical_caption_rows_v11(list(p)) for p in itertools.islice(itertools.permutations(DUP), 20)]
    ok = all(p == base for p in perms)
    print(f"  A5 canonical multiset invariant to source row permutation={ok}")
    return ok


def a6_multiplicity():
    # find a ytid where multiset (n=5) and dedup (n=4) pick different text
    diff_found = False
    for i in range(2000):
        y = f"y{i}"
        canon = canonical_caption_rows_v11(DUP)                       # n=5
        uniq = sorted(set(DUP), key=lambda c: c.encode())            # n=4
        h = hashlib.sha256(f"{SELECTION_SALT_V1}|CAPTION|{y}".encode()).digest()
        m = canon[int.from_bytes(h[:8], "big") % len(canon)]
        u = uniq[int.from_bytes(h[:8], "big") % len(uniq)]
        if m != u:
            diff_found = True
            break
    print(f"  A6 multiplicity can change pick vs dedup={diff_found}")
    return diff_found


def a7_det_index():
    a = choose_caption_v11("zz", DUP); b = choose_caption_v11("zz", DUP)
    ok = a == b and a["n_caption_rows"] == 5 and a["chosen_caption_index"] == (
        int.from_bytes(hashlib.sha256(f"{SELECTION_SALT_V1}|CAPTION|zz".encode()).digest()[:8], "big") % 5)
    print(f"  A7 deterministic index, n drives modulo={ok}")
    return ok


def a8_no_dedup_code():
    src = inspect.getsource(_canon) + inspect.getsource(choose_caption_v11)
    banned = ("set(", "fromkeys", "np.unique", "unique(")
    ok = not any(b in src for b in banned)
    print(f"  A8 no dedup primitive in V1.1 caption code={ok}")
    return ok


def a9_not_v10_ytid():
    ok = selection_key_v11("abc") != selection_order_key("abc")
    print(f"  A9 V1.1 ytid key != V1.0 ytid key={ok}")
    return ok


def a10_not_v10_caption():
    # V1.0 argmin over sha256(SALT|ytid|caption) on unique captions; V1.1 = modulo over rows.
    y = "zz"
    v10 = choose_caption(y, DUP)["caption"]
    v11 = choose_caption_v11(y, DUP)["caption"]
    # they need not always differ, but the algorithms must be distinct: V1.0 hashes caption text
    v10_key = min(sorted(set(DUP), key=lambda c: c.encode()),
                  key=lambda c: hashlib.sha256(f"{SELECTION_SALT_V1}|{y}|{c}".encode()).hexdigest())
    ok = v10 == v10_key and "caption_hash_hex" in choose_caption_v11(y, DUP)
    print(f"  A10 V1.0 argmin(text-hash) vs V1.1 modulo-rows are distinct algorithms={ok}")
    return ok


def a11_builder_stop():
    import reversal_v1_1_select_audiocaps as B11
    # monkeypatch load_universe to return a selected ytid with 4 rows -> must STOP
    import research_pruning.eval.reversal as R
    fake_caps = {f"yt{i:03d}": ["a", "b", "c", "d", "e"] for i in range(200)}
    # make one eligible ytid that will be selected have only 4 rows
    picked = R.select_prompts_v11(list(fake_caps), 96)
    fake_caps[picked[0]] = ["a", "b", "c", "d"]  # 4 rows
    orig = B11.V10.load_universe
    B11.V10.load_universe = lambda: (fake_caps, sorted(fake_caps), {})
    import csv as _csv
    orig_open = B11.csv.DictReader
    # empty train/music/kim via real files is fine; exclusions won't drop these synthetic ids
    raised = False
    try:
        B11.build("/tmp/should_not_write_v11.json")
    except SystemExit as e:
        raised = "!= 5" in str(e) or "caption-row count" in str(e)
    finally:
        B11.V10.load_universe = orig
    wrote = os.path.exists("/tmp/should_not_write_v11.json")
    print(f"  A11 builder STOPs on !=5 rows={raised} (no file written={not wrote})")
    return raised and not wrote


def a12_prod_uses_v11():
    import reversal_v1_1_select_audiocaps as B11
    src = inspect.getsource(B11)
    ok = "choose_caption_v11" in src and "select_prompts_v11" in src
    # and it must NOT call the V1.0 choose_caption / selection_order_key for selection
    ok = ok and "choose_caption(" not in src.replace("choose_caption_v11", "")
    print(f"  A12 production builder uses V1.1 multiset helpers={ok}")
    return ok


def main():
    checks = [a1_ytid_token, a2_caption_token, a3_multiset_keep, a4_sort_det, a5_perm_invariant,
              a6_multiplicity, a7_det_index, a8_no_dedup_code, a9_not_v10_ytid, a10_not_v10_caption,
              a11_builder_stop, a12_prod_uses_v11]
    res = [c() for c in checks]
    ok = all(res)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(res)}/{len(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
