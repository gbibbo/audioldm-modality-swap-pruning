#!/usr/bin/env python3
"""Source-leakage audit: recover the MusicCaps ytids that Kim-193 was built from.

Kim's captions ARE MusicCaps captions with a deterministic appended suffix
`" The subgenre of hip-hop is <SUBGENRE>"`. That suffix is why an exact-string dedup vs Kim found
0 hits in the first battery freeze (build_gate0_battery.py) even though every Kim example
originates from MusicCaps — the Jaccard≥0.85 near-dup rule caught them, but a fragile threshold is
not an acceptable disjointness guarantee for a PRIMARY evaluation artifact.

This audit strips ONLY that deterministic suffix (rule below, derived from the released dataset
format — NOT fuzzy post-hoc editing), matches the base caption EXACTLY against the MusicCaps CSV,
and freezes the recovered set of training-source ytids. Battery exclusion then operates primarily
by SOURCE YTID (this file), with caption exact/near-dup as a secondary safeguard.

Outputs configs/research/kim193_source_ytids.json (tracked provenance).
"""
import csv, json, re, sys

KIM = "configs/research/kim193_captions_unique.json"
MUSICCAPS = "artifacts/icassp_gate0/musiccaps-public.csv"
OUT = "configs/research/kim193_source_ytids.json"

# Deterministic suffix rule (released Kim format): "<MusicCaps caption> The subgenre of hip-hop is X"
SUFFIX_RE = re.compile(r"\s*The subgenre of hip-hop is\b.*$")


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def main():
    kim = json.load(open(KIM))["unique_captions"]
    rows = list(csv.DictReader(open(MUSICCAPS)))
    by_raw, by_norm = {}, {}
    for r in rows:
        by_raw.setdefault(r["caption"], []).append(r["ytid"])
        by_norm.setdefault(norm(r["caption"]), []).append(r["ytid"])

    recovered, unmatched, subgenres = {}, [], {}
    multi = []
    for c in kim:
        m = re.search(r"The subgenre of hip-hop is\s*(.+?)\s*$", c)
        sg = m.group(1) if m else "<none>"
        subgenres[sg] = subgenres.get(sg, 0) + 1
        base = SUFFIX_RE.sub("", c)
        if base in by_raw:
            yt = by_raw[base]; lvl = "exact_raw"
        elif norm(base) in by_norm:
            yt = by_norm[norm(base)]; lvl = "norm_only"
        else:
            unmatched.append({"tail": c[-70:], "base_tail": base[-70:]}); continue
        if len(yt) > 1:
            multi.append({"base_tail": base[-60:], "ytids": yt})
        recovered[c] = {"base": base, "subgenre": sg, "ytids": yt, "match_level": lvl}

    all_ytids = sorted({y for v in recovered.values() for y in v["ytids"]})
    report = {
        "n_kim_unique_captions": len(kim),
        "n_mapped": len(recovered),
        "n_unmatched": len(unmatched),
        "map_is_complete_45_of_45": len(recovered) == len(kim) and not unmatched,
        "n_recovered_source_ytids": len(all_ytids),
        "captions_with_multiple_ytid_matches": len(multi),
        "subgenre_distribution": subgenres,
        "suffix_rule": r"strip r'\s*The subgenre of hip-hop is\b.*$' then EXACT match vs MusicCaps caption",
    }
    out = {
        "name": "kim193_recovered_source_ytids",
        "report": report,
        "multi_match_examples": multi[:10],
        "unmatched": unmatched,
        "source_ytids": all_ytids,
        "per_caption": [{"kim_caption": k, **v} for k, v in recovered.items()],
    }
    if OUT and not (len(sys.argv) > 1 and sys.argv[1] == "--dry"):
        json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=2))
    if unmatched:
        print("\nUNMATCHED (investigate before freezing):")
        for u in unmatched:
            print("  ", u)
        return 1
    print("\nfrozen source_ytids ->", OUT, f"({len(all_ytids)} ytids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
