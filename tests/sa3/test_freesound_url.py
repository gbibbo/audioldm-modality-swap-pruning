#!/usr/bin/env python3
"""Pure CPU test for the Freesound search URL/filter construction (rc1.2 semantic repair).
No network. Run: .venv-sa3/bin/python tests/sa3/test_freesound_url.py

Guards the exact bug that made the first authenticated dry-list return 0: the OR-tag selection must
live in `filter=tag:(...)`, NOT in `query` (Freesound treats query terms as mandatory-AND)."""
import os, sys, urllib.parse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "scripts", "sa3"))
import fetch_freesound_domain as FS


def t1_filter_string_exact():
    f = FS.build_filter("impact_percussion")
    ok = f == 'license:"Creative Commons 0" tag:(impact OR hit OR percussion OR clap OR knock)'
    print(f"    T1 filter={f!r}")
    return ok


def t2_query_empty_or_in_filter_not_query():
    p = FS.build_search_params("water_liquid")
    ok = p["query"] == ""                                   # deterministic tag-based selection
    ok = ok and " OR " in p["filter"] and "tag:(" in p["filter"]
    ok = ok and " OR " not in p["query"]                    # THE bug: OR must NOT be in query
    ok = ok and p["sort"] == "downloads_desc"
    ok = ok and 'license:"Creative Commons 0"' in p["filter"]
    print(f"    T2 query={p['query']!r} filter has tag()+OR, sort={p['sort']}")
    return ok


def t3_endpoint_path():
    u = FS.search_url("impact_percussion", page_size=150)
    ok = "/apiv2/search/?" in u and "/search/text/" not in u
    path, _, qs = u.partition("?")
    q = urllib.parse.parse_qs(qs, keep_blank_values=True)
    # decode round-trips: filter carries license+tags, query is blank, sort/page_size correct
    ok = ok and q.get("filter", [""])[0] == 'license:"Creative Commons 0" tag:(impact OR hit OR percussion OR clap OR knock)'
    ok = ok and q.get("query", ["x"])[0] == "" and q.get("sort", [""])[0] == "downloads_desc"
    ok = ok and q.get("page_size", [""])[0] == "150"
    print(f"    T3 path={path[-20:]} filter/query/sort decode OK")
    return ok


def t4_all_domains_build():
    ok = True
    for d in FS.DOMAINS:
        p = FS.build_search_params(d)
        ok = ok and p["query"] == "" and p["filter"].startswith('license:"Creative Commons 0" tag:(')
    print(f"    T4 all {len(FS.DOMAINS)} domains build filter-based params")
    return ok


def main():
    checks = [("T1", t1_filter_string_exact), ("T2", t2_query_empty_or_in_filter_not_query),
              ("T3", t3_endpoint_path), ("T4", t4_all_domains_build)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
