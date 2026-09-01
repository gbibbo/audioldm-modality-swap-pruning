#!/usr/bin/env python3
"""Stage the gh-pages deployment tree for the listening study (v1.2). CPU only.

Consumes the self-contained pages from build_listening_deploy.py and lays them out under
OPAQUE random slugs so directory/file names never expose participant codes. Adds a
"Not found." root index, robots.txt, and .nojekyll. Writes a LOCAL-ONLY code->slug map
(gitignored, never committed). Verifies each staged page (live endpoint present, robots
noindex, embedded manifest, frozen study/protocol, audio embedded, no identity leak, no
recipient email, no unblinding-key values).

Outputs: artifacts/gh_pages_stage/  (staging; pushed to the orphan gh-pages branch by the caller)
         configs/research/listening_deploy_slugs.local.json  (LOCAL ONLY, gitignored)

Run: .venv-loudness/bin/python scripts/research/deploy_gh_pages.py
"""
import json, os, re, secrets, shutil, hashlib

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
SRC = os.path.join(ROOT, "artifacts/listening_study_deploy")
STAGE = os.path.join(ROOT, "artifacts/gh_pages_stage")
SLUGMAP = os.path.join(ROOT, "configs/research/listening_deploy_slugs.local.json")
PRIV = os.path.join(ROOT, "configs/research/listening_study_assignments_private.json")
STUDY = "LSTUDY-2026-08-31-v1.2"
PROTO = "fd53a5babda774efc8186a2601e2366f8e59f7ad8739ae5648a4f328a075719d"
FORBIDDEN = re.compile(r"recovered|pruned|\bdense\b|reconstr|_alt10s|realref|\bsev1\b|\bsev2\b|severity|bridge2|catch_kind|\"type\"|p1_pruned|p1_recovered", re.I)
B64 = re.compile(r"data:audio/flac;base64,[A-Za-z0-9+/=]+")
ROBOTS_META = '<meta name="robots" content="noindex,nofollow,noarchive">'

ROOT_INDEX = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
              + ROBOTS_META + "<title>Not found</title></head><body><p>Not found.</p></body></html>\n")
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"


def main():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "s"))
    open(os.path.join(STAGE, ".nojekyll"), "w").write("")
    open(os.path.join(STAGE, "robots.txt"), "w").write(ROBOTS_TXT)
    open(os.path.join(STAGE, "index.html"), "w").write(ROOT_INDEX)

    # unblinding-key values that must NOT be reconstructable from deployed content
    priv = json.load(open(PRIV))
    salt = priv.get("salt", "")

    slugmap = {}
    problems = []
    for c in ["P0%d" % i for i in range(1, 7)]:
        html = open(os.path.join(SRC, c + ".html")).read()
        # upgrade robots meta to include noarchive
        html = re.sub(r'<meta name="robots"[^>]*>', ROBOTS_META, html, count=1)
        if ROBOTS_META not in html:
            html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8">' + ROBOTS_META, 1)
        slug = secrets.token_hex(16)   # 128-bit
        d = os.path.join(STAGE, "s", slug)
        os.makedirs(d)
        open(os.path.join(d, "index.html"), "w").write(html)
        url = "https://gbibbo.github.io/audioldm-modality-swap-pruning/s/%s/" % slug
        slugmap[c] = {"slug": slug, "path": "s/%s/index.html" % slug, "url": url}

        # ---- per-page verification ----
        if "macros/s/AKfycby" not in html:
            problems.append(f"{c}: live endpoint missing")
        if ROBOTS_META not in html:
            problems.append(f"{c}: robots meta missing")
        if STUDY not in html:
            problems.append(f"{c}: study_version missing")
        if PROTO not in html:
            problems.append(f"{c}: protocol hash missing")
        if "data:audio/flac" not in html:
            problems.append(f"{c}: embedded audio missing")
        if salt and salt in html:
            problems.append(f"{c}: LEAK unblinding salt present")
        # strip the base64 audio blobs BEFORE any text scan (random base64 causes false hits)
        scan = B64.sub("data:audio/flac;base64,<omitted>", html)
        if re.search(r'[\w.-]+@[\w.-]+\.\w{2,}', scan):
            problems.append(f"{c}: possible email address present")
        # leak scan outside caption text (prompt_text values are natural English)
        try:
            man = json.loads(re.search(r'window\.EMBEDDED_MANIFEST=(\{.*?\});\n', html, re.S).group(1))
            for t in man["trials"]:
                scan = scan.replace(json.dumps(t["prompt_text"])[1:-1], "")
        except Exception as e:
            problems.append(f"{c}: manifest parse for leak-scan failed ({e})")
        leaks = sorted(set(m.group(0).lower() for m in FORBIDDEN.finditer(scan)))
        if leaks:
            problems.append(f"{c}: identity leak {leaks}")
        print(f"{c}: slug {slug[:8]}… | {os.path.getsize(os.path.join(d,'index.html'))/1e6:.2f} MB | leaks {leaks}")

    json.dump({"artifact": "listening_deploy_slugs.local", "WARNING": "LOCAL ONLY — do not commit",
               "base": "https://gbibbo.github.io/audioldm-modality-swap-pruning/",
               "map": slugmap}, open(SLUGMAP, "w"), indent=2, sort_keys=True)

    # staged tree listing + total size
    total = 0
    for dp, _, fs in os.walk(STAGE):
        for f in fs:
            total += os.path.getsize(os.path.join(dp, f))
    print("--- staged tree ---")
    for dp, dns, fs in os.walk(STAGE):
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(dp, f), STAGE)
            print("  ", rel)
    print(f"total staged: {total/1e6:.1f} MB | problems: {len(problems)}")
    for p in problems:
        print("  PROBLEM", p)
    print("slug map (LOCAL ONLY):", SLUGMAP)
    if problems:
        raise SystemExit("staging FAILED — fix problems before pushing gh-pages")


if __name__ == "__main__":
    main()
