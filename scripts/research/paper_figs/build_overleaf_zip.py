#!/usr/bin/env python3
"""Build the ready-to-upload Overleaf bundle with the version in its FILE NAME (CPU, 0 cr).

Before this script the bundle was always called `icassp_operating_point_overleaf.zip`, so two
downloads of different drafts were indistinguishable once they left the repo. The name now carries
the draft label, the build date and the short commit the sources came from:

    icassp/icassp_operating_point_draft6_2026-09-05_0a4c01f.zip
    icassp/icassp_operating_point_draft6_2026-09-05_0a4c01f-dirty.zip   (uncommitted changes)

The same stamp goes into a `VERSION.txt` inside the zip, so the bundle also identifies itself once
it is open in Overleaf. Older bundles are left alone; pass --prune to delete them.

The draft label is read from the last `%% draft<N>-...` marker in the manuscript, so it follows the
manuscript instead of being hard-coded.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/build_overleaf_zip.py
"""
import argparse
import datetime
import glob
import os
import re
import shutil
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ICASSP = os.path.join(ROOT, "icassp")
TEX = "icassp_operating_point.tex"
MEMBERS = [TEX, "spconf.sty", "IEEEbib.bst", "README_OVERLEAF.md", "figs/fig1_operating_points.pdf"]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def draft_label(tex_text):
    """Last `%% draft<N>-<something>` marker in the preamble, e.g. `%% draft6-layout` -> `draft6`."""
    labels = re.findall(r"^%%\s*(draft\d+)[-\w]*\s*$", tex_text, flags=re.M)
    return labels[-1] if labels else "draft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true",
                    help="delete older icassp_operating_point_draft*.{zip,pdf} bundles")
    args = ap.parse_args()


    tex_text = open(os.path.join(ICASSP, TEX), encoding="utf-8").read()
    label = draft_label(tex_text)
    date = datetime.date.today().isoformat()
    sha = git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(git("status", "--porcelain"))
    stamp = f"{label}_{date}_{sha}{'-dirty' if dirty else ''}"
    name = f"icassp_operating_point_{stamp}.zip"
    out = os.path.join(ICASSP, name)

    version = (
        f"ICASSP manuscript bundle\n"
        f"  draft      : {label}\n"
        f"  built      : {date}\n"
        f"  git commit : {sha}{' (WITH UNCOMMITTED CHANGES)' if dirty else ''}\n"
        f"  main file  : {TEX}   (Overleaf: Compiler -> pdfLaTeX)\n"
        f"  figure     : figs/fig1_operating_points.pdf (the only one embedded)\n"
        f"\nSee README_OVERLEAF.md in this bundle. Page-budget checks must use\n"
        f"scripts/research/paper_figs/pagecheck_times.py in the repo, not a plain tectonic build.\n"
    )

    if args.prune:
        for old in (glob.glob(os.path.join(ICASSP, "icassp_operating_point_draft*.zip"))
                    + glob.glob(os.path.join(ICASSP, "icassp_operating_point_draft*.pdf"))):
            if os.path.abspath(old) != os.path.abspath(out):
                os.remove(old)
                print("removed older bundle:", os.path.basename(old))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for m in MEMBERS:
            src = os.path.join(ICASSP, m)
            assert os.path.exists(src), f"missing bundle member: {m}"
            z.write(src, m)
        z.writestr("VERSION.txt", version)

    # a stamped copy of the local preview, so a downloaded PDF is identifiable too
    preview = os.path.join(ICASSP, "icassp_operating_point.pdf")
    stamped_pdf = os.path.join(ICASSP, f"icassp_operating_point_{stamp}.pdf")
    if os.path.exists(preview):
        shutil.copy(preview, stamped_pdf)
        print(f"wrote icassp/{os.path.basename(stamped_pdf)} (copy of the local Times-metric preview)")

    print(f"wrote icassp/{name}")
    for i in zipfile.ZipFile(out).infolist():
        print(f"  {i.file_size:9d}  {i.filename}")
    if dirty:
        print("NOTE: working tree is dirty; the bundle is stamped -dirty and is not reproducible from the commit.")


if __name__ == "__main__":
    main()
