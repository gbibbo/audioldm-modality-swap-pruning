#!/usr/bin/env python3
"""Rebuild the Draft-16 Overleaf delivery bundle from the repository sources (CPU, 0 cr).

The bundle keeps exactly the member layout Gabriel delivered in
`icassp_operating_point_draft16_camera_ready.zip`; this script only re-assembles it from the tracked
sources so that the zip can never drift from the repository, and adds the two generated Figure-1
panels. It also refreshes the preview PDF using the same Times-metric proxy as `pagecheck_times.py`
(fontspec + Liberation Serif, metric-compatible with the Nimbus Roman that Overleaf's pdfLaTeX
actually uses) -- a local `tectonic` build without that patch silently falls back to the wider Latin
Modern and misrepresents the page budget.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/build_draft16_zip.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pagecheck_times as PC  # noqa: E402  (reuse the validated Times-metric compile settings)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ICASSP = os.path.join(ROOT, "icassp")
TOP = "icassp_operating_point_draft16_camera_ready_delivery"
OUT = os.path.join(ICASSP, "icassp_operating_point_draft16_camera_ready.zip")

# zip member -> repository source
MEMBERS = {
    "icassp_operating_point.tex": "icassp/icassp_operating_point.tex",
    "sections/draft16_1.tex": "icassp/sections/draft16_1.tex",
    "sections/draft16_2.tex": "icassp/sections/draft16_2.tex",
    "sections/draft16_3.tex": "icassp/sections/draft16_3.tex",
    "sections/draft16_4.tex": "icassp/sections/draft16_4.tex",
    "figs/fig1a_duration.pdf": "icassp/figs/fig1a_duration.pdf",
    "figs/fig1b_intervention.pdf": "icassp/figs/fig1b_intervention.pdf",
    "spconf.sty": "icassp/spconf.sty",
    "IEEEbib.bst": "icassp/IEEEbib.bst",
    "README.md": "icassp/draft16_delivery_notes/DELIVERY_README.md",
    "README_OVERLEAF.md": "icassp/README_OVERLEAF.md",
    "PAPER_COMPANION.md": "icassp/PAPER_COMPANION.md",
    "PAPER_EXPANDED_RESULTS.md": "icassp/PAPER_EXPANDED_RESULTS.md",
    # `.gitignore` inherits an upstream `*.txt` rule, so the source is tracked as .md and
    # emitted into the bundle under the name Gabriel delivered.
    "VERSION.txt": "icassp/draft16_delivery_notes/VERSION.md",
    "docs/fourth_review_response_manuscript.md": "docs/fourth_review_response_manuscript.md",
    "paper-operating-point-recovery/README.md": "icassp/paper-operating-point-recovery/README.md",
}


def build_preview_pdf(dest):
    """Compile the manuscript with Times-compatible metrics and write the PDF to `dest`."""
    tmp = tempfile.mkdtemp(prefix="draft16_preview_")
    for f in ("spconf.sty", "IEEEbib.bst"):
        shutil.copy(os.path.join(ICASSP, f), tmp)
    shutil.copytree(os.path.join(ICASSP, "figs"), os.path.join(tmp, "figs"))
    shutil.copytree(os.path.join(ICASSP, "sections"), os.path.join(tmp, "sections"))
    shutil.copy(os.path.join(ICASSP, "icassp_operating_point.tex"),
                os.path.join(tmp, "icassp_operating_point.tex"))
    patched = False
    for rel in ["icassp_operating_point.tex"] + [f"sections/{f}" for f in sorted(os.listdir(os.path.join(tmp, "sections")))]:
        fp = os.path.join(tmp, rel)
        src = open(fp, encoding="utf-8").read()
        if PC.ANCHOR_RE.search(src):
            open(fp, "w", encoding="utf-8").write(
                PC.ANCHOR_RE.sub(lambda m: m.group(0) + "\n" + PC.PATCH, src, count=1))
            patched = True
            break
    assert patched, "package line not found; update pagecheck_times.ANCHOR_RE"
    r = subprocess.run([PC.TECTONIC, "-X", "compile", "icassp_operating_point.tex", "--outdir", "."],
                       cwd=tmp, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:])
        sys.exit(1)
    shutil.copy(os.path.join(tmp, "icassp_operating_point.pdf"), dest)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    for src in MEMBERS.values():
        p = os.path.join(ROOT, src)
        assert os.path.exists(p), f"missing source: {src}"

    tmp = tempfile.mkdtemp(prefix="draft16_zip_")
    pdf = os.path.join(tmp, "icassp_operating_point.pdf")
    build_preview_pdf(pdf)
    shutil.copy(pdf, os.path.join(ICASSP, "icassp_operating_point.pdf"))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for member, src in MEMBERS.items():
            z.write(os.path.join(ROOT, src), f"{TOP}/{member}")
        z.write(pdf, f"{TOP}/icassp_operating_point.pdf")
    shutil.rmtree(tmp, ignore_errors=True)

    with zipfile.ZipFile(OUT) as z:
        names = sorted(z.namelist())
        print(f"{OUT}  ({os.path.getsize(OUT)} bytes, {len(names)} members)")
        for n in names:
            print(f"  {z.getinfo(n).file_size:>9}  {n}")


if __name__ == "__main__":
    main()
