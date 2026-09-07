#!/usr/bin/env python3
"""Build the Draft-16 Overleaf bundle: FOUR files, no subdirectories (CPU, 0 cr).

WHY FLAT. The first Draft-16 bundle shipped `icassp_operating_point.tex` plus `sections/draft16_*.tex`.
Overleaf chooses the main file by looking for `\\documentclass`, found it in `sections/draft16_1.tex`,
compiled that section alone and aborted with `(job aborted, no legal \\end found)` -- the matching
`\\end{document}` was in `draft16_4.tex`. A single .tex removes the ambiguity by construction, so the
manuscript is now one flat file and the repository layout mirrors the bundle exactly.

    icassp_operating_point.tex   the whole manuscript
    spconf.sty                   the ICASSP class file (not on CTAN, must ship)
    fig1a_duration.pdf           Figure 1(a)
    fig1b_intervention.pdf       Figure 1(b)

`IEEEbib.bst` is deliberately NOT shipped: Draft 16 has no `\\bibliography{}`/`\\bibliographystyle`,
its references are an inline `thebibliography`, so BibTeX never runs (asserted below). The companion
markdown (`PAPER_COMPANION.md`, `PAPER_EXPANDED_RESULTS.md`, ...) is not shipped either; it lives in
the repository and is not needed to compile.

Before writing the zip, the bundle is unpacked into a scratch directory and compiled there with
Times-compatible metrics (see `pagecheck_times.py`), so the zip is never published unless exactly
those four files build on their own.

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
OUT = os.path.join(ICASSP, "icassp_operating_point_draft16_camera_ready.zip")
MEMBERS = ["icassp_operating_point.tex", "spconf.sty",
           "fig1a_duration.pdf", "fig1b_intervention.pdf"]


def check_sources():
    tex = open(os.path.join(ICASSP, "icassp_operating_point.tex"), encoding="utf-8").read()
    assert tex.count("\\documentclass") == 1, "expected exactly one \\documentclass"
    assert tex.count("\\begin{document}") == 1 and tex.count("\\end{document}") == 1, \
        "the manuscript must be one complete document"
    assert "\\input{" not in tex, "the manuscript must be flat: no \\input of section files"
    assert "\\bibliographystyle" not in tex and "\\bibliography{" not in tex, \
        "Draft 16 gained a BibTeX bibliography; IEEEbib.bst must then be shipped again"
    assert "{figs/" not in tex, "graphics paths must be flat (no figs/ directory in the bundle)"
    for name in MEMBERS:
        assert os.path.exists(os.path.join(ICASSP, name)), f"missing bundle member: {name}"


def compile_bundle(tmp):
    """Compile the unpacked bundle in place; returns the PDF path."""
    src = open(os.path.join(tmp, "icassp_operating_point.tex"), encoding="utf-8").read()
    assert PC.ANCHOR_RE.search(src), "package line not found; update pagecheck_times.ANCHOR_RE"
    open(os.path.join(tmp, "check.tex"), "w", encoding="utf-8").write(
        PC.ANCHOR_RE.sub(lambda m: m.group(0) + "\n" + PC.PATCH, src, count=1))
    r = subprocess.run([PC.TECTONIC, "-X", "compile", "check.tex", "--outdir", "."],
                       cwd=tmp, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2500:])
        sys.exit(1)
    return os.path.join(tmp, "check.pdf")


def main():
    check_sources()

    tmp = tempfile.mkdtemp(prefix="draft16_bundle_")
    for name in MEMBERS:
        shutil.copy(os.path.join(ICASSP, name), tmp)
    pdf = compile_bundle(tmp)

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf)
    pages = len(doc)
    page_of = lambda s: next((i + 1 for i in range(pages)
                              if s in doc[i].get_textpage().get_text_range()), None)
    fig_page, ref_page = page_of("Duration dependence and the symmetric"), page_of("REFERENCES")
    shutil.copy(pdf, os.path.join(ICASSP, "icassp_operating_point.pdf"))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in MEMBERS:
            z.write(os.path.join(ICASSP, name), name)
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"bundle compiles standalone: {pages} pages, Fig. 1 on page {fig_page}, "
          f"REFERENCES on page {ref_page}")
    with zipfile.ZipFile(OUT) as z:
        names = sorted(z.namelist())
        assert not any("/" in n for n in names), "the bundle must have no subdirectories"
        print(f"{OUT}  ({os.path.getsize(OUT)} bytes, {len(names)} files, no subdirectories)")
        for n in names:
            print(f"  {z.getinfo(n).file_size:>9}  {n}")


if __name__ == "__main__":
    main()
