#!/usr/bin/env python3
"""Page-length check with the TIMES metrics Overleaf actually uses (CPU, 0 cr).

WHY THIS EXISTS. `spconf.sty` sets `\renewcommand{\rmdefault}{ptm}` (Times). Overleaf compiles with
pdfLaTeX and gets URW NimbusRomNo9L, a Times clone. The local `tectonic` build runs XeTeX, finds no
`TUptm.fd`, and silently falls back to **Latin Modern**, which is wider: on Draft 5 that pushed the
references a full 0.6 column further down, so the local build looked "full to the last line" while
Gabriel's Overleaf build had ~half a page free on page 4. Judging the page budget from the plain
tectonic build therefore over-compresses the manuscript.

This script compiles a COPY of the manuscript with `fontspec` + Liberation Serif (metric-compatible
with Times New Roman / Nimbus Roman) and reports where the body ends. Validated against Gabriel's
Overleaf PDF of Draft 5: references start in the page-4 right column at y=506 pt here vs 475 pt there,
i.e. this proxy is optimistic by about 3 lines -- keep at least that much margin.

The manuscript itself is never modified.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/pagecheck_times.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pypdfium2 as pdfium

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ICASSP = os.path.join(ROOT, "icassp")
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")
if not os.path.exists(TECTONIC):
    TECTONIC = shutil.which("tectonic") or TECTONIC
PATCH = ("\\usepackage{fontspec}\n\\setmainfont{Liberation Serif}\n\\setmonofont{DejaVu Sans Mono}")
ANCHOR = "\\usepackage{spconf,amsmath,amssymb,graphicx,booktabs,array,multirow,url,cite}"
TEXT_BOTTOM_PT = 72.0   # spconf text block: 229 mm tall, 1 in top margin on US Letter


def main():
    tmp = tempfile.mkdtemp(prefix="pagecheck_times_")
    for f in ("spconf.sty", "IEEEbib.bst"):
        shutil.copy(os.path.join(ICASSP, f), tmp)
    shutil.copytree(os.path.join(ICASSP, "figs"), os.path.join(tmp, "figs"))
    tex = open(os.path.join(ICASSP, "icassp_operating_point.tex"), encoding="utf-8").read()
    assert ANCHOR in tex, "package line not found; update ANCHOR"
    open(os.path.join(tmp, "check.tex"), "w", encoding="utf-8").write(tex.replace(ANCHOR, ANCHOR + "\n" + PATCH, 1))
    r = subprocess.run([TECTONIC, "-X", "compile", "check.tex", "--outdir", "."], cwd=tmp,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:])
        sys.exit(1)
    doc = pdfium.PdfDocument(os.path.join(tmp, "check.pdf"))
    print(f"pages: {len(doc)}")
    ok = False
    for i in range(len(doc)):
        tp = doc[i].get_textpage()
        k = tp.get_text_range().find("REFERENCES")
        if k >= 0:
            x, y = tp.get_charbox(k)[0], tp.get_charbox(k)[1]
            col = "left" if x < 306 else "right"
            print(f"body ends: page {i+1}, {col} column, references heading at y={y:.0f} pt")
            ok = i + 1 <= 4 or (i + 1 == 5 and k <= 4)
            break
    else:
        print("REFERENCES heading not found")
    # the hard rule: page 5 may carry references only, so no body text may precede the
    # references heading on page 5 (if the heading is already on page 4, nothing can).
    if len(doc) >= 5:
        t5 = doc[4].get_textpage().get_text_range()
        k5 = t5.find("REFERENCES")
        n = len(t5[:k5].strip()) if k5 >= 0 else 0   # heading on page 4 -> page 5 is references only
        print(f"body text on page 5 before the references: {n} chars "
              f"({'OK' if n <= 4 else 'OVER THE LIMIT'})")
        ok = ok and n <= 4
    print("VERDICT:", "fits 4 content pages (Times metrics)" if ok else "DOES NOT FIT")
    print(f"(build kept at {tmp})")


if __name__ == "__main__":
    main()
