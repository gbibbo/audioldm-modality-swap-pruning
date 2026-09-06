# icassp/archive — superseded manuscript versions

Draft 14 (`icassp/icassp_operating_point.tex` → `icassp/sections/draft14_*.tex`, marker `%% draft14-second-review`)
is the current version since 2026-09-06 (Draft 13 was current from 2026-09-05 17:4x to 2026-09-06; Draft 12 from
2026-09-05 00:56 to 17:4x). Everything here is a previous version kept for reference; none of it is embedded or built.

| File | What it is | Reproducible from |
|---|---|---|
| `icassp_operating_point_draft6_e89235d.tex` | Draft 6 source (last version before the from-scratch rewrite) | `git show e89235d:icassp/icassp_operating_point.tex` |
| `README_OVERLEAF_draft6_e89235d.md` | Draft 6 Overleaf notes | `git show e89235d:icassp/README_OVERLEAF.md` |
| `icassp_operating_point_draft6_2026-09-05_e89235d.pdf` | Draft 6 Times-metric preview (gitignored; on disk only) | `pagecheck_times.py` at e89235d |
| `icassp_operating_point_draft6_2026-09-05_e89235d.zip` | Draft 6 Overleaf bundle (gitignored; on disk only) | `build_overleaf_zip.py` at e89235d |
| `icassp_operating_point_draft12_6b51775.tex` | Draft 12 source (from-scratch prose rewrite, delivered by Gabriel) | `git show 6b51775:icassp/icassp_operating_point.tex` |
| `README_OVERLEAF_draft12_6b51775.md` | Draft 12 Overleaf notes | `git show 6b51775:icassp/README_OVERLEAF.md` |
| `draft12_delivery_notes/` | Draft 12 VERSION / EDITORIAL_AUDIT / TEMPLATE_VERIFICATION | `git show 6b51775:icassp/draft12_delivery_notes/` |
| `icassp_operating_point_draft12_2026-09-05_6b51775.pdf` | Draft 12 PDF as committed at 6b51775 (gitignored; on disk only) | `git show 6b51775:icassp/icassp_operating_point.pdf` |
| `icassp_operating_point_draft12_from_scratch.zip` | Draft 12 delivery bundle (gitignored; on disk only) | Gabriel's delivery |
| `icassp_operating_point_draft13_f2a5d0c.tex` | Draft 13 main file (4 `\input`s; the prose is in `draft13_sections/`) | `git show f2a5d0c:icassp/icassp_operating_point.tex` |
| `draft13_sections/draft13_{1..4}.tex` | Draft 13 modular source (reviewer follow-up version, delivered by Gabriel; B1 camera-ready correction applied at f2a5d0c) | `git show f2a5d0c:icassp/sections/` |
| `README_OVERLEAF_draft13_f2a5d0c.md` | Draft 13 Overleaf notes | `git show f2a5d0c:icassp/README_OVERLEAF.md` |
| `draft13_delivery_notes/VERSION.md` | Draft 13 version note | `git show f2a5d0c:icassp/draft13_delivery_notes/VERSION.md` |
| `icassp_operating_point_draft13_2026-09-05_f2a5d0c.pdf` | Draft 13 PDF as committed at f2a5d0c (gitignored; on disk only) | `git show f2a5d0c:icassp/icassp_operating_point.pdf` |
| `icassp_operating_point_draft13_reviewer_followup.zip` | Draft 13 delivery bundle (gitignored; on disk only) | Gabriel's delivery |

Drafts 7–11 were written outside this repository (Overleaf / delivery bundles) and never committed here;
Drafts 12, 13 and 14 arrived as delivery bundles from Gabriel (the Draft 14 bundle is
`icassp/icassp_operating_point_draft14_second_review.zip`, gitignored, on disk). Number provenance: Draft 12 →
`scripts/research/paper_figs/verify_draft12_numbers.py`, Draft 13 → `verify_draft13_numbers.py` (run each against the
archived .tex), Draft 14 → `verify_draft14_numbers.py` (current). Draft 14 embeds no figure yet (two placeholder boxes
with a production specification in `sections/draft14_2.tex`); `icassp/figs/fig1_operating_points.*` is the Draft 13
figure, kept as the basis for panel (a) of the pending Draft 14 figure.
