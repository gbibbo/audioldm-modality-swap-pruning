# icassp/archive — superseded manuscript versions

Draft 13 (`icassp/icassp_operating_point.tex` → `icassp/sections/draft13_*.tex`, marker `%% draft13-reviewer-followup`)
is the current version since 2026-09-05 17:4x MVD (Draft 12 was current from 2026-09-05 00:56 to 17:4x). Everything here is a previous version kept for reference; none of it is embedded or built.

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

Drafts 7–11 were written outside this repository (Overleaf / delivery bundles) and never committed here;
Draft 12 arrived as a delivery bundle (now in this folder); its notes are in `draft12_delivery_notes/` here. Number
provenance for Draft 12: `scripts/research/paper_figs/verify_draft12_numbers.py` (run it against the archived .tex).
Draft 13 arrived as `icassp/icassp_operating_point_draft13_reviewer_followup.zip` (gitignored); its verifier is
`verify_draft13_numbers.py`.
