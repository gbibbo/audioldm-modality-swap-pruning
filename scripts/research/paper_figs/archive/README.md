# Archived scripts (discarded attempt, NOT applied to the manuscript)

`integrate_draft6_tighten.py`, `integrate_draft6_fit.py`, `integrate_draft6_fit_b.py` compressed the
Draft-5 prose to make room for the Draft-6 figure and the longer abstract/introduction. That compression
turned out to be **unnecessary**: the local page check was compiling with tectonic, which falls back from
Times (`ptm`) to the wider Latin Modern and inflated the page count by about 0.6 of a column. See the
2026-09-04 CORRECTION section of `icassp/MANUSCRIPT_NOTES.md`.

The manuscript was rebuilt from the Draft-5 text with only `../integrate_draft6_layout.py` applied, so
none of these three is applied to `icassp/icassp_operating_point.tex`. They are kept only as the record of
the discarded attempt. Do not run them.

For any page-budget decision use `../pagecheck_times.py`, not the plain tectonic build.
