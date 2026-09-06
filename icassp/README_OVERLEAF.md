# ICASSP 2027 Overleaf package

Main file is `icassp_operating_point.tex`.

This is Draft 14. It uses the ICASSP `spconf.sty` layout, 9 pt body text, two columns and a references-only fifth page.

## Figure placeholders

The manuscript intentionally does not include generated graphs in this version. Figure 1 reserves the intended final space with two empty boxes. Detailed TeX comments immediately above the figure in `sections/draft14_2.tex` specify the required plots, data relationships, uncertainty and annotations.

The final plotting step should replace each box with the generated vector figure while preserving the same approximate panel footprint of 0.49 textwidth by 4.80 cm. Do not delete the specification comments until the final artwork has been independently checked against the committed result artifacts.

## Compile

Upload `icassp_operating_point.tex`, `spconf.sty`, `IEEEbib.bst` and the `sections/` directory to Overleaf. The current placeholder draft requires no external figure file.
