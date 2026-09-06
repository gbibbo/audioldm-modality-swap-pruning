# Project card for gbibbo.github.io

Copy for the "Selected work" card of this project, matched to the site's existing
format (`article.selected-item`: `meta-line` → `h3` → `p.project-description` →
`p.tag-line` → `link-row`). House length: 2–3 sentences, 44–52 words.

## Card fields

| Field | Text |
|---|---|
| `meta-line` (left) | Text-to-audio · Model compression · Evaluation |
| `meta-line` (right) | 2026 |
| `h3` (title) | Recovery Gain in Pruned Text-to-Audio Diffusion |
| `tag-line` (5 keywords) | AudioLDM · Diffusion · Structured pruning · CLAP · PyTorch |
| image | `homepage_files/audioldm_recovery_operating_point.svg` (or `.png`) |
| link | https://github.com/gbibbo/audioldm-modality-swap-pruning |

### `project-description` (56 words, 3 sentences)

> This research measures how much of a pruned text-to-audio model's quality comes
> from the recovery fine-tuning applied after pruning. It re-evaluates released
> AudioLDM checkpoints at 65% and 83% pruning with paired generations across
> duration and prompt domain. Recovery closes 44% of the gap to the dense model at
> 3.84 s but 82% at 10.24 s.

Optional longer third sentence, if the falsification is worth the extra words:

> Recovery closes 44% of the gap to the dense model at 3.84 s but 82% at 10.24 s,
> and a matched 20,000-step intervention shows the fine-tuning duration does not
> explain the difference.

### Alternatives

* Title: *Operating-Point Evaluation of Pruned Audio Diffusion*.
* Keywords: swap `PyTorch` → `Bootstrap CI` to foreground the statistical protocol.

## Figure

`docs/figs/audioldm_recovery_operating_point.{png,svg}`, built by
`scripts/research/paper_figs/make_website_card_fig.py` (CPU, 0 cr).

* 16:10, matching the site's `.selected-media` box (`aspect-ratio: 16/10`,
  `object-fit: contain`, `clamp(300px, 38%, 370px)` wide). Type is sized to stay
  legible at 370 px; the SVG has text converted to paths, so it needs no web font.
* Every plotted value is read from the frozen artifact
  `configs/research/draft5_opsweep_result.json`
  (`artifact_sha256 = 3c4b8147db943f15…`), never typed by hand:
  `secondary.by_duration[d].levels` for the three curves, `.R.point` for the
  annotated gains, `.rho_dense.point` for the percentages.
* What it shows: absolute CLAP for the dense reference, the 83%-pruned checkpoint,
  and the same checkpoint after the released recovery fine-tuning, over the
  requested-duration sweep. The shaded band is the paired recovery gain, annotated
  at both endpoints (+0.085 / 44% of the dense gap at 3.84 s; +0.244 / 82% at
  10.24 s) — i.e. the headline of the manuscript, that recovery gain is
  operating-point dependent.
