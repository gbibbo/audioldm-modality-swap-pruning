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

Two renders, same data, both built by `scripts/research/paper_figs/make_website_card_fig.py`
(CPU, 0 cr) and both 16:10 for the site's `.selected-media` box
(`aspect-ratio: 16/10`, `object-fit: contain`, `clamp(300px, 38%, 370px)` wide).
The SVGs have text converted to paths, so they need no web font.

| File | What it adds |
|---|---|
| `docs/figs/audioldm_recovery_operating_point.{png,svg}` | One series: the paired recovery gain over the duration sweep. |
| `docs/figs/audioldm_recovery_operating_point_domain.{png,svg}` | Same, plus the flat out-of-domain hip-hop series (recommended). |

Design: a single panel, `y = R = CLAP(P+FT) − CLAP(P)` for the 83%-pruned
checkpoint, `x` = requested duration, four points on a clearly rising line, the
95% prompt-bootstrap interval as a recessive band, a dotted guide at the
fine-tuning duration, and the headline *"Recovery depends on the operating point"*
inside the plot. Absolute CLAP levels, the dense and real anchors, the chance
floors and the ρ_dense percentages are deliberately omitted — correct for the
paper, too much work for a portfolio preview.

Every plotted value is read from frozen artifacts, never typed by hand:
`configs/research/draft5_opsweep_result.json` (`artifact_sha256 = 3c4b8147db943f15…`)
for `R_by_duration` (point, lo, hi, n = 192), and
`configs/research/r2_posthoc_pooled_anchors.json` for the pooled hip-hop gains
(n = 127).

### Two accuracy notes on the annotations

* **"fine-tuning duration" at 10.24 s is correct** (the released recovery was
  fine-tuned at the native duration), but the guide must not be allowed to read as
  a *peak*: E1c gives `D4 = R(15.36) − R(10.24) = +0.021 [−0.023, +0.067]`,
  reading PLATEAU, and the ledger states plainly that "peaked at the fine-tuning
  duration" is not supported. The guide is drawn inside the axis, not at the plot
  edge, and nothing in the figure claims a maximum.
* **"held-out domain: recovery largely disappears" would be wrong.** Clotho is
  held out and still gives `R = +0.210 [+0.176, +0.243]` at 10.24 s, close to
  AudioCaps; the ledger records that "recovery transfers to held-out Clotho, so the
  hip-hop null is a content/style artefact, not 'recovers only on AudioCaps'".
  What collapses is the *music* battery specifically: hip-hop gives `+0.026` at
  3.84 s and `+0.027` at 10.24 s. The variant therefore labels that series
  "hip-hop prompts (n = 127)" and claims nothing about held-out domains in general.
