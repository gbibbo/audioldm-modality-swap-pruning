#!/usr/bin/env python3
"""Draft-6 page fit, last wave (CPU, 0 cr, idempotent). Wording and table spacing only.

Same rule as the two earlier passes: no number, claim, caveat or citation is removed.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_draft6_fit_b.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
MARK = "%% draft6-fit-b"

tex = open(TEX, encoding="utf-8").read()
if MARK in tex:
    print("already applied; nothing to do")
    sys.exit(0)
n = 0


def rep(old, new, count=1):
    global tex, n
    assert tex.count(old) == count, (tex.count(old), old[:90])
    tex = tex.replace(old, new)
    n += 1


rep(r"\renewcommand{\arraystretch}{0.98}", r"\renewcommand{\arraystretch}{0.95}", 2)

rep(r"""\item \textbf{Duration.} The recovery gain is several times larger at the fine-tuning duration
      ($10.24$\,s) than at $3.84$\,s, at both severities, and grows monotonically over four durations
      at severity~2. The interaction was pre-specified, replicates on a disjoint prompt set, and is not
      a scale or scorer effect (the chance floor moves by at most $0.025$ between durations).
\item \textbf{Domain.} At matched duration the gain is present in-domain (AudioCaps) and absent or
      negative on held-out hip-hop/rap captions---at both severities at $3.84$\,s and at both durations
      at severity~2---where alignment after fine-tuning sits $0.02$--$0.03$ above the chance floor,
      against $0.32$ in-domain.
\item \textbf{Where the gain lives.} Frame-level grounding shows the native gain spread uniformly over
      the clip; a crop analysis shows the short-duration deficit comes from \emph{generating} short clips,
      not from scoring short excerpts.""",
    r"""\item \textbf{Duration.} The gain is several times larger at the fine-tuning duration ($10.24$\,s) than
      at $3.84$\,s, at both severities, and grows monotonically over four durations at severity~2. It was
      pre-specified, replicates on a disjoint prompt set, and is no scale or scorer effect.
\item \textbf{Domain.} At matched duration the gain is present in-domain (AudioCaps) and absent or
      negative on held-out hip-hop/rap captions (both severities at $3.84$\,s, both durations at
      severity~2), where alignment after fine-tuning sits $0.02$--$0.03$ above the chance floor against
      $0.32$ in-domain.
\item \textbf{Where it lives.} Frame-level grounding finds the native gain spread uniformly over the
      clip; a crop analysis traces the short-duration deficit to \emph{generating} short clips, not to
      scoring short excerpts.""")

rep(r"""so the large native gain too is confined to the fine-tuning domain
(domain contrast $+0.239$; music duration interaction $-0.004$).""",
    r"""so the large native gain too stays in-domain (domain
contrast $+0.239$; music duration interaction $-0.004$).""")

rep(r"""At the published sampler recipe (DDIM $200$, guidance $3.5$; single generation) on the first
$64$ prompts the interaction holds,""",
    r"""At the published sampler recipe (DDIM $200$, guidance $3.5$) on the first $64$ prompts the interaction
holds,""")

rep(r"""Off the primary scorer (post-hoc), Human-CLAP, KL and PANNs capture all show $R$ rising at every step of the
sweep, the first two steps resolved and the last ($7.68\!\to\!10.24$\,s) not:""",
    r"""Off the primary scorer (post-hoc), Human-CLAP, KL and PANNs capture all show $R$ rising at every sweep
step, the first two resolved and the last ($7.68\!\to\!10.24$\,s) not:""")

tex = tex.replace("%% draft6-fit\n", "%% draft6-fit\n" + MARK + "\n", 1)
open(TEX, "w", encoding="utf-8").write(tex)
print(f"draft6 fit pass (b) applied ({n} edits)")
