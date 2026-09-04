#!/usr/bin/env python3
"""Draft-6 final fit to 4 content pages (CPU, 0 cr, idempotent). Wording only.

Second wave of the page-budget pass: the same rule as integrate_draft6_tighten.py -- no number, claim,
caveat or citation is removed, only words. Includes a light tightening of the two paragraphs the
Draft-6 layout pass added to the introduction and of the abstract's new opening, which are the largest
new consumers of space.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_draft6_fit.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
MARK = "%% draft6-fit"

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


# ------------------------------------------------------------------------------------ abstract
rep(r"""stage of \emph{recovery fine-tuning} on an in-domain caption corpus. What that repair is worth is
normally certified by one aggregate score, computed on the in-domain test set at one clip duration and
one sampler setting.""",
    r"""stage of \emph{recovery fine-tuning} on an in-domain caption corpus. What the repair is worth is
normally certified by one aggregate score on the in-domain test set, at one clip duration and one
sampler setting.""")
rep(r"""at one such point, so its benefit need not carry to another. We test this on the publicly released
pruned and recovery-fine-tuned AudioLDM-M checkpoints of a recent text-to-audio pruning study, at two
severities ($65\%$ and $83\%$ of U-Net parameters removed): training nothing, we measure the paired gain
that fine-tuning buys in text--audio alignment across clip duration and prompt domain, read against a
shuffled-caption chance floor and the real audio of the same prompts.""",
    r"""at one such point, so its benefit need not carry to another. We test this on the publicly released
pruned and recovery-fine-tuned AudioLDM-M checkpoints of a recent pruning study, at two severities
($65\%$ and $83\%$ of U-Net parameters removed): training nothing, we measure the paired gain
fine-tuning buys in text--audio alignment across clip duration and prompt domain, read against a
shuffled-caption chance floor and the real audio of the same prompts.""")

# --------------------------------------------------------------------------------- introduction
rep(r"""AudioLDM-M U-Net carries $416$\,M parameters and is evaluated once per sampler step for every clip it
produces~\cite{liu2023audioldm}. The standard way to shrink such a model is \emph{structured pruning},
which deletes whole channels so that parameters and compute fall together. Pruning alone costs quality,
so it is followed by \emph{recovery fine-tuning}: the pruned network is trained further, on an in-domain
caption corpus, until its scores approach those of the model it came from. The two stages together are
the established recipe for cheaper image diffusion""",
    r"""AudioLDM-M U-Net carries $416$\,M parameters and runs once per sampler step for every clip it
produces~\cite{liu2023audioldm}. The standard way to shrink it is \emph{structured pruning}, which
deletes whole channels so that parameters and compute fall together; since pruning alone costs quality,
it is followed by \emph{recovery fine-tuning}, further training of the pruned network on an in-domain
caption corpus. The two stages are the established recipe for cheaper image diffusion""")
rep(r"""an inference \emph{operating point}: how long a clip is asked for, what kind of prompt is given, how the
sampler is configured. Recovery fine-tuning is itself training at one such point, and nothing guarantees
that what it buys there is also bought elsewhere.""",
    r"""an inference \emph{operating point}: how long a clip is asked for, what prompt is given, how the sampler
is configured. Recovery fine-tuning is itself training at one such point, and nothing guarantees that
what it buys there is bought elsewhere.""")

# ------------------------------------------------------------------------------------ Sec. 2
rep(r"""Structural pruning with fine-tuning recovery is established for image diffusion
models~\cite{fang2023diffpruning,kim2024bksdm,fang2025tinyfusion} and self-supervised speech
encoders~\cite{lai2021parp,peng2023dphubert}. For TTA, Singh et al.~\cite{singh2026pruning} L1-prune the""",
    r"""The recipe is established for image diffusion
models~\cite{fang2023diffpruning,kim2024bksdm,fang2025tinyfusion} and for self-supervised speech
encoders~\cite{lai2021parp,peng2023dphubert}. For TTA, Singh et al.~\cite{singh2026pruning} L1-prune the""")

# ------------------------------------------------------------------------------------ Sec. 4
rep(r"""Fine-tuning raises CLAP alignment from $0.055$ to $0.299$ at the native
duration ($\Rn=+0.244$) but only from $0.015$ to $0.100$ at $3.84$\,s ($\Rs=+0.085$; resolved, not
absent); the pruned checkpoint gains just $s(\mathrm{P})=+0.040$ from the longer clip, the fine-tuned
one $s(\PFT)=+0.200$.""",
    r"""Fine-tuning raises alignment from $0.055$ to $0.299$ at the native duration
($\Rn=+0.244$) but only from $0.015$ to $0.100$ at $3.84$\,s ($\Rs=+0.085$; resolved, not absent); the
longer clip gains the pruned checkpoint just $s(\mathrm{P})=+0.040$, the fine-tuned one
$s(\PFT)=+0.200$.""")
rep(r"""(excess over dense $+0.043$ $\ci{-0.020}{+0.109}$). The recovery gain itself is small there (Table~\ref{tab:core}), and part of the pruned checkpoint's raw response is a
rise of its own floor ($+0.019$; the fine-tuned checkpoint's floor does not move), so on chance-corrected
scores the severity-1 interaction is resolved,""",
    r"""(excess over dense $+0.043$ $\ci{-0.020}{+0.109}$). The gain itself is small there
(Table~\ref{tab:core}), and part of the pruned checkpoint's raw response is a rise of its own floor
($+0.019$; the fine-tuned one's does not move), so on chance-corrected scores the severity-1 interaction
is resolved,""")
rep(r"""$+0.062$~\ci{+0.037}{+0.087}, $+0.043$~\ci{+0.014}{+0.073}): by the pre-specified shape rule the gain is \emph{monotone increasing}. The pruned
checkpoint is flat to $5.12$\,s ($-0.000$) while \PFT{} rises at every
step;""",
    r"""$+0.062$~\ci{+0.037}{+0.087}, $+0.043$~\ci{+0.014}{+0.073}): by the pre-specified shape rule the gain is
\emph{monotone increasing}. P is flat to $5.12$\,s ($-0.000$) while \PFT{} rises at every step;""")
rep(r"""To our ears the music null is no tie: we preferred \PFT{} on 8/8 music pairs and heard it as music on 5/8
(P on 1/8), while neither followed the long captions ($1.4$ and $2.5$ of $5$); musicality without alignment
is invisible to a text--audio scorer. The null is near chance for both checkpoints (no real hip-hop reference was available for a ceiling)
and not a caption-length effect: within AudioCaps the native gain is
uncorrelated with caption length (Spearman $\rho=+0.04$ $\ci{-0.12}{+0.18}$), and no caption is
truncated by the conditioner---though $47\%$ of the music captions exceed CLAP's $77$-token
pre-training length, a caption-style covariate inseparable from content.""",
    r"""To our ears the music null is no tie: we preferred \PFT{} on 8/8 music pairs and heard it as music on
5/8 (P on 1/8), while neither followed the long captions ($1.4$ and $2.5$ of $5$): musicality without
alignment is invisible to a text--audio scorer. The null sits near chance for both checkpoints (no real
hip-hop reference was available for a ceiling) and is no caption-length effect: within AudioCaps the
native gain is uncorrelated with caption length (Spearman $\rho=+0.04$ $\ci{-0.12}{+0.18}$), and the
conditioner truncates no caption---though $47\%$ of the music captions exceed CLAP's $77$-token
pre-training length, a caption-style covariate inseparable from content.""")
rep(r"""Human-CLAP~\cite{takano2025humanclap} (a CLAP fine-tuned on human ratings; automatic) gives $\Rn=+0.375$""",
    r"""Human-CLAP~\cite{takano2025humanclap} (a CLAP fine-tuned on human ratings) gives $\Rn=+0.375$""")

tex = tex.replace("%% draft6-tighten\n", "%% draft6-tighten\n" + MARK + "\n", 1)
open(TEX, "w", encoding="utf-8").write(tex)
print(f"draft6 fit pass applied ({n} edits)")
