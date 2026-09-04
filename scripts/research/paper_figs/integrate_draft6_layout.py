#!/usr/bin/env python3
"""Draft-6 EDITORIAL pass on icassp/icassp_operating_point.tex (CPU, 0 cr, idempotent).

Presentation only -- no experimental number is added, removed or changed. What it does:

1. Figures. The two column-width stacked figures had 0.7 in panels whose annotations collided with the
   axes and the legend. They become ONE full-text-width five-panel `figure*`
   (figs/fig1_operating_points.pdf, from make_draft6_figs.py): every panel roughly doubles in area,
   and one shared caption costs the page budget far less than two double-column floats would.
2. A third graph is added as Fig. 1(c): the domain result of Sec. 4.2 -- already reported in
   Tables 1-2 -- drawn as levels against each cell's own chance floor.
3. Abstract. Two opening sentences that state the problem (why TTA models are pruned, what recovery
   fine-tuning is) before any of the paper's own machinery, and the checkpoints are introduced as
   "publicly released ... of a recent TTA pruning study" instead of a bare "the released checkpoints".
4. Introduction. The same gradual build-up: pruning and recovery are explained before they are used.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_draft6_layout.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
MARK = "%% draft6-layout"

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


# ------------------------------------------------------------------ 0. float parameters for figure*
rep(r"""\setcounter{topnumber}{3}
\setcounter{totalnumber}{4}""",
    r"""\setcounter{topnumber}{3}
\setcounter{totalnumber}{4}
\setcounter{dbltopnumber}{2}
\renewcommand{\dbltopfraction}{0.92}
\renewcommand{\dblfloatpagefraction}{0.75}""")

rep(r"""\setlength{\abovecaptionskip}{3pt}""",
    r"""\setlength{\dbltextfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{3pt}""")

# ------------------------------------------------------------------------------------- 1. abstract
rep(r"""Structured pruning of text-to-audio diffusion models is followed by recovery fine-tuning, whose benefit
is usually certified by one score at one inference setting. Using the released pruned and fine-tuned
AudioLDM-M checkpoints at two pruning severities ($65\%$ and $83\%$ of U-Net parameters removed), we
measure the paired recovery gain in text--audio alignment across clip duration and prompt domain,
anchored by a shuffled-caption chance floor and the real audio of the same prompts. At
$83\%$ pruning, fine-tuning closes $63\%$ of the pruned checkpoint's gap to real audio at
its own $10.24$\,s fine-tuning duration but only $33\%$ at $3.84$\,s, and nothing on held-out
hip-hop captions; against the unpruned model, $82\%$ versus $44\%$ at $83\%$ and $52\%$ versus
$8\%$ at $65\%$.""",
    r"""Text-to-audio diffusion models are expensive to run, so they are increasingly compressed by
\emph{structured pruning}, which deletes whole channels of the U-Net and then repairs the damage with a
stage of \emph{recovery fine-tuning} on an in-domain caption corpus. What that repair is worth is
normally certified by one aggregate score, computed on the in-domain test set at one clip duration and
one sampler setting. A generative model, however, is used at an inference \emph{operating point} --- a
clip duration, a prompt domain, a sampler configuration --- and recovery fine-tuning is itself training
at one such point, so its benefit need not carry to another. We test this on the publicly released
pruned and recovery-fine-tuned AudioLDM-M checkpoints of a recent text-to-audio pruning study, at two
severities ($65\%$ and $83\%$ of U-Net parameters removed): training nothing, we measure the paired gain
that fine-tuning buys in text--audio alignment across clip duration and prompt domain, read against a
shuffled-caption chance floor and the real audio of the same prompts. At
$83\%$ pruning, fine-tuning closes $63\%$ of the pruned checkpoint's gap to real audio at
its own $10.24$\,s fine-tuning duration but only $33\%$ at $3.84$\,s, and nothing on held-out
hip-hop captions; measured instead against the unpruned model, $82\%$ versus $44\%$ at $83\%$ pruning
and $52\%$ versus $8\%$ at $65\%$.""")

# --------------------------------------------------------------------------------- 2. introduction
rep(r"""Structured pruning followed by recovery fine-tuning is the standard route to cheaper diffusion
models~\cite{fang2023diffpruning,kim2024bksdm,fang2025tinyfusion}, recently applied to text-to-audio
(TTA) generation with AudioLDM~\cite{liu2023audioldm,singh2026pruning}. Recovery is judged by one aggregate number on the in-domain test set at one sampler setting, yet a
generative model is used at an inference \emph{operating point}: a clip duration, a prompt domain, a
sampler configuration. Recovery fine-tuning is itself training at one operating point; nothing guarantees
that its gain holds elsewhere.""",
    r"""Latent diffusion has made text-to-audio (TTA) generation good enough to deploy, but not cheap: the
AudioLDM-M U-Net carries $416$\,M parameters and is evaluated once per sampler step for every clip it
produces~\cite{liu2023audioldm}. The standard way to shrink such a model is \emph{structured pruning},
which deletes whole channels so that parameters and compute fall together. Pruning alone costs quality,
so it is followed by \emph{recovery fine-tuning}: the pruned network is trained further, on an in-domain
caption corpus, until its scores approach those of the model it came from. The two stages together are
the established recipe for cheaper image diffusion
models~\cite{fang2023diffpruning,kim2024bksdm,fang2025tinyfusion} and have recently been carried over to
TTA with AudioLDM~\cite{singh2026pruning}.

What is reported of that second stage is one aggregate number --- FAD, KL or a CLAP score --- on the
in-domain test set, at one sampler setting and one clip length. A generative model, however, is used at
an inference \emph{operating point}: how long a clip is asked for, what kind of prompt is given, how the
sampler is configured. Recovery fine-tuning is itself training at one such point, and nothing guarantees
that what it buys there is also bought elsewhere.""")

# ------------------------------------------------- 3. one full-width, five-panel operating-point figure
rep(r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figs/fig1_interaction.pdf}
\caption{The recovery gain grows with clip duration. Mean CLAP cosine of the pruned (P, dashed) and
fine-tuned (\PFT, solid) checkpoints at the short and native points, (a)~severity~1, and (b)~severity~2
with the two sweep durations; whiskers: $95\%$ CI of the paired gain $R$
about the \PFT{} mean. Anchors: real audio of the same
prompts (triangles), each cell's chance floor (ticks) and the matched dense control (stars; same prompts
and scoring convention).}
\label{fig:interaction}
\end{figure}""",
    r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figs/fig1_operating_points.pdf}
\caption{The recovery gain across the operating points we measure ($n$ as in Sec.~3.2).
\textbf{Duration}, mean CLAP cosine of the pruned (P) and fine-tuned (\PFT) checkpoints against generated
clip duration at (a)~severity~1 and (b)~severity~2, with the real audio of the same prompts, each cell's
chance floor and the matched dense control (same prompts and scoring convention); whiskers are the
$95\%$ CI of the paired gain $R$ about the \PFT{} mean, and $J=\Rn-\Rs$.
\textbf{Domain}, (c)~the same two checkpoints on in-domain AudioCaps and on held-out hip-hop captions per
(severity, duration) cell, each against its own chance floor.
\textbf{Where the gain sits}, (d)~FineLAP frame-level grounding gain (\PFT{} $-$ P) vs.\ time in the
$10.24$\,s clip---uniform, not back-loaded---and (e)~$R$ on the generated $3.84$\,s clip, on the first
$3.84$\,s of the $10.24$\,s clip and on the full clip: a generation-length, not a scoring-window, effect.}
\label{fig:ops}
\end{figure*}""")

rep(r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figs/fig2_where.pdf}
\caption{(a)~FineLAP frame-level grounding gain (\PFT{} $-$ P) vs.\ time in the $10.24$\,s clip
: uniform, not back-loaded. (b)~$R$ ($95\%$ CI) on the generated $3.84$\,s
clip, the first $3.84$\,s of the $10.24$\,s clip and the full clip: a generation-length, not
scoring-window, effect.}
\label{fig:where}
\end{figure}

""", "")

# ------------------------------------------------------------------------ 4. cross-references
rep(r"interaction is clearly resolved, $J=+0.159$ (Fig.~\ref{fig:interaction}, Table~\ref{tab:core}).",
    r"interaction is clearly resolved, $J=+0.159$ (Fig.~\ref{fig:ops}a,b, Table~\ref{tab:core}).")
rep(r"the same prompts (Fig.~\ref{fig:interaction}b) gives $R=+0.085$",
    r"the same prompts (Fig.~\ref{fig:ops}b) gives $R=+0.085$")
rep(r"""\subsection{At matched duration the gain is domain-specific}
The gain does not transfer off the fine-tuning domain, at either duration. At severity~2 and""",
    r"""\subsection{At matched duration the gain is domain-specific}
The gain does not transfer off the fine-tuning domain, at either duration (Fig.~\ref{fig:ops}c). At severity~2 and""")
rep(r"$\ci{-0.024}{+0.020}$ (Fig.~\ref{fig:where}a).", r"$\ci{-0.024}{+0.020}$ (Fig.~\ref{fig:ops}d).")
rep(r"the separately generated $3.84$\,s clips (Fig.~\ref{fig:where}b).",
    r"the separately generated $3.84$\,s clips (Fig.~\ref{fig:ops}e).")

tex = tex.replace("%% author-listening-v2\n", "%% author-listening-v2\n" + MARK + "\n", 1)
open(TEX, "w", encoding="utf-8").write(tex)
print(f"draft6 layout pass applied ({n} edits)")
