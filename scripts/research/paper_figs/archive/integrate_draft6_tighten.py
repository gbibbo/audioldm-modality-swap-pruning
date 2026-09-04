#!/usr/bin/env python3
"""Draft-6 page-budget pass on icassp/icassp_operating_point.tex (CPU, 0 cr, idempotent).

The Draft-6 layout pass (integrate_draft6_layout.py) buys a much larger figure and a longer abstract
and introduction; ICASSP gives 4 pages of technical content, so the difference has to come back out.

NO experimental number, claim, caveat or citation is removed here. What changes:
  * float/table typography (array stretch, caption and item spacing) -- pure spacing;
  * topic sentences that only restate the subsection title they sit under;
  * wording, where the same statement can be made in fewer words.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_draft6_tighten.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
MARK = "%% draft6-tighten"

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


# ------------------------------------------------------------------------- typography (no content)
rep(r"\setlength{\abovecaptionskip}{3pt}", r"\setlength{\abovecaptionskip}{2pt}")
rep(r"\begin{itemize}\itemsep2pt \parskip0pt \topsep2pt",
    r"\begin{itemize}\itemsep1pt \parskip0pt \topsep2pt")
rep(r"\setlength{\tabcolsep}{1.9pt}\renewcommand{\arraystretch}{1.05}",
    r"\setlength{\tabcolsep}{1.9pt}\renewcommand{\arraystretch}{0.98}")
rep(r"\setlength{\tabcolsep}{1.4pt}\renewcommand{\arraystretch}{1.05}",
    r"\setlength{\tabcolsep}{1.4pt}\renewcommand{\arraystretch}{0.98}")

# ------------------------------------------------------------------------------ Sec. 3 (wording)
rep(r"""$71.08$\,M ($-82.9\%$; $39\%$ fewer multiply--accumulates~\cite{singh2026pruning}). At each severity we compare the""",
    r"""$71.08$\,M ($-82.9\%$; $39\%$ fewer multiply--accumulates). At each we compare the""")
rep(r"""$10^{6}$ steps of AudioCaps recovery fine-tuning of P~\cite{singh2026pruning} (EMA weights). The released pruned checkpoints carry no usable EMA weights, so P is the released L1 channel
selection applied to the dense EMA weights (bit-exact to the release at severity~1; at""",
    r"""$10^{6}$ steps of AudioCaps recovery fine-tuning of P (EMA weights). The released P has no usable
EMA weights, so it is the released L1 channel selection applied to the dense EMA weights (bit-exact at""")
rep(r"""primary, B$'$ sensitivity). ``Recovery'' names the fine-tuning stage, not an achieved restoration; we
train no system. Two of the present authors released the evaluated checkpoints~\cite{singh2026pruning}; the operating points,
batteries, estimands and gates below were specified independently.""",
    r"""primary, B$'$ sensitivity). ``Recovery'' names the stage, not an achieved restoration. Two of the
present authors released the evaluated checkpoints; the operating points, batteries, estimands and
gates below were specified independently.""")

rep(r"""Two durations: the \emph{native} point ($10.24$\,s, latent length $256$; the fine-tuning duration) and a
\emph{short} point ($3.84$\,s, latent length $96$); a sweep adds $5.12$ and $7.68$\,s (latent $128$, $192$)
at severity~2. Two domains: in-domain AudioCaps \textsc{test}
captions~\cite{kim2019audiocaps} and held-out hip-hop/rap captions from
MusicCaps~\cite{agostinelli2023musiclm}, a sub-domain with near-zero exposure in the recovery corpus
(``music''; captions seven times longer, median $56.5$ vs.\ $8$ words, so the domain axis bundles
content with caption style).""",
    r"""Two durations: the \emph{native} point ($10.24$\,s, latent $256$; the fine-tuning duration) and a
\emph{short} point ($3.84$\,s, latent $96$); a sweep adds $5.12$ and $7.68$\,s (latent $128$, $192$) at
severity~2. Two domains: in-domain AudioCaps \textsc{test} captions~\cite{kim2019audiocaps} and held-out
hip-hop/rap captions from MusicCaps~\cite{agostinelli2023musiclm}, a sub-domain with near-zero exposure
in the recovery corpus (``music''; captions seven times longer, median $56.5$ vs.\ $8$ words, so the
axis bundles content with caption style).""")
rep(r"""$3.84$\,s; the dense model is generated at both durations on both AudioCaps
sets (matched controls, same noise).""",
    r"""$3.84$\,s; dense is generated at both durations on both AudioCaps sets (matched
controls, same noise).""")

rep(r"""Within each operating point the compared systems receive identical generation noise $x_T$ per prompt
(common random numbers); across durations the latent shapes differ, so duration contrasts are
prompt-paired but not noise-paired.""",
    r"""Within an operating point the compared systems receive identical generation noise $x_T$ per prompt
(common random numbers); across durations the latent shapes differ, so duration contrasts are prompt-
but not noise-paired.""")
rep(r"""the \emph{real-audio ceiling}, the real AudioCaps clip of each
prompt, band-limited to $16$\,kHz and scored under the identical convention at full length and as its
first $3.84$\,s. The unit is the prompt (music replicates averaged first); intervals are $95\%$
prompt-level percentile bootstrap intervals ($B=10^{4}$). We also listened, blind (opaque ids, sealed key, seeded draw), to $8$ AudioCaps and $8$ music prompts at
severity~2; these clips, and more, are on the companion page (Sec.~6).""",
    r"""the \emph{real-audio ceiling}, the real AudioCaps clip of each prompt,
band-limited to $16$\,kHz and scored under the same convention at full length and as its first
$3.84$\,s. The unit is the prompt (music replicates averaged first); intervals are $95\%$ prompt-level
percentile bootstrap intervals ($B=10^{4}$). We also listened, blind (opaque ids, sealed key, seeded
draw), to $8$ AudioCaps and $8$ music prompts at severity~2; those clips, and more, are on the companion
page (Sec.~6).""")

rep(r"""matched duration is $\Rs-\Rm$; the smallest effect size of interest is $0.025$. ``Pre-specified'' means estimand, gate and prompt set were committed before any score was seen: the
severity-1 domain test (the original hypothesis; failed), the severity-1 duration follow-up, the severity-2 replication with seam
sensitivity (primary), the severity-2 music cell at $10.24$\,s and the secondary metrics. The FineLAP diagnostic, the severity-2 dense control, the duration sweep and the published-recipe
check were \emph{registered after the primary result}: committed, with their decision rules, before
their own scores were seen, after the primary endpoint had been read. The
severity-1 dense control, crop, rank-scale, Holm and anchor analyses and our listening are \emph{post-hoc}.""",
    r"""matched duration is $\Rs-\Rm$; the smallest effect size of interest is $0.025$. ``Pre-specified'' means
estimand, gate and prompt set were committed before any score was seen: the severity-1 domain test (the
original hypothesis; failed) and duration follow-up, the severity-2 replication with seam sensitivity
(primary), the severity-2 music cell at $10.24$\,s and the secondary metrics. \emph{Registered after the
primary result} --- committed with their decision rules before their own scores were seen, once the
primary endpoint had been read --- are the FineLAP diagnostic, the severity-2 dense control, the
duration sweep and the published-recipe check. The severity-1 dense control, crop, rank-scale, Holm and
anchor analyses and our listening are \emph{post-hoc}.""")

# ---------------------------------------------- Sec. 4: topic sentences that restate their own title
rep(r"""\subsection{The recovery gain grows with clip duration}
At the more severe operating point, recovery fine-tuning buys several times more alignment at its own
duration than at $3.84$\,s. On the disjoint $192$-prompt set, with estimands frozen before scoring, the
interaction is clearly resolved, $J=+0.159$ (Fig.~\ref{fig:ops}a,b, Table~\ref{tab:core}).""",
    r"""\subsection{The recovery gain grows with clip duration}
At the more severe operating point, fine-tuning buys several times more alignment at its own duration
than at $3.84$\,s. On the disjoint $192$-prompt set, estimands frozen before scoring, the interaction is
clearly resolved, $J=+0.159$ (Fig.~\ref{fig:ops}a,b, Table~\ref{tab:core}).""")

rep(r"""The effect is not an artefact of scale. The chance floor is close to zero and nearly duration-independent
(Table~\ref{tab:anchors}: $-0.048$ to $+0.020$ across the AudioCaps cells, moving by at most $0.025$
between durations for any system), so the interaction survives chance correction:""",
    r"""The effect is no artefact of scale: the chance floor is close to zero and nearly duration-independent
(Table~\ref{tab:anchors}: $-0.048$ to $+0.020$ across the AudioCaps cells, moving by at most $0.025$
between durations for any system), so the interaction survives chance correction,""")

rep(r"""The matched dense control separates the systems' duration response from the scorer's. At severity~1, on
the same $80$ prompts and under the same convention, the dense model's response""",
    r"""The matched dense control separates the systems' duration response from the scorer's. At severity~1, on
the same $80$ prompts and convention, the dense model's response""")

rep(r"""The gain grows monotonically with duration. A sweep of P, \PFT{} and dense over $5.12$ and $7.68$\,s on
the same prompts (Fig.~\ref{fig:ops}b) gives $R=+0.085$, $+0.139$, $+0.201$,""",
    r"""The gain also grows monotonically. A sweep of P, \PFT{} and dense over $5.12$ and $7.68$\,s on the same
prompts (Fig.~\ref{fig:ops}b) gives $R=+0.085$, $+0.139$, $+0.201$,""")

rep(r"""$10$\,s window without repeat-padding. Listening blind, we found the gain plainly audible at $10.24$\,s (\PFT{} preferred on 6/8 prompts; P mostly
noise) but not at $3.84$\,s (0/8: both checkpoints sound like noise), so the short gain CLAP resolves is one
the ear does not.""",
    r"""$10$\,s window without repeat-padding. Listening blind, we found the gain plainly audible at
$10.24$\,s (\PFT{} preferred on 6/8 prompts; P mostly noise) but not at $3.84$\,s (0/8: both checkpoints
sound like noise): the short gain CLAP resolves is one the ear does not.""")

rep(r"""Severity~1 is underpowered, not negative: its raw-scale interval narrowly includes zero (the follow-up
was powered for $J\ge0.065$ at $n=80$), but it is directionally consistent on every scale (post-hoc): the median per-prompt interaction is $+0.051$ $\ci{+0.012}{+0.077}$ and
the second scorer gives $J_{\mathrm{HC}}=+0.075$ $\ci{+0.012}{+0.137}$. A Holm correction over all $19$ contrasts, sweep steps and published-recipe check included,
leaves every severity-2 conclusion unchanged ($p<10^{-4}$; last sweep step $p=0.003$); at severity~1, $\Rn$ ($p=0.016$)
and $J$ ($p=0.052$) do not survive, so the severity-1 duration effect rests on the severity-2 replication.""",
    r"""Severity~1 is underpowered, not negative: its raw-scale interval narrowly includes zero (the follow-up
was powered for $J\ge0.065$ at $n=80$) but is directionally consistent on every scale (post-hoc): the
median per-prompt interaction is $+0.051$ $\ci{+0.012}{+0.077}$ and the second scorer gives
$J_{\mathrm{HC}}=+0.075$ $\ci{+0.012}{+0.137}$. A Holm correction over all $19$ contrasts, sweep steps
and published-recipe check included, leaves every severity-2 conclusion unchanged ($p<10^{-4}$; last
sweep step $p=0.003$); at severity~1, $\Rn$ ($p=0.016$) and $J$ ($p=0.052$) do not, so the severity-1
duration effect rests on the severity-2 replication.""")

rep(r"""The gain does not transfer off the fine-tuning domain, at either duration (Fig.~\ref{fig:ops}c). At severity~2 and
$3.84$\,s it is resolved in-domain ($\Rs=+0.085$) and null on music ($\Rm=+0.009$; $W=0.48$), a
matched-duration domain contrast of $+0.076$; at the native duration the music contrast is likewise null
($+0.005$; $W=0.53$), so the large native gain is confined to the fine-tuning domain (domain contrast $+0.239$; music duration interaction $-0.004$). At severity~1 the
pre-specified domain test ($96$ prompts)""",
    r"""The gain does not transfer off the fine-tuning domain, at either duration (Fig.~\ref{fig:ops}c). At
severity~2 and $3.84$\,s it is resolved in-domain ($\Rs=+0.085$) and null on music ($\Rm=+0.009$;
$W=0.48$), a matched-duration domain contrast of $+0.076$; at the native duration the music contrast is
likewise null ($+0.005$; $W=0.53$), so the large native gain too is confined to the fine-tuning domain
(domain contrast $+0.239$; music duration interaction $-0.004$). At severity~1 the pre-specified domain
test ($96$ prompts)""")

rep(r"""The chance floor reads the two severities as one statement
(Table~\ref{tab:anchors}): after fine-tuning, alignment on the hip-hop captions is $0.022$, $0.018$ and
$0.033$ above chance (severity~1 at $3.84$\,s; severity~2 at $3.84$ and $10.24$\,s), against $0.115$ and
$0.321$ in-domain at severity~2. Whether that registers as a ``penalty'' depends on whether the pruned checkpoint had music alignment to lose: the $65\%$-pruned one did ($0.061$ above chance;
chance-corrected $\Rm=-0.040$ $\ci{-0.060}{-0.020}$), the $83\%$-pruned one did not ($0.018$).""",
    r"""The chance floor reads the two severities as one statement (Table~\ref{tab:anchors}): after fine-tuning,
alignment on the hip-hop captions is $0.022$, $0.018$ and $0.033$ above chance (severity~1 at
$3.84$\,s; severity~2 at $3.84$ and $10.24$\,s), against $0.115$ and $0.321$ in-domain at severity~2.
Whether that is a ``penalty'' depends on whether the pruned checkpoint had music alignment to lose: the
$65\%$-pruned one did ($0.061$ above chance; chance-corrected $\Rm=-0.040$ $\ci{-0.060}{-0.020}$), the
$83\%$-pruned one did not ($0.018$).""")

rep(r"""Two analyses locate the native-duration gain. First, a prospectively frozen diagnostic scores the native clips with FineLAP~\cite{li2026finelap}, a frame-level language--audio grounding model (EAT
encoder; not CLAP-derived) reporting per $0.16$\,s frame how strongly the requested event is grounded
($110$/$49$ eligible prompts at severities 2/1, outcome-blind).""",
    r"""Two analyses locate the native-duration gain. First, a prospectively frozen diagnostic scores the native
clips with FineLAP~\cite{li2026finelap}, a frame-level language--audio grounding model (EAT encoder; not
CLAP-derived) reporting per $0.16$\,s frame how strongly the requested event is grounded ($110$/$49$
eligible prompts at severities 2/1, outcome-blind).""")

rep(r"""Second, a post-hoc crop analysis scores the first $3.84$\,s of each $10.24$\,s generation under the same
convention against the separately generated $3.84$\,s clips (Fig.~\ref{fig:ops}e). The crop carries a large part of the native gain: at severity~1 it equals $\Rn$ and exceeds $\Rs$ by""",
    r"""Second, a post-hoc crop analysis scores the first $3.84$\,s of each $10.24$\,s generation under that
convention against the separately generated $3.84$\,s clips (Fig.~\ref{fig:ops}e). The crop carries much
of the native gain: at severity~1 it equals $\Rn$ and exceeds $\Rs$ by""")

rep(r"""Three pre-specified hypotheses failed and are kept: (i)~that fine-tuning \emph{trades} in-domain for out-of-domain alignment: it requires an in-domain gain
at $3.84$\,s, which severity~1 lacks ($R_{\mathrm{AC}}\approx0$), so the music loss is real but the trade is not; (ii)~the severity-1 music
penalty did not replicate (Human-CLAP alone shows $-0.037$~$\ci{-0.068}{-0.005}$ at severity~2); (iii)~a ``late allocation'' account of the duration effect is rejected by FineLAP
($T\approx0$).""",
    r"""Three pre-specified hypotheses failed and are kept: (i)~that fine-tuning \emph{trades} in-domain for
out-of-domain alignment --- it needs an in-domain gain at $3.84$\,s, which severity~1 lacks
($R_{\mathrm{AC}}\approx0$), so the music loss is real but the trade is not; (ii)~the severity-1 music
penalty did not replicate (Human-CLAP alone shows $-0.037$~$\ci{-0.068}{-0.005}$ at severity~2);
(iii)~a ``late allocation'' account of the duration effect is rejected by FineLAP ($T\approx0$).""")

rep(r"""\emph{Limitations.} One case study (one model family, two severities, two domains).
Mechanistic attribution is blocked: the matched control (a dense model fine-tuned identically) no longer exists and the public dense text-fine-tuned release is not equivalent, so the
dependence cannot be attributed to pruning rather than to fine-tuning in general. Primary inference rests on CLAP-family scorers; our listening is informal and disagrees with CLAP at $3.84$\,s. Sampler settings differ from the published recipe; the interaction is reproduced at its DDIM $200$\,/\,$3.5$
on a $64$-prompt subset, best-of-3 selection was not. The sweep stops at the fine-tuning duration, so it cannot separate ``largest at the
training duration'' from ``larger for longer clips''. One held-out sub-genre; the severity-1 duration test was underpowered.""",
    r"""\emph{Limitations.} One case study (one model family, two severities, two domains). Mechanistic
attribution is blocked: the matched control (a dense model fine-tuned identically) no longer exists and
the public dense text-fine-tuned release is not equivalent, so the dependence cannot be attributed to
pruning rather than to fine-tuning in general. Primary inference rests on CLAP-family scorers; our
listening is informal and disagrees with CLAP at $3.84$\,s. Sampler settings differ from the published
recipe; the interaction is reproduced at its DDIM $200$\,/\,$3.5$ on a $64$-prompt subset, best-of-3
selection was not. The sweep stops at the fine-tuning duration, so it cannot separate ``largest at the
training duration'' from ``larger for longer clips''. One held-out sub-genre, and the severity-1
duration test was underpowered.""")

tex = tex.replace("%% draft6-layout\n", "%% draft6-layout\n" + MARK + "\n", 1)
open(TEX, "w", encoding="utf-8").write(tex)
print(f"draft6 tightening pass applied ({n} edits)")
