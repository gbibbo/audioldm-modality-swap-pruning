#!/usr/bin/env python3
"""Integrate DRAFT5-OPSWEEP-1 (duration sweep, E1) and DRAFT5-PUBRECIPE-1 (published-recipe check, E2b)
into the Draft-5 manuscript (CPU, 0 cr). Idempotent: refuses to run twice (own marker). Reads
configs/research/draft5_opsweep_result.json and configs/research/draft5_pubrecipe_result.json only.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_opsweep.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
SW = json.load(open(os.path.join(ROOT, "configs/research/draft5_opsweep_result.json")))
PR = json.load(open(os.path.join(ROOT, "configs/research/draft5_pubrecipe_result.json")))
MARK = "%% opsweep-integrated"
assert SW["SHAPE_VERDICT"] == "MONOTONE-INCREASING", SW["SHAPE_VERDICT"]   # the prose below states this verdict
assert PR["GATE_lo95_J_pub_gt_0"] is True                                   # and this gate
R, S, B = SW["R_by_duration"], SW["steps"], SW["secondary"]["by_duration"]


def pt(c, nd=3):
    return f"${c['point']:+.{nd}f}$"


def ci(c, nd=3):
    return f"\\ci{{{c['lo']:+.{nd}f}}}{{{c['hi']:+.{nd}f}}}"


def pct(c):
    return f"${100*c['point']:.0f}\\%$"


def tab_ci(c):
    return f"${100*c['point']:.0f}\\%$~$[{100*c['lo']:.0f},\\,{100*c['hi']:.0f}]$"


def floor_pair(d):
    f = B[d]["floors"]
    return f"${f['P']:+.3f}/{f['PFT']:+.3f}$"


s = open(TEX, encoding="utf-8").read()
if MARK in s:
    sys.exit("already integrated")


def rep(a, b):
    global s
    assert s.count(a) == 1, "MISSING/AMBIGUOUS: " + a[:100]
    s = s.replace(a, b)


# ---- abstract
rep("""The duration dependence was pre-specified, replicates on a disjoint prompt set, survives a
family-wise correction, is not a scorer artefact, and is reproduced at both durations by a second
scorer and by two event-level metrics outside the CLAP family.""",
    """The duration dependence was pre-specified, replicates on a disjoint prompt set, survives a
family-wise correction, is not a scorer artefact, grows monotonically over four durations, holds at the
published sampler recipe, and is reproduced at both durations by a second scorer and by two event-level
metrics outside the CLAP family.""")
# ---- intro bullet
rep("""\\item \\textbf{Duration.} The recovery gain is several times larger at the fine-tuning duration
      ($10.24$\\,s) than at $3.84$\\,s, at both severities. The interaction was pre-specified, replicates
      on a disjoint prompt set, and is not a scale or scorer effect (the chance floor moves by at most
      $0.025$ between durations).""",
    """\\item \\textbf{Duration.} The recovery gain is several times larger at the fine-tuning duration
      ($10.24$\\,s) than at $3.84$\\,s, at both severities, and grows monotonically over four durations
      at severity~2. The interaction was pre-specified, replicates on a disjoint prompt set, and is not
      a scale or scorer effect (the chance floor moves by at most $0.025$ between durations).""")
# ---- 3.2 operating points
rep("""Two durations: the \\emph{native} point ($10.24$\\,s, latent length $256$; the fine-tuning duration) and a
\\emph{short} point ($3.84$\\,s, latent length $96$).""",
    """Two durations: the \\emph{native} point ($10.24$\\,s, latent length $256$; the fine-tuning duration) and a
\\emph{short} point ($3.84$\\,s, latent length $96$); a sweep adds $5.12$ and $7.68$\\,s (latent $128$, $192$)
at severity~2.""")
rep("""Generation uses DDIM~\\cite{song2021ddim}, $50$ steps, guidance $2.5$, $\\eta=0$,
\\texttt{fp32}, single generation, EMA weights (off the published recipe, Sec.~5).""",
    """Generation uses DDIM~\\cite{song2021ddim}, $50$ steps, guidance $2.5$, $\\eta=0$,
\\texttt{fp32}, single generation, EMA weights (off the published recipe, Sec.~5; one check on the first
$64$ severity-2 prompts uses the published DDIM $200$, guidance $3.5$).""")
# ---- 3.4 status labels
rep("""The FineLAP diagnostic and the severity-2 dense control were \\emph{registered after the primary
result}: committed before their own scores were seen, after the primary endpoint had been read.""",
    """The FineLAP diagnostic, the severity-2 dense control, the duration sweep and the published-recipe
check were \\emph{registered after the primary result}: committed, with their decision rules, before
their own scores were seen, after the primary endpoint had been read.""")
# ---- 4.1: sweep paragraph after the severity-2 dense-control sentence
rep("""fine-tuning closes $44\\%$ of the gap to dense at $3.84$\\,s and $82\\%$ at $10.24$\\,s (Table~\\ref{tab:anchors}).
""",
    f"""fine-tuning closes $44\\%$ of the gap to dense at $3.84$\\,s and $82\\%$ at $10.24$\\,s (Table~\\ref{{tab:anchors}}).

The gain grows monotonically with duration. A sweep (P, \\PFT{{}} and dense at $5.12$ and $7.68$\\,s on the
same $192$ prompts; Fig.~\\ref{{fig:interaction}}b) gives $R={pt(R['3.84'])[1:-1]}$, {pt(R['5.12'])}, {pt(R['7.68'])},
{pt(R['10.24'])} at $3.84$, $5.12$, $7.68$, $10.24$\\,s, every step resolved ({pt(S['D1'])}~{ci(S['D1'])},
{pt(S['D2'])}~{ci(S['D2'])}, {pt(S['D3'])}~{ci(S['D3'])}): by the pre-specified shape rule the gain is
\\emph{{monotone increasing}}, neither peaked nor saturating below the fine-tuning duration. The pruned
checkpoint is flat to $5.12$\\,s ({pt(SW['s_response']['pruned2_A_3.84_to_5.12'])}) while \\PFT{{}} rises at every
step; $\\rho_{{\\mathrm{{dense}}}}$ climbs {pct(B['3.84']['rho_dense'])}, {pct(B['5.12']['rho_dense'])},
{pct(B['7.68']['rho_dense'])}, {pct(B['10.24']['rho_dense'])} (Table~\\ref{{tab:anchors}}).
""")
# ---- Fig. 1 caption
rep("""fine-tuned (\\PFT, solid) checkpoints at the short and native points, (a)~severity~1, (b)~severity~2;
whiskers: $95\\%$ CI of the paired gain $R$ about the \\PFT{} mean.""",
    """fine-tuned (\\PFT, solid) checkpoints at the short and native points, (a)~severity~1, and (b)~severity~2
with the two sweep durations ($R$ printed at the new points); whiskers: $95\\%$ CI of the paired gain $R$
about the \\PFT{} mean.""")
# ---- Table 2: two sweep rows
rep("""sev.\\,2 AC $3.84$\\,s & $-0.005/-0.015$ & $0.274$ & $44\\%$~$[36,\\,53]$ & $33\\%$~$[26,\\,39]$\\\\
""",
    f"""sev.\\,2 AC $3.84$\\,s & $-0.005/-0.015$ & $0.274$ & $44\\%$~$[36,\\,53]$ & $33\\%$~$[26,\\,39]$\\\\
sev.\\,2 AC $5.12$\\,s & {floor_pair('5.12')} & ${B['5.12']['levels']['real']:.3f}$ & {tab_ci(B['5.12']['rho_dense'])} & {tab_ci(B['5.12']['rho_real'])}\\\\
sev.\\,2 AC $7.68$\\,s & {floor_pair('7.68')} & ${B['7.68']['levels']['real']:.3f}$ & {tab_ci(B['7.68']['rho_dense'])} & {tab_ci(B['7.68']['rho_real'])}\\\\
""")
rep("""fraction $\\rho$ of the pruned checkpoint's gap to dense and to real audio closed by fine-tuning ($95\\%$
CI, in \\%). Dense control: severity~2 registered after the primary result, severity~1 post-hoc
(Sec.~3.4).}""",
    """fraction $\\rho$ of the pruned checkpoint's gap to dense and to real audio closed by fine-tuning ($95\\%$
CI, in \\%). Dense control and $5.12$/$7.68$\\,s sweep: registered after the primary result; severity-1
dense control post-hoc (Sec.~3.4).}""")
# ---- 4.4: published-recipe check
rep("""(seam-robust).

Three pre-specified hypotheses failed and are kept:""",
    f"""(seam-robust). At the published sampler recipe (DDIM $200$, guidance $3.5$; single generation) on the first
$64$ prompts the interaction holds, $J={pt(PR['J_pub'])[1:-1]}$~{ci(PR['J_pub'])} (pre-specified gate; the frozen
recipe on the same prompts gives {pt(PR['J_frozen_same64'])}, difference {pt(PR['J_pub_minus_J_frozen'])}~{ci(PR['J_pub_minus_J_frozen'])}).

Three pre-specified hypotheses failed and are kept:""")
# ---- discussion
rep("""tracks the operating point of the fine-tuning itself: largest at the fine-tuning duration,
confined to the fine-tuning domain,""",
    """tracks the operating point of the fine-tuning itself: growing monotonically toward the fine-tuning
duration, confined to the fine-tuning domain,""")
# ---- limitations
rep("""Sampler settings
differ from the published recipe (DDIM $50$ vs.\\ $200$, guidance $2.5$ vs.\\ $3.5$, single vs.\\
best-of-3; both systems evaluated identically).""",
    """Sampler settings
differ from the published recipe (DDIM $50$ vs.\\ $200$, guidance $2.5$ vs.\\ $3.5$; both systems evaluated
identically); the interaction is reproduced at DDIM $200$\\,/\\,$3.5$ on a $64$-prompt subset, but best-of-3
selection was not. The sweep stops at the fine-tuning duration, so it cannot separate ``largest at the
training duration'' from ``larger for longer clips''.""")
s = s.replace("\\documentclass{article}", MARK + "\n\\documentclass{article}", 1)
open(TEX, "w", encoding="utf-8").write(s)
print("integrated: shape", SW["SHAPE_VERDICT"], "| J_pub", pt(PR["J_pub"]), ci(PR["J_pub"]))
