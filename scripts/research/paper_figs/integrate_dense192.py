#!/usr/bin/env python3
"""Integrate the XSEV-DENSE-192-CONTROL result into the Draft-5 manuscript (CPU, 0 cr). Idempotent: refuses
to run twice (checks for its own marker). Reads configs/research/xsev_dense_192_control_result.json and
rewrites the sentences that were cross-set in Draft 5 into paired statements; Table 2 gains a dense column
and a rho_dense column (severity-1 values from the Draft-4 control, severity-2 from this result).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_dense192.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
D = json.load(open(os.path.join(ROOT, "configs/research/xsev_dense_192_control_result.json")))
DDC = json.load(open(os.path.join(ROOT, "configs/research/draft4_dense_duration_control_result.json")))
P, m = D["PRIMARY"], D["means"]
MARK = "%% dense192-integrated"


def pct(c):
    return f"${100*c['point']:.0f}\\%$"


def pct_ci(c):
    return f"${100*c['point']:.0f}\\%$~\\ci{{{100*c['lo']:+.0f}\\%}}{{{100*c['hi']:+.0f}\\%}}"


def pt(c):
    return f"${c['point']:+.3f}$"


def pt_ci(c):
    return f"${c['point']:+.3f}$ $\\ci{{{c['lo']:+.3f}}}{{{c['hi']:+.3f}}}$"


s = open(TEX, encoding="utf-8").read()
if MARK in s:
    sys.exit("already integrated")


def rep(a, b):
    global s
    assert a in s, "MISSING: " + a[:90]
    s = s.replace(a, b)


rho_s, rho_n = P["rho_short"], P["rho_native"]
# ---- abstract
rep("""hip-hop captions; at $65\\%$ pruning a matched dense control gives $52\\%$ versus $8\\%$ of the gap to the
unpruned model.""",
    f"""hip-hop captions; against the unpruned model, {pct(rho_n)} versus {pct(rho_s)} at $83\\%$ and $52\\%$ versus
$8\\%$ at $65\\%$.""")
# ---- intro bullet
rep("""$10.24$\\,s and $33\\%$ at $3.84$\\,s; at $65\\%$, $52\\%$ versus $8\\%$ of the gap to the unpruned
      model.""",
    f"""$10.24$\\,s and $33\\%$ at $3.84$\\,s ({pct(rho_n)} versus {pct(rho_s)} of the gap to the unpruned
      model; $52\\%$ versus $8\\%$ at $65\\%$).""")
# ---- 3.2: dense generated on both prompt sets
rep("""a matched $80$-prompt subset for the duration comparison (on which the dense
model is also generated at both durations: the matched dense control), and $64$ music prompts at $3.84$\\,s.""",
    """a matched $80$-prompt subset for the duration comparison, and $64$ music prompts at $3.84$\\,s. The dense
model is generated at both durations on the severity-2 $192$ prompts and on the severity-1 $80$ (matched
dense controls, same noise as the pruned pair).""")
# ---- 4.1 severity-2 paragraph: replace the cross-set sentence
rep("""$10.24$\\,s (Table~\\ref{tab:anchors}). At severity~2,
$s(\\mathrm{P})=+0.040$ is far below the dense response on the severity-1 prompts while $s(\\PFT)=+0.200$ is
of dense magnitude (cross-set, descriptive).""",
    f"""$10.24$\\,s (Table~\\ref{{tab:anchors}}). The same control on the severity-2 prompts ({m['dense_short']:.3f}\\!\\to\\!{m['dense_native']:.3f}) gives
$s(\\mathrm{{dense}})={P['s_dense']['point']:+.3f}$ $\\ci{{{P['s_dense']['lo']:+.3f}}}{{{P['s_dense']['hi']:+.3f}}}$: the pruned checkpoint's response is
{pt_ci(P['s_pruned_minus_s_dense'])} below it and the fine-tuned checkpoint's {pt_ci(P['s_postft_minus_s_dense'])} above
it (paired); fine-tuning closes {pct_ci(rho_s)} of the gap to dense at $3.84$\\,s and {pct_ci(rho_n)}
at $10.24$\\,s.""")
# ---- Table 2: add dense and rho_dense columns
rep("""\\begin{tabular}{@{}lccc@{}}
\\toprule
Setting & floor P\\,/\\,\\PFT & real & $\\rho_{\\mathrm{real}}$ [95\\% CI]\\\\
\\midrule
sev.\\,1 AudioCaps $3.84$\\,s & $-0.031\\,/\\,-0.033$ & $0.264$ & $5\\%$~\\ci{-16\\%}{+24\\%}\\\\
sev.\\,1 AudioCaps $10.24$\\,s & $-0.012\\,/\\,-0.036$ & $0.442$ & $27\\%$~\\ci{+6\\%}{+46\\%}\\\\
sev.\\,2 AudioCaps $3.84$\\,s & $-0.005\\,/\\,-0.015$ & $0.274$ & $33\\%$~\\ci{+26\\%}{+39\\%}\\\\
sev.\\,2 AudioCaps $10.24$\\,s & $+0.020\\,/\\,-0.022$ & $0.440$ & $63\\%$~\\ci{+56\\%}{+71\\%}\\\\
sev.\\,1 music $3.84$\\,s & $+0.055\\,/\\,+0.001$ & -- & --\\\\
sev.\\,2 music $3.84$\\,s & $-0.013\\,/\\,-0.004$ & -- & --\\\\
sev.\\,2 music $10.24$\\,s & $+0.070\\,/\\,+0.061$ & -- & --\\\\""",
    f"""\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\toprule
Setting & floor P\\,/\\,\\PFT & dense & real & $\\rho_{{\\mathrm{{dense}}}}$ [95\\% CI] & $\\rho_{{\\mathrm{{real}}}}$ [95\\% CI]\\\\
\\midrule
sev.\\,1 AC $3.84$\\,s & $-0.031\\,/\\,-0.033$ & ${DDC['means']['dense_short']:.3f}$ & $0.264$ & $8\\%$~\\ci{{-30\\%}}{{+36\\%}} & $5\\%$~\\ci{{-16\\%}}{{+24\\%}}\\\\
sev.\\,1 AC $10.24$\\,s & $-0.012\\,/\\,-0.036$ & ${DDC['means']['dense_native']:.3f}$ & $0.442$ & $52\\%$~\\ci{{+11\\%}}{{+103\\%}} & $27\\%$~\\ci{{+6\\%}}{{+46\\%}}\\\\
sev.\\,2 AC $3.84$\\,s & $-0.005\\,/\\,-0.015$ & ${m['dense_short']:.3f}$ & $0.274$ & {pct_ci(rho_s)} & $33\\%$~\\ci{{+26\\%}}{{+39\\%}}\\\\
sev.\\,2 AC $10.24$\\,s & $+0.020\\,/\\,-0.022$ & ${m['dense_native']:.3f}$ & $0.440$ & {pct_ci(rho_n)} & $63\\%$~\\ci{{+56\\%}}{{+71\\%}}\\\\
sev.\\,1 music $3.84$\\,s & $+0.055\\,/\\,+0.001$ & -- & -- & -- & --\\\\
sev.\\,2 music $3.84$\\,s & $-0.013\\,/\\,-0.004$ & -- & -- & -- & --\\\\
sev.\\,2 music $10.24$\\,s & $+0.070\\,/\\,+0.061$ & -- & -- & -- & --\\\\""")
rep("""\\caption{Anchors and recovery ratios (post-hoc; prompts and scoring as in Table~\\ref{tab:core}):
shuffled-caption chance floor of the P\\,/\\,\\PFT{} cells, mean CLAP of the real audio of the same prompts,
and the fraction $\\rho_{\\mathrm{real}}$ of the pruned checkpoint's gap to real audio closed by fine-tuning
($95\\%$ CI). Against the matched dense model (severity~1): $8\\%$~\\ci{-30\\%}{+36\\%} at $3.84$\\,s,
$52\\%$~\\ci{+11\\%}{+103\\%} at $10.24$\\,s.}""",
    """\\caption{Anchors and recovery ratios (prompts and scoring as in Table~\\ref{tab:core}; AC = AudioCaps):
shuffled-caption chance floor of the P\\,/\\,\\PFT{} cells, mean CLAP of the matched dense model and of the
real audio of the same prompts, and the fraction $\\rho$ of the pruned checkpoint's gap to dense and to
real audio closed by fine-tuning ($95\\%$ CI). Dense control at severity~1: post-hoc; at severity~2:
pre-specified design completion.}""")
rep("\\setlength{\\tabcolsep}{2.2pt}\\renewcommand{\\arraystretch}{1.05}\n\\scriptsize\n\\begin{tabular}{@{}lccccc@{}}",
    "\\setlength{\\tabcolsep}{1.6pt}\\renewcommand{\\arraystretch}{1.05}\n\\scriptsize\n\\begin{tabular}{@{}lccccc@{}}")
# ---- 4.4 not restored to dense: add severity 2
rep("""Finally, \\PFT{} is not restored to dense: at severity~1 and $10.24$\\,s the dense model
leads P by $+0.099$~$\\ci{+0.058}{+0.140}$ and \\PFT{} by $+0.048$~$\\ci{-0.000}{+0.096}$.""",
    f"""Finally, \\PFT{{}} is not restored to dense: at severity~1 and $10.24$\\,s the dense model
leads P by $+0.099$~$\\ci{{+0.058}}{{+0.140}}$ and \\PFT{{}} by $+0.048$~$\\ci{{-0.000}}{{+0.096}}$; at severity~2 it
leads \\PFT{{}} by {pt_ci(P['G_native_postft'])} at $10.24$\\,s and {pt_ci(P['G_short_postft'])} at $3.84$\\,s.""")
# ---- discussion
rep("""($0.020$ and $0.035$), and fine-tuning restores a dense-magnitude duration response and $63\\%$ of the gap
to real audio---only in the fine-tuning domain and only at the fine-tuning duration ($33\\%$ at
$3.84$\\,s),""",
    f"""($0.020$ and $0.035$), and fine-tuning restores a duration response above the dense model's and
{pct(rho_n)} of the gap to dense---only in the fine-tuning domain and only at the fine-tuning duration
({pct(rho_s)} at $3.84$\\,s),""")
# ---- limitations: drop the cross-set caveat
rep("""The dense control exists only on the
severity-1 prompts (severity-2 ratios are taken against real audio). Primary inference rests on""",
    """Primary inference rests on""")
rep("""the frame-level, crop, anchor and dense-control analyses are post-result.""",
    """the frame-level, crop, anchor and severity-1 dense-control analyses are post-result.""")
# ---- conclusion
rep("""at the fine-tuning duration, $33\\%$ at $3.84$\\,s, none on held-out music; at $65\\%$: $52\\%$ vs.\\ $8\\%$ of
the gap to dense).""",
    f"""at the fine-tuning duration, $33\\%$ at $3.84$\\,s, none on held-out music; {pct(rho_n)} vs.\\ {pct(rho_s)} of the
gap to dense, and $52\\%$ vs.\\ $8\\%$ at $65\\%$).""")
s = s.replace("\\documentclass{article}", MARK + "\n\\documentclass{article}", 1)
open(TEX, "w", encoding="utf-8").write(s)
print("integrated: rho_dense sev-2", pct_ci(rho_s), pct_ci(rho_n), "| s(dense)", pt_ci(P["s_dense"]), "| TOST native:", P["tost_native_postft_pm0.025"]["equivalent"])
