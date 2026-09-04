#!/usr/bin/env python3
"""Idempotent integration of AUTHOR-LISTENING-1 (one author's blinded informal listening; N=1; no gate)
into the manuscript, labelled as exactly that everywhere it appears. Reads
configs/research/author_listening_1_result.json. Marker: %% author-listening-integrated.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_author_listening.py
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
L = json.load(open(os.path.join(ROOT, "configs/research/author_listening_1_result.json")))
A, B, C = L["block_A_content_10p24s"], L["block_B_duration_pairs"], L["block_C_music_pairs_10p24s"]
assert B["10.24s"]["prefers_PFT"] == 6 and B["3.84s"]["prefers_PFT"] == 0 and C["prefers_PFT"] == 8   # the prose below
MARK = "%% author-listening-integrated"
s = open(TEX, encoding="utf-8").read()
if MARK in s:
    sys.exit("already integrated")
def rep(a, b):
    global s
    pat = re.sub(r"\\\s+", r"\\s+", re.escape(a)); m = list(re.finditer(pat, s)); assert len(m) == 1, (len(m), a[:80]); s = s[:m[0].start()] + b + s[m[0].end():]
n = B["n_pairs_per_duration"]
# abstract
rep("""lacking a matched dense fine-tuned control and human ratings, our claims concern evaluation, not mechanism.""",
    """lacking a matched dense fine-tuned control and a listening panel (one author's blinded listening agrees at
the native point, not at $3.84$\\,s), our claims concern evaluation, not mechanism.""")
# 3.3: the listening design
rep("""prompt-level percentile bootstrap intervals ($B=10^{4}$).""",
    """prompt-level percentile bootstrap intervals ($B=10^{4}$). Finally, one author listened \\emph{blind}
(opaque clip ids, sealed key, seeded prompt draw) to $8$ AudioCaps and $8$ music prompts at severity~2:
P and \\PFT{} singly at $10.24$\\,s, as pairs at both durations, and as music pairs. This is one listener, not
a perceptual study; it is reported only where it qualifies a scorer.""")
# 3.4 status
rep("""severity-1 dense control, crop, rank-scale, Holm and anchor analyses are \\emph{post-hoc}.""",
    """severity-1 dense control, crop, rank-scale, Holm and anchor analyses and the author listening are
\\emph{post-hoc}.""")
# 4.1: after the sweep paragraph (rho_real clause ends it)
rep("""$10$\\,s window without repeat-padding.""",
    f"""$10$\\,s window without repeat-padding. To the blinded listener the gain is audible at $10.24$\\,s
(\\PFT{{}} preferred {B['10.24s']['prefers_PFT']}/{n}, P mostly heard as noise) but not at $3.84$\\,s ({B['3.84s']['prefers_PFT']}/{n}; both heard as noise),
so the short-duration gain that CLAP resolves is below what one ear picks up.""")
# 4.2: after the music floor sentence "the 83%-pruned one did not (0.018)."
rep("""the $83\\%$-pruned one did not ($0.018$).""",
    f"""the $83\\%$-pruned one did not ($0.018$). The scorer's null is not a perceptual tie: the listener
preferred \\PFT{{}} on {C['prefers_PFT']}/{C['n_pairs']} music pairs and heard it as music on {C['sounds_like_music']['P+FT']['yes']}/{C['n_pairs']} (P {C['sounds_like_music']['P']['yes']}/{C['n_pairs']}), while neither
followed the long captions ($1.4$ and $2.5$ of $5$): musicality without alignment is invisible to a
text--audio scorer.""")
# discussion
rep("""At $83\\%$ pruning it is barely above chance at either duration ($0.020$ and $0.035$ above the floor),""",
    """At $83\\%$ pruning it is barely above chance at either duration ($0.020$ and $0.035$ above the floor; to
a listener, mostly noise),""")
# limitations
rep("""Primary inference rests on CLAP-family scorers; there is no human evaluation.""",
    """Primary inference rests on CLAP-family scorers; the only listening is one author's, blinded but
informal ($N=1$), and it disagrees with CLAP at $3.84$\\,s.""")
# page budget: trims that drop nothing a table does not carry
rep("""Finally, \\PFT{} is not restored to dense: at severity~1 and $10.24$\\,s the dense model leads P by $+0.099$~$\\ci{+0.058}{+0.140}$ and \\PFT{} by $+0.048$~$\\ci{-0.000}{+0.096}$; at severity~2 by $+0.055$ $\\ci{+0.021}{+0.088}$ at $10.24$\\,s ($+0.107$ at $3.84$\\,s).""",
    """Finally, \\PFT{} is not restored to dense: at $10.24$\\,s the dense model leads it by
$+0.048$~$\\ci{-0.000}{+0.096}$ at severity~1 and $+0.055$ $\\ci{+0.021}{+0.088}$ at severity~2 ($+0.107$ at $3.84$\\,s).""")
rep("""the gain in the first $3.84$\\,s equals the gain in the remainder: $D_{\\mathrm{early}}=+0.275$ vs.\\ $D_{\\mathrm{late}}=+0.273$, $T=-0.002$""",
    """the gain in the first $3.84$\\,s equals the gain in the remainder: $T=-0.002$""")
rep("""Sweep rows and severity-2 dense control registered after the primary result; severity-1 dense control post-hoc.}""", """Statuses as in Sec.~3.4.}""")
s = s.replace("\\documentclass{article}", MARK + "\n\\documentclass{article}", 1)
open(TEX, "w", encoding="utf-8").write(s); print("author listening integrated")
