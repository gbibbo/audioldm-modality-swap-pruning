#!/usr/bin/env python3
"""Reframe the author listening in the manuscript (Gabriel, 2026-09-04 16:36): present it as what we heard,
in the first person, as an addition -- not as a substitute for a panel and without repeated caveats -- and
point readers to the companion page where the same clips can be heard. Idempotent (marker
%% author-listening-v2). Numbers unchanged (configs/research/author_listening_1_result.json).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/integrate_author_listening_v2.py
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
MARK = "%% author-listening-v2"
s = open(TEX, encoding="utf-8").read()
if MARK in s:
    sys.exit("already applied")
def rep(a, b):
    global s
    pat = re.sub(r"\\\s+", r"\\s+", re.escape(a)); m = list(re.finditer(pat, s)); assert len(m) == 1, (len(m), a[:80]); s = s[:m[0].start()] + b + s[m[0].end():]
# abstract: an addition, not a lack
rep("""lacking a matched dense fine-tuned control and a listening panel (one author's blinded listening agrees at the native point, not at $3.84$\\,s), our claims concern evaluation, not mechanism.""",
    """our own blind listening agrees at the native point and hears both checkpoints as noise at $3.84$\\,s; lacking a
matched dense fine-tuned control, our claims concern evaluation, not mechanism.""")
# 3.3 design sentence -> what we did, with the page
rep("""Finally, one author listened \\emph{blind} (opaque ids, sealed key, seeded draw) to $8$ AudioCaps and $8$ music prompts at severity~2: one listener, not a perceptual study.""",
    """We also listened, blind (opaque ids, sealed key, seeded draw), to $8$ AudioCaps and $8$ music prompts at
severity~2; these clips, and more, are on the companion page (Sec.~6).""")
# 3.4 status list
rep("""Holm and anchor analyses and the author listening are \\emph{post-hoc}.""",
    """Holm and anchor analyses and our listening are \\emph{post-hoc}.""")
# 4.1
rep("""To the blinded listener the gain is audible at $10.24$\\,s (\\PFT{} preferred 6/8, P mostly heard as noise) but not at $3.84$\\,s (0/8; both heard as noise), so the short gain CLAP resolves is below what one ear picks up.""",
    """Listening blind, we found the gain plainly audible at $10.24$\\,s (\\PFT{} preferred on 6/8 prompts; P mostly
noise) but not at $3.84$\\,s (0/8: both checkpoints sound like noise), so the short gain CLAP resolves is one
the ear does not.""")
# 4.2
rep("""The scorer's null is not a perceptual tie: the listener preferred \\PFT{} on 8/8 music pairs and heard it as music on 5/8 (P 1/8), though neither followed the long captions ($1.4$ and $2.5$ of $5$); musicality without alignment is invisible to a text--audio scorer.""",
    """To our ears the music null is no tie: we preferred \\PFT{} on 8/8 music pairs and heard it as music on 5/8
(P on 1/8), while neither followed the long captions ($1.4$ and $2.5$ of $5$); musicality without alignment
is invisible to a text--audio scorer.""")
# discussion aside
rep("""($0.020$ and $0.035$ above the floor; to a listener, mostly noise)""", """($0.020$ and $0.035$ above the floor; to our ears, mostly noise)""")
# limitations: one plain clause, no drama
rep("""Primary inference rests on CLAP-family scorers; the only listening is one author's, blinded but informal ($N=1$), and it disagrees with CLAP at $3.84$\\,s.""",
    """Primary inference rests on CLAP-family scorers; our listening is informal and disagrees with CLAP at $3.84$\\,s.""")
# conclusion: name the page
rep("""none on held-out music. Audio: \\url{gbibbo.github.io/audioldm-modality-swap-pruning}.""",
    """none on held-out music. Listen: \\url{gbibbo.github.io/audioldm-modality-swap-pruning}.""")
s = s.replace("\\documentclass{article}", MARK + "\n\\documentclass{article}", 1)
open(TEX, "w", encoding="utf-8").write(s); print("author listening reframed (v2)")
