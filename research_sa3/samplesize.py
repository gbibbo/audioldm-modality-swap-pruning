"""Pilot-based sample-size rules (protocol section 2.3), pure decision functions.

N_main: pre-registered ladder N_j = 16*2^j, truncated at the pilot size. For each rung and each
criterion X, draw B pairs of prompt subsamples of size N_j, compute the greedy sets on each and
the disagreement d_X(k) = |R_X^a(k) △ R_X^b(k)| / 2. N_main = smallest rung at which the 95th
percentile of d_X(k) is 0 blocks for EVERY X and EVERY k. No rung qualifying => underpowered
(no curve is fitted). n_u: same rule under probe bootstrap, ladder {8,16,32,...}.

Here we implement the RULE (the pure selection given the disagreement arrays); the bootstrap
draw + greedy recomputation is orchestration in rq1_rq2_stage1.py.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import math


def ladder(pilot_size: int, base: int = 16) -> List[int]:
    """16, 32, 64, ... truncated at pilot_size (rungs strictly greater than pilot_size dropped;
    the last rung is capped at pilot_size only if pilot_size is itself a rung)."""
    out, j = [], 0
    while True:
        n = base * (2 ** j)
        if n > pilot_size:
            break
        out.append(n)
        j += 1
    return out


def n_u_ladder(max_probes: int, base: int = 8) -> List[int]:
    out, j = [], 0
    while True:
        n = base * (2 ** j)
        out.append(n)
        if n >= max_probes:
            break
        j += 1
    return out


def percentile(vals: List[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,100]); matches numpy 'linear' for our use."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    idx = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (idx - lo))


def choose_rung(
    disagreement: Dict[int, Dict[str, Dict[int, List[int]]]],  # rung -> criterion -> k -> [d over B pairs]
    ks=(2, 4, 6),
    q: float = 95.0,
) -> dict:
    """Return {'n_main': rung or None, 'trace': {rung: {crit: {k: p95}}}, 'qualifies': {rung: bool}}."""
    trace, qualifies = {}, {}
    chosen: Optional[int] = None
    for rung in sorted(disagreement):
        crit_ok = True
        rtrace = {}
        for crit in sorted(disagreement[rung]):
            ktrace = {}
            for k in ks:
                p = percentile(disagreement[rung][crit].get(k, []), q)
                ktrace[k] = p
                if not (p <= 0.0):
                    crit_ok = False
            rtrace[crit] = ktrace
        trace[rung] = rtrace
        qualifies[rung] = crit_ok
        if crit_ok and chosen is None:
            chosen = rung
    return {"n_main": chosen, "trace": trace, "qualifies": qualifies}
