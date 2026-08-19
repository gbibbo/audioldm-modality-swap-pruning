"""Paired audio-text saliency (P2 mean, P3 max) and the matched-budget P1/P2/P3
orchestration.

STATUS: orchestration implemented and CONTROL-MODEL tested (see
tests/research/test_taylor_saliency.py). It runs on whatever gated model and loss
closures it is handed and enforces the §5 matched gradient-evaluation budget. It
does NOT load the real pruned/L1 checkpoint or freeze any slot construction — that
is the M3B/M4 scientific run, blocked until the pilot protocol is frozen. P1 is
scientifically load-bearing; this path must pass `/auditar` before any real use.
"""
from .criteria import Criteria, compute_criteria

__all__ = ["Criteria", "compute_criteria"]
