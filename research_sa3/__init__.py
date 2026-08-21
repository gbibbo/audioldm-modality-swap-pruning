"""Stable Audio 3 analysis code (docs/sa3/analysis_protocol_rq1_rq2.md).

Runs in the separate `.venv-sa3` (torch 2.7.1, stable-audio-3 @ a0b57f54), never in the
frozen AudioLDM `.venv`. Nothing here trains, prunes for deployment, or recovers a model; it
measures (Step 0: verification; later: fields, probes, metrics).
"""
