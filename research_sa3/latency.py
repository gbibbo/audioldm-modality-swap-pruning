"""Wall-clock generation latency (protocol section 4.2), warm-up + repeats + cuda.synchronize.

Reports median and dispersion (IQR) so the latency-matched comparator (nearest measured latency)
can be built after the smoke. Also a nearest-latency comparator helper."""
from __future__ import annotations
from typing import Callable, Dict, List
import time
import torch


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn: Callable, device="cpu", warmup: int = 1, reps: int = 3) -> Dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        _sync(device); t0 = time.perf_counter()
        fn()
        _sync(device); ts.append(time.perf_counter() - t0)
    ts.sort()
    n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    return {"median_s": med, "min_s": ts[0], "max_s": ts[-1],
            "iqr_s": (ts[int(0.75 * (n - 1))] - ts[int(0.25 * (n - 1))]) if n >= 2 else 0.0,
            "reps": reps, "all_s": ts}


def nearest_latency_comparator(pruned_latency: float, dense_latencies: Dict[int, float]) -> Dict:
    """Given dense step->latency, return the dense step count with nearest latency (linear interp
    between the two bracketing step counts if needed)."""
    steps = sorted(dense_latencies)
    lats = [dense_latencies[s] for s in steps]
    # nearest by absolute latency
    best = min(steps, key=lambda s: abs(dense_latencies[s] - pruned_latency))
    # interpolated step (fractional) for reporting
    interp = None
    for a, b in zip(steps, steps[1:]):
        la, lb = dense_latencies[a], dense_latencies[b]
        if (la - pruned_latency) * (lb - pruned_latency) <= 0 and lb != la:
            interp = a + (pruned_latency - la) * (b - a) / (lb - la)
            break
    return {"nearest_step": best, "interp_step": interp, "pruned_latency_s": pruned_latency}
