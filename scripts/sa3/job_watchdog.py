#!/usr/bin/env python3
"""External cost watchdog for a Lightning job (Gabriel 2026-08-22: poll COST, not just status).

The lesson from sa3-smoke-t4-1: a job can finish its real compute + export and then idle-bill for
minutes (a lingering non-daemon thread keeps it "Running"). Status-only polling misses this. This
watchdog polls the job's settled `total_cost` and STOPS it the moment it crosses a hard cost ceiling
or a max runtime — so a hung job cannot silently drain credits. Run it (cloudspace python) right
after launching a training job:

    /home/zeus/miniconda3/envs/cloudspace/bin/python scripts/sa3/job_watchdog.py \
        --name sa3-L6-1 --max-cost 0.15 --max-minutes 20

Note (Gabriel): do NOT launch separate CPU-check Lightning jobs — free CPU checks belong in the
Studio/local venv. This watchdog is for the real GPU training jobs.
"""
from __future__ import annotations
import argparse, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-cost", type=float, required=True, help="hard cr ceiling; job is STOPPED when reached")
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--teamspace", default="general")
    ap.add_argument("--org", default="independentaudioresearch")
    a = ap.parse_args()
    from lightning_sdk import Job
    j = Job(name=a.name, teamspace=a.teamspace, org=a.org)
    term = ("completed", "failed", "stopped", "error", "crashed")
    t0 = time.time()
    run_t0 = None          # set when the job first reaches Running: max_minutes counts RUN time, not queue time
    stopped_by_us = None
    while True:
        status = str(j.status)
        s = status.lower()
        try:
            cost = float(j.total_cost or 0.0)
        except Exception:
            cost = float("nan")
        if run_t0 is None and "running" in s:
            run_t0 = time.time()
        elapsed_min = (time.time() - t0) / 60.0
        run_min = (time.time() - run_t0) / 60.0 if run_t0 is not None else 0.0
        print(f"[watchdog] {a.name} status={status} cost={cost:.4f} elapsed={elapsed_min:.1f}m run={run_min:.1f}m", flush=True)
        if any(t in s for t in term):
            print(f"[watchdog] job reached terminal state {status}; final cost {cost:.4f}")
            break
        if cost >= a.max_cost:
            stopped_by_us = f"cost {cost:.4f} >= max_cost {a.max_cost}"
        elif run_t0 is not None and run_min >= a.max_minutes:
            stopped_by_us = f"run {run_min:.1f}m >= max_minutes {a.max_minutes} (queue time excluded)"
        if stopped_by_us:
            print(f"[watchdog] STOPPING {a.name}: {stopped_by_us}", flush=True)
            try:
                j.stop()
            except Exception as e:
                print(f"[watchdog] stop() error: {e}")
            time.sleep(10)
            print(f"[watchdog] post-stop status={j.status} cost={float(j.total_cost or 0.0):.4f}")
            break
        time.sleep(a.poll_seconds)
    print(f"WATCHDOG_DONE name={a.name} final_status={j.status} final_cost={float(j.total_cost or 0.0):.4f} "
          f"killed={bool(stopped_by_us)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
