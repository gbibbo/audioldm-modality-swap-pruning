#!/usr/bin/env python3
"""Studio idle guard: STOP this Lightning Studio when nobody is using it (Gabriel, 2026-09-05).

Why. The CPU Studio bills ~0.27 cr/h from the same credit pool as the GPU jobs (lifetime reading
2026-09-05: 99.18 cr spent, of which only 27.2 cr were jobs). Lightning's own auto-sleep (600 s) never
fired because an attached VS Code / Claude Code session counts as activity. This guard applies OUR
definition of idle and stops the Studio through the SDK.

Rule (evaluated every --poll seconds, default 60):
  idle_user   = no user prompt AND no agent tool call for --idle-min minutes (UserPromptSubmit / PostToolUse hooks
                touch LAST_ACTIVITY / LAST_AGENT)
  quiet_cpu   = the 1-min load average stayed below --load-max for the last --quiet-min minutes
  unprotected = no valid HOLD file (scripts/ops/with_hold.sh) and no protected process
                (job_watchdog.py, a Lightning job launch, scoring/verdict/tests, downloads)
  idle_user AND quiet_cpu AND unprotected  ->  log the decision, then Studio().stop()

Run with the cloudspace python (the only interpreter that has lightning_sdk):
  /home/zeus/miniconda3/envs/cloudspace/bin/python scripts/ops/studio_idle_guard.py --daemon   # start
  ... --check    one evaluation, no action, prints the decision (safe)
  ... --status   is the daemon running? when was the last user activity? any hold?
  ... --stop-now stop the Studio immediately (asks no questions; use at the end of a session)
  ... --hold 120 protect the Studio for 120 minutes (long CPU scoring/downloads)
  ... --release  drop the hold
The SessionStart hook starts the daemon if it is not running; the UserPromptSubmit hook refreshes
LAST_ACTIVITY. Log: artifacts/logs/studio_idle_guard.log (gitignored).
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, sys, time

HOME = os.path.expanduser("~")
STATE = os.path.join(HOME, ".cache", "studio_idle_guard")
LAST_ACTIVITY = os.path.join(STATE, "last_user_activity")
LAST_AGENT = os.path.join(STATE, "last_agent_activity")     # touched by the PostToolUse hook: the agent is working
HOLD = os.path.join(STATE, "hold")
PID = os.path.join(STATE, "guard.pid")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(ROOT, "artifacts", "logs", "studio_idle_guard.log")
PROTECT = re.compile(r"job_watchdog\.py|lightning job run|_verdict\.py|_score|score_verdict|floor_ceiling|"
                     r"secondary_metrics|run_research_tests|pytest|finelap_temporal|curl .*zenodo|py7zr|"
                     r"e3_shortft_trainer|studio_idle_guard\.py --stop-now", re.I)


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + " | " + msg
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def touch_activity():
    os.makedirs(STATE, exist_ok=True)
    with open(LAST_ACTIVITY, "w") as f:
        f.write(str(time.time()))


def last_activity_age_min():
    """minutes since the most recent of: a user prompt, a tool call by the agent (both hook-touched)."""
    ts = [os.path.getmtime(f) for f in (LAST_ACTIVITY, LAST_AGENT) if os.path.exists(f)]
    if not ts:
        return None
    return (time.time() - max(ts)) / 60.0


def hold_remaining_min():
    try:
        until = float(open(HOLD).read().strip())
    except Exception:
        return 0.0
    return max(0.0, (until - time.time()) / 60.0)


def protected_processes():
    hits = []
    for pidfile in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(pidfile, "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except Exception:
            continue
        if PROTECT.search(cmd) and "--check" not in cmd and "--status" not in cmd:
            hits.append(cmd.strip()[:120])
    return hits


def loadavg1():
    return float(open("/proc/loadavg").read().split()[0])


def daemon_running():
    try:
        pid = int(open(PID).read().strip())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def stop_studio(reason):
    log("STOPPING STUDIO: " + reason)
    try:
        from lightning_sdk import Studio
        s = Studio()
        log(f"Studio {s.name} status={s.status} machine={s.machine}; calling stop()")
        s.stop()
        log("stop() returned")
    except Exception as e:
        log(f"stop() FAILED: {e!r}")
        return False
    return True


def evaluate(a, load_hist):
    age = last_activity_age_min()
    hold = hold_remaining_min()
    prot = protected_processes()
    la = loadavg1(); load_hist.append((time.time(), la))
    cutoff = time.time() - a.quiet_min * 60
    load_hist[:] = [x for x in load_hist if x[0] >= cutoff]
    quiet = len(load_hist) >= max(1, int(a.quiet_min * 60 / a.poll) - 1) and all(x[1] < a.load_max for x in load_hist)
    idle_user = age is not None and age >= a.idle_min
    decision = idle_user and quiet and hold <= 0 and not prot
    return {"last_user_activity_min": None if age is None else round(age, 1), "idle_user": idle_user,
            "loadavg1": la, "quiet_cpu": quiet, "hold_remaining_min": round(hold, 1),
            "protected_processes": prot, "WOULD_STOP": decision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--status", action="store_true"); ap.add_argument("--stop-now", action="store_true")
    ap.add_argument("--hold", type=float, default=None, help="minutes"); ap.add_argument("--release", action="store_true")
    ap.add_argument("--touch", action="store_true", help="record user activity now (hook use)")
    ap.add_argument("--idle-min", type=float, default=45.0); ap.add_argument("--quiet-min", type=float, default=10.0)
    ap.add_argument("--load-max", type=float, default=0.8); ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args()
    os.makedirs(STATE, exist_ok=True)
    if a.touch:
        touch_activity(); return 0
    if a.hold is not None:
        with open(HOLD, "w") as f: f.write(str(time.time() + a.hold * 60))
        log(f"HOLD set for {a.hold:.0f} min"); return 0
    if a.release:
        if os.path.exists(HOLD): os.remove(HOLD)
        log("HOLD released"); return 0
    if a.status:
        print(json.dumps({"daemon_pid": daemon_running(), **evaluate(a, []), "log": LOG}, indent=1)); return 0
    if a.check:
        print(json.dumps(evaluate(a, []), indent=1)); return 0
    if a.stop_now:
        return 0 if stop_studio("--stop-now requested") else 1
    if a.daemon:
        if daemon_running():
            print("guard already running, pid", daemon_running()); return 0
        with open(PID, "w") as f: f.write(str(os.getpid()))
        if not os.path.exists(LAST_ACTIVITY): touch_activity()
        log(f"guard started pid {os.getpid()} idle_min={a.idle_min} quiet_min={a.quiet_min} load_max={a.load_max}")
        hist = []; last_report = 0.0
        while True:
            try:
                ev = evaluate(a, hist)
                if time.time() - last_report > 1800:
                    log("heartbeat " + json.dumps({k: ev[k] for k in ("last_user_activity_min", "loadavg1", "hold_remaining_min")}) +
                        f" protected={len(ev['protected_processes'])}")
                    last_report = time.time()
                if ev["WOULD_STOP"]:
                    log("decision " + json.dumps(ev))
                    if stop_studio(f"idle {ev['last_user_activity_min']} min, load {ev['loadavg1']}, no hold, no protected process"):
                        time.sleep(300)
            except Exception as e:
                log(f"guard loop error: {e!r}")
            time.sleep(a.poll)
    ap.print_help(); return 2


if __name__ == "__main__":
    sys.exit(main())
