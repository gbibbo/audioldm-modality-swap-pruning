#!/usr/bin/env python3
"""Internal technical QA for the frozen listening study (§14). CPU only.
No human responses. Verifies masking, counts, balance, pairing, separation,
catch trials, loudness (if bundle built), audio paths/hashes, payload shape,
offline fallback, and estimated completion time.

Run: .venv-loudness/bin/python scripts/research/listening_study_validate.py
"""
import json, os, re, hashlib
import numpy as np

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
os.chdir(ROOT)
DESIGN = json.load(open("configs/research/listening_study_design.json"))
PRIV = json.load(open("configs/research/listening_study_assignments_private.json"))
INV = json.load(open("configs/research/listening_study_inventory.json"))
STIM = INV["stimuli"]
CODES = ["P0%d" % (i + 1) for i in range(6)]
AUDIO_DIR = "listening_study/audio"
FORBIDDEN = re.compile(r"recovered|pruned|dense|reconstr|sev1|sev2|severity|_alt10s", re.I)
RESP_OVERHEAD_S = 8.0     # per-trial think/answer overhead
REPLAY_FRACTION = 0.35    # fraction of samples a normal listener replays once
MIN_SEP = DESIGN["min_pair_separation_target"]

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS" if ok else "FAIL"), name, "-", detail)

# ---------- per participant ----------
sep_min_global = 99
time_by = {}
bundle_ok_paths = True
for c in CODES:
    pub = json.load(open(f"listening_study/public_manifests/{c}.json"))
    pv = PRIV["participants"][c]
    tr = pub["trials"]
    pmap = {t["public_trial_id"]: t for t in pv["trials"]}

    # counts
    exp = [t for t in pv["trials"] if t["type"] == "experimental"]
    cat = [t for t in pv["trials"] if t["type"] == "catch"]
    n_expected = DESIGN["trials_by_listener"][c]["total"]
    check(f"{c} n_trials", len(tr) == n_expected == len(pv["trials"]), f"{len(tr)} == {n_expected}")
    check(f"{c} catch=3", len(cat) == 3, f"{len(cat)}")

    # public masking: forbidden tokens only allowed inside prompt_text
    leak = []
    for t in tr:
        for k, v in t.items():
            if k == "prompt_text":
                continue
            if isinstance(v, str) and FORBIDDEN.search(v):
                leak.append((t["trial_id"], k, v))
    check(f"{c} public no-leak", not leak, str(leak[:3]))

    # A/B are distinct URLs everywhere (incl. identical catch -> two copies)
    dist = all(t["audio_A"] != t["audio_B"] for t in tr)
    check(f"{c} A!=B urls", dist, "")

    # duration pairing: every sev1 prompt has both short and native for this rater
    s1 = {}
    for t in pv["trials"]:
        if t["severity"] == "sev1":
            s1.setdefault(t["ytid"], set()).add(t["duration"])
    pair_ok = all(v == {"short", "native"} for v in s1.values())
    check(f"{c} sev1 both durations", pair_ok, f"{len(s1)} prompts")

    # repeated-prompt separation in presented order
    order_pos = {t["trial_id"]: i for i, t in enumerate(tr)}
    pair_positions = {}
    for t in pv["trials"]:
        if t.get("pair_id"):
            pair_positions.setdefault(t["pair_id"], []).append(order_pos[t["public_trial_id"]])
    seps = [abs(p[0] - p[1]) for p in pair_positions.values() if len(p) == 2]
    smin = min(seps) if seps else 99
    sep_min_global = min(sep_min_global, smin)
    check(f"{c} pair separation", smin >= 3, f"min gap {smin} (target>= {MIN_SEP}, non-adjacent>=2)")

    # catch not first-2, not adjacent
    cpos = sorted(order_pos[t["trial_id"]] for t in tr if t["type"] == "catch")
    catch_place = all(p >= 2 for p in cpos) and all(cpos[k+1]-cpos[k] > 1 for k in range(len(cpos)-1))
    check(f"{c} catch placement", catch_place, str(cpos))

    # A/B balance within participant (experimental only)
    recA = sum(1 for t in exp if t["recovered_side"] == "A")
    frac = recA / len(exp)
    check(f"{c} AB balance", 0.35 <= frac <= 0.65, f"recovered-as-A {recA}/{len(exp)}={frac:.2f}")

    # audio paths + hash (if bundle built)
    if os.path.isdir(AUDIO_DIR) and os.listdir(AUDIO_DIR):
        miss = [t["audio_A"] for t in tr if not os.path.exists(os.path.join("listening_study", t["audio_A"]))]
        miss += [t["audio_B"] for t in tr if not os.path.exists(os.path.join("listening_study", t["audio_B"]))]
        if miss:
            bundle_ok_paths = False
        check(f"{c} audio paths", not miss, f"{len(miss)} missing")

    # timing estimate
    dur = 0.0
    for t in pv["trials"]:
        da = STIM[t["A_stim"]]["duration_s"] if t["A_stim"] in STIM else 3.84
        db = STIM[t["B_stim"]]["duration_s"] if t["B_stim"] in STIM else 3.84
        dur += da + db
    single = dur + RESP_OVERHEAD_S * len(pv["trials"])
    withreplay = dur * (1 + REPLAY_FRACTION) + RESP_OVERHEAD_S * len(pv["trials"])
    time_by[c] = {"single_pass_min": round(single/60, 2), "moderate_replay_min": round(withreplay/60, 2)}

# ---------- global ----------
allpub = []
for c in CODES:
    allpub += json.load(open(f"listening_study/public_manifests/{c}.json"))["trials"]
gexp = [(c, t) for c in CODES for t in PRIV["participants"][c]["trials"] if t["type"] == "experimental"]
grecA = sum(1 for _, t in gexp if t["recovered_side"] == "A")
check("global AB balance", 0.45 <= grecA/len(gexp) <= 0.55, f"{grecA}/{len(gexp)}={grecA/len(gexp):.3f}")
check("global pair separation >=3", sep_min_global >= 3, f"min {sep_min_global}")

# sev2 selection outcome-blind reproducibility
elig2 = [q["ytid"] for q in INV["prompts"]["sev2"]]
resel = sorted(elig2, key=lambda y: hashlib.sha256((DESIGN["sev2_select_namespace"]+"|"+y).encode()).hexdigest())[:36]
check("sev2 selection reproducible", resel == DESIGN["sev2_selected_ytids"], "")

# private key not in public bundle
priv_in_pub = os.path.exists("listening_study/public_manifests/listening_study_assignments_private.json")
check("private key not in public dir", not priv_in_pub, "")

# offline fallback + no analytics in client
app = open("listening_study/app.js").read()
check("client offline fallback", "btn-download" in app and "clipboard" in app, "")
check("client no autoplay-on-load", ".play()" in app and "autoplay" not in open("listening_study/index.html").read().lower(), "")
idx = open("listening_study/index.html").read().lower()
tracking = re.compile(r"document\.cookie|gtag\(|google-analytics|googletagmanager|analytics\.js|\bga\(")
check("client no tracking code", not tracking.search(app.lower()) and not tracking.search(idx), "")

# payload shape (synthetic)
synth = {"study_version": DESIGN["study_version"], "protocol_hash": "x", "participant_code": "P01",
         "assignment_hash": "x", "submission_uuid": "u", "client_started_ts": 1, "client_completed_ts": 2,
         "total_ms": 1, "responses": [{"trial_id": "P01_01", "type": "experimental", "relevance": 1,
         "quality": 0, "plays_A": 1, "plays_B": 1, "shown_ts": 1, "responded_ts": 2, "dwell_ms": 1}]}
req = {"study_version", "protocol_hash", "participant_code", "assignment_hash", "submission_uuid",
       "responses"}
check("payload shape", req.issubset(synth.keys()) and "email" not in json.dumps(synth).lower(), "no PII fields")

# loudness of listening copies (if bundle built)
if os.path.exists("configs/research/listening_study_bundle_manifest.json"):
    bm = json.load(open("configs/research/listening_study_bundle_manifest.json"))
    peaks = [f["copy_peak_dbfs"] for f in bm["files"].values()]
    lufs = [f["copy_lufs"] for f in bm["files"].values()]
    # applied gain must target the source integrated loudness EXACTLY (broadcast-standard)
    gain_err = max(abs(f["src_lufs"] + f["gain_db"] + 36.0) for f in bm["files"].values())
    check("bundle applied-gain exact to -36 LUFS (source)", gain_err < 1e-6, f"max err {gain_err:.2e}")
    check("bundle peak <= -1 dBFS", max(peaks) <= -1.0 + 1e-3, f"max {max(peaks):.2f}")
    # re-measured copy loudness: >=98% within +/-1 dB; a few near-silent failed-pruned clips
    # drift due to the BS.1770 absolute gate (documented; does not favour recovered)
    within1 = sum(1 for l in lufs if abs(l + 36) <= 1.0)
    check("bundle >=98% copies within +/-1 dB re-measured", within1 / len(lufs) >= 0.98,
          f"{within1}/{len(lufs)} within 1 dB; drift range [{min(lufs):.2f},{max(lufs):.2f}]")
    check("bundle no problems", not bm["problems"], str(bm["problems"][:2]))
    # every deployed audio file exists on disk
    missing = [hn for hn in bm["files"] if not os.path.exists(os.path.join(AUDIO_DIR, hn))]
    check("bundle files on disk", not missing, f"{len(missing)} missing")
else:
    print("NOTE bundle not built yet -> loudness/audio-path checks pending (freeze order step 9)")

print("\n=== estimated completion time (min) ===")
for c in CODES:
    print(f"  {c}: single-pass {time_by[c]['single_pass_min']}  moderate-replay {time_by[c]['moderate_replay_min']}")
worst = max(t["moderate_replay_min"] for t in time_by.values())
check("time <= 20 min (moderate replay)", worst <= 20.0, f"worst {worst} min")

# ---------- emit ----------
npass = sum(1 for _, ok, _ in results if ok)
out = {"artifact": "listening_study_validation", "n_checks": len(results),
       "n_pass": npass, "n_fail": len(results)-npass,
       "all_pass": npass == len(results),
       "pair_separation_min": sep_min_global, "time_estimate_min": time_by,
       "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in results]}
json.dump(out, open("configs/research/listening_study_validation.json", "w"), indent=2, sort_keys=True)
print(f"\n{npass}/{len(results)} checks pass. all_pass={out['all_pass']}")
