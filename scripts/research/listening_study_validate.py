#!/usr/bin/env python3
"""Internal technical QA for the FROZEN listening study v1.1 (§12). CPU only.
No human responses. Verifies masking (incl. catch), counts, D3 bridge structure,
A/B balance, duration pairing, separation, catch design, playback enforcement,
public leakage, loudness (bundle + pair audit), payload shape, and timing.
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
FORBIDDEN = re.compile(r"recovered|pruned|dense|reconstr|catch|_alt10s|realref", re.I)
RESP_OVERHEAD_S = 8.0
MIN_SEP = DESIGN["min_pair_separation_target"]

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS" if ok else "FAIL"), name, "-", detail)

sep_min_global = 99
time_by = {}
# bridge structure: each bridge ytid rated by exactly 2 listeners
bridge_raters = {}

for c in CODES:
    pub = json.load(open(f"listening_study/public_manifests/{c}.json"))
    pv = PRIV["participants"][c]
    tr = pub["trials"]
    exp = [t for t in pv["trials"] if t["type"] == "experimental"]
    cat = [t for t in pv["trials"] if t["type"] == "catch"]
    nb = DESIGN["trials_by_listener"][c]
    check(f"{c} n_trials", len(tr) == nb["total"] == len(pv["trials"]), f"{len(tr)}=={nb['total']}")
    check(f"{c} catch=3", len(cat) == 3, f"{len(cat)}")
    check(f"{c} sev2 human absent", all(t["severity"] != "sev2" for t in pv["trials"]), "")

    # public masking: no 'type'; only allowed keys; forbidden tokens only in prompt_text
    pub_keys_ok = all(set(t.keys()) == {"trial_id", "prompt_text", "audio_A", "audio_B"} for t in tr)
    check(f"{c} public keys minimal", pub_keys_ok, "trial_id/prompt_text/audio_A/audio_B only")
    leak = []
    for t in tr:
        for k, v in t.items():
            if k == "prompt_text":
                continue
            if isinstance(v, str) and FORBIDDEN.search(v):
                leak.append((k, v))
    check(f"{c} public no-leak", not leak, str(leak[:2]))
    check(f"{c} A!=B urls", all(t["audio_A"] != t["audio_B"] for t in tr), "")

    # duration pairing for every rated sev1 prompt
    s1 = {}
    for t in pv["trials"]:
        if t["severity"] == "sev1":
            s1.setdefault(t["ytid"], set()).add(t["duration"])
            if t.get("bridge_role") == "bridge2":
                bridge_raters.setdefault(t["ytid"], set()).add(c)
    check(f"{c} sev1 both durations", all(v == {"short", "native"} for v in s1.values()), f"{len(s1)} prompts")

    # separation
    pos = {t["trial_id"]: i for i, t in enumerate(tr)}
    pp = {}
    for t in pv["trials"]:
        if t.get("pair_id"):
            pp.setdefault(t["pair_id"], []).append(pos[t["public_trial_id"]])
    seps = [abs(p[0] - p[1]) for p in pp.values() if len(p) == 2]
    smin = min(seps) if seps else 99
    sep_min_global = min(sep_min_global, smin)
    check(f"{c} pair separation", smin >= 3, f"min gap {smin} (target>={MIN_SEP})")

    catch_pub_ids = {x["public_trial_id"] for x in cat}
    cpos = sorted(pos[t["trial_id"]] for t in tr if t["trial_id"] in catch_pub_ids)
    check(f"{c} catch placement", all(p >= 2 for p in cpos) and
          all(cpos[k+1]-cpos[k] > 1 for k in range(len(cpos)-1)), str(cpos))

    # catch kinds
    kinds = sorted(t["catch_kind"] for t in cat)
    check(f"{c} catch kinds", kinds == ["identical_native", "identical_short", "matched_vs_unrelated_real"], str(kinds))
    # real-ref catch uses real refs with disjoint labels (via stim id realref)
    mr = [t for t in cat if t["catch_kind"] == "matched_vs_unrelated_real"][0]
    check(f"{c} realref catch is real", mr["A_kind"] == "real" and mr["B_kind"] == "real", "")

    # A/B balance
    recA = sum(1 for t in exp if t["recovered_side"] == "A")
    check(f"{c} AB balance", 0.35 <= recA/len(exp) <= 0.65, f"recA {recA}/{len(exp)}={recA/len(exp):.2f}")

    if os.path.isdir(AUDIO_DIR) and os.listdir(AUDIO_DIR):
        miss = [t["audio_A"] for t in tr if not os.path.exists(os.path.join("listening_study", t["audio_A"]))]
        miss += [t["audio_B"] for t in tr if not os.path.exists(os.path.join("listening_study", t["audio_B"]))]
        check(f"{c} audio paths", not miss, f"{len(miss)} missing")

    dur = 0.0
    for t in pv["trials"]:
        da = STIM[t["A_stim"]]["duration_s"] if t["A_stim"] in STIM else 10.242
        db = STIM[t["B_stim"]]["duration_s"] if t["B_stim"] in STIM else 10.242
        dur += da + db
    single = dur + RESP_OVERHEAD_S * len(pv["trials"])
    withreplay = dur * 1.35 + RESP_OVERHEAD_S * len(pv["trials"])
    time_by[c] = {"single_pass_min": round(single/60, 2), "moderate_replay_min": round(withreplay/60, 2)}

# ---- global ----
gexp = [(c, t) for c in CODES for t in PRIV["participants"][c]["trials"] if t["type"] == "experimental"]
grecA = sum(1 for _, t in gexp if t["recovered_side"] == "A")
check("global AB balance", 0.45 <= grecA/len(gexp) <= 0.55, f"{grecA}/{len(gexp)}={grecA/len(gexp):.3f}")
check("global pair separation >=3", sep_min_global >= 3, f"min {sep_min_global}")

# D3 bridge: 18 prompts, each rated by exactly 2 listeners (owner + bridge2)
n_bridge = len(bridge_raters)
check("bridge count = 18", n_bridge == DESIGN["bridge_prompts"], f"{n_bridge}")
# each bridge ytid should have its bridge2 rater distinct from owner (i.e., 2 total raters incl primary)
# verify total distinct listeners rating each bridge ytid == 2 (primary + one bridge2)
total_raters = {}
for c in CODES:
    for t in PRIV["participants"][c]["trials"]:
        if t["severity"] == "sev1":
            total_raters.setdefault(t["ytid"], set()).add(c)
bridge_ok = all(len(total_raters[y]) == 2 for y in bridge_raters)
nonbridge_single = all(len(v) == 1 for y, v in total_raters.items() if y not in bridge_raters)
check("bridge prompts rated by 2 listeners", bridge_ok, "")
check("non-bridge prompts rated by 1 listener", nonbridge_single, "")
# v1.2: unique-prompt estimand structure
check("80 unique sev1 prompts", len(total_raters) == 80, f"{len(total_raters)}")
n2 = sum(1 for v in total_raters.values() if len(v) == 2)
n1 = sum(1 for v in total_raters.values() if len(v) == 1)
check("18 bridge (2-rater) + 62 single-rater", n2 == 18 and n1 == 62, f"{n2}/{n1}")
check("study_version v1.2", DESIGN["study_version"] == "LSTUDY-2026-08-31-v1.2", DESIGN["study_version"])
an = open("scripts/research/listening_analyze.py").read()
check("analyzer unique-prompt primary (prompt is unit)",
      "unique-prompt" in an and "HUMAN-BOOTSTRAP|V1.2" in an and "!= 80" in an and "average within prompt" in an.lower(),
      "prompt-first estimator frozen; fails closed on != 80 unique prompts")
check("analyzer primary ignores catch trials",
      'type"] == "experimental"' in an and 'severity"] == "sev1"' in an, "primary uses experimental sev1 only")
rd = open("receiver/google_apps_script/README.md").read()
check("receiver README payload has no type / has completed_A",
      ("completed_A" in rd) and (", type," not in rd) and ("type, relevance" not in rd),
      "payload contract updated to v1.2")

# bridge selection reproducible
sev1_yt = [p["ytid"] for p in INV["prompts"]["sev1"]]
resel = sorted(sev1_yt, key=lambda y: hashlib.sha256(("LISTENING-STUDY|BRIDGE-SELECT|2026-08-31|"+y).encode()).hexdigest())
# design stores sorted selected; recompute selection then sort
sel = sorted(sorted(range(80), key=lambda i: hashlib.sha256(
    ("LISTENING-STUDY|BRIDGE-SELECT|2026-08-31|"+sev1_yt[i]).encode()).hexdigest())[:18])
check("bridge selection reproducible",
      sorted(sev1_yt[i] for i in sel) == DESIGN["bridge_selected_ytids"], "")

# private key not public; no tracking; offline fallback; playback enforcement
check("private key not in public dir",
      not os.path.exists("listening_study/public_manifests/listening_study_assignments_private.json"), "")
app = open("listening_study/app.js").read()
idx = open("listening_study/index.html").read().lower()
check("client offline fallback", "btn-download" in app and "clipboard" in app, "")
tracking = re.compile(r"document\.cookie|gtag\(|google-analytics|googletagmanager|analytics\.js|\bga\(")
check("client no tracking code", not tracking.search(app.lower()) and not tracking.search(idx), "")
# playback enforcement present
pb = ("canAnswer" in app and "completedA" in app and "completedB" in app
      and "if (!current.canAnswer) return" in app and "lockAnswers" in app)
check("client both-playback enforcement", pb, "answers locked until both clips completed")

# payload shape
synth = {"study_version": DESIGN["study_version"], "protocol_hash": "x", "participant_code": "P01",
         "assignment_hash": "x", "submission_uuid": "u",
         "responses": [{"trial_id": "P01_01", "relevance": 1, "quality": 0, "plays_A": 1,
                         "plays_B": 1, "completed_A": True, "completed_B": True,
                         "shown_ts": 1, "responded_ts": 2, "dwell_ms": 1}]}
req = {"study_version", "protocol_hash", "participant_code", "assignment_hash", "submission_uuid", "responses"}
check("payload shape", req.issubset(synth.keys()) and "email" not in json.dumps(synth).lower() and
      "type" not in synth["responses"][0], "no PII / no type leak")

# loudness bundle + pair audit
if os.path.exists("configs/research/listening_study_bundle_manifest.json"):
    bm = json.load(open("configs/research/listening_study_bundle_manifest.json"))
    peaks = [f["copy_peak_dbfs"] for f in bm["files"].values()]
    lufs = [f["copy_lufs"] for f in bm["files"].values()]
    ge = max(abs(f["src_lufs"] + f["gain_db"] + 36.0) for f in bm["files"].values())
    check("bundle applied-gain exact to -36 LUFS", ge < 1e-6, f"max err {ge:.2e}")
    check("bundle peak <= -1 dBFS", max(peaks) <= -1.0 + 1e-3, f"max {max(peaks):.2f}")
    within1 = sum(1 for l in lufs if abs(l+36) <= 1.0)
    check("bundle >=97% within +/-1 dB re-measured", within1/len(lufs) >= 0.97,
          f"{within1}/{len(lufs)}; range [{min(lufs):.2f},{max(lufs):.2f}]")
    check("bundle no problems", not bm["problems"], str(bm["problems"][:2]))
    check("bundle files on disk", not [h for h in bm["files"] if not os.path.exists(os.path.join(AUDIO_DIR, h))], "")
if os.path.exists("configs/research/listening_loudness_pair_audit.json"):
    pa = json.load(open("configs/research/listening_loudness_pair_audit.json"))
    check("loudness pair audit negligible", pa["verdict"].startswith("NEGLIGIBLE"),
          f"max|signed mean|={pa['max_abs_signed_mean_across_strata']} dB")
else:
    print("NOTE bundle/pair-audit not built yet")

print("\n=== est completion time (min) ===")
for c in CODES:
    print(f"  {c}: single {time_by[c]['single_pass_min']}  moderate-replay {time_by[c]['moderate_replay_min']}")
worst = max(t["moderate_replay_min"] for t in time_by.values())
check("time <= 20 min (moderate replay)", worst <= 20.0, f"worst {worst} min")

npass = sum(1 for _, ok, _ in results if ok)
out = {"artifact": "listening_study_validation", "study_version": DESIGN["study_version"],
       "n_checks": len(results), "n_pass": npass, "n_fail": len(results)-npass,
       "all_pass": npass == len(results), "pair_separation_min": sep_min_global,
       "time_estimate_min": time_by,
       "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in results]}
json.dump(out, open("configs/research/listening_study_validation.json", "w"), indent=2, sort_keys=True)
print(f"\n{npass}/{len(results)} checks pass. all_pass={out['all_pass']}")
