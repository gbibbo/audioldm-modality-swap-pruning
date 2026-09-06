#!/usr/bin/env python3
"""REVIEWER2-FOLLOWUP-EXT — CPU scoring of the three fine-tuned checkpoints' eval WAVs (0 cr), so the EXT2x2 verdict
can run. It reads the WAVs directly from the settled job dirs (validating count=192 and sample length per duration,
pairing captions from the frozen 192-prompt manifest by prompt_index) and scores each cell as ONE seed-once fused-CLAP
call with the shuffled-caption floor -- the same convention r2_verdict.score_groups uses -- writing gout("longft"),
gout("denseft_s"), gout("denseft_n"). The strict manifest-sha emit is bypassed for denseft-s ac_native only, whose
generation manifest was never written (the watchdog cut that job mid-eval); its 192 = 170 from r2-denseft-s + 22 from
r2-denseft-s-tail, same checkpoint (sha recorded), same frozen GEN_SALT seeds. Provenance = each job's saved checkpoint
sha (trainer_report) + this script + the frozen prompt manifest.
Run: OPENBLAS_CORETYPE=Haswell .venv-metrics/bin/python scripts/research/r2_ext2x2_score.py
"""
import glob, json, os, re, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import soundfile as sf
import r2_verdict as V

JOBS = "/teamspace/jobs"
AC = json.load(open(V.AC192))["prompts"]
CAP = {p["prompt_index"]: p["caption"] for p in AC}
YT = {p["prompt_index"]: p["ytid"] for p in AC}
ALLPI = sorted(CAP)[:192]
FR = {"ac_short": 61472, "ac_native": 163872}
# system -> (file prefix, list of (gen dir, context) sources). denseft-s ac_native merges two dirs.
def gd(job, out): return f"{JOBS}/{job}/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/{out}/gen"
SRC = {
 "longft":    ("longft",  {"ac_short":[gd("r2-longft","r2_longft")], "ac_native":[gd("r2-longft","r2_longft")]}),
 "denseft_n": ("denseft", {"ac_short":[gd("r2-denseft-n","r2_denseft_n")], "ac_native":[gd("r2-denseft-n","r2_denseft_n")]}),
 "denseft_s": ("denseft", {"ac_short":[gd("r2-denseft-s","r2_denseft_s")],
                            "ac_native":[gd("r2-denseft-s-tail2","r2_denseft_s_tail"), gd("r2-denseft-s-tail","r2_denseft_s_tail"), gd("r2-denseft-s","r2_denseft_s")]}),
}


def collect(prefix, ctx, dirs):
    """return items [{caption,wav,prompt_index}] for all 192 prompts; earlier dirs win (tail overrides)."""
    picked = {}
    for d in dirs:
        for w in glob.glob(f"{d}/{prefix}_{ctx}_p*_r0.wav"):
            pi = int(re.search(rf"{prefix}_{ctx}_p(\d+)_r0", w).group(1))
            if pi not in picked:
                info = sf.info(w)
                if info.frames != FR[ctx] or info.samplerate != 16000:
                    raise SystemExit(f"bad WAV {w}: frames {info.frames} (expect {FR[ctx]}) sr {info.samplerate}")
                picked[pi] = w
    miss = [pi for pi in ALLPI if pi not in picked]
    if miss:
        raise SystemExit(f"{prefix}/{ctx}: missing {len(miss)} prompts {miss[:8]}{'...' if len(miss)>8 else ''}")
    return [{"caption": CAP[pi], "wav": picked[pi], "prompt_index": pi, "ytid": YT[pi]} for pi in ALLPI]


def ck_sha(out):
    for job in os.listdir(JOBS):
        tr = f"{JOBS}/{job}/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/{out}/trainer_report.json"
        if os.path.exists(tr):
            r = json.load(open(tr))["report"]; return r.get("saved_unet", {}).get("sha256"), r.get("commit")
    return None, None


def main():
    prov = {}
    for job, (prefix, ctxs) in SRC.items():
        groups = []
        for ctx, dirs in ctxs.items():
            items = collect(prefix, ctx, dirs)
            groups.append({"name": f"{prefix}__{ctx}", "items": items, "reps": 1})
        outp = V.gout(job)
        out_name = {"longft": "r2_longft", "denseft_s": "r2_denseft_s", "denseft_n": "r2_denseft_n"}[job]
        sha, commit = ck_sha(out_name)
        prov[job] = {"checkpoint_sha256": sha, "gen_commit": commit, "sources": {c: d for c, d in ctxs.items()}}
        V.score_groups(groups, outp, {"ext2x2_provenance": prov[job], "note": "CPU rescoring of EXT2x2 eval WAVs (r2_ext2x2_score.py); floors = shuffled captions, seed-once fused-CLAP rev 365dea6e"})
        print(f"== {job}: scored {[g['name'] for g in groups]} -> {outp}")
    json.dump(prov, open("artifacts/icassp_gate0/_score_tmp/ext2x2_scoring_provenance.json", "w"), indent=1)
    print("provenance ->", "artifacts/icassp_gate0/_score_tmp/ext2x2_scoring_provenance.json")


if __name__ == "__main__":
    main()
