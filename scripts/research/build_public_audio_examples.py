#!/usr/bin/env python3
"""Build the PUBLIC audio-examples companion site (CPU only, no GPU, no new generation).

Non-interactive illustrative page: deterministic hash-selected representative model
outputs. NO scores are used for selection (anti-cherry-picking). Original generated WAVs
only (NOT the loudness-normalized listening copies); lossless FLAC (bit-identical samples).

Selection: rank a section's candidate ytids by SHA256(NAMESPACE | section | ytid), take
first N. NAMESPACE = 'PUBLIC-AUDIO-EXAMPLES|2026-09-01'.

Sections:
  A  severity-1 AudioCaps: 4 prompts x {3.84 s,10.24 s} x {pruned,recovered}   (Arm-D 80 pool)
  B  severity-2 AudioCaps: 4 prompts x {3.84 s,10.24 s} x {pruned2_A,recovered2} (xsev 192 pool)
  C  music context (severity-2 only): 2 prompts x 3.84 s x {pruned2_A,recovered2} (xsev music 64 pool)
     (severity-1 music audio was not retained on disk — only scores — so it is omitted; no new gen.)
  Dense reference omitted (dense@10.24 s coverage is incomplete: 73/80).

Outputs:
  public_examples/{index.html, .nojekyll, robots.txt, audio/*.flac}   (deploy tree; gitignored under artifacts? no — see note)
  configs/research/public_audio_examples_manifest.json                 (committed provenance)

Run: OPENBLAS_CORETYPE=Haswell .venv-loudness/bin/python scripts/research/build_public_audio_examples.py
"""
import json, os, io, hashlib, html, shutil
import numpy as np, soundfile as sf

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
NS = "PUBLIC-AUDIO-EXAMPLES|2026-09-01"
OUT = os.path.join(ROOT, "public_examples")
AUDIO = os.path.join(OUT, "audio")
MANIFEST = os.path.join(ROOT, "configs/research/public_audio_examples_manifest.json")
REPO_URL = "https://github.com/gbibbo/audioldm-modality-swap-pruning"

ARMD = "/teamspace/jobs/reversal-armd-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_armd_gen"
V11  = "/teamspace/jobs/reversal-v11-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_v1_1_gen"
XSEV = "/teamspace/jobs/reversal-xsev-gen-1/artifacts/audioldm-modality-swap-pruning/artifacts/icassp_gate0/reversal_xsev_gen"


def rows(p, key):
    d = json.load(open(p))
    r = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return {row[key]: row for row in r}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rank(section, ytids, n):
    return sorted(ytids, key=lambda y: hashlib.sha256((NS + "|" + section + "|" + y).encode()).hexdigest())[:n]


def flac_lossless(src_wav, dst_flac):
    x, sr = sf.read(src_wav, dtype="int16")
    buf = io.BytesIO(); sf.write(buf, x, sr, format="FLAC", subtype="PCM_16"); buf.seek(0)
    y, sr2 = sf.read(buf, dtype="int16")
    if not (sr == sr2 and np.array_equal(x, y)):
        raise SystemExit(f"FLAC not lossless for {src_wav}")
    with open(dst_flac, "wb") as f:
        f.write(buf.getvalue())
    return sr, len(x)


def main():
    os.chdir(ROOT)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(AUDIO)

    # ---- candidate pools (frozen manifests; ids only) ----
    subset = json.load(open("configs/research/op_duration_discriminator_1_subset.json"))["prompts"]
    subA_by_ytid = {p["ytid"]: p for p in subset}                 # sev1 pool (80)
    xsev_ac = json.load(open("configs/research/xsev_audiocaps_manifest.json"))["prompts"]
    xsevAC_by_ytid = {p["ytid"]: p for p in xsev_ac}              # sev2 AC pool (192)
    xsev_mus = json.load(open("configs/research/xsev_music_manifest.json"))["prompts"]
    xsevMus_by_ytid = {p["ytid"]: p for p in xsev_mus}           # sev2 music pool (64)

    # ---- gen manifests -> filename + source sha ----
    v11_rec = rows(os.path.join(V11, "gen_manifest_p1_recovered.json"), "prompt_index")
    v11_pru = rows(os.path.join(V11, "gen_manifest_p1_pruned_ema_reconstructed.json"), "prompt_index")
    armd_rec = rows(os.path.join(ARMD, "gen_manifest_p1_recovered.json"), "subset_prompt_index")
    armd_pru = rows(os.path.join(ARMD, "gen_manifest_p1_pruned_ema_reconstructed.json"), "subset_prompt_index")
    x_rec_n = rows(os.path.join(XSEV, "gen_manifest_recovered2_ac_native.json"), "prompt_index")
    x_pru_n = rows(os.path.join(XSEV, "gen_manifest_pruned2_A_ac_native.json"), "prompt_index")
    x_rec_s = rows(os.path.join(XSEV, "gen_manifest_recovered2_ac_short.json"), "prompt_index")
    x_pru_s = rows(os.path.join(XSEV, "gen_manifest_pruned2_A_ac_short.json"), "prompt_index")
    x_rec_m = rows(os.path.join(XSEV, "gen_manifest_recovered2_music.json"), "prompt_index")
    x_pru_m = rows(os.path.join(XSEV, "gen_manifest_pruned2_A_music.json"), "prompt_index")

    examples = {"A": [], "B": [], "C": []}
    files = []

    def emit(section, eid, ytid, caption, cells):
        """cells: list of (label_dur, label_sys, system_name, job_dir, gen_row, dur_expected)."""
        rec = {"example_id": eid, "section": section, "ytid": ytid, "caption": caption, "audio": []}
        for dur_lab, sys_lab, sysname, jobdir, row, dexp in cells:
            base = os.path.basename(row["wav"])
            src = os.path.join(jobdir, base)
            if not os.path.exists(src):
                raise SystemExit(f"missing source WAV {src}")
            src_sha = sha(src)
            if row.get("wav_sha256") and row["wav_sha256"] != src_sha:
                raise SystemExit(f"SHA mismatch {src}")
            fid = f"{eid}_{dur_lab}_{sys_lab}".lower().replace(".", "p")
            dst = os.path.join(AUDIO, fid + ".flac")
            sr, n = flac_lossless(src, dst)
            dur = round(n / sr, 3)
            if abs(dur - dexp) > 0.05:
                raise SystemExit(f"duration {dur}!={dexp} for {src}")
            flac_sha = sha(dst)
            entry = {"file": "audio/" + fid + ".flac", "duration_label": dur_lab, "system_label": sys_lab,
                     "source_system": sysname, "duration_s": dur, "sr": sr,
                     "source_wav_basename": base, "source_sha256": src_sha, "flac_sha256": flac_sha,
                     "lossless_roundtrip": True}
            rec["audio"].append(entry)
            files.append(entry)
        examples[section].append(rec)

    # ---- Section A: sev1 AudioCaps ----
    A_ids = rank("sev1_audiocaps", list(subA_by_ytid), 4)
    for i, y in enumerate(A_ids, 1):
        p = subA_by_ytid[y]; si = p["subset_prompt_index"]; vi = p["v1_1_prompt_index"]
        emit("A", f"a{i}", y, p["caption"], [
            ("3.84s", "Pruned", "p1_pruned_ema_reconstructed", V11, v11_pru[vi], 3.84),
            ("3.84s", "Recovered", "p1_recovered", V11, v11_rec[vi], 3.84),
            ("10.24s", "Pruned", "p1_pruned_ema_reconstructed", ARMD, armd_pru[si], 10.24),
            ("10.24s", "Recovered", "p1_recovered", ARMD, armd_rec[si], 10.24),
        ])

    # ---- Section B: sev2 AudioCaps ----
    B_ids = rank("sev2_audiocaps", list(xsevAC_by_ytid), 4)
    for i, y in enumerate(B_ids, 1):
        p = xsevAC_by_ytid[y]; idx = p["prompt_index"]
        emit("B", f"b{i}", y, p["caption"], [
            ("3.84s", "Pruned", "pruned2_A", XSEV, x_pru_s[idx], 3.84),
            ("3.84s", "Recovered", "recovered2", XSEV, x_rec_s[idx], 3.84),
            ("10.24s", "Pruned", "pruned2_A", XSEV, x_pru_n[idx], 10.24),
            ("10.24s", "Recovered", "recovered2", XSEV, x_rec_n[idx], 10.24),
        ])

    # ---- Section C: sev2 music ----
    C_ids = rank("sev2_music", list(xsevMus_by_ytid), 2)
    for i, y in enumerate(C_ids, 1):
        p = xsevMus_by_ytid[y]; idx = p["prompt_index"]
        emit("C", f"c{i}", y, p["caption"], [
            ("3.84s", "Pruned", "pruned2_A", XSEV, x_pru_m[idx], 3.84),
            ("3.84s", "Recovered", "recovered2", XSEV, x_rec_m[idx], 3.84),
        ])

    # ---- provenance manifest ----
    man = {
        "artifact": "public_audio_examples_manifest", "namespace": NS,
        "selection_rule": "rank ytid by SHA256(namespace|section|ytid) ascending, take first N; NO outcome/score used",
        "candidate_pools": {
            "sev1_audiocaps": {"source": "op_duration_discriminator_1_subset.json", "n": len(subA_by_ytid),
                                "ytids_sorted": sorted(subA_by_ytid)},
            "sev2_audiocaps": {"source": "xsev_audiocaps_manifest.json", "n": len(xsevAC_by_ytid),
                                "ytids_sorted": sorted(xsevAC_by_ytid)},
            "sev2_music": {"source": "xsev_music_manifest.json", "n": len(xsevMus_by_ytid),
                            "ytids_sorted": sorted(xsevMus_by_ytid)},
        },
        "selection_hashes": {
            sec: {y: hashlib.sha256((NS + "|" + sec + "|" + y).encode()).hexdigest()
                  for y in ids}
            for sec, ids in (("sev1_audiocaps", A_ids), ("sev2_audiocaps", B_ids), ("sev2_music", C_ids))
        },
        "selected": {"sev1_audiocaps": A_ids, "sev2_audiocaps": B_ids, "sev2_music": C_ids},
        "omitted": {"sev1_music": "audio not retained on disk (only scores); no new generation",
                     "dense_reference": "dense@10.24s coverage incomplete (73/80); omitted"},
        "n_audio_files": len(files),
        "examples": examples,
    }
    payload = json.dumps(man, indent=2, sort_keys=True)
    man["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json.dump(man, open(MANIFEST, "w"), indent=2, sort_keys=True)

    # ---- static page ----
    write_page(examples)
    open(os.path.join(OUT, ".nojekyll"), "w").write("")
    open(os.path.join(OUT, "robots.txt"), "w").write("User-agent: *\nDisallow: /\n")
    print(f"selected A={A_ids} B={B_ids} C={C_ids}")
    print(f"audio files: {len(files)} | manifest self_sha256 {man['self_sha256'][:16]}")
    print("out:", OUT)


def audio_tag(e):
    return ('<div class="sys"><span class="sys-label">%s</span>'
            '<audio controls preload="none" src="%s"></audio></div>') % (html.escape(e["system_label"]), e["file"])


def dur_block(rec, dur):
    cells = [e for e in rec["audio"] if e["duration_label"] == dur]
    return ('<div class="dur"><h4>%s</h4>%s</div>' %
            (dur.replace("p", ".") if "p" in dur else dur, "".join(audio_tag(e) for e in cells)))


def card(rec, durs):
    blocks = "".join(dur_block(rec, d) for d in durs)
    return ('<article class="card"><p class="prompt">%s</p><div class="grid">%s</div></article>'
            % (html.escape(rec["caption"]), blocks))


def write_page(examples):
    A = "".join(card(r, ["3.84s", "10.24s"]) for r in examples["A"])
    B = "".join(card(r, ["3.84s", "10.24s"]) for r in examples["B"])
    C = "".join(card(r, ["3.84s"]) for r in examples["C"])
    page = PAGE_TMPL.format(A=A, B=B, C=C, repo=REPO_URL)
    open(os.path.join(OUT, "index.html"), "w").write(page)


PAGE_TMPL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>AudioLDM Post-Pruning Recovery — Audio Examples</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1b2130;--muted:#5c6675;--line:#dde2ea;--accent:#2b5cb8;--chip:#eef2f8;}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--card:#1a1f27;--ink:#e7ebf1;--muted:#9aa4b2;--line:#2c333d;--accent:#6fa0ec;--chip:#222a35;}}}}
*{{box-sizing:border-box}}
html,body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:32px 18px 72px}}
h1{{font-size:1.7rem;margin:0 0 .15em;text-wrap:balance}}
.sub{{color:var(--muted);font-size:1.05rem;margin:0 0 1.3em}}
h2{{font-size:1.25rem;margin:1.9em 0 .2em;border-bottom:2px solid var(--line);padding-bottom:.25em}}
h2 .ctx{{color:var(--muted);font-weight:400;font-size:.95rem}}
.note{{background:var(--chip);border:1px solid var(--line);border-radius:10px;padding:12px 16px;color:var(--ink);margin:1em 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}}
.prompt{{font-size:1.06rem;font-weight:600;margin:.1em 0 .8em}}
.prompt::before{{content:"\\201C"}} .prompt::after{{content:"\\201D"}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.dur h4{{margin:.1em 0 .5em;font-size:.95rem;letter-spacing:.02em}}
.sys{{display:flex;flex-direction:column;gap:4px;margin-bottom:12px}}
.sys-label{{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
audio{{width:100%}}
.section-c .grid{{grid-template-columns:1fr}}
footer{{margin-top:2.5em;border-top:1px solid var(--line);padding-top:1em;color:var(--muted);font-size:.92rem}}
footer h3{{color:var(--ink);font-size:1.05rem;margin:.4em 0}}
a{{color:var(--accent)}}
@media (max-width:600px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<h1>AudioLDM Post-Pruning Recovery</h1>
<p class="sub">Representative Audio Examples</p>
<div class="note">These examples are illustrative companion material. Quantitative conclusions are based
on the complete evaluation sets, not on the examples shown here. Examples were selected deterministically
by a hash of their identifiers, without using any evaluation score, to avoid cherry-picking.</div>

<p>Each card shows one text prompt and the audio generated by the <b>Pruned</b> and <b>Recovered</b>
models at two clip lengths (<b>3.84&nbsp;s</b> and <b>10.24&nbsp;s</b>). Two pruning severities are shown,
plus a separate music context. Labels indicate the system and clip length; listen and compare.</p>

<h2>1. Severity&nbsp;1 &mdash; <span class="ctx">AudioCaps evaluation context</span></h2>
{A}

<h2>2. Severity&nbsp;2 &mdash; <span class="ctx">AudioCaps evaluation context</span></h2>
{B}

<h2>3. Music context <span class="ctx">(severity&nbsp;2)</span></h2>
<div class="note">These examples illustrate the separate music evaluation context. They are not matched
to the AudioCaps prompts above.</div>
<div class="section-c">
{C}
</div>

<footer>
<h3>Selection and provenance</h3>
<p>Examples were selected deterministically by ranking each section's frozen candidate prompts on
<code>SHA256(namespace | section | ytid)</code> and taking the first few; the selection used only prompt
identifiers and ignored every evaluation score (CLAP, Human-CLAP, FineLAP, KL, PANN, FAD, FD). All audio
is taken directly from the frozen experimental model outputs (original generated waveforms, converted to
lossless FLAC with bit-identical samples — no normalization, resampling, or gain change). Quantitative
results in the paper use the complete evaluation populations, not these examples. Severity-1 music audio
and a dense reference are not shown because those files were not fully retained; this does not affect any
reported result. Source and selection details: <a href="{repo}">{repo}</a>.</p>
</footer>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
