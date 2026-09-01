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



def _cell_file(rec, system, dur):
    for a in rec["audio"]:
        if a["system_label"] == system and a["duration_label"] == dur:
            return a["file"]
    return ""


def _player(file, label):
    return ('<div class="player" data-label="%s">'
            '<audio controls preload="none" src="%s"></audio>'
            '<span class="cell-tag">%s</span></div>') % (html.escape(label), file, html.escape(label))


def _matrix_card(rec, n):
    P384 = _cell_file(rec, "Pruned", "3.84s"); R384 = _cell_file(rec, "Recovered", "3.84s")
    P1024 = _cell_file(rec, "Pruned", "10.24s"); R1024 = _cell_file(rec, "Recovered", "10.24s")
    return ('<article class="card" id="{sec}">'
            '<div class="card-head"><span class="eyebrow">Example {n:02d}</span>'
            '<span class="badge">AudioCaps</span></div>'
            '<p class="prompt">{cap}</p>'
            '<div class="matrix" role="group" aria-label="Audio comparison">'
            '<div class="mx-corner" aria-hidden="true"></div>'
            '<div class="mx-col">3.84 s</div><div class="mx-col">10.24 s</div>'
            '<div class="mx-row"><span class="chip chip-pruned">Pruned</span></div>'
            '<div class="mx-cell">{p1}</div><div class="mx-cell">{p2}</div>'
            '<div class="mx-row"><span class="chip chip-recovered">Recovered</span></div>'
            '<div class="mx-cell">{r1}</div><div class="mx-cell">{r2}</div>'
            '</div></article>').format(
                sec=rec["example_id"], n=n, cap=html.escape(rec["caption"]),
                p1=_player(P384, "Pruned · 3.84 s"), p2=_player(P1024, "Pruned · 10.24 s"),
                r1=_player(R384, "Recovered · 3.84 s"), r2=_player(R1024, "Recovered · 10.24 s"))


def _music_card(rec, n):
    P = _cell_file(rec, "Pruned", "3.84s"); R = _cell_file(rec, "Recovered", "3.84s")
    return ('<article class="card card-music">'
            '<div class="card-head"><span class="eyebrow">Example {n:02d}</span>'
            '<span class="badge badge-music">Music</span></div>'
            '<details class="cap"><summary class="prompt">{cap}</summary></details>'
            '<div class="duo" role="group" aria-label="Audio comparison">'
            '<div class="duo-cell"><span class="chip chip-pruned">Pruned · 3.84 s</span>{p}</div>'
            '<div class="duo-cell"><span class="chip chip-recovered">Recovered · 3.84 s</span>{r}</div>'
            '</div></article>').format(
                n=n, cap=html.escape(rec["caption"]),
                p=_player(P, "Pruned · 3.84 s (music)"), r=_player(R, "Recovered · 3.84 s (music)"))


def write_page(examples):
    A = "".join(_matrix_card(r, i) for i, r in enumerate(examples["A"], 1))
    B = "".join(_matrix_card(r, i) for i, r in enumerate(examples["B"], 1))
    C = "".join(_music_card(r, i) for i, r in enumerate(examples["C"], 1))
    page = (PAGE_HTML.replace("__A__", A).replace("__B__", B)
            .replace("__C__", C).replace("__REPO__", REPO_URL))
    open(os.path.join(OUT, "index.html"), "w").write(page)


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>AudioLDM Post-Pruning Recovery — Audio Examples</title>
<style>
:root{
  --bg:#f7f8fa; --bg2:#eef1f5; --card:#ffffff; --ink:#171b23; --ink2:#3a4150; --muted:#6a7280;
  --line:#e4e7ee; --line2:#d7dbe4; --accent:#4f46e5; --accent-ink:#4f46e5;
  --pruned:#5b6472; --pruned-bg:#eceef3; --recovered:#0d8f80; --recovered-bg:#e5f4f1;
  --sev1:#5a6270; --sev2:#7a5ea8;
  --shadow:0 1px 2px rgba(20,25,40,.04),0 6px 20px rgba(20,25,40,.05);
  --radius:14px; --maxw:1140px;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d1016; --bg2:#12161d; --card:#161b23; --ink:#e8ebf1; --ink2:#c2c8d4; --muted:#8b93a3;
  --line:#242a34; --line2:#2e3542; --accent:#8b8cf9; --accent-ink:#a5a6ff;
  --pruned:#9aa3b4; --pruned-bg:#1e2531; --recovered:#3fbfae; --recovered-bg:#132a29;
  --sev1:#9aa3b4; --sev2:#b49ce0;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.28);
}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
body{background:
  radial-gradient(1100px 380px at 50% -140px, color-mix(in srgb, var(--accent) 12%, transparent), transparent 70%),
  var(--bg);}
a{color:var(--accent-ink);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 22px}

/* nav */
.nav{position:sticky;top:0;z-index:20;backdrop-filter:saturate(1.4) blur(10px);
  background:color-mix(in srgb, var(--bg) 82%, transparent);border-bottom:1px solid var(--line)}
.nav-in{max-width:var(--maxw);margin:0 auto;padding:10px 22px;display:flex;align-items:center;gap:16px}
.brand{font-weight:640;letter-spacing:-.01em;font-size:.95rem;white-space:nowrap}
.brand b{color:var(--accent-ink)}
.nav-links{margin-left:auto;display:flex;gap:6px;overflow-x:auto;scrollbar-width:none}
.nav-links::-webkit-scrollbar{display:none}
.nav-links a{color:var(--ink2);font-size:.86rem;font-weight:540;padding:6px 12px;border-radius:999px;
  white-space:nowrap;border:1px solid transparent}
.nav-links a:hover{background:var(--bg2);text-decoration:none}
.nav-links a.active{color:var(--accent-ink);background:color-mix(in srgb,var(--accent) 12%,transparent);
  border-color:color-mix(in srgb,var(--accent) 26%,transparent)}

/* hero */
.hero{padding:52px 0 26px}
.eyebrow{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent-ink);font-weight:680}
.hero h1{font-size:clamp(2rem,4.4vw,3rem);line-height:1.05;letter-spacing:-.025em;margin:.32em 0 .18em;
  font-weight:720;text-wrap:balance}
.hero .lead{font-size:1.12rem;color:var(--ink2);max-width:58ch;margin:0}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 4px}
.chips span{font-size:.82rem;color:var(--ink2);background:var(--card);border:1px solid var(--line);
  padding:5px 11px;border-radius:999px;box-shadow:var(--shadow)}
.cta{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}
.btn{display:inline-flex;align-items:center;gap:7px;font-size:.9rem;font-weight:580;padding:9px 16px;
  border-radius:10px;border:1px solid var(--line2);background:var(--card);color:var(--ink);box-shadow:var(--shadow)}
.btn:hover{text-decoration:none;border-color:var(--accent);color:var(--accent-ink)}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn.primary:hover{color:#fff;filter:brightness(1.06)}

/* callout */
.callout{display:flex;gap:14px;align-items:flex-start;background:var(--card);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:12px;padding:16px 18px;margin:22px 0 6px;box-shadow:var(--shadow)}
.callout .dot{flex:0 0 auto;width:24px;height:24px;border-radius:7px;display:grid;place-items:center;
  background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent-ink);font-weight:800}
.callout h3{margin:.1em 0 .3em;font-size:1rem}
.callout p{margin:0;color:var(--ink2);font-size:.95rem}
.callout .sub{margin-top:.5em;font-size:.88rem;color:var(--muted)}

/* sections */
section.sec{padding:30px 0 6px;scroll-margin-top:64px}
.sec-head{display:flex;align-items:baseline;gap:12px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:6px}
.sec-head h2{font-size:1.32rem;letter-spacing:-.02em;margin:0;font-weight:660}
.sec-head .tier{font-size:.8rem;font-weight:640;padding:3px 9px;border-radius:999px}
.tier-1{color:var(--sev1);background:color-mix(in srgb,var(--sev1) 14%,transparent)}
.tier-2{color:var(--sev2);background:color-mix(in srgb,var(--sev2) 16%,transparent)}
.sec-head .ctx{margin-left:auto;color:var(--muted);font-size:.86rem}

/* cards */
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  padding:18px 20px;margin:16px 0;box-shadow:var(--shadow)}
.card-head{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.card-head .eyebrow{color:var(--muted);letter-spacing:.1em}
.badge{margin-left:auto;font-size:.72rem;letter-spacing:.04em;color:var(--ink2);background:var(--bg2);
  border:1px solid var(--line);padding:3px 9px;border-radius:999px}
.badge-music{color:var(--sev2)}
.prompt{font-size:1.08rem;font-weight:560;color:var(--ink);margin:.1em 0 1em;line-height:1.5;max-width:70ch}
.prompt::before{content:"\201C"} .prompt::after{content:"\201D"}

/* comparison matrix: systems=rows, durations=cols */
.matrix{display:grid;grid-template-columns:minmax(96px,auto) 1fr 1fr;gap:12px;align-items:center}
.mx-col{font-size:.82rem;font-weight:640;color:var(--muted);text-align:center;letter-spacing:.02em}
.mx-row{display:flex}
.chip{display:inline-flex;align-items:center;gap:7px;font-size:.86rem;font-weight:620;padding:5px 11px;
  border-radius:8px;border:1px solid var(--line2)}
.chip::before{content:"";width:9px;height:9px;border-radius:3px;background:currentColor;opacity:.9}
.chip-pruned{color:var(--pruned);background:var(--pruned-bg)}
.chip-recovered{color:var(--recovered);background:var(--recovered-bg)}
.cell-tag{display:none}

/* duo (music) */
.duo{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.duo-cell{display:flex;flex-direction:column;gap:8px}

/* player */
.player{--pc:var(--muted)}
.mx-cell .player, .duo-cell .player{}
.player{background:var(--bg2);border:1px solid var(--line);border-radius:11px;padding:8px 10px}
.mx-cell:nth-child(5) .player,.mx-cell:nth-child(6) .player{border-left:3px solid var(--pruned)}
.mx-cell:nth-child(8) .player,.mx-cell:nth-child(9) .player{border-left:3px solid var(--recovered)}
.player.enhanced{display:flex}
.pctl{display:flex;align-items:center;gap:10px;width:100%}
.pp{flex:0 0 auto;width:38px;height:38px;border-radius:9px;border:1px solid var(--line2);background:var(--card);
  color:var(--ink);display:grid;place-items:center;cursor:pointer;transition:background .12s,border-color .12s}
.pp:hover{border-color:var(--accent);color:var(--accent-ink)}
.player.playing .pp{background:var(--accent);border-color:var(--accent);color:#fff}
.pp svg{width:16px;height:16px;display:block}
.pbar{flex:1 1 auto;height:7px;border-radius:6px;background:var(--line2);position:relative;cursor:pointer;overflow:hidden}
.pfill{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--accent);border-radius:6px}
.ptime{flex:0 0 auto;font-family:var(--mono);font-size:.76rem;color:var(--muted);min-width:78px;text-align:right;
  font-variant-numeric:tabular-nums}
audio{width:100%}

/* provenance */
.prov{margin-top:40px;padding-top:8px}
.prov .sec-head h2{font-size:1.18rem}
.prov p{color:var(--ink2);font-size:.95rem;max-width:78ch}
.prov code{font-family:var(--mono);font-size:.86em;background:var(--bg2);border:1px solid var(--line);
  padding:1px 6px;border-radius:6px}
details.tech{margin:14px 0;border:1px solid var(--line);border-radius:11px;background:var(--card);box-shadow:var(--shadow)}
details.tech summary{cursor:pointer;padding:12px 16px;font-weight:580;font-size:.92rem;list-style:none}
details.tech summary::-webkit-details-marker{display:none}
details.tech summary::after{content:"＋";float:right;color:var(--muted)}
details.tech[open] summary::after{content:"－"}
details.tech .body{padding:0 16px 14px;color:var(--ink2);font-size:.92rem}
details.cap>summary{cursor:pointer;list-style:none;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;
  overflow:hidden}
details.cap>summary::-webkit-details-marker{display:none}
details.cap[open]>summary{-webkit-line-clamp:unset}
details.cap>summary::after{content:" — show full prompt";color:var(--accent-ink);font-weight:600;font-size:.82rem;
  -webkit-line-clamp:unset;font-style:normal}
details.cap[open]>summary::after{content:" — show less"}

footer.foot{margin-top:44px;border-top:1px solid var(--line);padding:22px 0 40px;color:var(--muted);
  font-size:.9rem;display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
footer.foot b{color:var(--ink2);font-weight:620}

@media (max-width:820px){
  .sec-head .ctx{display:none}
}
@media (max-width:640px){
  .matrix{grid-template-columns:1fr;gap:10px}
  .mx-corner,.mx-col,.mx-row{display:none}
  .mx-cell .cell-tag{display:block;font-size:.76rem;font-weight:640;color:var(--muted);margin-bottom:5px}
  .mx-cell:nth-child(5) .cell-tag,.mx-cell:nth-child(6) .cell-tag{color:var(--pruned)}
  .mx-cell:nth-child(8) .cell-tag,.mx-cell:nth-child(9) .cell-tag{color:var(--recovered)}
  .duo{grid-template-columns:1fr}
  .ptime{min-width:70px}
}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
</style>
</head>
<body>
<nav class="nav"><div class="nav-in">
  <span class="brand">AudioLDM · <b>Recovery</b></span>
  <div class="nav-links">
    <a href="#sec1">Severity 1</a><a href="#sec2">Severity 2</a>
    <a href="#sec3">Music</a><a href="#prov">Provenance</a>
  </div>
</div></nav>

<div class="wrap">
<header class="hero">
  <span class="eyebrow">Research companion · Audio examples</span>
  <h1>AudioLDM Post-Pruning Recovery</h1>
  <p class="lead">Representative audio comparisons across pruning severity and generation duration.</p>
  <div class="chips">
    <span>AudioLDM</span><span>2 pruning severities</span><span>3.84 s / 10.24 s</span>
    <span>AudioCaps + music context</span><span>36 audio clips</span>
  </div>
  <div class="cta">
    <a class="btn primary" href="__REPO__" target="_blank" rel="noopener">View repository ↗</a>
    <a class="btn" href="#prov">Selection methodology ↓</a>
  </div>
</header>

<div class="callout">
  <span class="dot">✓</span>
  <div>
    <h3>Representative, not hand-picked</h3>
    <p>Examples were selected deterministically from the frozen evaluation sets using prompt identifiers
    only. Evaluation scores were never used for selection.</p>
    <p class="sub">All quantitative conclusions use the complete evaluation sets, not the examples shown here.</p>
  </div>
</div>

<section class="sec" id="sec1">
  <div class="sec-head"><h2>Severity 1</h2><span class="tier tier-1">Milder pruning</span>
    <span class="ctx">AudioCaps evaluation context</span></div>
  __A__
</section>

<section class="sec" id="sec2">
  <div class="sec-head"><h2>Severity 2</h2><span class="tier tier-2">Stronger pruning</span>
    <span class="ctx">AudioCaps evaluation context</span></div>
  __B__
</section>

<section class="sec" id="sec3">
  <div class="sec-head"><h2>Music context</h2><span class="tier tier-2">Severity 2</span>
    <span class="ctx">Separate evaluation context</span></div>
  <p style="color:var(--ink2);font-size:.95rem;margin:.4em 0 0;max-width:76ch">These examples illustrate the
  separate music evaluation context. They are not matched to the AudioCaps prompts above.</p>
  __C__
</section>

<section class="sec prov" id="prov">
  <div class="sec-head"><h2>Selection &amp; provenance</h2></div>
  <p>Examples were selected deterministically by ranking each section's frozen candidate prompts on
  <code>SHA256(namespace | section | ytid)</code> and taking the first few; the selection used only prompt
  identifiers and ignored every evaluation score (CLAP, Human-CLAP, FineLAP, KL, PANN, FAD, FD). All audio is
  taken directly from the frozen experimental model outputs — original generated waveforms converted to
  lossless FLAC with bit-identical samples (no normalization, resampling, or gain change). Quantitative
  results in the paper use the complete evaluation populations, not these examples.</p>
  <details class="tech"><summary>Technical notes</summary>
    <div class="body">Severity-1 music audio and a dense (unpruned) reference are not shown here because those
    files were not fully retained on disk; this affects only which examples can be displayed and does not affect
    any reported result. Audio is served as lossless FLAC (16-bit PCM, bit-identical to the source waveforms).</div>
  </details>
  <p style="margin-top:16px"><a class="btn" href="__REPO__" target="_blank" rel="noopener">Source repository ↗</a></p>
</section>

<footer class="foot">
  <span><b>AudioLDM Post-Pruning Recovery</b> · Companion audio examples · 2026</span>
  <a href="__REPO__" target="_blank" rel="noopener">Repository ↗</a>
</footer>
</div>

<script>
(function(){
  var PLAY='<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  var PAUSE='<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>';
  var audios=[];
  function fmt(t){t=Math.max(0,t||0);var m=Math.floor(t/60),s=Math.floor(t%60);return m+':'+(s<10?'0':'')+s;}
  Array.prototype.forEach.call(document.querySelectorAll('.player'),function(p){
    var a=p.querySelector('audio'); if(!a) return;
    a.removeAttribute('controls'); audios.push(a);
    var label=p.getAttribute('data-label')||'audio';
    var ctl=document.createElement('div'); ctl.className='pctl';
    var btn=document.createElement('button'); btn.type='button'; btn.className='pp';
    btn.setAttribute('aria-label','Play '+label); btn.innerHTML=PLAY;
    var bar=document.createElement('div'); bar.className='pbar'; bar.setAttribute('role','progressbar');
    bar.setAttribute('aria-label','Playback progress'); bar.setAttribute('aria-valuemin','0'); bar.setAttribute('aria-valuemax','100');
    var fill=document.createElement('div'); fill.className='pfill'; bar.appendChild(fill);
    var time=document.createElement('span'); time.className='ptime'; time.textContent='0:00 / --:--';
    ctl.appendChild(btn); ctl.appendChild(bar); ctl.appendChild(time);
    p.appendChild(ctl); p.classList.add('enhanced');
    function upd(){var d=a.duration||0; fill.style.width=(d?a.currentTime/d*100:0)+'%';
      bar.setAttribute('aria-valuenow',String(Math.round(d?a.currentTime/d*100:0)));
      time.textContent=fmt(a.currentTime)+' / '+(isFinite(d)&&d?fmt(d):'--:--');}
    btn.addEventListener('click',function(){ if(a.paused){audios.forEach(function(o){if(o!==a)o.pause();}); a.play();} else {a.pause();} });
    a.addEventListener('play',function(){btn.innerHTML=PAUSE; btn.setAttribute('aria-label','Pause '+label); p.classList.add('playing');});
    a.addEventListener('pause',function(){btn.innerHTML=PLAY; btn.setAttribute('aria-label','Play '+label); p.classList.remove('playing');});
    a.addEventListener('ended',function(){btn.innerHTML=PLAY; p.classList.remove('playing'); a.currentTime=0; upd();});
    a.addEventListener('timeupdate',upd); a.addEventListener('loadedmetadata',upd);
    bar.addEventListener('click',function(e){var r=bar.getBoundingClientRect(); var f=(e.clientX-r.left)/r.width;
      if(a.duration){a.currentTime=Math.min(1,Math.max(0,f))*a.duration;}});
  });
  // nav active state
  var links={}; Array.prototype.forEach.call(document.querySelectorAll('.nav-links a'),function(l){links[l.getAttribute('href').slice(1)]=l;});
  if('IntersectionObserver' in window){
    var obs=new IntersectionObserver(function(es){es.forEach(function(en){var l=links[en.target.id];
      if(l&&en.isIntersecting){for(var k in links)links[k].classList.remove('active'); l.classList.add('active');}});},
      {rootMargin:'-45% 0px -50% 0px'});
    Array.prototype.forEach.call(document.querySelectorAll('section.sec'),function(s){obs.observe(s);});
  }
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
