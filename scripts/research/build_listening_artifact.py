#!/usr/bin/env python3
"""Assemble per-participant Claude ARTIFACT HTML for the listening study (v1.2).

Artifact format: the file is <title> + <style> + page content (NO <!doctype>/<html>/
<head>/<body> — the platform wraps it). Each participant's manifest + all their audio
(lossless FLAC data: URIs) are embedded, so the page is fully self-contained and needs
no external fetch (the artifact CSP blocks external POST). Results are collected by the
manual copy/download flow (the `downloads` capability + clipboard); RESULTS_ENDPOINT is
empty. Fully blinded; each file < 16 MB (the artifact page cap).

Outputs artifacts/listening_study_artifact/P0{1..6}.html (gitignored) +
configs/research/listening_artifact_report.json (committed: sizes/sha, no unblinding).

Run: .venv-loudness/bin/python scripts/research/build_listening_artifact.py
"""
import json, os, io, re, base64, hashlib
import soundfile as sf

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
LS = os.path.join(ROOT, "listening_study")
PRIV = os.path.join(ROOT, "configs/research/listening_study_assignments_private.json")
OUTDIR = os.path.join(ROOT, "artifacts/listening_study_artifact")
REPORT = os.path.join(ROOT, "configs/research/listening_artifact_report.json")
CAP = 16 * 1024 * 1024
FORBIDDEN = re.compile(r"recovered|pruned|\bdense\b|reconstr|_alt10s|realref|severity|\"type\"|catch_kind", re.I)


def flac_data_uri(path, cache):
    if path not in cache:
        x, sr = sf.read(path, dtype="int16")
        buf = io.BytesIO(); sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
        cache[path] = "data:audio/flac;base64," + base64.b64encode(buf.getvalue()).decode()
    return cache[path]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    css = open(os.path.join(LS, "style.css")).read()
    appjs = open(os.path.join(LS, "app.js")).read()
    index = open(os.path.join(LS, "index.html")).read()
    main_html = re.search(r"<main[\s\S]*?</main>", index).group(0)
    priv = json.load(open(PRIV))
    cache = {}
    report = {"artifact": "listening_artifact_report", "page_cap_bytes": CAP,
              "collection": "manual (downloads capability + clipboard); external POST blocked by artifact CSP",
              "capabilities": {"downloads": True}, "files": {}}
    for c in ["P0%d" % i for i in range(1, 7)]:
        man = json.load(open(os.path.join(LS, "public_manifests", c + ".json")))
        urls = set()
        for t in man["trials"]:
            urls.add(t["audio_A"]); urls.add(t["audio_B"])
        urls.add(man["level_check_audio"])
        emb = {u: flac_data_uri(os.path.join(LS, "audio", os.path.basename(u)), cache) for u in sorted(urls)}
        title = "Listening Study %s" % c
        inline = ("<script>\n"
                  "window.STUDY_CONFIG={RESULTS_ENDPOINT:\"\"};\n"
                  "window.EMBEDDED_MANIFEST=%s;\n"
                  "window.EMBEDDED_AUDIO=%s;\n"
                  "</script>\n<script>\n%s\n</script>\n") % (json.dumps(man), json.dumps(emb), appjs)
        doc = "<title>%s</title>\n<style>\n%s\n</style>\n%s\n%s" % (title, css, main_html, inline)
        # blinding scan (ignore caption text)
        scan = doc
        for t in man["trials"]:
            scan = scan.replace(json.dumps(t["prompt_text"])[1:-1], "")
        leaks = sorted(set(m.group(0) for m in FORBIDDEN.finditer(scan)))
        out = os.path.join(OUTDIR, c + ".html")
        open(out, "w").write(doc)
        b = os.path.getsize(out)
        report["files"][c] = {"bytes": b, "mb": round(b/1e6, 2), "under_16mb": b < CAP,
                               "n_trials": len(man["trials"]), "n_audio": len(emb),
                               "title": title, "sha256": hashlib.sha256(doc.encode()).hexdigest(),
                               "blinding_leaks": leaks, "has_doctype": doc.lstrip().lower().startswith("<!doctype")}
        print(f"{c}: {b/1e6:.2f} MB | {len(man['trials'])} trials | <16MB {b<CAP} | leaks {leaks} | doctype {report['files'][c]['has_doctype']}")
    report["all_under_16mb"] = all(f["under_16mb"] for f in report["files"].values())
    report["all_blinded"] = all(not f["blinding_leaks"] for f in report["files"].values())
    report["no_doctype"] = all(not f["has_doctype"] for f in report["files"].values())
    payload = json.dumps(report, indent=2, sort_keys=True)
    report["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json.dump(report, open(REPORT, "w"), indent=2, sort_keys=True)
    print(f"all<16MB {report['all_under_16mb']} | all_blinded {report['all_blinded']} | no_doctype {report['no_doctype']}")


if __name__ == "__main__":
    main()
