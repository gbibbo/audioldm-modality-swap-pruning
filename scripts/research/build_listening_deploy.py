#!/usr/bin/env python3
"""Build host-agnostic SELF-CONTAINED deployment HTML for the listening study (v1.2).

Produces six standalone HTML files (P01.html … P06.html), each embedding CSS + JS +
that participant's blinded manifest + all their audio as LOSSLESS FLAC data: URIs. No
external assets, no ?p= dependency, no CORS; the only external request is the results
POST. Each file is verified < 20 MB (fits a single-HTML host) and fully blinded.

Audio is FLAC (lossless: decoded samples are bit-identical to the -36 LUFS listening
copies) purely to fit the size budget — it is NOT a stimulus change.

RESULTS_ENDPOINT is taken from the env var, else parsed from listening_study/config.js.
It is embedded in the DEPLOY artifact (necessary for the client) but the artifact is
gitignored, so the live webhook is never committed.

Outputs: artifacts/listening_study_deploy/P0{1..6}.html (gitignored)
         configs/research/listening_deploy_report.json (committed; sizes/sha, no endpoint value)

Run: RESULTS_ENDPOINT=... .venv-loudness/bin/python scripts/research/build_listening_deploy.py
"""
import json, os, io, re, base64, hashlib
import numpy as np, soundfile as sf

ROOT = "/teamspace/studios/this_studio/audioldm-modality-swap-pruning"
LS = os.path.join(ROOT, "listening_study")
PRIV = os.path.join(ROOT, "configs/research/listening_study_assignments_private.json")
OUTDIR = os.path.join(ROOT, "artifacts/listening_study_deploy")
REPORT = os.path.join(ROOT, "configs/research/listening_deploy_report.json")
MAXB = 20 * 1024 * 1024
FORBIDDEN = re.compile(r"recovered|pruned|\bdense\b|reconstr|_alt10s|realref|severity|\"type\"|catch_kind", re.I)


def endpoint():
    ep = os.environ.get("RESULTS_ENDPOINT", "").strip()
    if ep:
        return ep
    m = re.search(r'RESULTS_ENDPOINT:\s*"([^"]*)"', open(os.path.join(LS, "config.js")).read())
    return (m.group(1).strip() if m else "")


def flac_data_uri(path):
    x, sr = sf.read(path, dtype="int16")
    buf = io.BytesIO(); sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
    return "data:audio/flac;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ep = endpoint()
    priv = json.load(open(PRIV))
    render = priv["audio_render_map"]  # hash -> {src_path,...}
    css = open(os.path.join(LS, "style.css")).read()
    appjs = open(os.path.join(LS, "app.js")).read()
    index = open(os.path.join(LS, "index.html")).read()
    # strip external references from the template
    html_head = index.replace('<link rel="stylesheet" href="style.css">',
                              "<style>\n" + css + "\n</style>")
    html_head = html_head.replace('<script src="config.js"></script>', "")
    html_head = html_head.replace('<script src="app.js"></script>', "")

    # cache FLAC data URIs by source hash filename
    duri_cache = {}
    report = {"artifact": "listening_deploy_report", "endpoint_present": bool(ep),
              "max_bytes": MAXB, "files": {}}
    for c in ["P0%d" % i for i in range(1, 7)]:
        man = json.load(open(os.path.join(LS, "public_manifests", c + ".json")))
        # audio urls referenced by this participant
        urls = set()
        for t in man["trials"]:
            urls.add(t["audio_A"]); urls.add(t["audio_B"])
        urls.add(man["level_check_audio"])
        emb = {}
        for u in sorted(urls):
            hn = os.path.basename(u)   # "<hash>.wav"
            if hn not in duri_cache:
                duri_cache[hn] = flac_data_uri(os.path.join(LS, "audio", hn))
            emb[u] = duri_cache[hn]
        inline = ("<script>\n"
                  "window.STUDY_CONFIG={RESULTS_ENDPOINT:%s};\n"
                  "window.EMBEDDED_MANIFEST=%s;\n"
                  "window.EMBEDDED_AUDIO=%s;\n"
                  "</script>\n<script>\n%s\n</script>\n") % (
                      json.dumps(ep), json.dumps(man), json.dumps(emb), appjs)
        html = html_head.replace("</body>", inline + "</body>")
        # blinding scan (ignore caption prompt_text values, which are natural English)
        scan = html
        for t in man["trials"]:
            scan = scan.replace(json.dumps(t["prompt_text"])[1:-1], "")
        leaks = sorted(set(m.group(0) for m in FORBIDDEN.finditer(scan)))
        out = os.path.join(OUTDIR, c + ".html")
        open(out, "w").write(html)
        b = os.path.getsize(out)
        report["files"][c] = {"bytes": b, "mb": round(b / 1e6, 2), "under_20mb": b < MAXB,
                               "n_trials": len(man["trials"]), "n_audio_embedded": len(emb),
                               "sha256": hashlib.sha256(html.encode()).hexdigest(),
                               "blinding_leaks": leaks}
        print(f"{c}: {b/1e6:.2f} MB | {len(man['trials'])} trials | {len(emb)} audio | "
              f"<20MB {b<MAXB} | leaks {leaks}")
    report["all_under_20mb"] = all(f["under_20mb"] for f in report["files"].values())
    report["all_blinded"] = all(not f["blinding_leaks"] for f in report["files"].values())
    payload = json.dumps(report, indent=2, sort_keys=True)
    report["self_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json.dump(report, open(REPORT, "w"), indent=2, sort_keys=True)
    print(f"endpoint embedded: {bool(ep)} | all<20MB {report['all_under_20mb']} | all_blinded {report['all_blinded']}")
    print("report:", REPORT)


if __name__ == "__main__":
    main()
