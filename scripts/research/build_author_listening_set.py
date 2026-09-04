#!/usr/bin/env python3
"""AUTHOR-LISTENING-1 — blinded single-listener check by the first author (CPU, 0 cr; NOT a perceptual
experiment, NOT a replacement for the frozen six-listener panel; reported, if at all, as one author's
blinded informal listening).

Builds, from the frozen severity-2 WAVs already on disk, a blinded listening set that addresses three
questions the automatic scorers cannot settle:
  A  CONTENT   Is the 83 %-pruned checkpoint's output (near the chance floor on every scorer) degenerate
               audio (noise / silence / texture) or plausible audio that merely fails the caption? And what
               does P+FT produce for the same prompt?  -> 8 prompts x {P, P+FT} @10.24 s, single clips.
  B  DURATION  Is the recovery gain AUDIBLE, and does the audible margin grow with duration?
               -> the same 8 prompts, P vs P+FT pairs at 3.84 s and at 10.24 s (A/B order randomised).
  C  MUSIC     On held-out hip-hop captions (scorers: both systems near chance), does either output sound
               like music at all, and does either follow the caption?  -> 8 music prompts @10.24 s, pairs.
Prompt selection: seeded RNG, outcome-blind (no score consulted). Blinding: every clip is copied under an
opaque id; the key is written to a separate file whose sha256 is recorded in the manifest so that the
answers can be shown to have been given before unblinding. No loudness normalisation (raw generator
output; near-silent pruned outputs are part of the phenomenon and must stay audible as such).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/build_author_listening_set.py
Output: artifacts/icassp_gate0/author_listening/{index.html, responses_template.md, manifest.json, KEY_DO_NOT_OPEN.json, audio/}
"""
from __future__ import annotations
import hashlib, json, os, shutil, sys
import numpy as np

TMP = "artifacts/icassp_gate0/_score_tmp"
OUT = "artifacts/icassp_gate0/author_listening"
SEED_NS = "AUTHOR-LISTENING-1|SELECTION|2026-09-04"
N_AC, N_MUS = 8, 8


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    g = {x["name"]: x["items"] for x in json.load(open(f"{TMP}/xsev_sev2_groups_in.json"))["groups"]}
    m = {x["name"]: x["items"] for x in json.load(open(f"{TMP}/music_native_groups_in.json"))["groups"]}
    rng = np.random.default_rng(int(hashlib.sha256(SEED_NS.encode()).hexdigest()[:8], 16) % (2 ** 31))
    ac_idx = sorted(rng.choice(192, N_AC, replace=False).tolist())
    mus_idx = sorted(rng.choice(64, N_MUS, replace=False).tolist())
    os.makedirs(f"{OUT}/audio", exist_ok=True)
    key, items, counter = [], [], [0]

    def blind_copy(src):
        counter[0] += 1
        cid = hashlib.sha256(f"{SEED_NS}|{counter[0]}".encode()).hexdigest()[:10]
        dst = f"{OUT}/audio/{cid}.wav"
        shutil.copyfile(src, dst)
        return cid

    # Block A: single clips, P and P+FT @10.24 s, shuffled
    a_items = []
    for i in ac_idx:
        for sysname in ("pruned2_A", "recovered2"):
            it = g[f"{sysname}__ac_native"][i]
            cid = blind_copy(it["wav"])
            key.append({"block": "A", "clip": cid, "prompt_index": i, "system": sysname, "duration_s": 10.24, "src": it["wav"]})
            a_items.append({"clip": cid, "caption": it["caption"]})
    rng.shuffle(a_items)
    # Block B: pairs P vs P+FT at 3.84 and 10.24 s, order randomised
    b_items = []
    for i in ac_idx:
        for ctx, dur in (("ac_short", 3.84), ("ac_native", 10.24)):
            pair = [("pruned2_A", g[f"pruned2_A__{ctx}"][i]), ("recovered2", g[f"recovered2__{ctx}"][i])]
            if rng.random() < 0.5:
                pair = pair[::-1]
            cids = []
            for sysname, it in pair:
                cid = blind_copy(it["wav"]); cids.append(cid)
                key.append({"block": "B", "clip": cid, "prompt_index": i, "system": sysname, "duration_s": dur, "src": it["wav"]})
            b_items.append({"pair": cids, "caption": pair[0][1]["caption"], "duration_s": dur})
    rng.shuffle(b_items)
    # Block C: music pairs @10.24 s
    c_items = []
    for i in mus_idx:
        pair = [("pruned2_A", m["pruned2_A__music_native"][i]), ("recovered2", m["recovered2__music_native"][i])]
        if rng.random() < 0.5:
            pair = pair[::-1]
        cids = []
        for sysname, it in pair:
            cid = blind_copy(it["wav"]); cids.append(cid)
            key.append({"block": "C", "clip": cid, "prompt_index": i, "system": sysname, "duration_s": 10.24, "src": it["wav"]})
        c_items.append({"pair": cids, "caption": pair[0][1]["caption"]})
    rng.shuffle(c_items)

    key_path = f"{OUT}/KEY_DO_NOT_OPEN.json"
    json.dump({"seed_namespace": SEED_NS, "key": key}, open(key_path, "w"), indent=1)
    manifest = {"artifact": "author_listening_1", "status": "BLINDED SET BUILT; no answers yet",
                "class": ("single-listener informal check by the first author; NOT the frozen perceptual panel; "
                          "descriptive only; no gate; cannot change any verdict"),
                "selection": {"seed_namespace": SEED_NS, "audiocaps_prompt_index": ac_idx, "music_prompt_index": mus_idx,
                              "outcome_blind": True},
                "blocks": {"A": {"n_clips": len(a_items), "items": a_items},
                           "B": {"n_pairs": len(b_items), "items": b_items},
                           "C": {"n_pairs": len(c_items), "items": c_items}},
                "key_sha256": sha(key_path), "key_file": key_path,
                "audio_sha256": {k["clip"]: sha(f"{OUT}/audio/{k['clip']}.wav") for k in key},
                "loudness": "none applied (raw generator output)"}
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)

    # HTML player
    H = ["<!doctype html><meta charset='utf-8'><title>Author listening set (blinded)</title>",
         "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;line-height:1.4}"
         ".it{border:1px solid #ccc;padding:.6em 1em;margin:.6em 0}.cap{color:#333;font-style:italic}code{background:#eee}</style>",
         "<h1>Author listening set — blinded</h1>",
         "<p>Single listener (first author), informal. Clip ids are opaque; the key is sealed "
         f"(sha256 <code>{manifest['key_sha256'][:16]}…</code>). Write answers in <code>responses_template.md</code>. "
         "No loudness normalisation: near-silent clips are real outputs.</p>",
         "<h2>Block A — content (16 clips, 10.24 s)</h2><p>For each clip: (1) what do you hear, in a few words; "
         "(2) is it recognisable sound (5) … texture/noise/silence (1); (3) does it match the caption, 1–5.</p>"]
    for k, it in enumerate(a_items, 1):
        H.append(f"<div class='it'><b>A{k}</b> <code>{it['clip']}</code><br><span class='cap'>{it['caption']}</span><br>"
                 f"<audio controls preload='none' src='audio/{it['clip']}.wav'></audio></div>")
    H.append("<h2>Block B — duration (16 pairs, same prompts at 3.84 s and 10.24 s)</h2>"
             "<p>Which of the two matches the caption better: 1 / 2 / same? Confidence 1–3. Both clips of a pair have the same duration.</p>")
    for k, it in enumerate(b_items, 1):
        H.append(f"<div class='it'><b>B{k}</b> ({it['duration_s']} s)<br><span class='cap'>{it['caption']}</span><br>"
                 f"1: <audio controls preload='none' src='audio/{it['pair'][0]}.wav'></audio> "
                 f"2: <audio controls preload='none' src='audio/{it['pair'][1]}.wav'></audio></div>")
    H.append("<h2>Block C — music (8 pairs, 10.24 s, held-out hip-hop captions)</h2>"
             "<p>For each clip: does it sound like music at all (yes/partly/no)? Does it follow the caption (1–5)? Which is better: 1 / 2 / same?</p>")
    for k, it in enumerate(c_items, 1):
        H.append(f"<div class='it'><b>C{k}</b><br><span class='cap'>{it['caption']}</span><br>"
                 f"1: <audio controls preload='none' src='audio/{it['pair'][0]}.wav'></audio> "
                 f"2: <audio controls preload='none' src='audio/{it['pair'][1]}.wav'></audio></div>")
    open(f"{OUT}/index.html", "w").write("\n".join(H))
    # response template
    T = ["# AUTHOR-LISTENING-1 responses (blinded)", "", f"key sha256: `{manifest['key_sha256']}`", "",
         "## Block A — content", "| item | clip | what I hear | recognisable 1–5 | matches caption 1–5 |", "|---|---|---|---|---|"]
    T += [f"| A{k} | {it['clip']} |  |  |  |" for k, it in enumerate(a_items, 1)]
    T += ["", "## Block B — duration", "| item | dur | better (1/2/same) | confidence 1–3 | note |", "|---|---|---|---|---|"]
    T += [f"| B{k} | {it['duration_s']} |  |  |  |" for k, it in enumerate(b_items, 1)]
    T += ["", "## Block C — music", "| item | clip1 music? | clip2 music? | clip1 caption 1–5 | clip2 caption 1–5 | better | note |", "|---|---|---|---|---|---|---|"]
    T += [f"| C{k} |  |  |  |  |  |  |" for k, it in enumerate(c_items, 1)]
    T += ["", "## Free impressions", ""]
    open(f"{OUT}/responses_template.md", "w").write("\n".join(T) + "\n")
    print(f"built {OUT}: A {len(a_items)} clips, B {len(b_items)} pairs, C {len(c_items)} pairs, "
          f"{len(key)} WAVs; key sha256 {manifest['key_sha256']}")


if __name__ == "__main__":
    main()
