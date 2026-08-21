#!/usr/bin/env python3
"""Fetch a Stable Audio 3 HF repo (gated or not) into data/sa3/<name>/ using huggingface_hub.

The token is read ONLY from the environment (`HF_TOKEN`) or the standard HF token file
(`~/.cache/huggingface/token`, written by `hf auth login`). It is never printed, logged or
written anywhere by this script. Files are verified against the HF API sizes after download.

    .venv-sa3/bin/python scripts/sa3/fetch_sa3_repo.py --repo stabilityai/stable-audio-3-small-sfx --dest data/sa3/small-sfx
"""
from __future__ import annotations

import argparse
import os
import sys

FILES = ["model.safetensors", "model_config.json", "README.md", "LICENSE.md", "LICENSE_GEMMA.md", "NOTICE"]
T5 = ["config.json", "generation_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
      "tokenizer_config.json", "model.safetensors", "README.md"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--with-svd-bases", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    token_present = bool(os.environ.get("HF_TOKEN")) or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
    print(f"token source available: {token_present} (value never printed)")
    api = HfApi()
    try:
        info = api.model_info(args.repo, files_metadata=True)
    except (GatedRepoError, HfHubHTTPError) as e:
        print(f"cannot access {args.repo}: {type(e).__name__} — accept the license on the HF page and provide a token via `hf auth login`")
        return 2
    sizes = {s.rfilename: s.size for s in info.siblings}
    want = FILES + (["svd_bases.pt"] if args.with_svd_bases else []) + [f"t5gemma-b-b-ul2/{f}" for f in T5]
    os.makedirs(args.dest, exist_ok=True)
    bad = []
    for f in want:
        if f not in sizes:
            print(f"  (absent in repo) {f}")
            continue
        try:
            p = hf_hub_download(repo_id=args.repo, filename=f, local_dir=args.dest)
        except (GatedRepoError, HfHubHTTPError) as e:
            print(f"  FAILED {f}: {type(e).__name__}")
            return 2
        got = os.path.getsize(p)
        okf = sizes[f] is None or got == sizes[f]
        print(f"  {'ok ' if okf else 'BAD'} {f} {got} bytes (api {sizes[f]})")
        if not okf:
            bad.append(f)
    print("FETCH", "OK" if not bad else f"SIZE MISMATCH {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
