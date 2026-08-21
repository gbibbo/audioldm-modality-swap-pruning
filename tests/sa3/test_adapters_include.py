#!/usr/bin/env python3
"""Pure CPU test for the single-block include semantics the L_6/L_13 controls rely on (§5.1).
No torch/model needed. Run: .venv-sa3/bin/python tests/sa3/test_adapters_include.py

The scientific risk this guards: a single-block control must attach ONLY to its host block. The
upstream add_lora `include` is a plain substring match, so the include token `transformer.layers.6.`
must select block 6 and NOT block 16/26/... . And research_sa3.adapters._BLK must read the block id
back from a real module name so adapter_blocks() reports the right host block."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import adapters as A


def _matches_any(name, patterns):  # mirrors stable_audio_3.models.lora.model._matches_any (substring)
    return any(p in name for p in patterns)


# realistic module names under the ConditionedDiffusionModelWrapper
NAMES = [f"model.model.transformer.layers.{i}.self_attn.to_q" for i in range(20)] + \
        [f"model.model.transformer.layers.{i}.ff.linear_1" for i in range(20)] + \
        ["model.conditioner.conditioners.prompt.proj", "model.model.to_global_embed.0"]


def t1_single_block_include_is_exact():
    inc6 = ["transformer.layers.6."]
    hit = [n for n in NAMES if _matches_any(n, inc6)]
    blocks = {int(A._BLK.search(n).group(1)) for n in hit if A._BLK.search(n)}
    ok = blocks == {6}                                  # ONLY block 6, never 16
    ok = ok and all("layers.6." in n for n in hit)
    ok = ok and not any("layers.16." in n for n in hit)
    print(f"    T1 include 'transformer.layers.6.' -> blocks={sorted(blocks)} ({len(hit)} modules)")
    return ok


def t2_backbone_only_excludes_conditioner():
    inc = ["transformer.layers"]
    hit = [n for n in NAMES if _matches_any(n, inc)]
    ok = all("conditioner" not in n for n in hit)       # backbone-only never touches the conditioner
    ok = ok and len(hit) == 40
    print(f"    T2 backbone-only include -> {len(hit)} modules, 0 conditioner")
    return ok


def t3_block_regex_reads_id():
    ok = A._BLK.search("model.model.transformer.layers.13.ff.linear_1").group(1) == "13"
    ok = ok and A._BLK.search("model.conditioner.proj") is None
    print(f"    T3 _BLK reads block id ok={ok}")
    return ok


def main():
    checks = [("T1", t1_single_block_include_is_exact), ("T2", t2_backbone_only_excludes_conditioner),
              ("T3", t3_block_regex_reads_id)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
