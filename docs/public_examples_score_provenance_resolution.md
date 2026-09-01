# Public-Example CLAP Absolute-Value Discrepancy — Resolution

**Date:** 2026-09-01. **Type:** provenance resolution. No CLAP recomputed; no result changed.

## The discrepancy

A previous **chat** message reported severity-1 public examples with absolute CLAP values that differ
from the committed audit (`docs/public_audio_examples_score_audit.md`,
`configs/research/public_audio_examples_score_audit.json`):

| Example @10.24 s | chat message (WRONG) | committed audit (CANONICAL) | ΔCLAP (both) |
|---|---|---|---|
| Sev1 Ex1 `g5l3Bz6lWnc` | 0.081 → 0.304 | **0.1928 → 0.4155** | +0.223 |
| Sev1 Ex4 `emGPabOePzA` | 0.310 → 0.035 | **0.5184 → 0.2434** | −0.275 |

## Trace

The committed audit reads the sev-1 absolute cosines **directly** from the frozen experimental artifact
`configs/research/op_duration_discriminator_1_result.json` → `raw_cosines`
(`pruned_ctrl`/`recovered_ctrl` = 3.84 s; `pruned_alt`/`recovered_alt` = 10.24 s), indexed by the frozen
`op_duration_discriminator_1_subset.json` `subset_prompt_index`. Verified by re-reading those arrays:

* `g5l3Bz6lWnc` (k=37): `pruned_alt=0.1928`, `recovered_alt=0.4155` → exactly the committed audit.
* `emGPabOePzA` (k=47): `pruned_alt=0.5184`, `recovered_alt=0.2434` → exactly the committed audit.

## Verdict

**The committed audit is correct; the chat message's absolute columns were a transcription error.** The
audit script's console printed only ΔCLAP; the absolute values typed into the chat table were mis-entered
(they preserve the correct ΔCLAP but are not the frozen scores). There is **one** scorer context — the
canonical frozen convention (`laion/clap-htsat-fused` rev `365dea6e…`, seed-once per group, 4×80 matched,
unit = ytid, B=10000) used by the `op_duration` experiment. There is **no** second scorer context and
**no** mixing of conventions. **ΔCLAP (the reported quantity) was correct throughout**; only the absolute
columns in one chat message were wrong.

Canonical source of truth for public-example scores: `configs/research/public_audio_examples_score_audit.json`
(self-sha `5c2f54df…`). No number in any committed artifact changes.
