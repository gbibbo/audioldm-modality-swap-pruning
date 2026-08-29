# RECOVERY-REVERSAL-V1.1 — pre-data administrative amendment (deterministic selection rules)

```
STATUS: FROZEN PRE-DATA AMENDMENT
NO AUDIOCAPS WAV GENERATED
NO AUDIOCAPS MODEL SCORE OBSERVED
NO OUTCOME DATA USED IN THIS AMENDMENT
```

Frozen 2026-08-29 (MVD). Amends the FROZEN V1.0 preregistration `docs/recovery_reversal_v1.md`
(freeze commit `f531fb6`, manifest commit `9280d43`). A SHA256 of this file is recorded in
`docs/recovery_reversal_v1_1.md.sha256`. This is an **administrative/reproducibility correction, not
a new scientific hypothesis** and it does **not** reopen the scientific design.

## 1. What happened

The V1.0 freeze **transcribed the deterministic battery-selection procedure differently from the
previously supervisor-confirmed specification.** Because the discrepancy was discovered **before any
AudioCaps generation, scoring, or outcome observation**, V1.1 restores the previously specified
deterministic hash rules. The V1.0 manifest (`configs/research/reversal_v1_audiocaps_manifest.json`,
commit `9280d43`) **was instantiated but never used to generate or score any model output** — no WAV
exists, no CLAP/Human-CLAP score exists. V1.0 documents are preserved untouched as superseded
historical provenance; nothing is rewritten or deleted.

## 2. V1.1 changes ONLY two things

1. **ytid hash namespace** — the literal `|YTID|` token in the selection key;
2. **caption-selection hash algorithm** — a modulo pick over caption rows (below).

**Everything else is exactly as frozen in V1.0 and is NOT reopened:** N=96, 2 generation replicates,
3 backbones (dense_ema descriptive + p1_pruned_ema_reconstructed + p1_recovered), exclusions,
AudioCaps canonical universe (`audiocaps_test_label.json`, 964 ytids), operating point
(3.84 s/DDIM 50/eta 0/guidance 2.5/FP32/single-gen/no-adapter/no-best-of-3), `GENERATION_SALT`,
generation seed convention, primary CLAP scorer + its RNG convention (`np.random.seed(20260826)`
once per 192-item system), bootstrap (B=10000, PCG64(20260827)), R_AC / R_music / I definitions,
SESOI 0.025 on the R_AC point, the three PASS conditions, the Human-CLAP secondary, the waveform
secondary, the FAIL/no-rescue policy, and budget governance.

## 3. Corrected deterministic rules (frozen)

### 3a. Ytid selection

```python
selection_key = hashlib.sha256(f"{SELECTION_SALT}|YTID|{ytid}".encode("utf-8")).hexdigest()
```

Sort eligible unique ytids ascending lexicographically by `selection_key`; take exactly the first
96. No other ranking or tie-breaking. `SELECTION_SALT = "RECOVERY-REVERSAL-V1|AUDIOCAPS-TEST|2026-08-27"`.

### 3b. Caption selection — Convention B (annotation rows, multiplicity preserved)

```python
caption_hash  = hashlib.sha256(f"{SELECTION_SALT}|CAPTION|{ytid}".encode("utf-8")).digest()
caption_index = int.from_bytes(caption_hash[:8], "big") % n            # n = number of caption ROWS
selected_caption = canonical_captions[caption_index]
```

The caption **text never enters the hash**. `n` = number of canonical caption **rows** for the ytid
(for this universe, `n = 5` for every selected ytid). **Duplicate caption strings are NOT
deduplicated.**

### 3c. Canonical caption ordering

`audiocaps_test_label.json` has **no stable caption/row identifier** (rows carry only
`caption, labels, seg_label, wav`). Per the specified fallback: (1) collect all caption **row**
strings for the ytid, preserving multiplicity; (2) encode each as exact UTF-8 bytes; (3) sort
ascending by those bytes, preserving duplicates; (4) this ordered **multiset** is
`canonical_captions`. No stable row ID is imported from the non-canonical `test.csv` (its
`audiocap_id` is deliberately NOT used); the canonical V1 evaluation universe remains
`audiocaps_test_label.json`.

Implementation: `research_pruning/eval/reversal.py` — `selection_key_v11`,
`canonical_caption_rows_v11`, `choose_caption_v11` (no `set()`/dedup in the caption path);
tests `tests/research/test_reversal_v1_1.py` (12/12).

## 4. Caption multiset — scientific rationale (Convention B)

- The canonical source contains **five human annotation rows per test clip**; `n = 5` for every
  selected ytid (STOP if any selected ytid has a row count ≠ 5).
- **Some rows contain exact duplicate caption text.** V1.1 interprets *n captions* as **annotation
  rows, preserving multiplicity** — five annotation instances, not a unique-string set.
  Deduplicating would change the implicit annotation distribution (downweighting inter-annotator
  agreement, upweighting rarer formulations), so multiplicity is retained because it affects `n`.
- Equal duplicate strings need **no tie-breaker**: their mutual ordering is irrelevant because the
  selected prompt *text* is identical; only their multiplicity (via `n`) matters.
- This choice was made **before** any AudioCaps generation, scoring, or outcome observation.
- Relative to a deduplicated interpretation, Convention B changes the selected caption for **6 of
  the 96** V1.1 ytids (the 9 selected ytids with duplicate rows, of which 6 diverge); it does **not**
  affect ytid selection.

## 5. V1.0 → V1.1 administrative delta (provenance, not analysis)

```
common selected ytids           = 9
V1.0-only ytids                 = 87
V1.1-only ytids                 = 87
caption differences among the 9 common ytids = 7
```

Same 964-ytid eligible universe and identical exclusions for both; the divergence is purely the
corrected ytid-hash namespace (selection) and caption algorithm. No semantic characterization of the
changed prompts is made — this is provenance.

## 6. Chronology & provenance

Git order is `V1 freeze (f531fb6) → V1 manifest (9280d43) → V1.1 amendment → corrected V1.1
manifest`. The corrected manifest is written to a NEW path
`configs/research/reversal_v1_1_audiocaps_manifest.json`; the V1.0 manifest is never overwritten and
is marked superseded only through this document and the ledger. All CPU manifest/generation/scorer/
verdict/secondary preflights are rerun against V1.1. GPU remains BLOCKED (balance ≈ 0.72 cr < ~1.5 cr
envelope); no audio, no scoring, 0 credits.
