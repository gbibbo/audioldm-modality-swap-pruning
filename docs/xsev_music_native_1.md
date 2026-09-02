# XSEV-MUSIC-NATIVE-1 — frozen protocol (the missing factorial cell: held-out music @ 10.24 s, severity 2)

**STATUS: PROSPECTIVE FOLLOW-UP, FROZEN BEFORE ANY music@10.24 s OUTPUT EXISTS.** Motivated by the
Draft-2 review (`docs/review/2026-09-02_manuscript_draft2_scientific_review.md` §A4/B6); authorised by
Gabriel 2026-09-02 ("apruebo todo … siempre y cuando el consumo de créditos está minimizado").
No frozen verdict (V1.1, Arm-D, RECOVERY-CROSS-SEVERITY-REP-1 CASE C, FineLAP) is touched. This arm
**adds one cell**; it does not re-test any gate.

## 0. Why
At severity 2 the design has AudioCaps at {3.84 s, 10.24 s} but music only at 3.84 s, so the domain
contrast $K$ bundles domain and duration. Generating the music-64 battery at the native 10.24 s
completes the 2 × 2 (domain × duration) factorial and lets the paper report each axis with the other
held fixed.

## 1. Systems (severity 2)
* **pruned2_A** = prune_A′(dense EMA, [1,2,1,1]) — PRIMARY baseline (as in the frozen replication).
* **recovered2** = `l1_p1_dp1_finetuned_global_step_999999.ckpt`, EMA materialised.
* **pruned2_B is NOT generated** (credit minimisation). Justification: on every frozen severity-2
  estimand A′ and B′ differed by < 0.003 (R_music_A +0.0092 vs R_music_B +0.0092; J +0.159 vs +0.161),
  so the seam convention is not expected to matter for this cell; this is stated as a limitation of the
  arm, not assumed away.

## 2. Manifest, seeds, CRN
* Prompts: the frozen `configs/research/xsev_music_manifest.json` (manifest_sha256 `f5a26fbe…`),
  **64 prompts, replicate 0 only**.
* Integer seed = the manifest's `generation_seeds[0]` = `derive_paired_seed(GEN_SALT, ytid, 0)`
  (verified 2026-09-02 against the persisted `gen_manifest_recovered2_music.json` rows: identical).
  This is the same convention the AudioCaps short/native pair uses: same integer seed per ytid,
  x_T shape (1,8,256,16) ≠ the 3.84 s (1,8,96,16). Within the arm pruned2_A / recovered2 share x_T.
* Generator: `scripts/research/reversal_xsev_gen.py --context music_native` (new context; identical
  code path to `ac_native` with the music manifest). Entry: `scripts/research/run_xsev_music_native_gen.sh`.

## 3. Operating point
10.24 s (latent 256), DDIM 50, guidance 2.5, η 0, fp32, single generation, EMA weights — identical to
the frozen AudioCaps-native cell.

## 4. Compute (CPU-first rule; credit minimisation)
* **CPU first.** The Studio (4 cores) can run the generator; a timed CPU dry-run decides:
  if the projected wall time for 128 clips is ≤ ~3 h, generate on CPU (**0 cr**, device recorded
  in every manifest row). Otherwise a single T4 job (`run_xsev_music_native_gen.sh` with `DEV=cuda`,
  128 WAVs ≈ 0.45–0.6 cr at the measured native rate ≈ 0.0036 cr/WAV) with watchdog
  `--max-cost 0.70 --max-minutes 60`. **HARD MAX NEW GPU SPEND = 0.70 cr.**
* Device caveat (recorded, not hidden): if generated on CPU, the primary estimand $R_{\text{music,nat}}$
  is device-internal (both systems on CPU); the cross-duration contrast $J_{\text{music}}$ compares
  CPU-generated native clips with T4-generated short clips (fp32 both; numerics differ at rounding
  level). $J_{\text{music}}$ is therefore secondary.

## 5. Scoring (frozen convention)
CLAP `laion/clap-htsat-fused` rev `365dea6e…`, one seed-once group per system: **64 items**, canonical
`prompt_index` order, `np.random.seed(20260826)` reset once per group (`gate0_clap_scorer.py
--score-groups`). 10.24 s clips are centre-cropped to 10.0 s by the scorer, as for AudioCaps-native.
Secondary: Human-CLAP on the same 64-item groups (corroborative, no role).

## 6. Estimands (frozen)
Orientation: + favours the fine-tuned checkpoint. Bootstrap: unit = prompt, B = 10000, percentile 95 %,
seed namespace `XSEV-MUSIC-NATIVE-1|BOOTSTRAP|2026-09-02` → `PCG64(int(sha256(ns)[:8],16) % 2**31)`.
* **PRIMARY: $R_{\text{music,nat}} = C_{\text{rec}}(\text{music},10.24) - C_{\text{prunedA}}(\text{music},10.24)$**,
  paired per prompt (64), cluster CI.
* Secondary: $J_{\text{music}} = R_{\text{music,nat}} - R_{\text{music,short}}$ (paired per prompt; the
  frozen short value is the per-prompt mean over 3 replicates → replicate-count asymmetry noted);
  $D_{\text{nat}} = R_{\text{nat}}(\text{AudioCaps}) - R_{\text{music,nat}}$ (independent two-sample
  bootstrap, the matched-duration domain contrast at 10.24 s); absolute means; win-rate.
* SESOI 0.025 (as everywhere in the project). **No PASS/FAIL gate** — every branch is reportable and
  none rescues or alters a frozen result.

## 7. Outcome branches (frozen before scoring)
* **(a) Domain-specific native gain:** $R_{\text{music,nat}}$ CI includes 0 and |point| < 0.025 →
  "the native-duration recovery gain does not transfer to held-out music; the gain is specific to the
  fine-tuning domain as well as its duration".
* **(b) Generic duration gain:** lo95($R_{\text{music,nat}}$) > 0 and point ≥ 0.025 → "part of the
  native-duration gain is domain-generic (length conditioning), the rest domain-specific
  ($D_{\text{nat}}$ reported)".
* **(c) Native music penalty:** hi95($R_{\text{music,nat}}$) < 0 → report as a negative; no rescue.
* Anything between (CI includes 0, |point| ≥ 0.025) → "unresolved at n = 64", reported as such.

## 8. What this arm cannot do
It does not provide a dense fine-tuned control (mechanism stays blocked), does not test DDIM-200 or
guidance 3.5, and does not change CASE C, $J$, $K$ or any gate. It is a design completion.

## 9. Execution order (binding)
1. Commit this protocol (+ `.sha256`) and the generator/entry changes with a **clean tree**.
2. CPU dry-run (`--dry-run-cpu`) → timing → device decision recorded in the ledger.
3. Generate (CPU, or one T4 job under the cap) → structural validation (128/128, n_samples 163872,
   sha256 per WAV, seeds match the manifest).
4. Emit 64-item groups → frozen scorer (CPU) → verdict script → `configs/research/xsev_music_native_1_result.json`.
5. Ledger + PROGRESS + manuscript.
