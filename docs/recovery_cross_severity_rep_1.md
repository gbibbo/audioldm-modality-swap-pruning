# RECOVERY-CROSS-SEVERITY-REP-1 — frozen protocol

**STATUS: PROSPECTIVE INDEPENDENT REPLICATION. DESIGNED AFTER SEVERITY-1 RESULTS WERE OBSERVED;
SEVERITY-2 OUTCOMES NOT OBSERVED.** Frozen before any (1,2,1,1) generation/scoring. No manuscript work.
V1.1 `PASS=FALSE` and all severity-1 artifacts are untouched.

## 0. Goal
Independently replicate the severity-1 pattern at a materially stronger pruning severity **(1,2,1,1)**
(~83% U-Net reduction), with independent AudioCaps prompts, independent music prompts, and independent
generation seeds. Only the intended model factor changes: pruning severity/checkpoint.

## 1. Systems (severity 2, [1,2,1,1])
Built by the validated source-agnostic operator `research_pruning/diagnostics/prune_operator.py`
(oracle-validated; `configs/research/xsev_prune_validate.json`):
* **pruned2_A = prune_A′(dense EMA, [1,2,1,1])** — PRIMARY baseline; method-consistent with severity-1's
  `p1_pruned_ema_reconstructed` (A′ oracle: RAW→[1,2,3,1]==published p1 690/690; EMA→[1,2,3,1]==frozen
  sev-1 690/690).
* **pruned2_B = prune_B′(dense EMA, [1,2,1,1])** — PRE-SPECIFIED SENSITIVITY baseline (published-dp1 seam
  convention; B′ oracle: RAW→[1,2,1,1]==published dp1 688/688). Differs from pruned2_A in **exactly 3
  decoder-seam tensors** (§ provenance). NOT primary; NEVER rescues a failed A′ gate.
* **recovered2 = `l1_p1_dp1_finetuned_global_step_999999.ckpt`** (md5 `5d7da150…`, EMA-materialized,
  global_step 1,000,000). Shared across A′ and B′ contrasts — **not regenerated for B′**.

## 2. Manifests (frozen, independent)
* AudioCaps: `configs/research/xsev_audiocaps_manifest.json` — **manifest_sha256 `4da90661…`**,
  prompts_sha256 `e33d6613…`, **N=192** unique ytids, salt `RECOVERY-CROSS-SEVERITY-REP-1|AUDIOCAPS|
  2026-08-30`, caption salt `…|CAPTION|…`; excludes V1.1's 96 + train + music64 + kim44 (868 eligible);
  1 caption/ytid (V1.1 five-row multiset convention); **0 overlap with V1.1**.
* Music: `configs/research/xsev_music_manifest.json` — **manifest_sha256 `f5a26fbe…`**, prompts_sha256
  `cbea7e8b…`, **64** prompts × **3** seeds, salt `…|MUSIC|2026-08-30`; identical gate0 eligibility
  (keyword+kim-source+exact/neardup0.85+selfdedup) + excludes the frozen severity-1 64 (127 eligible);
  **0 overlap with frozen 64**. No rule loosening.
* Generation salt `RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30`.

## 3. Operating points
| context | duration | latent_t | DDIM | guidance | eta | prec | gen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AudioCaps SHORT | 3.84 s | 96 | 50 | 2.5 | 0 | fp32 | single |
| AudioCaps NATIVE | 10.24 s | 256 | 50 | 2.5 | 0 | fp32 | single |
| music | 3.84 s | 96 | 50 | 2.5 | 0 | fp32 | single (gate0 recipe) |

Within each (ytid, operating_point), pruned2_A / pruned2_B / recovered2 receive **identical x_T** (CRN).
Across SHORT/NATIVE the same integer seed is used per ytid, but x_T tensors are **not** identical
(shapes (1,8,96,16) vs (1,8,256,16)). EMA weight convention throughout.

## 4. Dense@10.24 s control (severity-1 completion)
`dense_ema` on the **existing Arm-D 80 ytids + r0 seeds**, 10.24 s / DDIM50 / g2.5 / eta0 / fp32 /
single (80 WAVs). Reports **G_pruned = C_dense − C_pruned@10.24** and **G_recovered = C_dense −
C_recovered@10.24** (paired-ytid CIs) — the practical dense ceiling. No "restored" language.

## 5. Primary confirmatory (A′ only; frozen before any sev-2 output)
CLAP contrasts, oriented so + favors recovered. `R_native_A = C_rec_native − C_prunedA_native`,
`R_music_A = C_rec_music − C_prunedA_music`, `R_short_A = C_rec_short − C_prunedA_short`.
* **PRIMARY — contextual dependence: `K_A = R_native_A − R_music_A`; PASS ⇔ `lo95(K_A) > 0`.** (K_A>0
  does **not** by itself establish sign reversal.) Bootstrap resamples the AudioCaps-192 and music-64
  populations **independently**.
* **Sign-pattern replication (conjunction, both required):** `H_native_A`: `R_native_A ≥ +0.025 ∧
  lo95(R_native_A) > 0`; `H_music_A`: `R_music_A ≤ −0.025 ∧ upper95(R_music_A) < 0`.
* **Key secondary confirmatory — duration interaction: `J_A = R_native_A − R_short_A`; gate
  `lo95(J_A) > 0`** (N=192 → ~84% power for the sev-1 magnitude 0.044; gate unchanged from severity 1).
* **Secondary — short-OP equivalence:** TOST / 90% CI within `[−0.025, +0.025]` for `R_short_A`
  (adequately powered at N=192 if the true effect ≈ 0). Not co-primary; reported regardless.
Bootstrap: unit = ytid (AudioCaps) / prompt (music), B=10000, **new frozen seed PCG64(20260831)**.
Music: average the 3 replicates to a per-prompt contrast **before** the prompt-level bootstrap (do NOT
treat 192 music WAVs as 192 independent observations).

## 6. B′ sensitivity (pre-specified; identical estimands/gates with pruned2_B)
Repeat every estimand as `recovered2 − pruned2_B`: `R_native_B, R_music_B, K_B, R_short_B, J_B`. Primary
inference is A′ only; **B′ never rescues a failed A′ result.**
* **SEAM-ROBUST** may be claimed for a given condition only if (i) the A′ gate passes AND (ii) the B′
  analysis preserves the same substantive conclusion (for sign-pattern: `R_native_B ≥ +0.025 ∧
  lo95>0` AND `R_music_B ≤ −0.025 ∧ upper95<0`; for K: `lo95(K_B)>0`; for J: `lo95(J_B)>0`).
* **A′ passes, B′ does not** → state the severity-2 conclusion is **sensitive to the source checkpoint's
  three-tensor decoder-seam convention ambiguity** (report, do not hide).
* **A′ fails, B′ passes** → primary result remains FAILED; do NOT rescue; report the source-convention
  produces a different secondary result but the method-consistent preregistered replication did not pass.

## 7. Metrics
CLAP `laion/clap-htsat-fused` rev `365dea6e…` is the ONLY confirmatory family. Matched scoring: exactly
one 192-item seed-once scorer call per (system, context) — AudioCaps 192 ytids×1; music 64×3=192 —
canonical order, `np.random.seed(20260826)` reset once per group, pinned model/preprocessing. Secondary
corroborative (no rescue, no vote): Human-CLAP, and for AudioCaps KL / PANN top-10 capture / FAD / FD
(validated metric-audit pipeline). Music KL/PANN/FAD/FD only if real reference clips are obtainable
under the exact frozen battery with valid provenance; reference availability must not alter the primary
music set.

## 8. Source-artifact provenance (design-affecting; not a paper claim)
**Cross-checkpoint pruning-convention inconsistency:** the published severity-1 `l1_p1` and severity-2
`l1_p1_dp1` checkpoints encode **contradictory** channel-selection conventions for the **same 3
identically-shaped decoder concat-seam tensors** (`output_blocks.0.0.in_layers.2.weight`,
`output_blocks.1.0.in_layers.2.weight`, `output_blocks.2.0.in_layers.2.bias`): at (1,2,3,1) the two
weights are positional and the bias ranked; at (1,2,1,1) the reverse. Proven that no single deterministic
operator reproduces both published RAW checkpoints bit-exact. Neutral terminology only; no misconduct
inferred; no cause inferred. This motivates the pre-specified A′/B′ seam-sensitivity design.

## 9. Interpretation (frozen)
* **Case A** — K_A pass ∧ H_native_A ∧ H_music_A ∧ J_A pass: strongest — independent cross-severity
  replication of contextual dependence, native-scale recovered advantage, OOD penalty, and positive
  temporal-scale interaction.
* **Case B** — K_A + sign-pattern pass, J_A fails: cross-context sign pattern replicates; formal temporal
  interaction not independently established (do NOT rescue J).
* **Case C** — K_A passes, one/both sign-pattern components fail: contextual dependence replicates; the
  specific native-positive/music-negative pattern does not fully replicate.
* **Case D** — K_A fails: the principal cross-context phenomenon does not independently replicate at the
  second severity/sample. Report and STOP. No third severity, no tuning.
Each overlaid with the §6 seam-robustness qualifier.

## 10. Budget
1808 new WAVs (SHORT 576 + NATIVE 576 + music 576 + dense 80). Conservative **~4.88 cr**; likely ~4.2
with the smaller (1,2,1,1) U-Net. **TARGET ≤ 5.0 cr; ABSOLUTE HARD MAX NEW GPU SPEND = 5.5 cr** (watchdog
terminates before exceeding). N is frozen at 192; not reduced post-freeze for budget. ~6 cr available.

## 11. Finality
This is the last planned experiment for the current paper plan. No DDIM-200, no third severity, no
manuscript work until explicit supervisor GO after adversarial audits.
