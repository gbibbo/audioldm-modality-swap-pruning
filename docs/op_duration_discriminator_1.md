# OP-DURATION-DISCRIMINATOR-1 (Arm D) — frozen protocol

**STATUS: PROSPECTIVELY SPECIFIED POST-V1.1 FOLLOW-UP. V1.1 OUTCOME ALREADY OBSERVED; NEW
DURATION-ARM OUTCOMES NOT OBSERVED.**

* Outcome-motivated follow-up (motivated by the V1.1 pre-registered negative + the metric-audit result);
  **NOT** an independent confirmation of the original recovery-reversal hypothesis.
* **V1.1 `PASS = FALSE` remains permanently true and is untouched.** Nothing here modifies any V1/V1.1 or
  metric-audit artifact.
* Arm D addresses **one remaining factor only: temporal extent** (duration 3.84 s → 10.24 s). It is the
  **final planned GPU experiment for this paper**; after it, any untested inference factor (notably
  DDIM 200) is a limitation/future-work item, not a run.
* Frozen and committed **before** the 80-ytid subset is instantiated and before any generation.

## 1. Operating points (only duration differs)

| | Control (already generated in V1.1) | Alternate (new GPU) |
| --- | --- | --- |
| duration | **3.84 s** (latent_t 96) | **10.24 s** (latent_t 256) |
| DDIM steps | 50 | 50 |
| guidance | 2.5 | 2.5 |
| eta | 0 | 0 |
| precision | fp32 | fp32 |
| generation | single (no best-of) | single (no best-of) |
| weight convention | EMA (materialized) | EMA (materialized) |

Latent_t 256 is divisible by 8 (U-Net skip symmetry ✓, per KIM-CLIP-LENGTH). All other framework
settings are held at the V1.1 control values, so the interaction isolates **duration alone**.

## 2. Systems, N, outputs

* Systems: **`p1_pruned_ema_reconstructed`** and **`p1_recovered`** only (no dense; not needed to test
  whether the pruned-vs-recovered ordering moves).
* **N = 80 ytids × 1 replicate** (r0 only).
* New GPU outputs: **80 × 2 = 160 WAVs** at 10.24 s. No dense generation, no DDIM-200 arm, no guidance
  arm, no best-of-3 arm.

## 3. Subset (instantiated in a SEPARATE commit AFTER this freeze — see §F2)

* Eligible universe = the exact frozen 96 V1.1 ytids (`reversal_v1_1_audiocaps_manifest.json`).
* `SUBSET_SALT = "OP-DURATION-DISCRIMINATOR-1|SUBSET|2026-08-30"`.
* Selection: `sha256(f"{SUBSET_SALT}|YTID|{ytid}".encode("utf-8")).hexdigest()`, sort ascending, take
  first 80. **No** outcome-dependent selection; **no** caption/content/label filtering.
* Reuse each selected ytid's already-frozen **V1.1 replicate-0 integer generation seed**
  (`generation_seed(ytid, 0)` = `manifest.generation_seeds[0]`). Same integer PRNG seed across operating
  points; within the ALT arm the same x_T is shared across pruned/recovered per ytid (common-random-
  number). **x_T tensors are NOT identical between 3.84 s and 10.24 s** (shapes (1,8,96,16) vs
  (1,8,256,16) differ) — only the integer seed is shared; this is stated, not hidden.

## 4. Matched CLAP rescoring (D1) — mandatory

Because fused-CLAP preprocessing is RNG/batch/order-sensitive, the interaction must **not** compare an
80-item ALT score against the historical 192-item V1.1 score. For the 80 selected ytids, score **four
matched groups**, each as exactly **one 80-item scorer call**, identical ytid ordering,
`np.random.seed(20260826)` reset once per group, same pinned CLAP (`laion/clap-htsat-fused` rev
`365dea6e`), same preprocessing:

```
CONTROL 3.84 s:  pruned  V1.1 r0  (80 existing WAVs)   |  recovered V1.1 r0  (80 existing WAVs)
ALT    10.24 s:  pruned  new r0   (80 new WAVs)         |  recovered new r0   (80 new WAVs)
```

These are follow-up-specific **matched rescored control values**; V1.1's own 192-item scores stay frozen
and are not reused for the primary contrast.

## 5. CLAP temporal truncation (D2) — documented

The fused-CLAP pipeline truncates/crops the ~10.24 s generated signal to ≈10.0 s under the existing
implementation. Therefore: the ALT audio is 10.24 s, but CLAP evaluates its frozen ≈10 s view; the
**same** preprocessing applies to pruned and recovered, so it does **not** confound the paired
recovered-vs-pruned contrast. Do not describe CLAP as evaluating all 10.24 s if it does not. The exact
implementation path/version is captured in the run provenance. Because the real AudioCaps references are
≈10 s, KL/PANN/FAD at the ALT duration also avoid the gross 3.84-s-vs-10-s mismatch present in V1.1 —
described as **improved duration matching**, not a guarantee of validity.

## 6. Corrected sensitivity (D3) — GATE PASSED

Matched r0-only paired design, n=80 (`scripts/research/armd_sensitivity.py`,
`configs/research/armd_sensitivity.json`). From the V1.1 r0 CLAP contrast (SD 0.147 over all 96;
0.144 over the selected 80): **conservative MDE(J_CLAP, 80% power, ρ=0) = 0.065 ≤ 0.075 → PASS**
(ρ=0.5 → 0.046; ρ=0.7 → 0.036). CI95 half-width ≈ 0.046 at ρ=0. N is fixed at 80; not to be changed
post-freeze without returning to the supervisor.

## 7. Primary analysis (G)

For ytid i (r0 only):
```
r_ctrl_i = CLAP(recovered_ctrl_r0_i) − CLAP(pruned_ctrl_r0_i)    # matched 80-item rescored control
r_alt_i  = CLAP(recovered_alt_i)      − CLAP(pruned_alt_i)        # 10.24 s
j_i      = r_alt_i − r_ctrl_i
J_CLAP   = mean_i j_i
```
Paired percentile bootstrap: unit = ytid, N = 80, B = 10000, **new frozen follow-up seed PCG64(20260830)**;
resample ytids once per iteration, identical sampled indices for control and ALT. Also report
`R_ctrl_80` and `R_alt` with CI95.

## 8. Interpretation (H) — interaction ≠ recovery; NO rescue

* **Duration interaction supported** iff `J_CLAP > 0 AND lo95(J_CLAP) > 0`. If CI excludes 0 negatively,
  duration affects the relative behavior in the opposite direction. If CI crosses 0: no resolved
  interaction.
* **Material recovered advantage at 10.24 s** requires the stronger, separate condition
  `R_alt ≥ +0.025 AND lo95(R_alt) > 0`.
* J+ & R_alt materially+ ⇒ "ordering is temporal-scale-sensitive AND at the recovery model's
  train/eval duration recovered shows a material advantage" (supports temporal-scale conditionality for
  this checkpoint). J+ & R_alt not materially+ ⇒ "duration changes the relationship but does not
  establish substantive recovery" (do NOT call recovery restored). **J null ⇒ matching temporal scale
  does not resolve the tie; do NOT infer guidance/best-of-3 carries the gain — DDIM 50 vs 200 remains an
  unresolved category-A difference.** J negative ⇒ report directly. **No rescue in any branch.**

## 9. Secondaries (I) — run after full ALT generation regardless of J; none affects the primary

Human-CLAP (matched 80-item control-r0 vs ALT interaction, corroborative); KL (validated metric-audit
impl., oriented `R_KL = KL_pruned − KL_recovered`, `J_KL = R_KL_alt − R_KL_ctrl`, paired CI, secondary);
PANN top-10 capture (same paired interaction, descriptive); FAD/FD (same evaluator, descriptive at
n=80). No composite, no majority vote, no promotion to primary.

## 10. Budget

Projected ≈ **1.02 cr**; **HARD MAX NEW GPU SPEND = 1.20 cr**. Do not launch if projected > 1.20 under
current pricing; do not shrink N post-freeze for budget. Expected post-run reserve ≥ 0.26 cr. See
`docs/compute_budget.md` (2026-08-30 reconciliation).

## 11. Finality

**This is the last GPU experiment for the current paper plan.** After Arm D, do NOT automatically
propose or run DDIM 200, guidance changes, best-of-3, another severity, denoiser drift, more adapters,
more domains, human listening, or more prompts. The DDIM-step mismatch becomes a LIMITATIONS item.
