# RECOVERY-CROSS-SEVERITY-REP-1 — Post-Result Adversarial Audit

**Type:** post-result scientific audit (NOT manuscript prose). CPU-only, 0 GPU, 0 new generation.
**Scope rule:** no frozen result, gate, threshold, CASE C verdict, or raw score was altered. Frozen
protocol history and self-hashed verified result artifacts are preserved byte-identical. Corrections
below are metadata/description only and are recorded here + in the living state docs; the
supervisor-verified `configs/research/xsev_result.json` (sha `02e3bd11…`) and
`xsev_secondary_metrics.json` (sha `e5a167fb…`) are NOT rewritten — this document **supersedes** the
descriptive `note` wording in them.

Audit date: 2026-08-31 (America/Montevideo). Final result commit under audit: `8f158067`.

---

## 1. Reporting corrections (metadata/description only; no number changed)

### A. FAD / FD are descriptive, not inferential
The durable summaries stated, for KL / PANN / FD / FAD, wording equivalent to "all four corroborate …
CIs exclude 0". **This is false.** The implementation (`scripts/research/xsev_secondary_metrics.py`)
computes a paired ytid bootstrap CI (PCG64(20260831), B=10000) ONLY for the two contrasts `R_KL` and
`R_cap`. FD (PANN-2048 Fréchet distance) and FAD (VGGish Fréchet Audio Distance) are each a **single
descriptive distributional statistic** (one scalar per system computed over the whole 192-clip set);
they have **no paired CI** in the current implementation.

Corrected durable statement (applied to PROGRESS.md and the ledger; supersedes the JSON `note`):
* **KL and PANN-capture** contrast CIs exclude zero — `R_KL +2.224 [1.926, 2.530]`,
  `R_cap +0.859 [0.698, 1.021]` (paired ytid bootstrap).
* **FD and FAD** are descriptive and directionally favor recovered (FD 48.7 vs 107.2; FAD 6.92 vs
  27.4) — **no inferential CI claim**.

### B. PANN terminology — a count, not a "capture rate"
The PANN quantity is, per clip, `len( set(top-10 predicted labels) ∩ set(ground-truth labels) )`,
which can exceed 1 (a clip may have several ground-truth labels). Its mean (1.464 recovered / 0.604
pruned) is therefore a **captured-label count**, not a rate/fraction. Precise term adopted:
**"PANN top-10 captured-label count"** = mean number of ground-truth labels appearing in the PANN
top-10 predictions. No durable artifact ever used the literal string "capture rate" (verified by
grep); the values are unchanged, only the descriptive term is fixed to avoid a rate misreading.

---

## 2. Chronology (immutable; commit SHA · Montevideo time · were sev-2 primary scores known yet)

| Stage | Commit | Time | Sev-2 primary scores known? |
|---|---|---|---|
| Protocol freeze (proto sha `19c50cc3`) | `3f08059` | 01:18 | No |
| Generation code (sev-2 generator) | `afe76f8` | 01:31 | No |
| Generation launched | `72d4c30` | 01:32 | No |
| Generation complete (1728/1728) + resume filter | `e251523`/`adf6d5c` | 12:36/12:51 | No (structural only) |
| Pre-primary scoring machinery (verdict + dense-union validator) | `94b868c` | 13:15 | No |
| Dense completion (union 80/80 PASS) | `ea6282d` | 13:26 | No |
| Human-CLAP secondary **scorer** | `b723e58` | 13:27 | (before result commit) |
| **Primary scoring → primary result (CASE C)** | `60cad5c` | 13:42 | **Yes (this is the result)** |
| KL/PANN/FAD/FD secondary **driver** | `7d42217` | 13:44 | **Yes — descends from `60cad5c`** |
| Human-CLAP secondary **result** | `e0c65af` | 13:51 | Yes |
| Final KL/PANN/FAD/FD secondary **result** | `8f15806` | 14:01 | Yes |

**Blunt answers to the audit questions:**
1. **Any primary CLAP endpoint/gate/bootstrap/scoring changed after sev-2 primary scores observed? —
   NO.** The primary verdict script `xsev_score_verdict.py` is the **identical git blob**
   (`b21aca77…`) at pre-result `94b868c` and at result `60cad5c`. The scoring machinery
   (`xsev_score_emit.py`, `gate0_clap_scorer.py`, `cluster_bootstrap.py`, `reversal.py`) was committed
   at `94b868c` (13:15) before the result (13:42); the protocol froze at `3f08059` (01:18) before
   generation. No endpoint/gate/bootstrap/threshold moved post-observation.
2. **Any generated observations dropped/replaced after inspection? — NO.** All 1728/1728 severity-2
   WAVs were used; structural validation was operational-only (byte/finiteness/CRN; no CLAP peeking).
   The 7 completed dense-tail WAVs (73→80) belong to the SECONDARY dense control, not the primary A′
   battery. No primary observation was dropped or replaced.
3. **Any primary manifests changed post-freeze? — NO.** AudioCaps `4da90661…` and music `f5a26fbe…`
   are the frozen (`3f08059`) manifests; the result artifact records those same prefixes.
4. **Was B′ ever used to select or reinterpret A′? — NO.** `classify()` reads only A′ gates;
   `seam_robust()` requires A′ pass AND B′ preserve; B′ can never rescue a failed A′ (enforced in
   code + `primary_note`).
5. **Secondary implementations before vs after primary results:**
   * Human-CLAP **scorer** implementation (`b723e58`) predates the primary **result commit**; its
     result (`e0c65af`) is after. Corroborative only, no gate role.
   * **KL/PANN/FAD/FD driver (`7d42217`) was implemented AFTER the primary result** — it is a git
     descendant of the result commit `60cad5c`. Its commit-message tag "(pre-results)" is
     **inaccurate as to implementation timing** and is corrected here.

**Correct classification of the KL/PANN/FAD/FD secondary:** *pre-specified secondary* (named in the
frozen protocol §7, sha `19c50cc3`, lines 81–83: "Secondary corroborative (no rescue, no vote):
Human-CLAP, and for AudioCaps KL / PANN top-10 capture / FAD / FD"), **implementation written after
the primary result**, **no primary/gate/rescue role**. The metric choice is preregistered; the code
that computed it was written post-result. This is scientifically acceptable (the metric cannot change
the gate) but must not be described as "pre-results".

**Primary implementation chronology verdict: CLEAN.**

---

## 3. Primary reproduction audit (CPU)

Re-ran `scripts/research/xsev_score_verdict.py` from the persisted frozen-scorer outputs
(`xsev_sev2_groups_out.json`, `xsev_dense_groups_out.json`).

**Result: exact reproduction. Max numerical discrepancy = 0.0 (bit-for-bit)** against both the
persisted verdict artifact and the durable `xsev_result.json` across `PRIMARY_A`, `SENSITIVITY_B`,
and `DENSE_CONTROL`.

Verified invariants (by code inspection of `cluster_bootstrap.py` / `reversal.py` /
`xsev_score_verdict.py` / `xsev_score_emit.py`):
* AudioCaps bootstrap unit = **ytid** (192×1 grid → 192 per-prompt scalars); music **64×3 reduced to
  per-prompt (mean over 3 replicates) BEFORE bootstrap** (`_paired_prompt_diff`, then
  `cluster_percentile_ci`).
* **K** = `interaction_ci` resamples AudioCaps (192) and music (64) **independently** (two index
  draws in one PCG64(20260831) stream).
* **J** = `cluster_percentile_ci(native_diff − short_diff)` — paired per-ytid contrast, n=192.
* **B = 10000**, **PCG64 seed 20260831**, **SESOI = ±0.025** exactly; 90% CI for the short-equivalence
  TOST; scorer group cardinality 192 per (system,context) canonical order; scorer seed reset
  `np.random.seed(20260826)` once per group.

Reproduced headline (unchanged): R_native +0.2443 [+0.2145,+0.2729] (H_native TRUE); R_music +0.0092
[−0.0134,+0.0319] (H_music FALSE); R_short +0.0849 [+0.0659,+0.1051]; K_A +0.2351 [+0.1974,+0.2719]
PASS; J_A +0.1594 [+0.1309,+0.1873] PASS; short-equiv FALSE; **CASE C**. B′ near-identical; seam-robust
(K TRUE, J TRUE, sign-pattern FALSE under both A′ and B′). Dense: G_pruned +0.0994 [+0.0576,+0.1399],
G_recovered +0.0476 [−0.00035,+0.0957].

**PRIMARY AUDIT: PASS. Max reproduction error 0.0.**

---

## 4. CASE C interpretation — claim-status table

| # | Statement | Verdict |
|---|---|---|
| A | Context dependence of recovered-vs-pruned replicated at severity 2 | **SUPPORTED** — K_A +0.235 [0.197,0.272], lo95>0, seam-robust |
| B | Native-positive / music-negative **sign-pattern** replicated | **FALSE** — H_native TRUE but H_music FALSE (R_music ~null); the conjunction is the defining failure of CASE C |
| C | Positive temporal-scale interaction replicated at severity 2 | **SUPPORTED** — J_A +0.159 [0.131,0.187], lo95>0, seam-robust (sev-1 J was borderline; sev-2 resolves it) |
| D | Recovery advantage **disappears at 3.84 s** | **FALSE** — at sev-2 short, R_short +0.085 [0.066,0.105], lo95>0 (short-equivalence FAILS): the advantage is present and resolved at 3.84 s |
| E | Recovery advantage is **larger at 10.24 s than 3.84 s** | **SUPPORTED** — this is exactly J_A = R_native − R_short +0.159 [0.131,0.187] |
| F | Recovery causes an **OOD music penalty** | **FALSE** (at severity 2) — R_music +0.009 [−0.013,+0.032], H_music FALSE; recovered ≈ pruned on music |
| G | Result is **robust to the three-tensor seam convention** | **SUPPORTED** — K and J seam_robust (A′ and B′ both pass); sign-pattern fails under BOTH (failure is not a seam artifact) |
| H | Recovered is **restored to dense at 10.24 s** | **FALSE** — dense−recovered +0.048 [−0.00035,+0.096]; not resolved from 0 ≠ equivalence; also a severity-1 control, not the sev-2 systems (see §8) |
| I | Recovery behavior **differs with pruning severity** | **descriptively SUGGESTED, NOT formally established** — the §5 cross-severity test is POST-HOC, exploratory, and confounds severity with experiment/prompt-set/checkpoint; there is no preregistered cross-severity inferential test |

---

## 5. POST-HOC cross-severity heterogeneity (EXPLORATORY — cannot become preregistered evidence)

Independent two-sample percentile bootstrap (disjoint prompt populations: sev-1 = Arm-D 80 ytids @
10.24/3.84 + frozen music-64; sev-2 = xsev 192 ytids + independent music-64, 0 overlap), B=10000,
**NEW explicitly post-hoc seed PCG64(2026083101)**. Prompt units preserved; no pretense of pairing
across severity.

| Contrast | Point | 95% CI | CI excludes 0 |
|---|---|---|---|
| Δseverity_native = R_native(sev2) − R_native(sev1) | +0.1925 | [+0.1421, +0.2453] | yes |
| Δseverity_short  = R_short(sev2)  − R_short(sev1)  | +0.0773 | [+0.0401, +0.1140] | yes |
| Δseverity_music  = R_music(sev2)  − R_music(sev1)  | +0.1033 | [+0.0661, +0.1411] | yes |
| ΔJ = J(sev2) − J(sev1) | +0.1152 | [+0.0630, +0.1689] | yes |

(sev-1: R_native +0.0518, R_short +0.0076, R_music −0.0941, J +0.0442; sev-2: R_native +0.2443,
R_short +0.0849, R_music +0.0092, J +0.1594.)

**Reading (bounded):** at the stronger severity (1,2,1,1) every recovered-vs-pruned effect is larger
in-context (native, short) and the music **penalty vanishes** (−0.094 → ~0). All four deltas have
post-hoc CIs excluding zero, so the data **support a future limited exploratory statement** —
"*the magnitude and cross-context profile of recovery appears severity-dependent*" — and NOT a
primary/causal severity claim. **Confound (must accompany any use):** severity is entangled with two
different experiments, prompt sets, n, and checkpoints; this is exploratory, post-hoc, and cannot be
promoted to preregistered evidence.

---

## 6. Music low-score / floor diagnostic (CPU, descriptive)

Severity-2 music **raw clip CLAP scores** (n=192/system):

| System | min | p05 | p25 | med | p75 | p95 | max | mean | SD | frac≤0 | frac≤0.01 | frac≤0.02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recovered2 music | −0.287 | −0.171 | −0.081 | +0.005 | +0.097 | +0.224 | +0.350 | +0.014 | 0.124 | 0.490 | 0.516 | 0.526 |
| pruned2_A music | −0.369 | −0.193 | −0.067 | +0.001 | +0.062 | +0.250 | +0.370 | +0.005 | 0.124 | 0.495 | 0.542 | 0.589 |
| paired rec−pru | −0.326 | −0.176 | −0.068 | +0.004 | +0.089 | +0.217 | +0.355 | +0.009 | 0.120 | 0.490 | 0.521 | 0.583 |

Comparators (sev-2 raw): AC-short pruned mean +0.015 SD 0.116; AC-short recovered +0.099 SD 0.118;
AC-native recovered +0.299 SD 0.177; AC-native pruned +0.055 SD 0.122.

**Verdict: "near scorer floor" is NOT supported.** Both music systems are centered near 0 with the
**full CLAP spread** (SD ≈ 0.124, essentially identical to the AC-short pruned distribution), roughly
symmetric, with ~49% of clips **below zero** and p95 ≈ +0.22–0.25. A floor would show mass piled at a
minimum with collapsed variance; instead the variance is full and the distribution straddles zero.
The correct statement is **"very low absolute CLAP scores (mean ≈ 0) with full spread for both
systems"**, reflecting poor text–audio alignment on OOD hip-hop music — **not** scorer saturation. The
absence of a recovered music penalty (F FALSE) is a genuine near-null contrast, not a floor artifact.

---

## 7. K interpretation — conceptual decomposition

`K_A = R_native(AudioCaps, 10.24 s) − R_music(music, 3.84 s)` changes **both** the domain/prompt
population (in-domain AudioCaps ↔ OOD music) **and** the temporal operating point (10.24 s ↔ 3.84 s).
K alone therefore **confounds domain and duration**.

* **Safe terminology from K alone:** "cross-context dependence", "context dependence" (context = the
  bundled domain+duration operating point).
* **Unsafe from K alone:** "domain interaction", "OOD penalty", "pure domain effect" — the native-vs-
  music gap also spans a duration change and cannot be attributed to domain alone.

**Decomposition available in this experiment (unlike a single K):**
K = [R_native − R_short] + [R_short − R_music] = **J (duration, same domain)** + **(domain at matched
3.84 s)**. Numerically J = +0.159 dominates; the matched-duration domain gap R_short − R_music ≈
+0.076 (music ~null, not negative). So **K's magnitude is duration-dominated**, and there is no
resolved OOD music penalty at matched duration. Contrast with the historical severity-1 AudioCaps-vs-
music comparison, which was evaluated at the **same** short (3.84 s) operating point and was therefore
a cleaner domain contrast (and there music was negative, −0.094). Use "context dependence" for K; do
not label it a domain interaction or OOD penalty.

---

## 8. Dense control interpretation

`dense − recovered = +0.0476 [−0.00035, +0.0957]` (n=80).

* No **statistically resolved** dense-vs-recovered difference at this n (lower CI marginally includes
  0 at −0.00035).
* Absence of significance is **NOT equivalence** (no TOST; point +0.048 is not trivially small).
* **No "restored to dense" claim.**
* The descriptive "closes ~half the gap" ratio (G_recovered 0.048 vs G_pruned 0.099) is descriptive
  only and is not promoted.
* **Additional caveat:** this control uses the **severity-1** Arm-D systems (`recovered_sev1`,
  `pruned_sev1`) against dense@10.24 s; it is a severity-1 completion, decoupled from the severity-2
  primary systems (which have no dense@10.24 s comparison).

**Maximum defensible statement:** "At 10.24 s, pruning opens a resolved gap below dense
(G_pruned +0.099 [0.058,0.140]); the recovered checkpoint is descriptively closer to dense
(G_recovered +0.048) but the data neither resolve a residual recovered-vs-dense deficit nor establish
equivalence to dense (this is the severity-1 lineage)."

---

## 9. Secondary metric semantics

* **KL** — lower = better (KL divergence of PANNs label distributions vs real refs). Stored contrast
  `R_KL = KL_pruned − KL_recovered = +2.224 > 0` ⇒ recovered has the **lower** KL ⇒ better. Paired
  ytid bootstrap CI [1.926, 2.530] excludes 0. **Recovered better, inferential.**
* **PANN** — quantity = per-clip `|top-10 predicted ∩ ground-truth|`, a **captured-label count**
  (can exceed 1), mean 1.464 vs 0.604. Renamed "PANN top-10 captured-label count". `R_cap` paired CI
  [0.698, 1.021] excludes 0. **Recovered better, inferential.**
* **FD / FAD** — lower = better; **descriptive only**, one scalar per system, **no paired CI** in the
  current implementation. FD 48.7 vs 107.2; FAD 6.92 vs 27.4 — directionally favor recovered, no
  inferential claim.
* **Human-CLAP** — corroborative, different scale, no SESOI, no PASS role. HC music
  `R_music_A = −0.037 [−0.068, −0.005]` (CI excludes 0 — HC does show the music-negative direction).
  **This cannot override the primary CLAP `H_music = FALSE`:** HC is a secondary scorer with no gate
  role; the primary endpoint is CLAP. The music sign-pattern remains **not established**; CASE C
  stands.

---

## 10. Music battery eligibility funnel (reconciled)

From the frozen music manifest diagnostics (`xsev_music_manifest.json`, manifest_sha `f5a26fbe…`):

```
keyword-eligible candidates (intermediate)         235   (diagnostics.keyword_filtered)
  − Kim-source ytid exclusions                     − 44   (dropped_source_ytid)
  − prior frozen severity-1 music-64 exclusion     − 64   (dropped_frozen64)
  − exact duplicates                               −  0   (dropped_exact_dup)
  − near-duplicates (threshold 0.85)               −  0   (dropped_neardup)
  − self-dedup                                     −  0
  = final eligible                                  127
  selected (deterministic, new salt)                 64
```
Arithmetic checks: 235 − 44 − 64 = **127**; 0 removed by exact/near-dup/self-dedup.

* The "**262**" rough count mentioned in earlier design-audit discussion **does not appear in any
  durable artifact** (grep-verified); the recorded intermediate is the keyword-eligible **235**. Treat
  262 as an earlier unrecorded/unfiltered candidate-pool figure, not a frozen number.
* **No rule loosening** (eligibility "IDENTICAL to gate0 battery … new salt; no rule loosening";
  the selector refuses if eligible < N rather than relaxing).
* **No outcome-dependent filtering** (selection is a deterministic hash order under a pre-frozen salt,
  fixed before generation).
* **0 overlap with frozen severity-1 music-64** — guaranteed by the explicit `dropped_frozen64 = 64`
  exclusion step.

---

## 11. Would DDIM200 materially change the paper? — **LOW (low end; upper-LOW / lower-MODERATE)**

Based only on the now-complete evidence (no design, not planned):
1. **Objection it would close:** "your operating point (DDIM50, short/native clips, guidance 2.5,
   single-gen) is off Singh's published recipe (DDIM200, 10 s, guidance 3.5, best-of-3), so your
   effects may be a low-step artifact." DDIM200 closes **one of ~four** op-point axes (sampler steps).
2. **Fatal or limitation?** A **limitation**, not fatal — the claim is a controlled recovered-vs-
   pruned contrast with both systems evaluated identically (internal validity intact); it does not
   assert reproduction of Singh's absolute FAD.
3. **Claim it would unlock:** that the native-scale recovered advantage **persists at the published
   sampler setting** — an external-validity strengthening only; it cannot change the CASE C verdict.
4. **Would it distinguish generic AudioCaps fine-tuning from pruning+recovery?** **No (as expected).**
   DDIM200 changes the sampler, not the design; it adds no fine-tuning-only control arm, so the more
   serious confound is untouched.
5. **Does the strong, multiply-corroborated, cross-severity native result reduce DDIM200's value
   relative to before this replication?** **Yes.** The native advantage is now large and corroborated
   by CLAP + HC + KL + PANN + FD + FAD and replicated across severity; one more sampler setting is a
   marginal robustness check.

**Verdict: LOW value.** It closes a limitation-grade objection on one axis and does not address the
central confound. The single most valuable missing experiment is a **fine-tuning-only control**
(pruned backbone given an equal AudioCaps fine-tune budget without the recovery-specific recipe), not
DDIM200. Not recommending spend.

---

## 12. Publication-readiness (scientific material only) — **2.5 / 5**

Adversarial ICASSP reviewer, after all current results (manuscript excluded from this audit).

**Strengths:** prospective preregistration and frozen protocol; bit-exact reproducible pipeline;
independent cross-severity replication of context-dependence (K) and the duration interaction (J);
seam-robustness (A′/B′); native-scale advantage corroborated across six metric families; honest
preregistered negatives (V1.1 reversal rejected; music sign-pattern not replicated → CASE C).

**Strongest remaining rejection argument (the crux):** **the fine-tuning confound.** "Recovered" =
pruned + additional AudioCaps fine-tuning, and no arm isolates the pruning-aware recovery mechanism
from generic extra same-domain fine-tuning. The surviving positive result ("recovered beats pruned at
native scale, context/duration-dependent") is consistent with "more in-domain fine-tuning helps at
native resolution" — a weaker, less-novel claim than the original OOD-reversal thesis, which was
rejected. Secondary weaknesses: off-recipe operating point vs Singh (limitation); single scorer family
(CLAP + HC both CLAP-derived), no human evaluation.

**What could move the score by ≥0.5:** a **fine-tuning-only control arm** that isolates the recovery
mechanism — if recovered still beats an equal-budget generic AudioCaps fine-tune, the novelty is
established (→ ~3.0–3.5). Alternatively, reframing the contribution as a rigorous **preregistered-
negative + temporal-scale/context-dependence findings** paper could aid acceptance as an analysis
contribution. **DDIM200 alone would NOT move the score ≥0.5.**

---

## Provenance

* Reproduction: `scripts/research/xsev_score_verdict.py` re-run from persisted scorer outputs; max
  diff 0.0.
* §5/§6 computed from persisted raw cosines (`xsev_sev2_groups_out.json`,
  `op_duration_discriminator_1_result.json` raw_cosines, `reversal_v1_r_music_clap.json`).
* Frozen protocol `docs/recovery_cross_severity_rep_1.md` sha `19c50cc3…` (intact, unaltered).
* Result artifacts `xsev_result.json` (`02e3bd11…`), `xsev_secondary_metrics.json` (`e5a167fb…`),
  `xsev_hc_secondary.json` (`beddaf9a…`) preserved byte-identical; this audit supersedes their
  descriptive `note` wording per §1.
