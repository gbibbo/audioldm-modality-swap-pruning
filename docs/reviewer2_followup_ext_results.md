# REVIEWER2-FOLLOWUP-EXT — the 2×2 fine-tuning control: results

**Date:** 2026-09-06 (MVD). **Compute:** three T4 training+eval jobs + one preempted tail; ~16.4 cr GPU, all
scoring/bootstrap on CPU (0 cr). **The campaign hit `USER_STOP_WORKLOAD_REASON_OUT_OF_FUNDS`** (total_spent 135.56
vs 114.95 at authorization = 20.6 cr, i.e. Gabriel's 20-cr pool + Studio, exhausted). **Provenance:** protocol
`docs/reviewer2_followup_ext.md` (frozen + sha256 before any output); verdict `configs/research/r2_EXT2x2_result.json`;
scorer `scripts/research/r2_ext2x2_score.py`; fused CLAP rev `365dea6e`, prompt bootstrap B=10⁴, seed namespace
`REVIEWER2-FOLLOWUP|BOOTSTRAP|2026-09-05|EXT2x2`. Intervals are 95% CIs. **Nothing in the manuscript is changed yet;
`/auditar` is recommended before folding this in (it changes a paper claim and the dense arm is a surprising negative).**

Notation: R = CLAP(fine-tuned) − CLAP(base); J = R(10.24 s) − R(3.84 s); ΔJ = J(trained@10.24) − J(trained@3.84),
paired per prompt (the base terms cancel). The 3.84-trained pruned cell is E3 (already in the paper); the other three
cells are new.

## Pruned arm (item 1 — the reviewer's requested symmetric control) — CLEAN

Base = P (severity-2 pruned). Both cells 192 prompts.

| Checkpoint | R(3.84 s) | R(10.24 s) | J |
|---|---|---|---|
| shortft (trained @3.84, = E3) | +0.009 [−0.006,+0.024] | +0.075 [+0.053,+0.097] | **+0.065 [+0.044,+0.087]** |
| longft (trained @10.24, NEW) | +0.017 [+0.001,+0.033] | +0.048 [+0.025,+0.070] | **+0.031 [+0.010,+0.052]** |

**ΔJ = J_lf − J_sf = −0.035 [−0.064, −0.004] (resolved negative).**

Reading. Both checkpoints recover **more at 10.24 s** than at 3.84 s (both J resolved positive). Training at the
longer duration did **not** tilt the gain toward the longer duration — if anything it tilted slightly *less* (ΔJ
resolved negative). This is the **opposite of the specialization prediction** (which needs ΔJ > 0). The reviewer's
alternative reading — "20 000 steps can't adapt short generation, so the gain only appears at 10.24 s because that
is the only duration the U-Net handles" — is **not supported**: longft, trained at 10.24 s where the base model works,
still gains only +0.048 at 10.24 s and a resolved +0.017 at 3.84 s; neither checkpoint concentrates its gain at its
own training duration. Both converge to a similar operating-point-dependent profile (≈+0.01–0.02 at 3.84 s, ≈+0.05–0.08
at 10.24 s) regardless of where they were trained. **The symmetric control strengthens the anti-specialization /
operating-point conclusion rather than weakening it.**

Wording consequence: the reviewer asked to drop "contradicting" to "does not support" *until this control existed*.
It now exists and points against specialization (ΔJ < 0, not > 0). "Does not support training-duration specialization"
is now **directly demonstrated by a paired symmetric control**, not merely asserted from one arm.

## Dense arm (item 2 — reduced-scale dense analogue) — COMPROMISED BY DEGRADATION (negative result)

Base = dense AudioLDM-M-Full (CLAP 0.207 at 3.84 s, 0.354 at 10.24 s). denseft_short ac_native is 170/192 (OUT_OF_FUNDS
cut the tail); all contrasts on the common set.

| Checkpoint | level(3.84) | level(10.24) | G(3.84) | G(10.24) | J |
|---|---|---|---|---|---|
| denseft_short (trained @3.84) | 0.025 | 0.121 | −0.182 [−0.207,−0.157] | −0.235 [−0.271,−0.199] | −0.051 [−0.090,−0.013] |
| denseft_native (trained @10.24) | 0.041 | 0.131 | −0.166 [−0.190,−0.141] | −0.223 [−0.255,−0.191] | −0.057 [−0.091,−0.023] |

**ΔJ_dense = J_dn − J_ds = −0.005 [−0.034, +0.024] (within SESOI → training-duration-independent).**

Reading. **The 20 000-step full-parameter fine-tune degraded the dense model badly** — CLAP fell from 0.354 to 0.131
at 10.24 s — so every G is strongly negative. The fine-tuned dense checkpoints land at ≈ the same low profile as the
fine-tuned *pruned* checkpoints (0.03 / 0.12), i.e. the aggressive short fine-tune pulls both dense and pruned models
toward a common operating-point-dependent attractor irrespective of the starting quality. This was the pre-registered
risk (M-Full was already AudioCaps-fine-tuned 0.25 M steps; little headroom; `docs/reviewer2_followup_ext.md` §3
"UNINFORMATIVE… declared possible"). **Consequence:** the dense arm does **not** provide a clean "dense recovers
similarly" analogue and must be reported as a negative/limitation, not as support. It does still show **no
training-duration specialization** in the dense model (ΔJ_dense ≈ 0), consistent with the pruned arm's direction.

## Cross-arm

pruned-minus-dense duration response: trained@3.84 +0.011 [−0.008,+0.029] (n=170), trained@10.24 −0.019 [−0.044,+0.005]
(n=192) — the pruned and dense fine-tunes have statistically similar duration responses at matched budget, unsurprising
given both collapse to the same attractor.

## Bottom line for the manuscript (decision pending an audit)

1. **Item 1 is answered well.** The symmetric 10.24-s control confirms recovery is operating-point-dependent, not
   training-duration-specialized, and rebuts the "short is globally broken" alternative. Sec. 4.2 / the abstract can
   state this from a paired control.
2. **Item 2 did not deliver a clean matched analogue.** The dense reduced-scale fine-tune was unstable and degraded the
   model; report it honestly as a negative result and a limitation, not as the "2×2 the paper lamented lacking." The
   matched dense control (Singh's deleted 10⁶-step checkpoint) remains the only clean version and remains unavailable.
3. **denseft_short ac_native is n=170 (OUT_OF_FUNDS).** A ~0.3-cr top-up would finish the 22 WAVs to 192; the result is
   already resolved at 170.

## Audit (2026-09-06, CPU, 0 cr) — is the result trustworthy?

Checked before recommending any manuscript change:

* **The dense degradation is NOT a training blow-up.** denseft training loss is stable and comparable to the pruned
  runs: mean(last 50) = 0.210 (denseft_short), 0.184 (denseft_native), 0.192 (longft); no divergence (diffusion MSE
  is high-variance per timestep, min 0.001 / max 0.79–0.99 for all three). So the CLAP drop (0.354→0.131) is a
  distribution shift, not a diverged optimizer.
* **Two clean explanations, both pre-declared limitations, not artifacts of the pipeline:** (a) **raw vs EMA** — the
  released dense baseline is EMA-averaged; our fine-tune exports RAW weights (protocol §1, "raw, no EMA at this
  horizon"). Raw weights at 20 k steps underperform EMA, which alone costs alignment. (b) **no headroom** — M-Full was
  already AudioCaps-fine-tuned 0.25 M steps, so further AudioCaps steps cannot raise in-domain CLAP and the raw-weight
  penalty dominates. Together they fully account for the dense arm landing below its EMA baseline.
* **Item 1 (ΔJ_pruned) is immune to the raw-vs-EMA concern.** ΔJ = (L10−L3) − (S10−S3): the P baseline cancels
  entirely, and BOTH longft and shortft are raw-weight exports of the SAME pruned backbone with the SAME recipe and
  step count, differing ONLY in training duration. So ΔJ_pruned = −0.035 [−0.064,−0.004] is a clean, symmetric,
  raw-vs-raw paired contrast. It stands.
* **Provenance verified:** each eval loaded the fine-tuned U-Net whose sha256 is recorded in its job's trainer_report
  (`ext2x2_scoring_provenance.json`); the frozen dense/P baselines are the committed XSEV-DENSE-192-CONTROL / xsev
  cells (CRN-paired by prompt).

**Audit verdict:** item 1 is sound and reportable; item 2 is a genuine negative that is *expected* (raw-vs-EMA + no
headroom) and must be reported as a limitation, not as evidence for or against dense recovery. No full adversarial
re-run is warranted (and none is affordable). This supersedes the earlier "‎/auditar recommended" flag for the parts
checked here.
