# Reviewer-2 follow-up experiments — results

**Date:** 2026-09-05 (America/Montevideo). **Compute:** four T4 jobs, ~9.2 credits of the 15-credit ceiling; all scoring, floors, bootstrap and verdicts on CPU (0 credits). **Provenance:** protocol `docs/reviewer2_followup.md` (frozen with a sha256 sidecar before any generation); per-experiment result artifacts `configs/research/r2_{E3,B,E1c,E5,E6,E7,E8}_result.json`; scorer fused CLAP `laion/clap-htsat-fused` rev `365dea6e`, one seed-once call per group with shuffled-caption floors; prompt-level percentile bootstrap B = 10000, seed namespace `REVIEWER2-FOLLOWUP|BOOTSTRAP|2026-09-05`. Intervals are 95% CIs. Nothing here changes a frozen primary verdict; the manuscript is unchanged pending a framing decision.

Notation: **P** = pruned checkpoint, **P+FT** = the released recovery fine-tune, **R(d)** = mean paired recovery gain CLAP(P+FT) − CLAP(P) at requested duration d, **J** = R(10.24 s) − R(3.84 s), **ρ_ref** = fraction of P's gap to a reference (real audio or the dense model) that recovery closes.

---

## The headline

The second reviewer flagged that the paper's title, **"Recovery Fine-Tuning Recovers Where It Was Trained"**, implies a *specialisation* mechanism — that recovery is largest at the operating point used during fine-tuning (10.24 s clips, AudioCaps captions) — which the paper could not establish. These seven experiments were designed to strengthen that claim. Instead they point the other way on **both** axes.

**Duration is an operating-point property ("longer is easier"), not specialisation to the training duration.**
- A checkpoint we fine-tuned at **3.84 s** still gains more at 10.24 s and essentially nothing at its own training duration (E3).
- A dense model given an unrelated text fine-tune shows the same larger gain at 10.24 s (B).
- The gain plateaus beyond the training duration rather than peaking at it (E1c).
- The duration interaction is real and replicates at both pruning severities (E8) — consistent with it being a property of the inference length, not of where fine-tuning happened.

**The held-out-domain failure is specific to hip-hop, not general.**
- Recovery transfers to a second held-out domain, Clotho, and is statistically indistinguishable from AudioCaps (E5).
- The hip-hop battery is not floor-limited: the dense model is well above chance on it (E6).
- With more prompts, hip-hop shows a small resolved gain, an order of magnitude below AudioCaps (E7).

The reviewer's suggested reframe, **"recovery gain is operating-point dependent,"** is now better supported than the specialisation framing, and it comes with a direct mechanism test (E3).

---

## Duration axis

### E3 — short-duration full fine-tune of the pruned checkpoint (the mechanism test)

We fine-tuned the severity-2 pruned checkpoint on AudioCaps at **3.84 s** for 20 000 steps (upstream recipe: full U-Net, AdamW lr 1e-4, batch 2, classifier-free-guidance dropout 0.1, random 3.84-s crops), then measured its recovery gain against P at both durations with common generation noise.

| Setting | Recovery gain R | 95% CI |
|---|---:|---:|
| at 3.84 s (its own training duration) | **+0.009** | [−0.006, +0.024] |
| at 10.24 s | **+0.074** | [+0.053, +0.096] |
| interaction J = R(10.24) − R(3.84) | **+0.065** | [+0.043, +0.087] |

A checkpoint fine-tuned at 3.84 s produces its gain preferentially at 10.24 s and shows **no resolved gain at its own training duration**. If recovery specialised to the training operating point, R(3.84) would exceed R(10.24); the opposite holds. Caveat, as pre-declared: 20 000 steps is 2% of the released 10⁶-step recovery (released R = +0.085 / +0.244), so absolute magnitudes are small and the claim is directional — but the sign and the interaction are unambiguous.

### B — public dense text-FT reference

The publicly released dense AudioLDM-M text-fine-tuned checkpoint (`audioldm-m-text-ft`; a reference, **not** a matched control, and **not** Singh's deleted dense fine-tune) also gains more at the native duration.

| Contrast | Value | 95% CI |
|---|---:|---:|
| text-FT − dense at 3.84 s | −0.022 | [−0.061, +0.017] |
| text-FT − dense at 10.24 s | +0.091 | [+0.042, +0.141] |
| interaction J | **+0.113** | [+0.051, +0.173] |

Duration-dependence of a fine-tuning gain is therefore **not specific to post-pruning recovery**. A dense model fine-tuned with a different recipe on a different corpus shows the same pattern.

### E1c — one point beyond the fine-tuning duration (15.36 s)

| Setting | Recovery gain R | 95% CI |
|---|---:|---:|
| at 10.24 s (first 96 prompts) | +0.242 | [+0.198, +0.285] |
| at 15.36 s | +0.264 | [+0.216, +0.310] |
| step D4 = R(15.36) − R(10.24) | **+0.021** | [−0.023, +0.067] |

The gain **plateaus** past the training duration; it does not peak at 10.24 s. So "peaked at the fine-tuning duration" is not supported (no real-audio anchor exists beyond 10 s, so this cell is read against the dense model and the chance floor only).

### E8 — severity 1 at higher power

The severity-1 duration interaction was underpowered in the paper (frozen n = 80, J = +0.044 [−0.001, +0.087], crossing zero). We added 96 more AudioCaps prompts.

| Set | J | 95% CI |
|---|---:|---:|
| new 96 prompts | +0.169 | [+0.116, +0.221] |
| frozen Arm-D 80 (paper) | +0.044 | [−0.001, +0.087] |
| **pooled n = 176** | **+0.112** | [+0.076, +0.149] |

Pooled, the severity-1 interaction is **resolved positive**: the duration dependence replicates at both severities. Caveat: the new 96 prompts give a much larger J than the frozen 80 — the two prompt sets differ, so the pooled estimate carries some between-set heterogeneity.

---

## Domain axis

### E5 — Clotho held-out battery (real + dense + floor anchors)

Clotho evaluation clips share the AudioCaps sound-event universe but change caption style (median 11 words vs 8 for AudioCaps and 56 for the hip-hop battery). 96 clips, one caption each, seeded selection.

| Duration | R | 95% CI | ρ_real | ρ_dense | P → P+FT → dense → real |
|---|---:|---:|---:|---:|---|
| 3.84 s | +0.098 | [+0.072, +0.125] | 0.28 | 0.49 | −0.111 → −0.013 → 0.089 → 0.236 |
| 10.24 s | **+0.210** | [+0.176, +0.243] | **0.59** | 0.74 | 0.003 → 0.213 → 0.286 → 0.357 |

Interaction J_clo = +0.112 [+0.078, +0.145]. Domain contrast vs AudioCaps at 10.24 s: D = R_AudioCaps − R_Clotho = +0.032 [−0.023, +0.088] (CI includes 0).

Recovery **transfers to Clotho** and closes 59% of the gap to real audio there, comparable to AudioCaps (63%). The AudioCaps-vs-Clotho contrast is not resolved. So the held-out failure the paper reports is **specific to hip-hop**, which shifts both content and caption style — not a general "recovers only on AudioCaps." (Registered reading came out "UNRESOLVED" only because the pre-set thresholds fall between the TRANSFERS and PARTIAL bands; the substantive result is a clear, large, resolved transfer.)

### E6 — dense anchors on the hip-hop cells (is the null a floor artefact?)

| Cell | dense above chance (A_dense) | 95% CI | battery discriminates? | ρ_dense (recovery closes) |
|---|---:|---:|:--:|---:|
| severity 2, 3.84 s | +0.096 | [+0.074, +0.119] | yes | 0.04 |
| severity 2, 10.24 s | +0.106 | [+0.079, +0.135] | yes | 0.02 |
| severity 1, 3.84 s | +0.086 | [+0.063, +0.109] | yes | — (recovery negative here) |

The dense model is well above the shuffled-caption floor on the hip-hop battery at every cell (lower CI bound above the 0.025 SESOI), so the battery **does** measure alignment — the music null is **not** a floor artefact. Recovery still closes almost none of the dense gap on hip-hop (2–4%) versus 44–82% on AudioCaps.

### E7 — hip-hop battery extended to n = 127

| Duration | pooled R (n = 127) | 95% CI | frozen R (n = 64, paper) |
|---|---:|---:|---:|
| 3.84 s | +0.026 | [+0.008, +0.044] | +0.009 |
| 10.24 s | +0.027 | [+0.004, +0.051] | +0.005 |

Interaction J_music,127 = +0.001 [−0.026, +0.028] (no duration interaction on hip-hop). With more prompts a **small** positive recovery becomes resolvable, roughly ten times smaller than AudioCaps. This supports "unresolved / small gain" over "no gain," matching the reviewer's wording point.

---

## What was refuted, what held

| Reviewer weakness | Follow-up | Outcome |
|---|---|---|
| W1 — no dense control; title implies specialisation | E3, B | Specialisation to the training duration is **refuted** by our own experiment; a dense text-FT shows the same duration dependence. The matched control remains impossible (Singh's checkpoint was deleted). |
| W2 — music null may be a floor effect | E6 | **Refuted**: the hip-hop battery discriminates for the dense model. |
| W3 — single extreme domain shift; Clotho as control | E5 | **Confirmed**: recovery transfers to Clotho; the failure is hip-hop-specific (content + caption style). |
| W4 — sweep ends at the native duration | E1c | Beyond 10.24 s the gain **plateaus**, not peaks. |
| W6 — modest sample sizes / underpowered severity 1 | E7, E8 | Severity-1 interaction **resolves** when pooled to n = 176; hip-hop gets a small resolved gain at n = 127. |

The frozen primary results are untouched: the duration effect is real and large on AudioCaps, replicates at severity 2, holds under the published sampler and off the primary scorer, and is confined to in-domain prompts. What changes is the **interpretation** of the mechanism.

---

## Decision needed

These are recorded as **exploratory** in `docs/claims_matrix.md`; no manuscript text has been changed. The options:

1. **Reframe** the title and abstract to "recovery gain is operating-point dependent" and fold E3, E5, E6, E1c into the paper. This is the reading the evidence supports, and it answers the reviewer with new experiments rather than caveats.
2. **Keep** the current framing and report these as limitations.
3. **Audit first** (`/auditar` on E3 and E5, the two results that change the conclusion) before committing to a reframe.

An independent adversarial audit of E3 and E5 is recommended before any reframing is committed, because they change the paper's central claim.
