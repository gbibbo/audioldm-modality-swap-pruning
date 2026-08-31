# Dense Fine-Tuned Baseline (Singh) — Availability & Control Audit

**Type:** zero-GPU / read-only audit (2026-08-31, MVD). No generation, no manuscript work, no frozen
artifact modified, no new scientific claim. Supersedes nothing; refines the framing-dependent
readiness assessment of `docs/xsev_postresult_adversarial_audit.md` §12 (see §6 below).

**Conceptual correction adopted (Gabriel, 2026-08-31):** in Singh et al. (arXiv 2607.13330),
*recovery = AudioCaps fine-tuning of the pruned U-Net (1M steps)*. A "pruned + generic fine-tuning"
control is therefore ill-posed — it would re-run recovery. The scientifically correct control is
**dense_pretrained vs dense + AudioCaps fine-tuning**, using the fine-tuned unpruned baseline the
paper reports having trained.

---

## 1. Source evidence (explicit vs inferred)

**Explicit (verbatim, arXiv 2607.13330 full text, fetched 2026-08-31):**
* "For a fair comparison, we also apply finetuning to the unpruned baseline model and evaluate its
  performance relative to that of the pruned models." → the dense-FT baseline **was trained**.
* "Although finetuning improves the performance of the unpruned baseline model, it still requires a
  higher parameter count and more compute in finetuning than the pruned model, even though both
  achieve comparable performance." → outcome reported **qualitatively only**.
* "The pruned AudioLDM-M-Full models are finetuned on the AudioCaps training dataset for 1M steps."
  → 1M steps stated **for the pruned models**.
* Unpruned (no-FT) figures: FAD 3.95 / KL 2.16. **No FAD/KL is published for the fine-tuned unpruned
  model** (checked twice; only the qualitative "comparable performance" sentence).
* Repo README pipeline (explicit): `latent_diffusion.py -c audioldm_original_medium.yaml
  --reload_from_ckpt …` on AudioCaps train — full-model fine-tuning (no adapter/freezing mentioned).

**Inferred (not stated in the sources):** the dense-FT step count (presumably the same 1M recipe),
and the exact trainable-module set for the dense FT (presumably the same full-pipeline trainer).
Also prior ledger fact: M-Full itself was already AudioCaps-finetuned 0.25M steps by the AudioLDM
authors, so "dense pretrained" is not AudioCaps-naive — the control isolates the *additional* 1M-step
specialization.

## 2. Artifact inventory — **the dense-FT checkpoint is NOT released**

Live Zenodo API, record **21977996** ("Checkpoints for pruned and Unpruned (baseline) AudioLDM-M-Full
model", 2026-08-17, latest of concept 21376821; earlier version 21376822 checked too): **8 files,
none is a fine-tuned unpruned model** —

| file | bytes | md5 | identity |
|---|---|---|---|
| audioldm-m-full.ckpt | 4,571,683,377 | 46bad9f1… | dense **pretrained** (md5 = official AudioLDM record 7884686 ⇒ provably NOT Singh-fine-tuned) |
| Unet_model-m.ckpt | 1,664,094,433 | e44eaa7c… | pretrained U-Net only |
| l1_audioldm-m-full_p1.ckpt | 3,490,506,986 | 2666e6fc… | pruned (1,2,3,1), no FT |
| l1_audioldm-m-full_p1_dp1.ckpt | 3,192,129,326 | 2427ffb5… | pruned (1,2,1,1), no FT |
| l1_unet_pruned_p2_dp2.pt | 540,384,123 | 9ca234f8… | pruned U-Net-only (1,2,2,2) |
| l1_p1_finetuned_global_step_999999.ckpt | 4,446,514,762 | cfb7ca3f… | **pruned+FT (recovered, sev-1)** — local |
| l1_p1_dp1_finetuned_global_step_999999.ckpt | 3,244,099,100 | 5d7da150… | **pruned+FT (recovered, sev-2)** — local |
| sorted_indexes_dict.pkl | 59,112 | a4cd11ff… | L1 ranking |

Local `data/checkpoints/` holds the base + both recovered checkpoints (md5-verified previously);
nothing dense-FT-like. GitHub README: no mention of releasing a dense-FT checkpoint. Zenodo search
(record versions + keyword search) found no other candidate record. **Ambiguous candidates
eliminated:** `audioldm-m-full.ckpt` (md5-proven pretrained), `Unet_model-m.ckpt` (pretrained),
`l1_p1*finetuned*` (pruned lineage by name AND by our own bit-exact pruning-oracle validations).

**Verdict: NOT AVAILABLE.** The model exists (trained per the paper) but was never published.
Obtainable only by author request (zero-GPU) or by retraining ~1M steps (out of scope for ICASSP).

## 3. Compatibility (conditional — the file does not exist to inspect)

**UNKNOWN-pending-artifact; expected PASS by inference:** a dense-FT full-pipeline `.ckpt` from the
same trainer would carry `channel_mult [1,2,3,5]` and the same key structure as
`audioldm-m-full.ckpt` (which our loader strict-loads) and the same in-ckpt layout as the two
`*_finetuned_global_step_999999.ckpt` we already load and EMA-reconstruct (state_dict + EMA shadows +
in-ckpt optimizer). Our existing CPU structural-inspection tooling would settle EMA-vs-raw identity
before any use. Nothing verified until a file exists.

## 4. Scientific control definition (no preregistration yet)

Question: *does AudioCaps fine-tuning of the UNPRUNED dense model itself induce the temporal/context
sensitivity observed in the published pruned→recovered checkpoints?*

Frozen batteries usable with **zero prompt selection**, and existing dense-pretrained WAVs:

| Battery | dense_pretrained WAVs | Informative contrast |
|---|---|---|
| Arm-D 80 @10.24s (sev-1, ⊂V1.1-96) | EXIST (dense union 80/80, scored) | ΔFT_native = C_denseFT − C_dense, paired per-ytid, CRN x_T reusable |
| Arm-D 80 @3.84s | EXIST (V1.1 dense_ema, 2 reps) | ΔFT_short; and **J_FT = ΔFT_native − ΔFT_short** (temporal interaction of pure FT) |
| Frozen music-64 @3.84s (sev-1) | EXIST (phenomenon dense arm, 64×3, C 0.197) | ΔFT_music (does +1M AudioCaps FT degrade held-out music in the dense model?) |
| xsev AudioCaps-192 / xsev music-64 (sev-2) | NONE | would need dense AND dense_FT generation → not minimal; skip |

Comparing J_FT / ΔFT_music against the recovered-vs-pruned interactions is **descriptive/suggestive,
not causal**: one training run per condition, no randomization over training seeds, different start
states. It can *dissociate* (dense-FT flat ⇒ profile is pruning-trajectory-associated) or *deflate*
(dense-FT reproduces the profile ⇒ "recovery inherits generic fine-tuning specialization"); it cannot
assign mechanism.

## 5. Minimum useful experiment (CONDITIONAL — only if the exact checkpoint is obtained; NOT launched)

Generate **dense_FT only** (all dense-pretrained comparators already exist; zero new dense-pretrained
generation): 80 native (10.24s) + 80 short (3.84s) + 64×3 music = **352 WAVs**.
Cost from settled jobs: native 0.0036 cr/WAV (Arm-D 0.5766/160) → ≈0.29 cr; 3.84s-class 0.00219
cr/WAV (V1.1 1.262/576) → 272 WAVs ≈0.60 cr; **total ≈0.9 cr, cap ~1.3 cr** incl. provisioning.
Precision (empirical, same batteries): paired n=80 CI half-width ≈ ±0.031–0.043; music n=64 ≈ ±0.030
— adequate for the profile-scale questions (native uplifts elsewhere are +0.15–0.25; music collapse
at sev-1 was −0.094), underpowered only if ΔFT effects sit near ±0.03.
Result patterns: (a) dense-FT reproduces the profile (large ΔFT_native, negative ΔFT_music, J_FT>0)
⇒ reinterpret as generic-FT specialization — evaluation claim intact, mechanistic claim dies;
(b) dense-FT ~flat ⇒ strong dissociation, pruning-trajectory-associated; (c) mixed ⇒ partial.
All three outcomes are informative → the experiment is decision-relevant either way.

## 6. Publication value — framing-split readiness (refines post-result audit §12)

* **A. Evaluation-of-recovery framing** ("a single-scalar, single-operating-point evaluation does not
  characterize the released recovered artifact across temporal scales and prompt domains"):
  **3.5/5.** The object of study is the released recovered artifact itself; H1-vs-H2 attribution is a
  distinct mechanistic question and its absence is a limitation, not a fatality. Supported by clean
  prospective implementation, independent cross-severity replication, K+J seam-robust, multi-metric
  concordance. Residual risks: framing novelty, single scorer family, op-point limitation.
* **B. Pruning-specific mechanistic framing** ("pruning+recovery causes this specialization"):
  **2/5.** H_generic vs H_pruning-associated undistinguished; no dense-FT control; the confound is
  fatal *for this framing*.
* The earlier single score (2.5/5) implicitly priced the mechanistic reading; under the evaluation
  framing 3.5/5 is the defensible number. Recorded here as a refinement, not a rewrite, of the
  post-result audit.
* **Impact of a successful dense-FT control:** mechanistic 2 → ~3.5 if dissociation (pattern b);
  stays ~2 if pattern (a). Evaluation framing 3.5 → ~3.75–4 under ANY outcome (adds an interpretive
  anchor either way).

## 7. Recommendation

**NEEDS CHECKPOINT FIRST.** Not downloadable anywhere; do NOT retrain (1M steps, out of scope).
The only zero-GPU path is requesting the fine-tuned unpruned checkpoint from the authors (Gabriel's
call; nothing sent). If obtained → the ≈0.9-cr, 352-WAV control in §5 is a strong candidate for the
final experiment. If not obtained → adopt the evaluation framing with the dense-FT limitation stated
explicitly.

**[RESOLVED same day — see §8: the author confirmed the checkpoint no longer exists. The §5
conditional experiment is VOID; the checkpoint branch is closed.]**

## 8. CLOSURE — direct author confirmation (2026-08-31)

> Author confirmation, 2026-08-31:
> Arshdeep Singh states that the fine-tuned unpruned AudioLDM-M-Full
> baseline checkpoint was deleted due to cluster storage/memory constraints
> and is no longer available.

**This establishes:** the checkpoint existed; the checkpoint is no longer available.
**This does NOT establish:** the exact dense-FT step count, the exact trainable modules, or recipe
identity with the pruned runs — those remain unknown unless the author answers separately. Gabriel is
asking a follow-up (same configuration? same step count?); any later answer is
**provenance/limitation information only** and does not reopen experiments unless a usable checkpoint
unexpectedly becomes available.

### Experimental decision tree — CLOSED (supervisor decision, 2026-08-31)

**No further GPU experiments.** Explicitly NOT worth running: DDIM200; a third pruning severity; more
prompts; additional Arm-D samples; new scorers; dense retraining; **approximate dense-FT
reconstruction** (the source recipe is underspecified — step count and trainable modules unknown —
and a new approximation would not be equivalent to evaluating the source artifact; do not attempt to
recreate it).

Reason: the surviving scientific-material contribution is already supported by prospective frozen
experiments; an independent second-severity replication; a positive temporal interaction at
severity 2; a strong native recovered advantage; seam robustness; multi-metric corroboration; an
honest negative sign-pattern replication; the explicit post-hoc severity analysis; and the dense
severity-1 reference. The only high-value mechanistic discriminator was the source dense-FT artifact,
and it is unavailable. No GPU credits on lower-value substitutes.

### Final framing boundary

The contribution is framed as **evaluation of the post-pruning recovered artifacts**.
Allowed conceptual target: *"Does evaluation at a single operating point adequately characterize the
behavior of published post-pruning recovered models?"* Supported answer: *"No. The
recovered-vs-pruned contrast depends strongly on evaluation context and temporal operating point, and
this behavior is observed prospectively at a second pruning severity."*
Forbidden framings (mechanism attribution blocked by the unavailable dense-FT control): "pruning
causes specialization"; "recovery-specific training causes specialization"; "pruning uniquely causes
the temporal interaction". The generic fine-tuning explanation remains a **limitation**, not a
falsification of the evaluation claim.

### Final status

```
EXPERIMENTAL PHASE: COMPLETE
GPU PHASE: CLOSED
```

Reopen only by explicit supervisor decision. Manuscript remains frozen pending explicit supervisor
GO. Final scientific-material readiness stays framing-split and unmerged: **evaluation-of-recovery
~3.5/5 (before manuscript quality); pruning-specific mechanistic ~2/5.** No new experiment is
recommended to improve the evaluation framing at acceptable cost.
