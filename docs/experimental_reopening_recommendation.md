# Experimental Reopening Recommendation — Temporal Semantic Recovery + Public Dense-FT Reference

**Status:** design/recommendation only; no new generated-audio outcome inspected; no GPU launch
authorized by this document.

**Date:** 2026-08-31.

**Decision context:** Gabriel explicitly rejected treating the experimental phase as irrevocably
closed and requested a literature- and artifact-grounded extension that could materially improve an
ICASSP contribution without restarting the project.

## 1. Bottom line

Do **not** reopen the experiment list that was already rejected (DDIM200, a third pruning severity,
more prompts, more Arm-D samples, or another aggregate scorer). Those mostly add robustness or
precision to the same claim.

The best compact reopening is a two-part **temporal semantic recovery** study:

1. **CPU / existing audio:** use the already pinned and locally validated FineLAP frame-level
   grounding model to determine *where in time* the recovered-vs-pruned semantic advantage arises in
   the existing 10.24-s generations, independently at both published pruning severities.
2. **One small GPU reference arm, conditional on a CPU checkpoint audit:** evaluate the official
   public `audioldm-m-text-ft` companion of `audioldm-m-full` on the existing Arm-D 80-prompt short /
   native-duration battery. This is a published **dense text-fine-tuning reference**, not Singh's
   unavailable matched dense-FT control. It asks whether duration dependence also appears after a
   documented dense text fine-tune without pruning.

Together these change the scientific question from merely *"the recovered advantage depends on clip
length"* to:

> **What temporal behavior does post-pruning recovery restore, and is that behavior distinguishable
> from the duration profile of a public dense text-fine-tuned AudioLDM?**

That is audio-specific, mechanism-facing, and directly relevant to ICASSP. It does not require new
training, a new dataset, a third severity, or reconstruction of Singh's deleted checkpoint.

## 2. Why this is a real gap rather than another metric sweep

The current paper has already established a recovered-over-pruned advantage at 10.24 s and a smaller
or absent advantage at 3.84 s, prospectively replicated at a second pruning severity. It has **not**
established what changes inside the longer generated signal. Clip-level CLAP, KL, PANN capture, FAD,
and FD cannot distinguish at least three explanations:

* recovery produces stronger local realizations of the requested event;
* recovery maintains/repeats the event over more of the timeline (**semantic coverage/persistence**);
* recovery improves other long-window acoustic properties that raise clip-level scores without a
  frame-localized semantic change.

Recent primary literature makes this distinction timely:

* [FineLAP (ACL 2026)](https://aclanthology.org/2026.acl-long.473/) shows that global CLAP-style
  embeddings are inadequate for frame-level applications and releases frame-level text-audio
  grounding machinery.
* [Kuan et al. (2026)](https://arxiv.org/abs/2607.13408) argue that global similarity/perceptual
  metrics miss event completeness and temporal relations in generated audio.
* [T-CLAP](https://arxiv.org/abs/2404.17806) documents limited temporal sensitivity in conventional
  CLAP-style retrieval embeddings.
* [Stable Audio](https://arxiv.org/abs/2402.04825) modifies evaluation for variable-length audio,
  reinforcing that duration is part of the endpoint rather than harmless presentation metadata.
* [Singh et al.](https://arxiv.org/abs/2607.13330) report event-family recovery at one 10-s operating
  point, but do not analyze frame-level semantic coverage or duration-conditioned recovery.

The bounded search performed for this recommendation found temporal grounding/evaluation work and
post-pruning audio recovery work, but no primary work combining **frame-level semantic allocation,
post-pruning recovery, and cross-severity replication**. Any manuscript must retain the careful
wording "we found no prior work" rather than claiming absolute priority.

## 3. Existing artifact substrate (verified)

No new outcome was computed. The following feasibility facts come only from manifests, files, and
previously frozen results:

* Severity 1 Arm D: 80 paired 10.24-s pruned/recovered outputs, completed dense 10.24-s reference,
  and matched existing 3.84-s outputs.
* Severity 2: 192 paired 10.24-s and 3.84-s recovered/pruned-A outputs; pruned-B is a three-tensor seam
  sensitivity.
* Strict requested-event eligibility, defined using the already frozen AudioSet-label + caption-alias
  rule and **without reading generated audio**, yields:
  * severity 1: **49 prompts / 63 requested-event occurrences**;
  * severity 2: **110 prompts / 131 requested-event occurrences**.
* FineLAP is already local and pinned (`AndreasXi/FineLAP`, weight sha256
  `13b9646c9f9d48513c0145bed75e654179e83f0fd8d49ed4ffc5d6b8f3353fb4`). The prior CPU validity smoke
  passed **5/5** own-event-vs-distractor checks and returns 64 temporal frames for 10.24-s audio.
* The threshold `tau = 0.5` and occupancy interpretation were already frozen before these generated
  outputs existed. They can be reused as a secondary endpoint without tuning.
* The official AudioLDM release provides `audioldm-m-text-ft.ckpt` (4,571,676,474 bytes;
  md5 `036bc9b547a50f78b960ef8f14d0e1fb`; sha256
  `d77d5a61785af82012edb8a72158d52592ac7c76d7f6ed51a048ec2dec8d5eca`) and documents it as the medium
  model fine-tuned with AudioCaps and MusicCaps audio-text pairs. Sources:
  [official Zenodo record](https://zenodo.org/records/7813012) and
  [official AudioLDM repository](https://github.com/haoheliu/AudioLDM).

## 4. Part A — Temporal Semantic Recovery Profile (CPU, zero new WAVs)

### 4.1 Construct

For every label-confirmed event explicitly requested in the caption, obtain FineLAP frame scores
`s(system, prompt, event, t)` on the **existing 10.24-s waveform**. When a prompt contains several
eligible events, average event-level estimands within the prompt before inference so the independent
unit remains the prompt.

The 3.84-s boundary is fixed by the prior experiment, not selected from the new data:

* early window: `[0, 3.84 s)`;
* late window: `[3.84, 10.24 s]`.

### 4.2 Primary endpoint

For severity `s`, define the within-prompt temporal recovery interaction:

```text
T_s = mean_late(score_recovered - score_pruned)
      - mean_early(score_recovered - score_pruned)
```

Primary hypothesis: `T_s > 0`, tested separately at severity 1 and severity 2 with a prompt-clustered
paired bootstrap (`B=10000`, new frozen seed namespace). The scientifically promotable result requires
the point estimate to be positive at **both** severities and the 95% CI to exclude zero at severity 2;
severity 1 supplies directional replication because its metadata-eligible n is only 49.

This endpoint asks whether the recovered advantage is disproportionately allocated after the short
operating-point boundary *within the same long generation*. It does **not** pretend that cropping a
10.24-s waveform recreates a separately generated 3.84-s sample.

### 4.3 Secondary endpoints (fixed; no fishing)

* **occupancy:** fraction of frames with score `>= 0.5`;
* **quarter coverage:** fraction of four fixed 2.56-s quarters containing at least one active frame;
* **peak evidence:** maximum frame score;
* **semantic mass:** mean frame score over the full 10.24 s;
* severity-1 dense and real-reference distributions as descriptive anchors;
* severity-2 pruned-B as seam sensitivity only.

Interpretation rule: call the gain **coverage/persistence-dominant** only if occupancy or quarter
coverage improves while peak evidence is approximately unchanged or materially smaller in effect.
Otherwise report a generic frame-level semantic gain. No onset/timing-accuracy claim is allowed because
AudioCaps captions do not specify target timestamps.

### 4.4 Outcome branches

* **A1 — positive `T` at both severities:** promote temporal semantic allocation as a main finding.
* **A2 — full-window semantic mass/occupancy improves but `T` is null:** recovery improves local event
  evidence, but not specifically late evidence; retain a weaker frame-level interpretation.
* **A3 — FineLAP null while clip metrics are positive:** the duration effect is not explained by
  localized requested-event evidence under this evaluator; report the disagreement as a bounded
  negative and do not make a temporal-mechanism claim.

## 5. Part B — Public dense text-FT duration reference (one bounded GPU arm)

### 5.1 Why it is useful and what it is not

`audioldm-m-text-ft` was absent from the prior dense-FT availability decision. It is a real public
artifact, not a reconstruction. It can test whether duration dependence is also present in an upstream
dense text-fine-tuned companion.

It is **not** Singh's deleted dense-FT checkpoint and is not a matched causal control: its documented
training data include MusicCaps as well as AudioCaps, and the exact step/module recipe is not reported
as identical to Singh's. Therefore:

* allowed: "public dense text-fine-tuning reference" / "dissociation from one published dense-FT
  companion";
* forbidden: "generic fine-tuning ruled out" / "pruning uniquely causes the interaction" / "matched
  dense control".

### 5.2 CPU provenance gate before any generation

Download by immutable hash, then require all of:

1. md5/sha256/byte size match the official release;
2. config reconstructs the same `(1,2,3,5)`, `model_channels=192` architecture and strict-loads;
3. checkpoint inventory identifies which components differ from `audioldm-m-full` and how EMA is
   represented;
4. production generator performs a CPU wiring dry-run at latent lengths 96 and 256;
5. no generated/recovery outcome is inspected before protocol, manifest, endpoint, and seed are
   frozen and committed.

Failure of any structural gate ends Part B at zero GPU spend.

### 5.3 Minimal generation

Generate **only the new dense text-FT system** on the already frozen Arm-D 80-prompt subset:

* 80 WAVs at 3.84 s;
* 80 WAVs at 10.24 s;
* DDIM50, guidance 2.5, eta 0, fp32, EMA convention resolved by the CPU audit;
* one generation per prompt and the exact existing common `x_T` per `(ytid, duration)`.

Existing `audioldm-m-full` outputs are reused byte-for-byte. No music arm is recommended because the
public text-FT artifact explicitly includes MusicCaps exposure, making an OOD-music contrast difficult
to interpret.

Primary reference estimand:

```text
J_dense_textFT = (textFT - dense)_10.24s - (textFT - dense)_3.84s
Q = J_recovery_sev1 - J_dense_textFT
```

`Q` uses the same 80 prompts and can be prompt-paired across the four systems. Report the full CI, not
only a binary verdict. Reuse the project's existing `0.025` CLAP SESOI as the pre-data equivalence
margin: call the duration profiles **equivalent** only if the 90% CI for `Q` lies wholly inside
`[-0.025, +0.025]`; call the recovery interaction **larger** only if the two-sided 95% CI for `Q`
lies above zero. Any other geometry is unresolved/mixed. A final power calculation under these fixed
rules is required before authorizing generation; lack of power stops Part B rather than relaxing the
margin or expanding the sample after inspection.

### 5.4 Interpretation branches

* **B1 — `Q` equivalent within `+/-0.025`:** temporal-scale dependence is not unique to the
  pruned recovery artifact. This weakens pruning-specific language but **broadens** the paper into a
  stronger evaluation warning about text fine-tuning/recovery in conditional audio diffusion.
* **B2 — 95% CI for `Q` lies above zero:** the recovered artifact's duration interaction is larger
  than this public dense text-FT reference. If `J_dense_textFT` also passes its own TOST against
  `+/-0.025`, it may additionally be described as near zero. This supports
  "pruning-trajectory-associated" wording, never causal uniqueness.
* **B3 — opposite/mixed:** report the interaction geometry and keep mechanism open.

Every branch adds an interpretive anchor that the current manuscript lacks.

## 6. Compute and schedule

### CPU work

Checkpoint inspection, FineLAP inference, bootstrap/statistics, manifests, and all scoring run in the
free CPU Studio. Expected paid cost: **0 cr**. No GPU is kept alive for these stages.

### GPU work

Part B requires exactly **160 new WAVs**. Using settled project rates:

* 80 native WAVs x approximately 0.00360 cr/WAV = approximately 0.29 cr;
* 80 short WAVs x approximately 0.00219 cr/WAV = approximately 0.18 cr;
* expected total including provisioning: approximately **0.47-0.60 cr**;
* proposed hard cap: **0.70 cr** on the smallest compatible T4 class.

There is currently no reliable live balance. A top-up and explicit launch authorization are required;
an OUT_OF_FUNDS launch is not useful. Estimated implementation + CPU analysis: 1-2 focused days;
generation: under one hour based on settled jobs.

## 7. Why this outranks the alternatives

| Candidate | New scientific construct? | Central confound addressed? | Paid cost | Recommendation |
|---|---:|---:|---:|---|
| More prompts / third severity | no | no | medium-high | reject |
| DDIM200 | no; external-validity check | no | high | reject |
| New aggregate scorer | weak | no | low | reject |
| Approximate Singh dense retraining | yes in principle | compromised by recipe mismatch | prohibitive | reject |
| Recovery-delta layer ablations | potentially | no; hybrid weights hard to interpret | medium | defer |
| **Temporal profile on existing audio** | **yes, audio-specific** | explains duration finding | **0 cr** | **do** |
| **Public dense text-FT duration reference** | **yes, comparative anchor** | **partially, with explicit mismatch** | **<=0.70 cr** | **do if CPU gate passes** |

## 8. Expected paper-level change

If Part A is positive and Part B supplies either a shared or dissociated duration profile, the paper can
move from a single-artifact evaluation warning to a more distinctive ICASSP contribution:

1. a paired cross-severity study of post-pruning recovery;
2. a frame-level **temporal semantic recovery profile** showing what the long-duration advantage
   consists of;
3. a public dense text-fine-tuning reference that bounds how pruning-specific the profile can be.

The bounded title direction would become something like:

> **What Does Post-Pruning Recovery Restore? Temporal Semantic Coverage in Compressed Text-to-Audio
> Diffusion**

The adapter-transfer material should then be removed from the four-page main story rather than compete
with the temporal contribution.

## 9. Recommendation / authorization boundary

**Recommend reopening experimentation for this package only.** First freeze and run Part A on CPU.
In parallel, fetch and structurally audit the public `audioldm-m-text-ft` artifact. Return with the
Part-A result, the Part-B compatibility verdict, a final power calculation, and an exact <=0.70-cr
launch command. Do not launch Part B until Gabriel explicitly authorizes the top-up and GPU run.
