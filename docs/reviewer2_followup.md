# REVIEWER2-FOLLOWUP — frozen protocol (E5, E6, E7, E8, E1c, B, E3)

**Status: FROZEN BEFORE ANY OUTPUT.** Nothing in this file may be edited after the first WAV or the first
training step of any job below exists. The sha256 sidecar `docs/reviewer2_followup.md.sha256` is committed
together with this file and before launch.

## 0. Authorization and scope

* **Trigger.** The second external review of Draft 12 (`docs/review/2026-09-05_reviewer2_methodological_response.md`)
  listed methodological weaknesses. Each was re-verified against the record on 2026-09-05; the correctable
  ones and their costs are in that document and in `docs/compute_budget.md` (2026-09-05 entry).
* **Gabriel, 2026-09-05 01:22 MVD:** "tenemos unos 15 créditos restantes para correr todo lo que queda.
  Puedes acomodarlo para que quepa en la menor cantidad de créditos posible y lanzar los jobs que
  identificaste?" → authorization to launch, with a **total ceiling of 15 cr for everything that remains**
  (GPU jobs + the Studio hours the CPU stages consume). The package below is sized to ≈ 9.3 cr point /
  ≈ 11.2 cr in watchdog caps, leaving ≥ 3.5 cr for Studio time and the settlement uplift.
* **Plan change recorded (AGENTS.md).** Four items on the closed list of 2026-08-31 (DENSE-FT-CLOSURE:
  "NO more prompts", "NO text-FT generation", "NO new scorers", and by extension no training) are
  **reopened by Gabriel's instruction for this package only**, each as a pre-specified addition to an
  evaluation whose primary verdicts are frozen and untouched: V1.1 PASS=FALSE, XSEV CASE C, FineLAP A2,
  DRAFT5-OPSWEEP-1 MONOTONE-INCREASING, DRAFT5-PUBRECIPE-1 gate PASS, XSEV-MUSIC-NATIVE-1 branch (a).
  Nothing here can change those verdicts. Not reopened: the human listening study (ethics/co-author
  decision), a third severity, best-of-3, DDIM200 beyond E2b.
* **What is NOT claimed by any branch.** No causal claim about pruning; no "restored to dense"; no
  "matched dense control" (Singh's is deleted); E3 is a *directional* single-arm test of the specialisation
  reading at 2 % of the released recovery budget and is reported as such.

## 1. Shared conventions (identical to the frozen runs unless stated)

* Sampler: DDIM 50 / guidance 2.5 / eta 0 / fp32 / single generation. Weights: EMA for every released
  checkpoint (`dense_ema`, `pruned2_A` = A′ L1 selection on the dense EMA, `recovered2_dp1_ema`,
  `p1_pruned` = L1 selection on the dense EMA at (1,2,3,1), `recovered1_p1_ema`, `textft_ema`). E3's
  checkpoint is evaluated with its **raw** weights (see §8).
* Scorer: fused CLAP `laion/clap-htsat-fused` rev `365dea6e`, one seed-once fixed-order call per group,
  shuffled-caption floor from the same embeddings (`draft5_opsweep_verdict.py` conventions). Unit = prompt;
  replicates averaged within prompt; percentile bootstrap `B = 10000`, seed namespace
  `REVIEWER2-FOLLOWUP|BOOTSTRAP|2026-09-05` (a PCG64 seed derived per estimand from that namespace).
  SESOI 0.025 CLAP (project-wide).
* CRN pairing: within a (context, prompt, replicate) every system shares `x_T`. Frozen AudioCaps cells keep
  `GEN_SALT = RECOVERY-CROSS-SEVERITY-REP-1|GENERATION|2026-08-30` (so `textft`, `p1_*`, `shortft` and the
  15.36 s cell are noise-paired with the existing `dense` / `pruned2_A` / `recovered2` clips of the same
  prompt, and duration cells are prompt-paired only). New batteries use
  `R2_GEN_SALT = REVIEWER2-FOLLOWUP|GENERATION|2026-09-05`.
* "First 96 prompts" = the first 96 entries of the frozen 192-prompt manifest in `prompt_index` order
  (outcome-blind, the same rule E2b used for its 64).
* Device rule: every WAV of a job comes from one Lightning T4 job. Jobs A and B regenerate 4 frozen
  `pruned2_A / ac_native` clips (indices 0–3) as a device-consistency check; expectation from the last three
  T4 jobs: within 1 int16 LSB, max |ΔCLAP| ≤ 1e-5 (not bit-identical — the frozen protocol's "bit-identical"
  expectation was corrected on 2026-09-03). Job C uses severity-1 checkpoints whose frozen clips came from
  T4 jobs too; it carries no device check (recorded limitation).
* Structural validation before any score: sha256 + ytid/clip id + sample count per WAV
  (3.84 s → 61 472; 10.24 s → 163 872; 15.36 s → the count the CPU dry-run reports, recorded in the
  generation manifest and re-checked at validation).

## 2. E5 — Clotho held-out battery (reviewer W3)

**Question.** Does the recovery gain transfer to a second held-out prompt domain whose *content* stays in
the AudioCaps sound-event universe while the *caption style* changes (Clotho, Freesound audio, 8–21-word
captions), or is the hip-hop null specific to a domain that changes both content and style?

* **Battery.** `configs/research/r2_clotho_manifest.json` (sha `536d45d5…`): 96 clips of the Clotho v2.1
  evaluation split chosen by seeded hash of `file_name`, one of the five captions per clip chosen by a second
  seeded hash, **no content filtering**. Caption words median 11 (AudioCaps battery 8, hip-hop 56.5).
* **Generation.** `pruned2_A`, `recovered2`, `dense` × 96 × {3.84 s (latent 96), 10.24 s (latent 256)} =
  **576 WAVs**, R2 salt, replicate 0.
* **Anchors (CPU, 0 cr).** Shuffled-caption floor per cell; **real-audio reference** = the Clotho evaluation
  WAV of the same clip, resampled to 16 kHz, first 10.24 s (163 872 samples) and first 3.84 s (61 472
  samples) — the same crop conventions as the AudioCaps references; `rho_real(d)` and `rho_dense(d)`.
* **Estimands.** `R_clo(d)` = mean per-prompt `CLAP(P+FT) − CLAP(P)`; `J_clo = R_clo(10.24) − R_clo(3.84)`;
  domain contrast `D_clo(d) = R_AC|96(d) − R_clo(d)` where `R_AC|96(d)` is the frozen AudioCaps gain on the
  first 96 prompts (unpaired two-sample bootstrap over the two prompt sets).
* **Pre-specified reading** (no gate; all outcomes reported):
  * **TRANSFERS** — `lo95(R_clo(10.24)) > 0` and `|point(D_clo(10.24))| < 0.025`.
  * **PARTIAL** — `lo95(R_clo(10.24)) > 0` and `lo95(D_clo(10.24)) > 0.025`.
  * **NO TRANSFER** — the CI of `R_clo(10.24)` contains 0 (as on hip-hop).
  * anything else → UNRESOLVED. The interpretation "caption-style vs content" is stated in the paper only
    as consistent/inconsistent with each branch, never as established.
* **Declared caveats.** AudioLDM's pre-training included Freesound, so Clotho content is unseen by the
  recovery stage only; Clotho clips are 15–30 s, so the real reference is a crop.

## 3. E6 — dense anchors on the hip-hop cells (reviewer W2)

**Question.** Is the hip-hop battery floor-limited (dense also near chance) or does it discriminate?

* **Generation.** `dense` × frozen `xsev_music_manifest.json` (64 prompts) × 3 replicates at 3.84 s (the
  frozen music cell's convention) + 64 × replicate 0 at 10.24 s (the frozen `music_native` convention) =
  **256 WAVs**, frozen `GEN_SALT` seeds (noise-paired with the existing P / P+FT music clips).
* **CPU, 0 cr, in the same verdict:** the existing `gate0-gen-1` dense clips on the severity-1 hip-hop
  battery (`dense_noadapter_p*_r*.wav`, 192 WAVs, 3.84 s) are scored under the same convention with their
  own floor → dense anchor for the severity-1 music cell; the 193 Kim clips (real 4.0-s hip-hop excerpts
  with MusicCaps captions, `artifacts/icassp_gate0/kim193_wav_3p84s/`) scored against their own captions
  give a **domain-level** real-audio ceiling at 3.84 s (not prompt-matched; declared).
* **Estimands.** `A_dense(cell) = mean(dense − floor_dense)` with CI (above-chance margin);
  `rho_dense(music, d) = R_music(d) / (dense(d) − P(d))` with paired-ratio CI as in the floor–ceiling
  artifact; `dense − P+FT` per cell.
* **Pre-specified reading.** The battery **discriminates** for the dense model iff `lo95(A_dense) > 0.025`
  at that cell; then the music null is reported as "recovery closes `rho_dense` of the dense gap" rather
  than "unresolved". If `A_dense` is not resolved above SESOI, the reviewer's floor reading is adopted for
  that cell and the paper says so.

## 4. E7 — hip-hop battery extension (reviewer W6, sample size)

* **Battery.** `configs/research/r2_music_ext_manifest.json` (sha `2e63b3f6…`): **all 63** remaining
  MusicCaps hip-hop/rap prompts eligible under the frozen rules after excluding both frozen 64-prompt
  batteries (the pool is exhausted; nothing was chosen).
* **Generation.** `pruned2_A`, `recovered2`, `dense` × 63 × {3.84, 10.24} s × replicate 0 = **378 WAVs**,
  R2 salt.
* **Estimands.** `R_music,127(d)` on the pooled battery (64 frozen + 63 new; per-prompt means, so the
  frozen prompts' 3 replicates at 3.84 s and the new prompts' single replicate enter with equal prompt
  weight); `J_music,127`; original-vs-extension difference (descriptive heterogeneity check). Floors,
  `rho_dense` via E6/E7 dense clips.
* **Reading.** The paper's music statement follows the pooled CI: "absent" if the CI of `R_music,127(10.24)`
  contains 0 with `|point| < 0.025`; "small positive/negative gain" if resolved; the frozen 64-prompt values
  remain reported as the pre-specified primary.

## 5. E8 — severity 1 on 96 more prompts (reviewer W6, power)

* **Generation.** `p1_pruned`, `p1_recovered` × first 96 AudioCaps prompts of the frozen 192 manifest ×
  {3.84, 10.24} s × replicate 0 = **384 WAVs**, frozen `GEN_SALT` seeds (noise-paired with the severity-2
  systems on the same prompts).
* **Estimands.** `R1(d)` and `J1_96` on the 96 new prompts; **pooled** `J1_176` over Arm-D 80 + new 96
  (prompt-level bootstrap over the union; the two sets are disjoint by construction and scored under the
  same convention). MDE at n = 176 ≈ 0.044 at the Arm-D variance.
* **Pre-specified rule.** `lo95(J1_176) > 0` → the severity-1 duration interaction is **resolved** (the paper
  drops "supporting context" for "replicated at both severities"); CI containing 0 → stays "directionally
  consistent, underpowered" with the pooled estimate reported; `hi95 < 0` would be a reversal and is
  reported as such. The frozen Arm-D value (+0.044 [−0.001, +0.087]) stays the pre-specified primary.

## 6. E1c — one duration beyond the fine-tuning duration (reviewer W4)

* **Generation.** `pruned2_A`, `recovered2` × first 96 prompts × **15.36 s (latent 384)** = **192 WAVs**,
  frozen `GEN_SALT` seed per ytid (sweep convention). No dense at this point (budget); no real-audio
  reference exists beyond 10 s (AudioCaps clips are 10 s), so the cell is read against its floor and the
  frozen 10.24 s cell only.
* **Estimand.** `D4 = R(15.36) − R(10.24)` on the same 96 prompts, paired; `R(10.24)|96` from the frozen
  clips.
* **Pre-specified rule** (extends `docs/draft5_opsweep.md` §2): **PEAKED AT THE FINE-TUNING DURATION** if
  `hi95(D4) < 0`; **STILL INCREASING** if `lo95(D4) > 0`; **PLATEAU** if the CI contains 0 and
  `|point(D4)| < 0.025`; otherwise UNRESOLVED. Floor-corrected `D4_c` as secondary. The fused-CLAP scorer
  already uses its > 10-s fusion path at 10.24 s, so 15.36 s is scored under the same regime (declared).

## 7. B — public dense text-FT reference (reviewer W1, reference not control)

* **System.** `audioldm-m-text-ft.ckpt` (TEXTFT-CHECKPOINT-AUDIT: hash PASS, strict-load 690/690, EMA
  present; dense architecture; fine-tuned by the AudioLDM authors on AudioCaps + MusicCaps for an unknown
  number of steps). **Permitted role: one public dense text-fine-tuning reference. NOT Singh's deleted
  dense-FT, NOT recipe-matched, NOT a causal control** — wording fixed by that audit and kept here.
* **Generation.** `textft` × first 96 prompts × {3.84, 10.24} s = **192 WAVs**, frozen `GEN_SALT` seeds
  (noise-paired with the existing dense clips of the same prompts).
* **Estimands.** `G_tf(d) = CLAP(textft) − CLAP(dense)` paired per prompt; `J_tf = G_tf(10.24) − G_tf(3.84)`.
* **Pre-specified reading.** `lo95(J_tf) > 0` → "a dense model that underwent a text fine-tune also gains
  more at 10.24 s than at 3.84 s on AudioCaps" (consistent with a generic fine-tuning account; weakens the
  specialisation-after-pruning reading); CI containing 0 or `hi95 < 0` → reported as is. Forbidden in every
  branch: "generic fine-tuning ruled out / established", "matched dense control".

## 8. E3 — short-duration full fine-tune of the pruned checkpoint (reviewer W1, directional test)

**Question.** If P (severity 2) is fine-tuned on AudioCaps **at 3.84 s** instead of 10.24 s, does the recovery
gain it produces follow the training duration (specialisation) or still favour the longer clip?

* **Recipe** (Singh's = upstream `audioldm_original_medium.yaml`, with the duration changed):
  start from `pruned2_A` (A′ L1 selection on the dense EMA, [1,2,1,1]); train **all U-Net parameters**
  (VAE, CLAP conditioner, vocoder frozen — as in the upstream trainer, where only the U-Net is optimised);
  AdamW lr 1e-4 (constant; upstream has no scheduler), betas (0.9, 0.999), weight decay 0.01 (torch default,
  as upstream), batch 2, CFG dropout `unconditional_prob_cfg 0.1`, eps-parameterisation, 1000 timesteps,
  FP32; data = the preprocessed AudioCaps **training** split (49 502 clips, `audiocaps_train_label.json`),
  **random 3.84-s crops** of each 10-s clip (the dataset's own `random_segment_wav`), latent 96;
  **N = 20 000 optimizer updates** (= 2 % of the released 10⁶). Seeds fixed (`20260905`). Checkpoint saved
  every 5 000 steps (insurance), final raw U-Net state saved as `shortft_unet.pt` with sha256.
* **Self-gate inside the job.** The first 200 steps are timed. If `sec/step × 20 000 / 3600 × 0.89 cr/h`
  exceeds **2.0 cr**, the job writes the benchmark and stops **without training further** (reported as
  "benchmark only"). Otherwise it continues.
* **Evaluation in the same job (CRN).** `shortft` × 192 prompts × {3.84, 10.24} s = **384 WAVs**, frozen
  `GEN_SALT` seeds (noise-paired with P and P+FT).
* **Estimands.** `R_sf(d) = CLAP(shortft) − CLAP(P)` paired; `J_sf = R_sf(10.24) − R_sf(3.84)`; secondary:
  `R_sf(d)` vs the released `R(d)` (descriptive, different training budgets).
* **Pre-specified reading.** branches, evaluated on the 95 % CI of `J_sf`:
  * `hi95(J_sf) < 0` → the short-trained checkpoint gains **more at 3.84 s**: consistent with
    specialisation to the training duration.
  * `lo95(J_sf) > 0` → it still gains more at 10.24 s: consistent with "longer clips are easier to recover
    regardless of training duration".
  * CI containing 0 with `lo95(R_sf(3.84)) > 0` → gain present but duration-neutral at this budget.
  * `R_sf(3.84)` and `R_sf(10.24)` both unresolved → **UNINFORMATIVE** (2×10⁴ steps changed too little);
    declared possible in advance and reported as such, not rescued.
* **Declared limits.** Single arm (no matched 10.24-s arm at the same step count: budget); raw weights (no
  EMA at this horizon); absolute `R_sf` is not comparable with the released recovery; batch 2 on one T4
  matches the upstream config's batch but Singh's cluster batch is unknown.

## 9. Budget (Gabriel's 15-cr ceiling for everything that remains)

Per-WAV model §A10 (`0.001329 + 9.0e-6·L`), 0.145 cr per job, training at the measured 0.307 s/step ×
0.89 cr/h; caps = point × 1.2 (E3 cap includes the 2.0-cr training gate).

| Job | Content | WAVs | Point (cr) | **Hard cap** | max-min |
|---|---|---:|---:|---:|---:|
| `r2-gen-a` | E6 (256) + E7 (378) + B (192) + device check (4) | 830 | 2.47 | **3.00** | 300 |
| `r2-gen-b` | E5 (576) + E1c (192) + device check (4) | 772 | 2.76 | **3.30** | 300 |
| `r2-gen-c` | E8 (384) | 384 | 1.26 | **1.50** | 180 |
| `r2-shortft` | E3: 200-step bench + 20 000 steps + 384 eval WAVs | 384 | 2.8 | **3.40** | 420 |
| **total** | | **2370** | **9.3** | **11.2** | |

Studio hours for CPU scoring/verdicts/manuscript work: ≈ 0.27 cr/h (idle guard active; stop the Studio
between stages). If a watchdog fires, the job is STOPPED and its partial state reported as partial — no
silent overrun, no rescue run. Live funded balance is not exposed by the SDK; the 15-cr figure is Gabriel's.

## 10. Compute-discipline record (AGENTS.md, required before every GPU launch)

1. **Why CPU is unsuitable.** 2 370 diffusion clips (1.7 s per DDIM step at latent 96 on this CPU; ×2.7–4
   at the longer latents) ≈ several days of Studio time at 0.27 cr/h — slower AND dearer than the T4.
   E3's 20 000 training steps on CPU are infeasible in wall time.
2. **GPU-only work.** Diffusion sampling and the fine-tune. Everything else (validation, scoring, floors,
   real references, bootstrap, verdicts, manuscript) is CPU.
3. **Smallest compatible class.** T4 on-demand (device rule; every frozen clip came from T4).
4. **Ceilings.** Per-job caps above, enforced by `scripts/sa3/job_watchdog.py` from the Studio (the idle
   guard treats a running watchdog as protected work).

## 11. Order of execution and reporting

Jobs A, B, C launch together after the CPU dry-runs of every new (system, context) path pass and this file
is committed with its sidecar. `r2-shortft` launches after its trainer's CPU dry-run passes and its own
sidecar-protected section (this §8) is unchanged. Scoring and verdicts run on CPU after each job settles
(`scripts/research/r2_verdict.py`, written against this protocol). Every result — including UNRESOLVED and
UNINFORMATIVE — is recorded in `docs/experiment_ledger.md` and `docs/claims_matrix.md`; the manuscript is
changed only after the verdicts exist.
