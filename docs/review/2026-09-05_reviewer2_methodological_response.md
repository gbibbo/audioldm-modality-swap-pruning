# Second external review of the ICASSP manuscript (Draft 12): methodological weaknesses, re-verified

Date: 2026-09-05 (MVD). Scope set by Gabriel: take the second reviewer's report on
`icassp/icassp_operating_point.pdf` (Draft 12, "from scratch" rewrite), keep only the criticisms that concern
**methodology**, and for each one re-check whether the design decision it attacks is still forced by a
constraint (compute, checkpoints, participants, data) or whether it can now be corrected. Presentation
items (title/abstract wording, figure density, a table in the body, anonymity) are listed at the end and
deliberately **not** acted on in this pass.

Everything below was re-verified against the repository record today (CPU, 0 cr, nothing launched). Cost
figures use the §A10 per-WAV model of `docs/compute_budget.md` (`cr(L) = 0.001329 + 9.0e-6·L` per WAV,
0.145 cr per job, cap = point × 1.2), which over-estimated the two settled jobs it has since been tested on
by 23–27 %, so the figures are conservative. **Available credit is unknown**: the SDK exposes only the
lifetime counter (`total_spent` = 99.18 cr on 2026-09-05 04:00 UTC; the `balance` field is a static 5.0).
Every "correctable" verdict below is therefore conditional on Gabriel confirming funds.

Review scores as received: novelty 3/5, technical correctness 4/5, experimental validation 3/5, clarity 4/5,
reproducibility 4/5, overall 3.5/5 (Weak Accept), reviewer confidence 4/5.

---

## Summary table

| # | Reviewer's weakness | Record says | Re-verified today | Verdict | Cheapest correction |
|---|---|---|---|---|---|
| W1 | No dense model with the same 10⁶-step fine-tune, so nothing can be said about pruning; the title implies a mechanism | Singh's dense-FT checkpoint was deleted (author confirmation 2026-08-31); retraining 10⁶ steps ≈ 500 cr; decision tree closed | Deletion confirmed in `docs/dense_ft_baseline_availability_audit.md` §8; **training throughput HAS been measured** (0.307 s/step, T4, FP32, batch 2, latent 96 — contradicting the §A10 note "never measured") | **Matched control: NOT correctable.** Reviewer's *directional* alternatives: correctable at 3–7 cr | (a) public dense text-FT reference, 160 WAVs ≈ 0.6 cr (already audited, B-GO-CANDIDATE); (b) short-duration fine-tune of P at 3.84 s, 2×10⁴ steps ≈ 3 cr single-arm / ≈ 5.6 cr two-arm |
| W2 | Music result may be a floor effect; panel (c) lacks dense and real audio for hip-hop | Real MusicCaps audio never obtained (0/64 refs); dense on the sev-2 music battery never generated | **Dense on the sev-1 music battery already exists** (job `gate0-gen-1`, 192 WAVs, mean CLAP 0.197 vs P 0.117 / P+FT 0.023 / floors ≈ +0.05): the battery is NOT floor-limited for the dense model. YouTube download is blocked from the Studio (bot check) | **Correctable** (dense: 0 cr for sev-1, ≈ 0.5 cr for sev-2). Real hip-hop audio: only via a manual download by Gabriel, or a domain-level ceiling from the 193 Kim clips already on disk (0 cr) | rescore existing dense WAVs (CPU) + one 128-WAV T4 job (0.52 cr, cap 0.62) |
| W3 | Single, extreme domain shift; Clotho is the natural control | Clotho never considered (no mention anywhere in the record) | Clotho captions (0.4 MB) and evaluation audio (1.25 GB) are downloadable from Zenodo now; captions median 11 words vs AudioCaps 8 vs hip-hop 56 | **Correctable**, ≈ 1.8 cr (96 prompts, 3 systems, 2 durations) / 3.5 cr (192) | freeze a Clotho-eval battery, generate P / P+FT / dense, real-audio and floor anchors |
| W4 | Sweep ends at the native duration; both systems are noise at 3.84 s, so the short gain may be noise alignment | E1c (15.36 s) costed at 2.90 cr, not launched; "sweep stops at the fine-tuning duration" declared as a limitation | The generator takes any latent length (96/128/192/256 already used; the U-Net has no absolute positional embedding); real audio is unavailable beyond 10 s (AudioCaps clips are 10 s) so only dense/floor anchors apply. The "all noise at 3.84 s" reading is already contradicted by existing data: dense scores 0.207 at 3.84 s (0.23 above chance, 76 % of real audio) and KL/PANNs gains at 3.84 s exclude 0 | **Correctable** (one point beyond native ≈ 2.0–2.9 cr); the noise objection is answerable from existing evidence (0 cr) | E1c: P / P+FT (+ dense) × 192 × 15.36 s |
| W6 | n = 64 music, n = 80 severity 1; listening is 8 pairs by an author | Sev-1 battery frozen at 80; music pool 127 eligible after exclusions; six-listener study designed, frozen and CANCELLED pre-launch because the co-author did not approve running it without ethics clearance | Pool for +63 music prompts confirmed; sev-1 checkpoints on disk; the listening study remains fully prepared | **Sample sizes: correctable** (music → 127 prompts ≈ 0.9–1.25 cr; sev-1 → 272 prompts ≈ 2.4 cr). **Listening: NOT correctable by compute** — it needs ethics approval / co-author consent | run the two generation jobs; re-open the listening question with the co-authors |
| W5 | Too much evidence lives in the repository | Draft 11/12 design choice (paper/repo split) | Verifier: 87/87 numbers in the .tex + companion reproduced from artifacts | presentation → deferred | half-column table with R, J, CIs per duration and scorer |
| W7 | "no alignment gain" vs "unresolved"; Fig. 1 density; repo URL vs anonymity | ICASSP 2027 is single-anonymous (checked 2026-09-03) | — | presentation → deferred (wording fix is trivial and should be done) | replace "yields no alignment gain" with "leaves the gain unresolved" in the abstract |

---

## W1 — the missing dense control and the mechanism implied by the title

**What the reviewer asks.** Without a dense AudioLDM-M given the same 10⁶-step AudioCaps fine-tune, the
paper cannot separate "recovery after pruning specialises to the training operating point" from "AudioCaps
fine-tuning at 10.24 s helps most on AudioCaps at 10.24 s, for any network". Improvement #1: a partial dense
fine-tune (10⁴–10⁵ steps) to show the direction of the effect; or, cheaper, a short fine-tune of the pruned
checkpoint at 3.84 s to test the specialisation mechanism directly.

**What the record says.**

* `DENSE-FT-BASELINE-AUDIT` (ledger 2026-08-31): Singh et al. did fine-tune the unpruned model ("for a fair
  comparison") but published no numbers for it and released no checkpoint; Zenodo 21977996 holds 8 files,
  none dense-FT; the released `audioldm-m-full.ckpt` is byte-identical to the official pretrained release.
* `DENSE-FT-CLOSURE` (same day): **Arshdeep Singh confirmed the fine-tuned unpruned checkpoint was deleted
  for cluster storage reasons and no longer exists** (`docs/dense_ft_baseline_availability_audit.md` §8).
  The supervisor closed the decision tree: no retraining, no approximate reconstruction (recipe
  under-specified: step count, trainable modules and schedule of the dense run were never published).
* `EXPERIMENTAL-REOPENING-DESIGN-1` + `TEXTFT-CHECKPOINT-AUDIT`: the official `audioldm-m-text-ft.ckpt`
  (dense, fine-tuned on AudioCaps + MusicCaps, unknown step count) was downloaded, hash-verified,
  strict-loads (690/690) and runs at both latent lengths. Its permitted role is "one public dense
  text-fine-tuning reference", explicitly **not** a matched control. Part-B generation (80 Arm-D prompts
  × 2 durations = 160 WAVs, ≈ 0.6 cr) reached B-GO-CANDIDATE and was never launched.
* `docs/compute_budget.md` §A10 (2026-09-03): "short-duration fine-tune NOT estimable — no training
  throughput has ever been measured"; the assumption-based numbers were 20 k steps ≈ 10 cr, 10⁶ steps
  ≈ 508 cr.

**Re-verification today.**

1. The deletion is a fact established by the author; nothing in the record has changed. The exact
   matched control **cannot be recovered**, and rebuilding it means Singh's full 10⁶-step recipe on the dense
   model: at the measured step time below that is ≈ 85 GPU-hours on a T4 *if* it fit in memory (it does not
   at batch 2 in FP32), i.e. ≈ 75–150 cr — two orders of magnitude above any job this project has run
   (largest settled job: 3.58 cr; lifetime spend 99 cr). **Not correctable.**
2. The §A10 premise is wrong: **training throughput was measured** on 2026-08-26 (job `gate0-smoke-1`,
   `docs/compute_budget.md` "Gate-0 M-Full smoke"): 0.30735 s per optimizer step on a T4, FP32, batch 2,
   latent 96 (3.84 s), dense AudioLDM-M-Full with LoRA, peak 5.4 GB. A LoRA step back-propagates through the
   whole dense U-Net, so a full-parameter step on the **83 %-pruned** U-Net (71 M parameters, blocks 3–4
   narrowed) at the same latent length should cost no more than that plus the optimizer update. The
   real-data trainer exists (`scripts/research/gate0_trainer.py`: manifest loader, CRN-free real batches,
   scheduler proof, resume) and the preprocessed AudioCaps training set is on disk (49 502 clips,
   `data/dataset/metadata/audiocaps/datafiles/audiocaps_train_label.json`, 31 GB of WAVs). Singh's recipe is
   the upstream one (`audioldm_original_medium.yaml`: lr 1e-4, batch 2, CFG dropout 0.1, duration 10.24 s;
   Singh: 10⁶ steps). A 200-step measured benchmark (≈ 0.15 cr) remains mandatory before any of the numbers
   below are used for a launch decision.
3. Re-derived costs at 0.89 cr/GPU-h (T4 on-demand, the rate behind every settled job here):

   | Option | Steps | GPU time | Train (cr) | + eval of the new checkpoint (192 prompts × 2 durations, CRN vs P) | Total point / cap |
   |---|---:|---:|---:|---:|---:|
   | (b) FT of P (sev 2) at **3.84 s**, single arm | 1×10⁴ | 0.85 h | 0.76 | 1.26 | **2.2 / 2.6** |
   | (b) same | 2×10⁴ | 1.7 h | 1.52 | 1.26 | **3.0 / 3.6** |
   | (b) same | 5×10⁴ | 4.3 h | 3.8 | 1.26 | 5.2 / 6.3 |
   | (b′) two arms: FT of P at 3.84 s AND at 10.24 s, matched steps | 2×10⁴ each | 1.7 h + 4.5 h | 1.52 + 4.0 | 2.5 | **8.2 / 9.8** |
   | (a) dense partial FT at 10.24 s (reviewer's first choice) | 1×10⁴ | ≈ 2.3 h (batch 1 or mixed precision: the FP32 optimizer state alone is 6.7 GB) | ≈ 2–4 | 1.26 | ≈ 3.5–5.5 |
   | (a) same | 1×10⁵ | ≈ 23 h | ≈ 20–40 | 1.26 | **not at project scale** |
   | (c) public dense text-FT reference (Part B, already designed) | — | — | — | 0.61 (80 prompts) / 1.26 (192 prompts) | **0.6 / 0.7** or 1.3 / 1.5 |

   The 10.24-s arm is 2.7× dearer per step (latent 256 vs 96); a full-parameter dense run at 10.24 s does
   not fit a 16 GB T4 at batch 2 in FP32 and would need batch 1, mixed precision or a 24 GB GPU (≈ 1.5–2×
   the hourly rate) — each a documented recipe deviation.

**Verdict.** The *matched* control is **not correctable** (checkpoint deleted, retraining unaffordable). The
reviewer's *directional* substitutes are **correctable at 3–8 cr** and would convert the limitation into
evidence about direction, not into a causal claim:

* **(b) is the recommended experiment** because it tests the mechanism the title implies with the cheapest
  training: fine-tune P (severity 2, A′ convention) on AudioCaps **at 3.84 s** (random 3.84-s crops of the
  10-s training clips, latent 96) with Singh's hyper-parameters for N = 2×10⁴ steps, then evaluate the
  resulting checkpoint against P with common noise on the frozen 192 prompts at 3.84 and 10.24 s.
  Pre-registered contrast: `J_short-FT = R(10.24) − R(3.84)` for the short-trained checkpoint. Specialisation
  predicts `J_short-FT` ≤ 0 (or at least far below the released +0.159); "longer clips are simply easier"
  predicts `J_short-FT` > 0 of the released magnitude. The two-arm variant (b′) adds the same N steps at
  10.24 s so the training duration is the only factor that differs at matched steps; it is the clean design
  and costs ≈ 8 cr. Interpretation limits to state in advance: 2×10⁴ steps is 2 % of the released recovery,
  so absolute R values are not comparable with the released checkpoint; only the *sign and ratio* of J
  across training durations are read. A null (both J ≈ 0 because 2×10⁴ steps change little) is
  uninformative and must be declared as such in the protocol.
* **(c) is the cheapest addition** (≈ 0.6 cr, everything audited): it shows whether a dense model that
  underwent *a* text fine-tune also gains more at 10.24 s than at 3.84 s. It stays a reference, not a control
  (different data, unknown steps), exactly as the TEXTFT audit ruled.
* **(a) at 10⁵ steps is not affordable**; at 10⁴ steps it is a weaker version of (b′)'s second arm on a
  model that is not the object of study.

Independently of compute, the reviewer's request to sell the protocol and the magnitude rather than the
mechanism is a wording decision (title, abstract) that the record already supports: the closure entry
forbids "recovery-specific training causes specialisation" as a framing.

## W2 — is the music null a floor effect?

**What the reviewer asks.** P+FT sits 0.02–0.03 above chance on the hip-hop battery. If dense and real audio
are also near chance there, CLAP cannot resolve anything in that domain and the conclusion must change from
"the gain does not transfer" to "the metric is blind here". Panel (c) shows no dense or real anchors for
music. Improvement #2: add them and report ρ_ref for music.

**What the record says.** `DRAFT5-FLOOR-CEILING`: music cells have chance floors (P/P+FT: sev-1 +0.055/+0.001;
sev-2 −0.013/−0.004 at 3.84 s, +0.070/+0.061 at 10.24 s) but "real audio: not available", "gap closed: not
defined" — no dense on the sev-2 music battery was ever generated, and "0/64 MusicCaps real refs" (ledger
RECOVERY-CROSS-SEVERITY-REP-1-RESULT: music KL/PANN/FAD unavailable for the same reason).

**Re-verification today.**

1. **Dense on hip-hop captions already exists for severity 1.** Job `gate0-gen-1` generated
   `dense_noadapter_p{0..63}_r{0..2}.wav` (192 WAVs, 3.84 s) on the frozen 64-prompt hip-hop battery — the very
   battery of the severity-1 music cell (the phenomenon job's P/P+FT groups). Its scored mean is **0.197**
   (`artifacts/icassp_gate0/gate0_verdict.json`, `mean_base_cosine`, same fused-CLAP revision and Option-B
   convention as the P/P+FT music scores). Against P = 0.117 and P+FT = 0.023 with floors ≈ +0.05, the dense
   model is ≈ 0.15 above chance on these captions, so the battery **does** discriminate for the dense model.
   The floor-effect reading is refuted for severity 1 at 3.84 s: recovery did not fail to register, it moved
   the pruned model from 0.06 above chance to chance while the dense model sits 0.15 above it
   (ρ_dense(music, sev 1) = R/(dense − P) ≈ −0.094/0.080 ≈ −1.2). The files are on
   `/teamspace/jobs/gate0-gen-1/artifacts/.../gen_gate0/`; re-scoring them under the frozen floor–ceiling
   convention (with its own shuffled-caption floor) is a CPU job of a few minutes, 0 cr, and would put the
   dense anchor into panel (c) for that cell.
2. **Dense on the severity-2 music battery** (64 different prompts, both durations) needs 128 new WAVs:
   64 × (0.00219 + 0.00363) + 0.145 = **0.52 cr point / 0.62 cap**, one T4 job with the existing
   `reversal_xsev_gen.py --system dense` path (the `dense_native` context already exists; a `music`/
   `music_native` dense context is a two-line addition mirroring the pruned ones). This yields ρ_dense for
   both sev-2 music cells and lets the paper say "closes x % of the gap to dense on hip-hop" instead of
   "unresolved".
3. **Real hip-hop audio.** MusicCaps clips are YouTube segments. `yt-dlp` from the Studio fails with
   "Sign in to confirm you're not a bot" (tested today on `V9EFYFKlYbE`), so the 64 + 64 reference clips
   cannot be fetched from here. Two ways out: (i) Gabriel downloads the 128 segments on his own machine
   with browser cookies (`start_s`/`end_s` are in `artifacts/icassp_gate0/musiccaps-public.csv`) and drops
   the WAVs into `artifacts/icassp_gate0/real_refs_music/`; scoring is then CPU, 0 cr; (ii) a
   **domain-level** ceiling from what is already on disk: the 193 Kim clips
   (`artifacts/icassp_gate0/kim193_wav_3p84s/`, real 4.0-s hip-hop excerpts with their MusicCaps captions,
   45 unique captions / 44 source ytids) scored against their own captions under the frozen convention.
   Caveat: those 44 ytids were *excluded* from both batteries (leakage rule), so this is the level real
   hip-hop reaches on hip-hop captions in general, not the prompt-matched ceiling of the battery. It
   answers the reviewer's question ("can CLAP resolve anything here?") at 0 cr.

**Verdict. Correctable.** 0 cr for the severity-1 dense anchor and the domain-level real ceiling; 0.52 cr
for the severity-2 dense anchors; prompt-matched real audio only with a manual download.

## W3 — a single, extreme domain shift; Clotho as the natural control

**What the reviewer asks.** Long MusicCaps captions on a model recovered on AudioCaps confound caption
style with content. Clotho keeps the sound-event content and changes the caption style.

**What the record says.** Clotho is mentioned nowhere in the project record (grep over docs, ledger,
configs, scripts: only inside the vendored CLAP training code). §A10 costed a variant, E4 "held-out domain
with AudioCaps-length captions", at 3.50 cr for 192 prompts × 3 systems × 2 durations, never launched. The
manuscript declares the confound (Sec. 3.2, Sec. 5) instead of controlling it.

**Re-verification today.**

* Zenodo record 4783391 (Clotho v2.1) is reachable from the Studio: `clotho_captions_evaluation.csv`
  (0.4 MB, downloaded to scratch, sha256 `0e116233…`), `clotho_audio_evaluation.7z` (1.25 GB; needs
  `py7zr`, disk has 221 GB free). Evaluation split: 1 045 clips × 5 captions; caption length median 11
  words (8–21) vs the AudioCaps battery's 8 (mean 9.7) and the hip-hop battery's 56.5. Clotho audio is
  Freesound (field recordings, effects; no music emphasis), so the *content* stays in the AudioCaps event
  universe while the *style* changes moderately — the intermediate step the reviewer wants between
  AudioCaps and MusicCaps. Two caveats to write into the protocol: AudioLDM's pre-training corpus included
  Freesound (AudioLDM paper), so Clotho content is not unseen by the dense model, only by the recovery
  stage (AudioCaps only); Clotho clips are 15–30 s, so the real-audio anchor is a 10.24-s / 3.84-s crop.
* Design: seeded-hash selection of 96 (or 192) evaluation clips with one caption each (caption index also
  seeded), no content filtering; P, P+FT and dense at 3.84 and 10.24 s with common noise; shuffled-caption
  floors; real-audio ceiling from the cropped Clotho WAV; ρ_real and ρ_dense as for AudioCaps. Pre-specified
  reading: if the hip-hop null is caption-style-driven, R_Clotho ≈ R_AudioCaps at both durations; if it is
  exposure/content-driven, R_Clotho falls between AudioCaps and hip-hop. The duration interaction on
  Clotho is a free secondary.
* Cost: 96 × 3 × (0.00219 + 0.00363) + 0.145 = **1.82 cr / 2.19 cap**; 192 prompts: 3.50 / 4.20; P and
  P+FT only, 96 prompts: 1.26 / 1.52.

**Verdict. Correctable**, ≈ 1.8 cr, with a real-audio anchor included (unlike the music battery).

## W4 — the sweep stops at the native duration; "noise at 3.84 s"

**What the reviewer asks.** Because 10.24 s is both the training and the maximum tested duration, "peak at
the training duration" and "longer is better" are indistinguishable; and since both systems sound like noise
at 3.84 s, the short-point CLAP gain may be alignment of noise.

**What the record says.** §A10 E1c: one point beyond the training length (15.36 s), 3 systems × 192
prompts = 576 WAVs, 2.90 cr / 3.48 cap, not launched. The limitation is declared in Sec. 5.

**Re-verification today.**

* Feasibility of > 10.24 s: the generator sets `latent_t_size` and the preprocessing duration per context
  (`reversal_xsev_gen.py`), and the sweep already ran at four latent lengths (96/128/192/256); the AudioLDM
  U-Net has no absolute positional embedding on the latent axis (`AttentionPool2d` in `openaimodel.py` is
  only instantiated for the class-conditional pooling head, which this model does not use), so 384 (15.36 s)
  or 512 (20.48 s) latents run without code changes beyond a new context entry. Anchors: **no real-audio
  reference exists beyond 10 s** (AudioCaps clips are 10 s), so the cell is read against dense and the
  shuffled-caption floor only; fused CLAP already uses its > 10-s fusion path at 10.24 s, so 15.36 s is
  scored under the same regime. Cost: 3 systems × 192 × 0.00479 + 0.145 = **2.90 / 3.48**; P and P+FT
  only: 1.98 / 2.38; 3 systems × 96 prompts: 1.52 / 1.83. Adding 20.48 s as a second point: +3.57.
  Pre-specified rule (extends `docs/draft5_opsweep.md`): D4 = R(15.36) − R(10.24); peaked if hi95(D4) < 0,
  monotone if lo95(D4) > 0, plateau otherwise.
* The "noise at 3.84 s" objection is answerable from existing evidence at 0 cr and the answer belongs in the
  paper, not only in the repo: the short point is **not** broken for every model — dense scores 0.207 at
  3.84 s on the same prompts (floor −0.028, real crop 0.274: 76 % of real audio,
  `xsev_dense_192_control_result.json`, `draft5_floor_ceiling_result.json`); what is broken is the
  83 %-pruned pair (P 0.015, P+FT 0.100, i.e. 0.02 and 0.115 above chance). The paired gains at 3.84 s
  also exist off CLAP: KL +0.66 [+0.42, +0.92], PANNs capture +0.19 [+0.06, +0.32]
  (`xsev_secondary_metrics_short.json`), and the crop analysis shows the first 3.84 s of a native
  generation carries R = +0.172. The listening remark ("both noise at 3.84 s") is one blinded author on 8
  pairs and the paper already says CLAP resolves a gain the ear does not.

**Verdict.** Beyond-native point: **correctable**, 2.0–2.9 cr. Noise objection: **answerable now** with
numbers already committed (wording only).

## W6 — sample sizes and the author listening

**Sample sizes.** Music n = 64 per severity and severity 1 n = 80 were budget choices (ledger: batteries
frozen before generation; "NO more prompts" in the closure). Re-verified pools: the hip-hop keyword filter
leaves 127 eligible MusicCaps prompts after the Kim-source and frozen-64 exclusions
(`xsev_music_manifest.json`, `candidates_after` 127), so **+63 prompts** are available under the frozen
rule: P/P+FT × 2 durations = 252 WAVs → 0.73 + 0.145 = **0.88 cr / 1.05 cap** (with dense: 1.25 / 1.50);
music n goes 64 → 127 (CI width ÷ 1.4). Severity 1 can be run on the 192 severity-2 prompts (disjoint from
its 80): P/P+FT × 192 × 2 durations = 768 WAVs → 2.24 + 0.145 = **2.38 cr / 2.86 cap**; sev-1 n goes 80 → 272
and the MDE from 0.065 to ≈ 0.035, enough to resolve the +0.044 sev-1 interaction if it is real. Both
checkpoints (`l1_audioldm-m-full_p1.ckpt`, `l1_p1_finetuned_global_step_999999.ckpt`) are on disk.
**Verdict: correctable**, 0.9 + 2.4 cr.

**Listening.** A six-listener blinded expert panel (design D3, fixed panel, power analysis, loudness
protocol, catch trials, platform) was designed, frozen (`aa906c2` → `7019ea5`) and prepared, then
**cancelled before recruitment because the co-author did not approve running a listening test without
ethics/institutional approval** (ledger LISTENING-STUDY-CANCELLED, 2026-09-01; 0 participants, 0 responses).
Nothing about that constraint is a compute or data problem: the study is ready to launch the day an
approval or a co-author decision exists. The 8-pair author check was the substitute and the paper labels it
as informal. **Verdict: not correctable by this project alone**; it is a co-author/ethics decision Gabriel
must reopen with KCL if he wants human evidence. If reopened, the frozen design already answers the
reviewer's "sanity check, not evidence" (powered for the native-point preference, 6 listeners × 2
durations).

## Presentation items (deferred, not acted on)

* W5 — a half-column table with R, J and CIs for the four durations and three scorers (the values exist in
  `PAPER_EXPANDED_RESULTS.md` and are verified by `verify_draft12_numbers.py`; page budget allows ≈ 3 lines
  today, so a table means trimming prose).
* W7 — abstract "yields no alignment gain" → "leaves the gain unresolved" (the body already says
  "unresolved"; the interval [−0.028, +0.039] admits half of R_short). Fig. 1 density. Anonymity: not an
  issue — ICASSP 2027 is single-anonymous (checked 2026-09-03), the reviewer assumed double-blind.
* W1 (wording half) — title/abstract to sell the protocol and the magnitude ("recovery gain is
  operating-point dependent") rather than the mechanism; consistent with the closure's forbidden framings.

## Priority if credit is confirmed

| Rank | Action | cr (point / cap) | What it changes in the paper |
|---|---|---:|---|
| 1 | W2: rescore the existing dense hip-hop WAVs (sev 1) + Kim real-clip domain ceiling | 0 | dense + real anchors in panel (c) for one cell; refutes the floor reading at sev 1 |
| 2 | W2: dense on the sev-2 music battery, both durations | 0.52 / 0.62 | ρ_dense for the two sev-2 music cells ("closes x % of the dense gap on hip-hop") |
| 3 | W1(c): public dense text-FT reference on the 192 prompts | 1.26 / 1.52 | a dense fine-tuned reference point for the duration response (reference, not control) |
| 4 | W3: Clotho battery, 96 prompts, 3 systems | 1.82 / 2.19 | second domain with real audio; separates style from content |
| 5 | W6: music +63 prompts (with dense) | 1.25 / 1.50 | music n = 127 |
| 6 | W4: 15.36-s point, P / P+FT / dense | 2.90 / 3.48 | peak vs monotone beyond the training duration |
| 7 | W6: severity 1 on the 192 prompts | 2.38 / 2.86 | sev-1 n = 272, J resolvable |
| 8 | W1(b): short-duration fine-tune of P, single arm, 2×10⁴ steps (after a 0.15-cr benchmark) | 3.0 / 3.6 | direction of the specialisation mechanism |
| 8′ | W1(b′): two matched arms | 8.2 / 9.8 | the clean mechanism test |

Ranks 1–4 together ≈ 3.6 cr point / 4.3 cap; 1–7 ≈ 10 cr / 12 cap; everything ≈ 18 cr / 22 cap. Order of
execution matters for the budget: the Studio itself billed ≈ 8.7 cr of idle CPU time between 2026-09-03
20:05 UTC and 2026-09-05 04:00 UTC with no job running (≈ 0.27 cr/h, see `docs/compute_budget.md`
2026-09-05 entry) — more than any single item above except the fine-tune.

## Provenance of this review

* Reviewer text: received from Gabriel in chat on 2026-09-05 00:56 MVD (Spanish); this file paraphrases it.
* Record consulted: `PROGRESS.md`, `docs/experiment_ledger.md` (entries DENSE-FT-BASELINE-AUDIT,
  DENSE-FT-CLOSURE, EXPERIMENTAL-REOPENING-DESIGN-1, TEXTFT-CHECKPOINT-AUDIT, LISTENING-STUDY-PREP,
  LISTENING-STUDY-CANCELLED, DRAFT5-REVIEW-ACTIONS…A10, XSEV-MUSIC-NATIVE-1-*, DRAFT5-OPSWEEP-1-RESULT),
  `docs/compute_budget.md`, `docs/claims_matrix.md`, `docs/dense_ft_baseline_availability_audit.md`,
  `icassp/MANUSCRIPT_NOTES.md`, the result artifacts under `configs/research/`.
* Live checks: Lightning SDK `billing_service_get_user_balance()` (read-only), job list with settled
  `total_cost` (48 jobs, sum 27.20 cr), `Studio().machine` = CPU; Zenodo API for Clotho; `yt-dlp` metadata
  probe (blocked); file inventory of `/teamspace/jobs/gate0-gen-1`; `data/checkpoints/`, `data/dataset/`.
* No GPU launched, no WAV generated, no score computed, no frozen artifact touched.
