# Recovery-reversal audit (2026-08-27, zero-GPU)

Scientific audit ordered by Gabriel/supervisor after the STOP-2 falsifier negative and the
PHENOM-VALIDITY-GEOM findings. Scope: (1) re-audit the "AudioCaps has 0% music" premise;
(2) tighten artifact↔paper correspondence; (3) downgrade the "retraining from scratch"
wording; (4) literature collision matrix; (5) evaluate the "recovery reversal under domain
shift" candidate thesis and identify the cheapest adequately powered discriminating
experiment. No repo mutation beyond audit/docs; no GPU; frozen STOP-2 verdict untouched.

## 1. AudioCaps music-exposure re-audit (supervisor correction ACCEPTED)

The prior claim "AudioCaps has 0.00% music" (M4-SCREEN exposure table, echoed in
PHENOM-VALIDITY-GEOM) was **wrong as stated**: it counted only the exact `Music` label.
Corrected method: ontology-closure join of the AudioSet MIDs carried per-clip in the exact
in-repo metadata the upstream finetuning pipeline consumes
(`data/dataset/metadata/audiocaps/datafiles/audiocaps_{train,val,test}_label.json`),
against `configs/research/audioset_ontology.json` (closures: Music 189 ids, Music genre 66,
Hip hop music 4, Musical instrument 92). Denominators complete: train 49,502/49,502 unique
ytids joined, 0 unknown MIDs.

| split | n | exact `Music` label | Music-or-descendant | Music-genre desc | Hip-hop desc |
|---|---|---|---|---|---|
| train | 49,502 | 0 (0.0000%) | **375 (0.7575%)** | 0 (0.0000%) | 0 (0.0000%) |
| val | 2,475 rows / 495 ytids | 0 | 10 (0.40%) | 0 | 0 |
| test | 4,820 rows / 964 ytids | 0 | 55 (1.14%) | 0 | 0 |

All 375 train music-descendant hits enter via **Musical instrument** descendants; zero via
genres. Lexical caption scan (weaker measure, labelled as such): "music" 2.14% of train
captions; **rap/hip-hop 9/49,502 (0.018%)**; song/sing 0.93%; instrument words 0.30%.

**Corrected premise:** the recovery corpus is not music-free, but its exposure to music at
all is <1% by ontology (≈2% lexically), with **zero genre labels and ≈zero hip-hop
presence**, while our battery is 100% hip-hop music captions. Caveat: AudioSet labels are
noisy; unlabeled background music cannot be excluded. The phrase "corpus without music"
must not be used; "near-zero exposure to the tested subdomain" is supported.

## 2. Artifact↔paper correspondence (tightened; supervisor's corrections verified)

Paper: Singh, Yuan, Chen, Wang, Plumbley, "Efficient Text-to-Audio Generation via
Pruning", arXiv 2607.13330 (read this session via subagent; quotes verbatim).

* **FAD/KL location corrected:** unpruned M-Full FAD 3.95 / KL 2.16; (1,2,3,1) after
  finetuning **FAD 1.57 / KL 1.678** — reported in **Section 5 running text, NOT Table 3**.
  Table 3 is the per-event PANNs loss/recovery analysis (e.g. safety-critical events: loss
  73.5%, recovery 76.0%). The earlier ledger line (M4-era, entry ~828) that attributed
  "FAD 1.57" to Table 3 is hereby corrected.
* **Pre-finetuning pruned absolute FAD/KL are NOT in the paper** — only Fig. 3 deltas.
* **Eval protocol per paper:** AudioCaps test (964 pairs), 10 s clips, **200 inference
  steps**; guidance scale and sampler **unspecified in the paper**.
* **Eval protocol per the framework the README prescribes** ("follow the official AudioLDM
  repository for evaluation"): `audioldm_original_medium.yaml` `evaluation_params` =
  guidance **3.5**, `ddim_sampling_steps` **200**, `n_candidates_per_samples` **3** with
  **best-of-3-by-CLAP selection** (`ddpm.py:1939-1948`, argmax of `clap.cos_similarity`).
  Inferred, clearly labelled as framework default rather than paper statement.
* **M-Full pre-specialization confirmed:** "The model is further finetuned on the AudioCaps
  dataset for an additional 0.25M steps." So dense-vs-recovered is not finetuned-vs-not;
  the variable is **amount of additional same-domain specialization** (0.25M shared + 1M
  recovery-only) — and the dense/pruned lineage retains music (CLAP 0.197 / 0.117) while
  recovered does not (0.023).
* **Recipe:** paper says 1M steps, U-Net only, VAE/CLAP frozen, "same finetuning
  configuration as used for training AudioLDM-M-Full"; lr/batch not stated. **Mechanical
  match established in-repo:** the ckpt's own optimizer state (lr 1e-4 constant, wd 0.01,
  betas (0.9,0.999), epoch 40, global_step 1,000,000) equals the prescribed config
  (`base_learning_rate: 1.0e-4`, `batchsize: 2` → 24,751 steps/epoch → 40.4 epochs at 1M).
  Artifact = the paper's (1,2,3,1) 1M-step endpoint, by md5 (Zenodo 21977996) + recipe.
* **The paper contains NO music / OOD / CLAP-score evaluation whatsoever** (verified: the
  strings "music", "out-of-domain", "CLAP score" do not appear). Its "forgetting mostly
  recovered" claim (Table 3) is strictly in-domain. Our thesis would **bound**, not
  contradict, their claim.

### Operating-point mismatch (4 axes, authors' framework vs our falsifier)

| axis | authors (framework default) | falsifier (frozen) |
|---|---|---|
| clip duration | 10.24 s (10 s in paper) | 3.84 s |
| inference steps | 200 | 50 (DDIM, eta 0) |
| guidance | 3.5 | 2.5 |
| selection | best-of-3 by CLAP | single gen per seed (preregistered) |

Best-of-3-by-CLAP is itself an alignment-maximizing selection that would partially mask
alignment collapse in reported numbers; our protocol measures the unselected distribution.

## 3. Wording downgrade (accepted)

"Far-from-init retraining / functionally a different model" is **withdrawn** as a causal
description. The measured facts stand exactly as recorded (cos weighted-mean 0.376 vs p1
init, median rel. displacement 1.25, 58.8% of tensors displaced > own norm, EMA sane,
recipe as above) and are now described as **large post-pruning recovery drift** —
parameter-space evidence only; functional replacement would require function/representation
-space evidence not yet collected. Scale/permutation symmetries (GroupNorm + wd 0.01) can
inflate parameter distances; the functional collapse on music is real but its *cause*
(domain specialization vs operating-point brittleness) is exactly what remains to be
discriminated.

## 4. Literature collision matrix (12+ works, 3 subagent sweeps, quotes on file)

Columns: post-pruning recovery / generative / multimodal conditioning / ID-vs-OOD measured
/ cross-modal alignment / legacy adapters / audio / metric reversal.

| Work | PPR | Gen | MM-cond | ID/OOD | X-modal align | Legacy adapters | Audio | Reversal |
|---|---|---|---|---|---|---|---|---|
| Self-Data Distillation (MLSys'25, 2410.09982) | Yes | Yes (LLM) | No | Partial | No | No | No | **Partial — pattern in tables, unframed** |
| TinyFusion (CVPR'25, 2412.01199) | Yes | Yes | No | No | No | No | No | No (stage, not domain) |
| 2ndMatch (CVPR'26, 2506.05398v2) | Yes | Yes | Partial (1 T2I exp) | No | Yes (CLIP score) | No | No | No (monotone) |
| CAR-LoRA (ICLR'26) | Partial (pre-hoc) | Yes (LLM) | No | No | No | **Yes (premise)** | No | No |
| ARF (CVPR'24, 2404.06244) | No | No | Partial | Yes | Partial | No | No | Partial (FT-induced, no pruning) |
| OGEN (ICLR'24, 2401.15914) | No | No | Partial | Yes | No | No | No | Partial (FT-induced) |
| IIMM (ICCV'25, 2407.15731) | No | No | Partial | Yes | Yes | No | No | Partial (predictive law, FT-induced) |
| MoPE-CLIP (CVPR'24, 2403.07839) | Yes | No | Partial | Partial | Yes | No | No | No |
| ZCLIP (CVPRW'26) | Yes | No | Partial | Yes | Yes | No | No | Partial (concedes mechanism; **no pruned-only arm**) |
| Diff-Tuning (NeurIPS'24, 2406.00773) | No | Yes | Partial | No | No | Partial | No | No |
| Kumar LP-FT (ICLR'22, 2202.10054) | No | No | No | Yes | No | Partial | No | **Yes (the reversal logic; discriminative, no pruning)** |
| LLM-Sieve (2505.18350) | Yes | Partial | No | Partial | No | Partial | No | Partial (narrows generality; no rank-flip) |
| Singh et al. (2607.13330) | Yes | Yes | Yes | **No** | No | No | **Yes** | No (never measured off-domain) |
| Concept-suppr. pruned DM (CVPR'25, 2412.15341) | Yes | Yes | Yes | No | Partial | No | No | No (side effect = concepts) |
| LoRA-X (2501.16559) / ProLoRA / Cross-LoRA | Partial | Yes | Yes | No | No | Yes (assumed premise) | No | No |

FAD-vs-CLAP as distinct axes (for the "metric-blind" leg): av-benchmark toolkit; Gui et
al. ICASSP'24 (FAD reliability); KAD ICML'25 ("No More FAD"); FAD encoder-bias
(2602.23958); Human-CLAP (2506.23553, CLAP↔human r≈0.28). Established, citable, and we do
NOT claim "FAD is bad" — only that the axes are non-substitutable.

### Novelty verdict

**The full conjunction is unoccupied**: no work demonstrates a cross-domain RANKING
REVERSAL (recovered > pruned in the recovery domain, recovered < pruned held-out) induced
by post-pruning recovery in a text-conditioned generative model — let alone audio, with a
semantic-alignment metric, plus measured legacy-adapter breakage. The two pruning+recovery
CLIP works never include a pruned-no-recovery arm off-domain; the works with reversals are
finetuning-of-uncompressed (Kumar, ARF, OGEN, IIMM).

**Mandatory wording bounds:** (i) Self-Data Distillation's tables already CONTAIN the
reversal pattern for text-benchmark LLM pruning (SFT-on-GSM8k: 0→62 in-domain while
ARC-C/MMLU fall below pruned-no-FT) as unframed collateral — we cannot claim "first to
show recovery can land below the pruned baseline off-domain" in general; we CAN claim the
first *demonstrated and characterized* cross-domain reversal for conditional generation /
audio / alignment metrics, citing SDD as text-domain collateral precedent. (ii) Kumar et
al. is the structural precedent for the reversal logic (discriminative, no pruning) — cite
and differentiate. (iii) Adapter breakage exists as *assumed premise* (LoRA-X, CAR-LoRA);
our contribution is *measuring* it as a finding (0.046→0.003 under paired protocol).
(iv) Generic "finetuning hurts OOD" is not ours to claim.

## 5. Candidate thesis evaluation + cheapest discriminating experiment (NOT launched)

Thesis ("recovery reversal under domain shift") status: **plausible, unproven, and worth
one cheap experiment**. Already in hand: music arm paired contrast **C_rec − C_pruned =
−0.0941, CI95 [−0.1241, −0.0646]** (n=64 prompts, frozen bootstrap; 79.7% of prompts
negative). Missing: the in-domain arm under the SAME recipe/metric, so that *domain* is
the only variable.

Two rival mechanisms to discriminate: **domain specialization/forgetting** (predicts
recovered > pruned on held-out AudioCaps at OUR recipe) vs **operating-point brittleness**
(predicts recovered ≤ pruned even in-domain at 3.84 s / 50 steps / 2.5 / single-gen; the
published behavior would then live at 10.24 s / 200 / 3.5 / best-of-3).

Cheapest adequately powered sketch (sizing only — full prereg happens only if authorized):

* 64 held-out AudioCaps-test captions (from the 964-ytid split; disjoint from the
  finetuning train split by construction, verified earlier: 0 ytid overlap), frozen seed
  selection; 3 replicates; {p1_pruned_ema_reconstructed, p1_recovered} standalone = **384
  clips ≈ 0.80 cr** at the realized phenom rate (1.6020/768 ≈ 0.00209 cr/clip). Optional
  dense anchor +192 ≈ +0.40 cr. A 2-replicate variant = 256 clips ≈ 0.54 cr with modestly
  wider CIs.
* Power: music-arm paired-contrast half-width was ±0.030 at n=64×3; an in-domain advantage
  ≥ ~0.04 is comfortably detectable (lo > 0). Interaction (difference of the two
  independent within-domain paired contrasts) half-width ≈ ±0.042.
* Gates (to be preregistered pre-data if authorized): reversal = music contrast < 0
  (already established) AND AudioCaps contrast > 0 (new). If the AudioCaps arm shows
  recovered ≤ pruned, the domain story is unsupported and brittleness becomes lead
  hypothesis (next probe would be a small arm at the authors' operating point, ≈10× per
  clip vs ours — priced separately if ever needed).
* Budget: central chain remaining 0.5332 cr → the 3-rep version requires a budget
  amendment (Gabriel's decision; Lightning balance field known stale).

## 6. Open items

1. Guidance/sampler for the paper's numbers are inferred from the framework, not stated —
   keep the label "framework default" in any comparison.
2. Pruned-only FAD/KL exist only as Fig. 3 deltas — if quoted, read off the figure or
   measure ourselves (out of current scope).
3. Label noise: ontology exposure is a lower bound on audible music in AudioCaps.
4. No repo/GPU action authorized beyond this audit; falsifier verdict and Gate-0 remain
   frozen and untouched.
