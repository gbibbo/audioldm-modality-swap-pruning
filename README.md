# Recovery Gain Is Operating-Point Dependent in Pruned Text-to-Audio Diffusion

Research code, pre-registrations, provenance, evaluation artifacts and manuscript material for a study of structured pruning followed by recovery fine-tuning in text-to-audio latent diffusion. The evaluated systems are released AudioLDM-M checkpoints from Singh et al.; the paper studies how the **gain produced by recovery fine-tuning** changes across inference conditions.

The current ICASSP manuscript is an evaluation paper. It does not introduce a new pruning method and does not claim that pruning causes the duration dependence described below.

## Main finding

A single post-recovery score can substantially misrepresent how much performance fine-tuning has restored. At 83% pruning, the paired CLAP gain grows from +0.085 at 3.84 s to +0.244 at 10.24 s. The effect is reproduced by Human-CLAP, KL-to-reference and PANNs event capture, and it remains large under the published sampler recipe.

Reviewer-follow-up experiments sharpen the interpretation. Fine-tuning the pruned model at 3.84 s does **not** move the largest gain to 3.84 s. The resulting checkpoint still gains more at 10.24 s, with an interaction of +0.065 [+0.043, +0.087]. A public dense text-fine-tuned AudioLDM reference shows the same direction, and the recovery gain plateaus rather than peaks when evaluation is extended to 15.36 s. The paper therefore frames the result as **operating-point dependence of fine-tuning gain**, not specialization to the training duration.

Domain transfer is also graded rather than binary. Recovery transfers strongly to Clotho, with R = +0.210 [+0.176, +0.243] at 10.24 s and 59% of the gap to real audio closed. On hip-hop captions the dense model is well above chance, so the battery is informative, but recovery closes only a small part of the dense gap. With 127 prompts the hip-hop gain is positive but much smaller than on AudioCaps.

## Paper and reviewer follow-up

- [ICASSP manuscript source](icassp/icassp_operating_point.tex)
- [Paper companion](PAPER_COMPANION.md), mapping each paper section to repository evidence
- [Expanded results](PAPER_EXPANDED_RESULTS.md), with complete intervals and secondary analyses
- [Reviewer follow-up protocol](docs/reviewer2_followup.md)
- [Reviewer follow-up results](docs/reviewer2_followup_results.md)
- [Reviewer response map](docs/reviewer2_response_manuscript.md)
- [Claims matrix](docs/claims_matrix.md)

The manuscript is designed to be understood without opening the repository. The repository is the audit and reproduction layer.

## Systems under study

| System | Definition | U-Net parameters |
|---|---|---:|
| dense | AudioLDM-M-Full public release | 415.96 M |
| P, severity 1 | released `(1,2,3,1)` pruned checkpoint | 145.67 M, 65.0% removed |
| P+FT, severity 1 | same pruned architecture after released recovery fine-tuning | 145.67 M |
| P, severity 2 | released `(1,2,1,1)` pruned checkpoint | 71.08 M, 82.9% removed |
| P+FT, severity 2 | same pruned architecture after released recovery fine-tuning | 71.08 M |

`Recovery` names the fine-tuning stage. It does not imply complete restoration.

## Evaluation design

The main estimand is the paired recovery gain

`R = score(P+FT) - score(P)`.

The duration interaction compares that gain across requested clip lengths. Scores are evaluated with common generation noise, prompt as the statistical unit and prompt-level percentile bootstrap intervals. Shuffled-caption floors, the dense checkpoint and real audio provide anchors for interpreting score magnitude.

The primary scorer is fused LAION-CLAP. Corroborating analyses include Human-CLAP, KL-to-reference, PANNs event capture, FineLAP frame-level grounding, crop controls and the published sampler recipe. Analysis status and protocol hashes are retained in the committed artifacts.

## Reviewer-follow-up experiments

The ICASSP review motivated a pre-specified follow-up battery stored in `docs/reviewer2_followup.md`.

| Experiment | Question answered | Main result |
|---|---|---|
| E3 | Is the duration effect specialization to the recovery-training duration? | No. A 3.84-s recovery fine-tune still gains more at 10.24 s. |
| B | Is duration-dependent fine-tuning gain unique to a pruned model? | No. A public dense text-FT reference shows the same direction. |
| E1c | Does gain peak at the 10.24-s training duration? | No resolved peak. Gain plateaus at 15.36 s. |
| E8 | Does the duration interaction replicate at severity 1 with higher power? | Yes. Pooled n=176 gives J = +0.112 [+0.076, +0.149]. |
| E5 | Does recovery transfer to a milder held-out domain? | Yes. Clotho transfer is strong and close to AudioCaps. |
| E6 | Is the hip-hop result a floor artifact? | No. Dense audio-text alignment is clearly above chance. |
| E7 | Was the original hip-hop battery simply too small? | More power resolves a small positive gain, still far below AudioCaps. |

Machine-readable results are in `configs/research/r2_*_result.json`.

## Reproducibility

Core CPU-side verification commands include

```bash
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/run_research_tests.py --all
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_score_verdict.py
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_opsweep_verdict.py --exp sweep --verdict
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_opsweep_verdict.py --exp pubrecipe --verdict
```

Public AudioLDM artifacts are fetched from their original records and checksum-verified rather than redistributed. The repository retains frozen prompt manifests, pre-registrations, result JSONs, scorer revisions and bootstrap namespaces so that reported contrasts can be traced back to their inputs.

## Repository structure

```text
audioldm_train/     upstream AudioLDM training code plus the documented local patch
audioldm_peft/      parameter-efficient recovery infrastructure
research_pruning/   pruning diagnostics and evaluation utilities
scripts/research/   experiment, scoring, verification and figure entry points
tests/research/     CPU research test suite
configs/research/   frozen manifests, pre-registrations and result artifacts
icassp/             ICASSP manuscript source
docs/               protocols, claims matrix, ledgers, audits and reviewer follow-up
listening_study/    blinded perceptual-study tooling; protected audio and keys excluded
```

## Scope and limitations

The strongest conclusion is methodological. Recovery should be evaluated as a function of operating point rather than certified by one benchmark setting. The current evidence does not identify a pruning-specific mechanism because the exact dense checkpoint given Singh et al.'s recovery fine-tune is unavailable. The public dense text-FT model is informative as a reference but is not a matched causal control.

The domain evidence spans AudioCaps, Clotho and one hip-hop battery. Clotho shows substantial transfer, while hip-hop shows much weaker transfer. These results do not justify a general claim that recovery fails out of domain.

The small author listening exercise remains a descriptive sanity check only. It is not used as inferential evidence.

## Frozen upstream references

- `haoheliu/AudioLDM-training-finetuning`, commit `702a638d023b008a2d9a45cdf1e1f4fcdc590dfc`, preserved as `upstream-frozen`
- `Arshdeep-Singh-Boparai/PruningAudioLDM`, commit `6f65f628fabc4ad27770753698fc81944e820f9f`, preserved as `pruning-reference-frozen`

Upstream code is MIT. Pretrained AudioLDM checkpoints are CC-BY-NC-4.0 according to the upstream project and are not redistributed here.
