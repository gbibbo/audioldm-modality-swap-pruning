# Paper companion

This page is the repository companion to **Recovery Gain Is Operating-Point Dependent in Pruned Text-to-Audio Diffusion**.

The ICASSP manuscript is designed to be read without this repository. It contains the scientific question, the experimental design, the figure needed to see the main result, the effect sizes needed to support the claims, and the limitations needed to interpret them. This companion keeps the material that is useful for audit and reproduction but too detailed for the four technical pages.

Paper files

* [`icassp/icassp_operating_point.tex`](icassp/icassp_operating_point.tex), the canonical manuscript source on this branch
* [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md), which contains the complete numerical tables and corroborating analyses removed from the paper body

## How the paper maps to the repository

| Paper section | Scientific question | Repository evidence |
|---|---|---|
| Sec. 1, Introduction | Why is one post-recovery score insufficient? | This page and the project [`README.md`](README.md) give the wider project context and distinguish the final checkpoint from the gain caused by recovery. |
| Sec. 2, Background and motivation | How do pruning, recovery and TTA evaluation fit together? | The frozen upstream implementations and evaluated checkpoints are documented in [`UPSTREAM_README.md`](UPSTREAM_README.md), [`docs/claims_matrix.md`](docs/claims_matrix.md), and the provenance material described in the root README. |
| Sec. 3.1, Case study and causal scope | What are P, P+FT and dense, and what comparison is causally justified? | Checkpoint reconstruction and provenance are documented in the root README under baseline reproduction and provenance. The released checkpoints are treated as the object of study. |
| Sec. 3.2, Operating-point grid | Which durations and domains are compared? | Frozen prompt partitions, manifests and preregistrations live under [`configs/research/`](configs/research/) and [`docs/`](docs/). |
| Sec. 3.3, Pairing, scoring and anchors | How is sampling variance controlled and how are scores interpreted? | Result artifacts preserve the scorer revision, generation recipe, prompt identity, pairing convention, bootstrap settings and chance-floor definitions. See [`configs/research/draft5_floor_ceiling_result.json`](configs/research/draft5_floor_ceiling_result.json). |
| Sec. 3.4, Estimands and analysis status | What do R, J and the recovery ratio measure, and which analyses were specified prospectively? | The exact status of candidate claims is recorded in [`docs/claims_matrix.md`](docs/claims_matrix.md). The result artifacts retain protocol hashes and analysis status. |
| Sec. 4.1, Duration | How strongly does recovery gain depend on requested duration, and is that dependence specialization to the recovery-training duration? | Full values are in [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md). The original sweep is in [`configs/research/draft5_opsweep_result.json`](configs/research/draft5_opsweep_result.json). The 3.84-s fine-tuning intervention, 15.36-s extension, dense text-FT reference and higher-power severity-1 replication are in `configs/research/r2_{E3,E1c,B,E8}_result.json`. |
| Sec. 4.2, Domain | How does recovery transfer across AudioCaps, Clotho and hip-hop prompts? | Full values are in [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md). Clotho, hip-hop dense anchors and the expanded hip-hop battery are in `configs/research/r2_{E5,E6,E7}_result.json`. The author listening remains a descriptive sanity check in [`configs/research/author_listening_1_result.json`](configs/research/author_listening_1_result.json). |
| Sec. 4.3, Generation length | Is the short-duration deficit caused by late content or by the short scoring window? | FineLAP and crop-analysis values are indexed in [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md#section-43-where-the-duration-effect-arises). |
| Sec. 5, Discussion and limitations | How far may the result be generalized? | Negative results and wording constraints are preserved in [`docs/claims_matrix.md`](docs/claims_matrix.md). The paper deliberately does not claim a pruning-specific mechanism because a dense model given the same recovery fine-tune is unavailable. |

## Reviewer-follow-up layer

The ICASSP reviewer raised causal, floor, domain-shift, duration-range, in-paper evidence and power concerns. We ran a pre-specified follow-up battery before changing the manuscript framing. The protocol is [`docs/reviewer2_followup.md`](docs/reviewer2_followup.md), the self-contained result report is [`docs/reviewer2_followup_results.md`](docs/reviewer2_followup_results.md), and [`docs/reviewer2_response_manuscript.md`](docs/reviewer2_response_manuscript.md) maps each concern to the evidence and manuscript change.

These follow-ups changed the interpretation in two ways. Duration dependence remains strong, but the 3.84-s fine-tuning intervention argues against specialization to the training duration. Domain transfer is not binary: recovery transfers strongly to Clotho and only weakly to hip-hop.

## What moved out of the paper

The previous manuscript placed the complete CLAP table, anchor table and many sensitivity intervals inside the four-page body. Those values have not been discarded. They are reproduced in [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md) and remain backed by committed JSON artifacts.

The paper now keeps a value when it performs at least one of three jobs.

1. It quantifies a headline claim.
2. It is needed to distinguish two scientific interpretations.
3. It is needed to understand the scale of an effect.

Intermediate sweep values, complete anchor rows, scorer-by-scorer corroboration and sensitivity intervals stay here because they are valuable for audit but are not required to understand the argument.

## Reproducing the evidence

The repository already contains the generation, scoring and verification entry points. The core CPU-side checks include

```bash
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/run_research_tests.py --all
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_score_verdict.py
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_opsweep_verdict.py --exp sweep --verdict
OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_opsweep_verdict.py --exp pubrecipe --verdict
```

The manuscript figure is generated from committed result artifacts by the figure-building scripts under [`scripts/research/paper_figs/`](scripts/research/paper_figs/). Public AudioLDM artifacts are fetched and checksum-verified by the existing repository workflow rather than redistributed.

## Reading the expanded results

Start with Sec. 4 of the paper. Use [`PAPER_EXPANDED_RESULTS.md`](PAPER_EXPANDED_RESULTS.md) only when you want an exact interval, an intermediate operating point, a sensitivity result, or the artifact that produced a reported value. This keeps the paper readable while retaining a complete numerical audit trail next to the code.
