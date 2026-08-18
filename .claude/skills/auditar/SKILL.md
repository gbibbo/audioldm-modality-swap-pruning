---
name: auditar
description: Evidence-first audit protocol for debugging, technical review, scientific result review, surprising metrics, reproducibility checks, or any conclusion that could change a project gate or claim.
---

# Auditar

Use this before reaching an important technical or scientific conclusion.

## Protocol

1. State the question as a falsifiable claim. Name what was observed, what was expected, and what observation would distinguish the plausible explanations.
2. Inspect or reproduce before theorizing. Read the exact code/config/artifact and run the narrowest command that can expose the issue.
3. Track provenance. For every result that matters, identify producing code path, Git commit, resolved config, dataset/manifest, checkpoint/hash, seed policy, and runtime/GPU metadata when applicable.
4. Generate only materially plausible hypotheses. For each one, define the discriminating test that would make you change your mind.
5. Prefer tests that isolate one relevant variable. A correlation or before/after observation without adequate controls is not a causal result.
6. Re-derive arithmetic and counters when metrics drive a decision. Do not trust copied prose as verification of a number.
7. Attack the provisional conclusion as a hostile reviewer. Check for confounds, stale artifacts, mismatched seeds/configs, metric leakage, cherry-picked timesteps, unequal budgets, wrong code paths, and claims broader than the evidence.
8. For master-plan gates, evaluate exactly the preregistered acceptance condition. Do not move thresholds after seeing results.
9. Before accepting an experiment as reproducible, verify that `docs/experiment_ledger.md` contains the required provenance and points to the raw outputs.

## Verdict labels

* **OBSERVED EVIDENCE**: directly inspected or measured in this session.
* **INFERENCE**: follows from observed evidence but is not directly measured.
* **HYPOTHESIS**: plausible and not yet discriminated; name the deciding experiment.
* **INCONCLUSIVE**: evidence is insufficient or instrumentation is not trustworthy.

A negative or inconclusive result is valid. Do not rescue a failed claim by silently changing the question.

## Output

Lead with the verdict. Then give the smallest evidence set that lets another reviewer reproduce or attack the conclusion. Include exact commands, paths, hashes, or raw outputs when they matter.
