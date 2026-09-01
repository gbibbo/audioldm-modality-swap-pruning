# Public Audio Examples — Per-Example Score Audit (INTERNAL)

**Status:** internal descriptive audit. **NOT deployed to gh-pages.** No score appears on the public page; the deterministic selection is unchanged. Reads only frozen scorer outputs (no recompute).

CLAP = `laion/clap-htsat-fused` rev `365dea6e…` (absolute cosine). ΔCLAP = post-fine-tuning − pruned. Near-zero band `|ΔCLAP| < 0.025` (project SESOI, fixed before inspection). Human-CLAP per-example exists only for severity 2 (severity-1 HC is aggregate-only → unavailable per example). FineLAP frame-level grounding is native-only and AudioCaps-eligible-only; values are mean frame grounding (not probability/quality).

| Section | Ex | ytid | Dur | CLAP pruned | CLAP post-FT | ΔCLAP | ΔHumanCLAP | FineLAP mean (pruned/post-FT, native) | Band |
|---|--:|---|--:|--:|--:|--:|--:|---|---|
| Sev1 AC | 1 | `g5l3Bz6lWnc` | 10.24 s | +0.193 | +0.415 | **+0.223** | —(unavail) | p/r 0.216/0.238 | post_ft_higher |
| Sev1 AC | 1 | `g5l3Bz6lWnc` | 3.84 s | +0.176 | +0.157 | **-0.019** | —(unavail) | — | near_zero |
| Sev1 AC | 2 | `a3GzZKxUTy8` | 10.24 s | +0.008 | +0.373 | **+0.365** | —(unavail) | p/r 0.013/0.430 | post_ft_higher |
| Sev1 AC | 2 | `a3GzZKxUTy8` | 3.84 s | -0.081 | -0.030 | **+0.051** | —(unavail) | — | post_ft_higher |
| Sev1 AC | 3 | `6aWnK1GyeJY` | 10.24 s | +0.167 | +0.318 | **+0.150** | —(unavail) | p/r 0.236/0.792 | post_ft_higher |
| Sev1 AC | 3 | `6aWnK1GyeJY` | 3.84 s | -0.044 | +0.208 | **+0.252** | —(unavail) | — | post_ft_higher |
| Sev1 AC | 4 | `emGPabOePzA` | 10.24 s | +0.518 | +0.243 | **-0.275** | —(unavail) | p/r 0.058/0.001 | pruned_higher |
| Sev1 AC | 4 | `emGPabOePzA` | 3.84 s | +0.098 | +0.092 | **-0.006** | —(unavail) | — | near_zero |
| Sev2 AC | 1 | `gkWd1HugK2w` | 10.24 s | +0.082 | +0.137 | **+0.055** | -0.018 | p/r 0.000/0.000 | post_ft_higher |
| Sev2 AC | 1 | `gkWd1HugK2w` | 3.84 s | +0.127 | +0.142 | **+0.015** | -0.088 | — | near_zero |
| Sev2 AC | 2 | `B3O476LeuXY` | 10.24 s | -0.049 | +0.126 | **+0.174** | +0.114 | n/a | post_ft_higher |
| Sev2 AC | 2 | `B3O476LeuXY` | 3.84 s | -0.162 | -0.170 | **-0.008** | +0.066 | — | near_zero |
| Sev2 AC | 3 | `lVr-PxhZo8s` | 10.24 s | -0.093 | +0.203 | **+0.296** | +0.618 | p/r 0.000/0.050 | post_ft_higher |
| Sev2 AC | 3 | `lVr-PxhZo8s` | 3.84 s | -0.022 | +0.040 | **+0.062** | +0.416 | — | post_ft_higher |
| Sev2 AC | 4 | `vfNKduToki4` | 10.24 s | -0.052 | +0.424 | **+0.476** | +0.327 | p/r 0.000/0.630 | post_ft_higher |
| Sev2 AC | 4 | `vfNKduToki4` | 3.84 s | -0.013 | +0.101 | **+0.114** | +0.172 | — | post_ft_higher |
| Sev2 music | 1 | `V9EFYFKlYbE` | 3.84 s | +0.015 | -0.103 | **-0.117** | -0.172 | — | pruned_higher |
| Sev2 music | 2 | `u6tgeRXOxnU` | 3.84 s | -0.037 | -0.043 | **-0.006** | -0.233 | — | near_zero |

## Aggregate over the 18 displayed CLAP comparisons

- ΔCLAP **post-fine-tuning higher**: **11**
- **near-zero** (|ΔCLAP| < 0.025): **5**
- ΔCLAP **pruned higher**: **2**

## Severity-1 special cases vs Gabriel's informal listening (descriptive only)

Gabriel's informal impressions are NOT scientific annotations and are not treated as labels; they are compared descriptively to the frozen metrics.

- **Example 1 @10.24 s** (`g5l3Bz6lWnc`): ΔCLAP **+0.223** (post-fine-tuning higher); FineLAP native mean pruned 0.2163 → post-FT 0.2385. Metrics **agree** with the impression that post-fine-tuning is reasonably decent.
- **Example 3 @10.24 s** (`6aWnK1GyeJY`): ΔCLAP **+0.150** (post-fine-tuning higher); FineLAP native mean pruned 0.2357 → post-FT 0.792 (strongly grounded). Metrics **agree**; the requested event is far more grounded after fine-tuning.
- **Example 4 @10.24 s** (`emGPabOePzA`): ΔCLAP **-0.275** — **pruned is higher**; FineLAP native mean pruned 0.0584 vs post-FT 0.0014 (≈0). **Both CLAP and FineLAP rank pruned > post-fine-tuning here, agreeing with the informal impression that post-fine-tuning sounds worse.** (Human-CLAP is unavailable per-example for severity 1.) So for this prompt the semantic metrics do **not** conflict with perception — no scorer/perception mismatch is demonstrated; the metrics correctly flag the worse post-fine-tuning sample.

## Interpretation (conservative)

- The public examples show substantial **sample-level heterogeneity**: across the 18 displayed comparisons, ΔCLAP is post-fine-tuning-higher in most, near-zero in several, and pruned-higher in a minority (notably Example 4 @10.24 s).
- The population-level post-fine-tuning advantage (reported elsewhere on the complete evaluation sets) does **not** imply monotonic improvement for every prompt.
- Semantic-alignment metrics (CLAP) and frame-level grounding (FineLAP) here **agree** with the informal perceptual ranking on the inspected severity-1 examples, including the worse case (Example 4); these 10 examples do not, by themselves, demonstrate a scorer/perception mismatch.
- These examples remain **representative** because selection was outcome-independent (identifier hash only). No conclusion is drawn about CLAP being 'wrong', about fine-tuning improving or damaging audio quality, or about causal mechanism, from these examples.

Artifacts: `configs/research/public_audio_examples_score_audit.json` (self-sha `5c2f54dfa4fd…`). Public manifest unchanged (`723982bb8482…`).
