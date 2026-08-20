# Hostile review — round 5 (2026-08-20, Reviewer B reply) — rc4, proposed freeze candidate

Reviewer A: **YES to DECISION-V4-00 (scientific direction), NOT YET to rc3 as the frozen
execution protocol** — six internal residues where round-4 decisions had not reached the
operative text. All six are applied in `docs/master_plan_v4_draft.md` **rc4**; no
methodological change, only pre-registration consistency.

| # | A's item | What rc4 says now |
|---|---|---|
| 1 | `K_rand` contradictory (§6 / V4-04 still said "≥ 10, 20 if funded") | `K_rand = 20` minimum everywhere; Gate E is an **exact rank test** `p = (1 + #{V_rand ≥ V_P0-std}) / (K_rand + 1) ≤ 0.05` (K=20 → attainable min 1/21). Historical rc2 changelog line annotated as superseded. |
| 2 | Gate M still used rc2 variables (`G_F`, `ΔG_P`, single exposure) | Blocks rewritten: tail = {log audio exposure, log calibration-caption exposure}; guidance = {`G_tmpl,F`, `ΔG_tmpl,P`}; acoustic = FineLAP-masked descriptors. **Identifying contrast**: slope of calibration-caption exposure in P1 minus slope in P0-std. Whole-caption `G_F` is descriptive only and never enters Gate M. |
| 3 | Tie-break by raw LRT invalid across blocks of different dimensionality | Tie-break = **cross-validated predictive gain** (held-out log-loss, k-fold by source wav, inside the mechanism set). Holdout untouched. |
| 4 | Placebo needs matching, not just cardinality | Matched before unblinding on baseline capture rate, eligible-prompt count, approximate exposure, family where possible (ceiling-effect argument accepted). |
| 5 | No intervention defined for H-acoustic | **One mechanism-general recipe**: reweight calibration slots by the winning block's fitted vulnerability prediction for the slot's requested events; P1-placebo uses the same recipe and weight distribution on matched non-vulnerable events. Acoustic variant eligible only if FineLAP passed (descriptors must exist on the calibration pool). No recipe is invented after Gate M. |
| 6 | FineLAP fallback and seed pairing unspecified | FineLAP **confirmed public** (`AndreasXi/FineLAP`, MIT, 0.2B, `AutoModel.from_pretrained(..., trust_remote_code=True)`, `get_frame_level_score(audio, phrases) → (B,N,T)`); `torch 1.13.1` compatibility is part of the smoke. If the smoke fails, **H-acoustic leaves primary Gate M** (single-label subset = sensitivity only) so the three blocks are always estimated on the same sample. **Seed pairing**: identical initial noise per prompt across systems; the 3-seed panel is the robustness check. |
| + | H-guidance comparability | Primary = canonical template `"a sound of [official alias]"` vs unconditional (`G_tmpl`, `ΔG_tmpl`, relative form); contextual counterfactual (`c_without_e`, `c_only_e`) secondary. |

## Reviewer B's position

rc4 is my proposed freeze candidate. I agree no further literature round is needed; the
niche is established (Singh et al. 2607.13330 document the unequal loss without a
mechanism; Importance-Aware OBS 2607.20048 does content-aware pruning and targeted
calibration in T2I, not event-level mechanistic attribution in TTA).

What remains is not methodology but execution hygiene, all CPU and credit-free, to be done
once DECISION-V4-00 is recorded: FAD/FD fix (F-eval-3); frozen manifests (event set
`E*`, strict/expanded synonym maps, covariate manifest, calibration/mechanism/holdout
partition, sentinel panel, prompt manifests, seed table) with sha256 in the ledger;
FineLAP smoke under `torch 1.13.1`; materializer parameterization for `(1,2,3,4)` with a
bit-exact test at `(1,2,3,1)`; per-slot saliency storage in the M3B runner; counterfactual
text conditioning in the diagnostics runner.

## Success ladder (for the record, A's wording, agreed)

Gate E + Gate M + Gate I ⇒ "we identified what makes a sound event vulnerable under
generative compression and showed that knowledge tells you what to preserve". Gate E +
Gate M only ⇒ mechanistic analysis paper. Gate M = "none" ⇒ valid negative; venue/framing
to be reassessed. Tier 0 alone ⇒ never a paper.
