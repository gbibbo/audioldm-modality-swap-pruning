# Camera-ready edit list for Draft 13 (all CPU, 0 cr) — the 2×2 result + the round-2 reviewer's conditions

Exact edits to apply in Overleaf (or I apply them to `icassp/sections/draft13_*.tex` on your word and re-verify numbers
+ page budget). Every number is from a committed artifact. Grouped by priority.

## A. Fold in the 2×2 fine-tuning control (`configs/research/r2_EXT2x2_result.json`; audit passed)

**A1 — Sec. 4.2 (draft13_4.tex), after "It produces the opposite ordering."** Add the symmetric control:

> A matched intervention fine-tuned the same pruned checkpoint for the same $20{,}000$ steps at $10.24$\,s.
> It gains $+0.048$ \ci{+0.025}{+0.070} at $10.24$\,s and $+0.017$ \ci{+0.001}{+0.033} at $3.84$\,s, an interaction
> $J=+0.031$ \ci{+0.010}{+0.052}. Training at the longer duration therefore did not move the gain toward that
> duration; the interaction is if anything smaller than for the $3.84$\,s intervention ($\Delta J=-0.035$
> \ci{-0.064}{-0.004}). Both checkpoints recover more at $10.24$\,s regardless of where they were trained, which is
> what an operating-point account predicts and a training-duration-specialization account does not.

**A2 — Abstract (draft13_1.tex).** The clause "contradicting the prediction of training-duration specialization" is now
supported by a *paired* control, so it may stay; to state the evidence rather than assert it, replace that clause with:

> A checkpoint fine-tuned at $3.84$\,s and a matched one fine-tuned at $10.24$\,s both recover more at $10.24$\,s, so
> the favorable duration does not follow the training duration.

**A3 — Sec. 5 limitations (draft13_4.tex), in the "missing matched dense fine-tune" paragraph.** Add the dense 2×2 as a
declared negative (do NOT present it as a clean analogue):

> A reduced-scale attempt to build the paired design directly — fine-tuning the dense model itself for $20{,}000$
> steps at each duration — is uninformative here: the dense model was already AudioCaps-fine-tuned, and a raw-weight
> $20{,}000$-step fine-tune lowers its CLAP alignment rather than raising it, so it cannot separate dense from pruned
> recovery. It does, however, show the same absence of training-duration specialization ($\Delta J=-0.005$
> \ci{-0.034}{+0.024}). A matched dense control at the released budget is still the missing experiment.

## B. Round-2 reviewer camera-ready conditions (independent of the 2×2)

**B1 — primary interaction upper bound (Sec. 4.1 + Table 1).** `+0.159 \ci{+0.131}{+0.188}` → `+0.159 \ci{+0.131}{+0.187}`
(the frozen artifact `xsev_result.json` gives 0.187; 0.188 was a different bootstrap seed). Two occurrences.

**B2 — "plateaus" overstates D4** (Sec. 4.1 "The gain therefore plateaus…" and Sec. 5 "The plateau at $15.36$\,s…").
Replace "plateaus"/"plateau" claims with the weaker, data-supported form, e.g. Sec. 4.1: "The gain therefore does not
clearly increase beyond $10.24$\,s" and Sec. 5: "recovery does not clearly increase beyond $10.24$\,s rather than
peaking there." (D4 $=+0.021$ \ci{-0.023}{+0.067} admits a $+0.067$ step.)

**B3 — severity-1 heterogeneity** (Sec. 4.1, one sentence; `configs/research/r2_posthoc_pooled_anchors.json`). Report both
subsets and their difference: "The pooled severity-1 interaction is $+0.112$ \ci{+0.076}{+0.149} ($n=176$); the two
prompt sets differ (original $80$: $+0.044$; new $96$: $+0.169$; difference $+0.124$ \ci{+0.058}{+0.194}), so the
pooled value carries between-set heterogeneity."

**B4 — hip-hop dense anchor to full coverage** (Table 1 footnote / Sec. 4.3; `r2_posthoc_pooled_anchors.json`). The dense
anchor now covers all $127$ prompts: $\rho_{\rm dense}=0.11$ ($3.84$\,s) / $0.12$ ($10.24$\,s) pooled (was $\approx
0.02$–$0.04$ on the $64$-prompt subset). Update the "closes only about $2\%$" wording to "≈ $11$–$12\%$ pooled over
$127$ prompts, with between-subset spread"; conclusion (an order of magnitude below AudioCaps) unchanged.

**B5 — define "PANNs capture"** (first use in Sec. 4.1 / Table 1 caption): add "(the fraction of the top-10 PANNs event
classes of the real clip that the generated clip also ranks in its top-10)".

**B6 — uncited references [6] [7] [17].** Either cite them where relevant (lai2021parp / peng2023dphubert on speech
pruning in the Background; liu2024audioldm2 near the AudioLDM mention) or drop them from the bibliography.

**B7 — presentation** (deferred earlier; optional for camera-ready): trim the "For replication" and "This protocol is
intentionally smaller" paragraphs if space is needed; the repository URL is allowed (ICASSP 2027 regular review is not
double-blind, per the Paper Kit — see `docs/reviewer2_response_manuscript.md` §7).

## Notes

* A1–A3 change what the paper claims; they are derived directly from the audited result but are your prose to sign off.
* denseft-s ac_native is $n=170$ (OUT_OF_FUNDS cut $22$ WAVs); the result is already resolved. A ~$0.3$-cr top-up would
  finish it to $192$ if you want the round number, but no conclusion depends on it.
* After applying, re-run `verify_draft13_numbers.py` and `pagecheck_times.py` (both 0 cr).
