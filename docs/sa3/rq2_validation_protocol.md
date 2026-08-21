# RQ2 real-LoRA validation — pre-registration **rc1** (2026-08-21 18:18, Gabriel's review)

**Status: pre-registration of the next phase, written before any data is fetched or any LoRA is
trained (rule S4: decision table before intervention). Gabriel's GO of 2026-08-21 authorises the
real-LoRA *validation* phase — NOT RQ3, NOT a CASE-C declaration, NOT stage-2.** All definitions are
grounded in `docs/sa3/analysis_protocol_rq1_rq2.md` (§3.4, §3.5, §4.1, §4.3, §5.1, §5.2, §8, §11);
nothing here invents a new statistic or threshold. Hard cap: **5 GPU credits all-time** (2.23 spent).

**rc1 amendments (2026-08-21 18:18, Gabriel's second review — applied BEFORE any Freesound data is
sourced; rc0 text preserved, changes marked `[rc1]`).** Six pre-data corrections, all tightening the
falsification and none inventing a new statistic:

1. **Controls `L_6`/`L_13` pass criterion re-specified (§3).** Requiring `A_eco(b;L_b)` to be the
   top-1 over **all** `g≠b` is **not** justified as a STOP: a single-block adapter's *parameters*
   live in block `b`, but its *function* can depend on downstream blocks, so removing some `g≠b`
   may perturb the effect as much as removing `b`. The correct positive control uses the algebraic
   fact that removing `b` **physically deletes the whole adapter** (`δF^{−b}(L_b)=0`): the STOP gate
   becomes `A_eco(b)≈1` (sanity) **and** `ΔT_{L_b}(post^{−b})≈0` within uncertainty **and** some
   external removals `g≠b` retain measurable uplift. Top-1 ranking is reported **descriptively**,
   not as a STOP.
2. **`FD_openl3` is descriptive only (§5).** With 5–10 eval clips the 512-dim Fréchet covariance is
   rank-deficient (already observed at N=16). `FD_openl3` **never** enters a gate, contract, or
   adapter decision. Primary scalar stays CLAP audio–audio; CLAP text–audio and retrieval secondary.
3. **`k=6` frozen as the primary `A_tan→A_eco` prediction test (§6).** The pre-gate divergence was
   robust only at k=6 (k=2 unresolved, k=4 marginal). Explicit discrete verdict rules
   CONFIRM/CONTRADICT/AMBIGUOUS are written below **before** the data; k=2/4 are secondary; rank
   correlation is corroborative and never substitutes for the discrete gate.
4. **Two `A_eco` readings (§6).** `A_eco^generic` on the frozen generic panel (the strong
   adapter-agnostic test — compression is blind to the domain) **and** `A_eco^domain` on the adapter's
   held-out `prompts_L` (new dense-post states, same seeds), the latter with a signal/precision guard
   on `‖δF(L)‖²`; an adapter with no measurable field effect is not interpreted.
5. **Per-adapter cost staging (§2, §7).** For each ecological adapter: dense base→post contract →
   **field `A_eco` first** → only if the field link survives, spend on end-to-end `ΔT_L`. No
   expensive audio generation after a field-level failure.
6. **End-to-end control uses the uplift `ΔT`, never absolute `T` (§5).**

**Domain predefinition (before browsing Freesound, to avoid post-hoc selection).** The two
falsification-phase domains are fixed in advance: **`impact/percussion`** and **`water/liquid`**
(well-separable SFX). Pre-ordered fallback if a domain yields `< N_min^clip = 20` valid CC0 44.1 kHz
clips: **`mechanical → animal → ambience`**. Domains are chosen by this rule, **not** by which
adapter turns out easiest to train.

---

## 0. What survived, and what it was NOT

The A_tan gate at N=32 (SA3-ATAN-N32-001) resolved **RQ2's structural pre-gate**, not RQ2:

> the **synthetic LoRA-tangent sensitivity proxy `A_tan` exhibits a stable structural ranking
> difference from standalone field sensitivity `D_P`** (candidate tails {11–16} vs {9–14}, δ=2
> robust at k=6, identical N=16→N=32, both rankings stable). **Whether this reflects real adapter
> compatibility remains untested.**

Two things this does **not** yet establish (Gabriel, 2026-08-21):

1. It is **not** "adaptability occupies distinct structural resources." That phrasing is retired.
   `A_tan` is a tangent-regime proxy computed **without seeing any real adapter**; calling it
   *adaptability* requires the ecological link `A_tan → A_eco → ΔT_L`.
2. `{11–16}` and `{9–14}` are **LOO-ranking candidate tails** (`removal_set()` = the k lowest
   single-block scores), **not** the protocol's sequential-greedy masks `R_X^greedy(k)` (§3.5: 105
   evaluations, re-evaluated from the new architecture at each step because pruning is
   **non-additive**). They are two candidate structural tails — not "the adaptability-aware mask vs
   the deployment mask." Greedy masks are a later step and are not authorised here.

`n_u` stays **frozen at 8** (SA3-ATAN-001). The `n_u_main=4` re-derived on the fuller N=32 data is a
diagnostic; it does not retroactively replace the value the experiment used.

## 1. The two filters RQ2 must still pass

RQ1 is closed (I_PT redundant with D_P → no PT-aware line; no further RQ1 credit). RQ2 now needs:

* **Filter 1 — `A_tan ≠ D_P`:** favourable evidence (the pre-gate above). **Necessary, not
  sufficient.**
* **Filter 2 — the ecological link (the crucial, still-missing one):**
  `A_tan ⟶ A_eco ⟶ ΔT_L`. Does a proxy computed blind to real adapters predict where **trained,
  held-out** adapters actually live (`A_eco`), and does respecting it preserve the adapter's
  **task-level uplift** `ΔT_L` under pruning?

If Filter 2 holds on unseen adapters, we have a criterion computed **without accessing downstream
adapters that predicts which backbone structure can be pruned without destroying LoRA reuse** — the
first genuinely publishable result of this line. If it fails, the line closes for a scientifically
interesting reason (a sophisticated proxy that does not transfer), at ~2.3 cr.

## 2. Ordered protocol (binding; CPU-first, controls-first)

Each GPU step is preceded by a CPU dry-run + a short VRAM/sec-step measurement + a predicted-and-
accumulated cost line in the ledger, from a clean pushed commit. STOP conditions are honoured.

0. **CPU, 0 cr — data + machinery (this phase's Step 0).** Fetch/curate the CC0 44.1 kHz clips
   (§4), build manifests with per-clip license + sha256, freeze train/eval splits and held-out
   prompts **before training anything**. The eval split is **never** used to select checkpoints or
   hyper-parameters. Build the two missing pieces of machinery (§8): `train_control_loras.py`
   (wraps `_external/stable-audio-3/scripts/train_lora.py`, backbone-only `--include`) and the
   `A_eco`/`ΔT_L` driver (reuses the persisted `S_traj` states + `research_sa3/metrics.py:a_eco`).
   CPU dry-runs + tests green.
1. **Positive controls FIRST (§5).** Train `L_6` and `L_13` (single-block, standard `lora` r16,
   backbone-only, §5.1 recipe). Before launching the full step budget, measure VRAM/sec-step and
   estimate cost; proceed only if it sits comfortably under the cap.
2. **Localisation test (§5).** Apply both controls base→dense-post; verify a measurable functional
   uplift; then the localisation pass. **If the controls do not localise a known adaptation → STOP
   RQ2. Do not train any ecological adapter.**
3. **≤2 ecological adapters (§6–§7), not ≥4.** Only if the controls pass. Two distinct SFX domains
   — a cheap falsification phase; they do not yet support the final claim.
4. **Per adapter, contract first (§6).** Verify the dense base→dense post uplift `ΔT_L`. An adapter
   whose uplift does not transfer to the dense post **cannot** validate compatibility under pruning
   and is dropped from the axis (§4.3).
5. **`[rc1]` Field `A_eco` FIRST, then (only if it survives) end-to-end `ΔT_L`.** Compute
   `A_eco^generic` (persisted states) and `A_eco^domain` (new dense-post states on `prompts_L`, with
   the precision guard) over the 20 single-block removals; run the frozen prediction check (§6: k=6
   CONFIRM/CONTRADICT/AMBIGUOUS). **Only if the field link does not already die** spend GPU on the
   end-to-end `ΔT_L` generations — no expensive audio after a field-level failure. No new threshold
   is invented after seeing the adapters.
6. **Branch (§7).** Expected relationship on both adapters → expand to the pre-registered **≥4
   independent domains** and repeat held-out. Clear contradiction on the first two → **stop and
   close the line**; do not buy more adapters hunting for a favourable one.
7. **Not now:** full sequential greedy, `E`-greedy, RQ3 design, stage-2 enumeration, `lora-xs`,
   `dora-rows`. First demonstrate the synthetic proxy predicts trained adapters.

## 3. Positive controls `L_6`, `L_13` (§5.1 — mandatory before RQ2 is interpreted)

* **Train:** `L_6` = `--include "transformer.layers[6]"`, `L_13` = `--include
  "transformer.layers[13]"` (one block in each depth half, away from the boundaries), standard
  **`lora` r16, backbone-only**, on `small-sfx-base`, same recipe/data regime as the ecological
  adapters (§5.2). Apply to the **dense post** at strength 1.0.
* **Pass (per control `L_b`) — `[rc1]` re-specified.** The STOP gate rests on the algebraic fact
  that removing block `b` **physically deletes the entire single-block adapter** (`δF^{−b}(L_b)=0`,
  §1.3), so the instrument must observe the adapter vanish when `b` is removed and survive when it
  is not. All three required:
  1. **Sanity:** `A_eco(b; L_b) ≈ 1` (the whole field effect of `L_b` disappears on removing `b`),
     within the bootstrap CI of 1.0;
  2. **Uplift collapse:** `ΔT_{L_b}(post^{−b}) ≈ 0` within uncertainty (the uplift the adapter added
     is gone once its host block is removed);
  3. **Observability:** at least some **external** removals `g ≠ b` keep a **measurable** `ΔT_{L_b}`
     (the pipeline can still see an uplift when the adapter is present) — proves the metric is not
     dead.
* **Descriptive (not a STOP):** whether `b` is the top-1 of the 20 single-block `A_eco` scores. A
  single-block adapter's *parameters* live in `b`, but its *function* can route through downstream
  blocks, so a `g ≠ b` removal may perturb the effect as much as removing `b`; that is a real
  property, not a broken instrument. Report the ranking; do not gate on it.
* **STOP RQ2** only if the three required conditions fail: if the adapter does **not** vanish when
  its host block is removed, or the pipeline can **never** observe its uplift, the measurement chain
  cannot localise a known adaptation from outputs. Report and close; do not train ecological adapters.

## 4. Data prerequisite (§5.2 — the only external-data dependency)

* **Source (pre-registered):** CC0, **Freesound**. 20–50 captioned **44.1 kHz** clips per domain.
* **Domains:** distinct SFX domains (the unit of generalisation is the *domain*, not `lora` vs
  `dora`). Two for the falsification phase; ≥4 only after §7 confirms.
* **Manifest discipline:** per clip — source URL/id, CC0 license proof, sha256, duration, caption;
  frozen **train/eval split**; a held-out **prompt** list per domain. Audio and generated wavs are
  **gitignored** (never committed); only manifests/hashes/splits are tracked.
* **No leakage:** eval never selects checkpoints or hyper-parameters. Until the data exists, RQ2
  cannot be interpreted (RQ1 and the A_tan tables already ran without it).

## 5. Adapter function — the dense-transfer compatibility band on the uplift (§4.3)

* **Uplift:** `ΔT_L(S) = T(S + L) − T(S)` (same prompts+seeds, with/without `L`). Absolute
  `T(S+L)` is never the contract (base and post differ in backbone quality on the domain).
* **Task metric `T` (frozen before training):** tuple = (a) CLAP audio–audio to `eval_L`
  [**primary scalar**], (b) CLAP text–audio to `prompts_L`, (d) in-domain retrieval rate.
  **`[rc1]` `FD_openl3` is reported descriptively only** — with 5–10 eval clips its 512-dim
  covariance is rank-deficient, so it **never** enters a gate, contract, or adapter decision.
* **`[rc1]` Only the uplift `ΔT` carries a decision.** Every control and contract below is stated on
  `ΔT_L(S) = T(S+L) − T(S)`; absolute `T(S+L)` is never a gate.
* **Contract check before any pruning:** `ΔT_L(dense post)` must be **positive** and within CI of
  `ΔT_L(dense base)`'s sign and order of magnitude. Otherwise the adapter does not transfer → the
  adapter is dropped from the axis (RQ2 "no contract to preserve"); RQ1 stands.
* **Compatibility band** `[ΔT_lo, ΔT_hi]` spanned by base/post uplifts; **pruning-induced extra
  loss** `ℓ(M;L)=max(0, ΔT_lo − ΔT_L(post^{−M}))`. **Only `ΔT` carries the function claim;** field
  `A_eco` is reported alongside as mechanism.

## 6. `A_eco` and the primary prediction check (§3.4, §4.1)

* **Field `A_eco`:** `A_eco(g; L) = ‖δF(L) − δF^{−g}(L)‖² / ‖δF(L)‖²`
  (`research_sa3/metrics.py:a_eco`). `L` is **never** used to choose probes, `κ`, `N_main`, `n_u`,
  or any set. **`[rc1]` computed in two readings:**
  * **`A_eco^generic`** — on the already-persisted generic `S_traj` states (the strong
    adapter-agnostic test: compression never saw the downstream domain). Load-bearing.
  * **`A_eco^domain`** — on the adapter's held-out `prompts_L`, with **new dense-post states**
    captured on those prompts (same seeds). **Signal/precision guard:** interpret the block ranking
    only if `‖δF(L)‖²` clears the precision floor (`metrics.py:precision_ok`); a domain LoRA whose
    field effect on generic prompts is near-zero can make the `A_eco^generic` denominator collapse,
    which is exactly why the domain reading exists.
* **Primary question (pre-registered), `[rc1]` decision rules frozen now:** does `A_tan` (blind, from
  `U_gen` r16) predict `A_eco` **better than `D_P`**?
  * **Primary test = `k = 6`** (the only k where the pre-gate divergence was robust). Let
    `δ_A(6) = δ_{A_tan,A_eco}(6)`, `δ_D(6) = δ_{D_P,A_eco}(6)`. **`[rc1.1]` Each comparison uses the
    §4.1 floors of BOTH criteria it involves — A_eco's own floor never drops out:**
    `F_A = max(f_{A_tan}(6), f_{A_eco}(6))`, `F_D = max(f_{D_P}(6), f_{A_eco}(6))`. Per adapter:
    * **CONFIRM** ⇔ `δ_A(6) ≤ F_A` **and** `δ_D(6) > F_D`: `A_tan` is inside the stability floor of
      `A_eco` and `D_P` is outside.
    * **CONTRADICT** ⇔ the inverse: `δ_D(6) ≤ F_D` **and** `δ_A(6) > F_A`.
    * **AMBIGUOUS** ⇔ anything else (both inside, both outside, or mixed). **`δ_A(6) < δ_D(6)` is
      NOT the gate** — it is a secondary descriptive analysis only (it is not equivalent to the
      floor-based rule; a noisy `f_{A_eco}` can absorb a small `δ_D`).
  * **Secondary (corroborative, never override the gate):** k∈{2,4} set-disagreement; rank
    correlations ρ(A_tan, A_eco) vs ρ(D_P, A_eco).
* **Frozen correspondence:** `U_gen` → standard `lora` r16 (primary). No best-of-two; the probe
  family that "predicts better" is never selected after the fact.

## 7. Decision branches (written before the data)

* **CONTROLS FAIL** (§3) → **STOP RQ2.** Report; no ecological adapters. The instrument cannot
  localise a known adaptation.
* **CONTROLS PASS, 2 ecological CONFIRM** (`A_tan` predicts `A_eco` better than `D_P`, §6; and the
  §5 band holds) → expand to **≥4 independent domains**, repeat held-out. Only then is RQ2's Filter
  2 a candidate; RQ3 design still needs `E`-greedy (§8 of the analysis protocol) and is a separate
  authorisation.
* **CONTROLS PASS, 2 ecological CONTRADICT** `A_tan` → **stop and close the line.** A sophisticated
  proxy that does not transfer is a clean negative at ~2.3 cr. Do not buy more adapters to find a
  favourable one.
* **Ambiguous** → report as such; does not trigger expansion by itself.

## 8. Budget discipline

* Hard cap **5 cr all-time** (2.23 spent → ~2.77 headroom). Register predicted + accumulated cost
  before each training/generation.
* **Priority under the cap:** controls (`L_6`,`L_13`) → 1st ecological → 2nd ecological. If the four
  ecological adapters would threaten the cap, **do not degrade the design to complete four** — stop
  at the controls + as many ecological adapters as fit, and report.

## 9. Explicitly NOT authorised by this document

RQ3 design; CASE-C declaration; sequential-greedy full paths; `E`-greedy adversary; stage-2 mask
enumeration; `lora-xs`/`dora-rows` adapters; `small-music` replication; any expansion beyond 2
ecological adapters before §7 confirms; converting a negative into a new hypothesis.

## 10. Machinery status (2026-08-21)

* **Exists:** `research_sa3/metrics.py:a_eco`; the persisted `S_traj` state-store; `probes.py`
  (`build_probe`/`set_strength`); `score_e_metrics.py` (CLAP/KL/FD scorer, `.venv-metrics`);
  `_external/stable-audio-3/scripts/train_lora.py` (training entry point).
* **To build (CPU, Step 0):** `scripts/sa3/train_control_loras.py` (backbone-only / single-block
  `--include` wrapper); the `A_eco`/`ΔT_L` driver (apply a trained `L` over the states + the
  with/without-`L` task generations); their tests + CPU dry-runs.
* **`[rc1]` Built 2026-08-21 18:18 (CPU, this commit):**
  * `research_sa3/adapters.py` — apply a trained `.safetensors` LoRA onto the base/post
    `ConditionedDiffusionModelWrapper` (upstream `add_lora` + `remap_lora_state_dict`), reusing
    `probes.set_strength`/`restrict_to_surviving` for the `δF(L)` / `δF^{−g}(L)` toggles.
  * `research_sa3/aeco_predict.py` — pure decision core: the §6 k=6 CONFIRM/CONTRADICT/AMBIGUOUS
    verdict, the §3 control-localisation verdict (`[rc1]` re-spec), Spearman corroborator.
  * `scripts/sa3/train_control_loras.py` — training wrapper (standard `lora` r16, backbone-only,
    single-block `--include`, local `small-sfx-base`), `--dry-run-cpu`.
  * `scripts/sa3/rq2_aeco_driver.py` — `A_eco^generic`/`A_eco^domain` + `ΔT_L` manifest driver,
    `--dry-run-cpu`; staging (field first, ΔT only if it survives).
  * `scripts/sa3/build_domain_manifest.py` — CC0 license + 44.1k-mono sha256 manifest, deterministic
    resample-to-44.1 + duration/silence/dedup filters + derived captions (spec §3–§4), 80/20 split +
    `prompts_L`, disjointness checks, `--selftest` (no external data; run in `.venv-metrics`).
  * Tests: `tests/sa3/{test_aeco_predict,test_adapters_include}.py` + `build_domain_manifest.py --selftest`.
* **`[rc1.1]` amended 2026-08-21 18:52 (Gabriel's review of `9b56b71`):** (1) prediction gate uses
  **pair-specific floors** `F_A=max(f_Atan,f_Aeco)`, `F_D=max(f_DP,f_Aeco)` (a single shared floor
  was wrong; `δ_A<δ_D` is secondary-descriptive, not the gate); (2) control STOP decided from
  **measured CIs** (A_eco(b) CI ∋ 1 + precision guard; ΔT(post^{−b}) CI ∋ 0; some external removal
  lower-CI > 0) — no `0.10`/`0.02` constants; (3) A_eco precision guard reads the **frozen η** from
  the smoke (`--eta-config`, pooled `max(η_i)=6.667e-5`), no silent default; (4) T4 training uses
  **`16-mixed` + base fp16** (not bf16); (5) intra-domain selection frozen in
  `docs/sa3/freesound_selection_spec.md` (queries/tags/sort/N/duration/silence/dedup/caption/resample,
  training crop 10 s) — **before** any Freesound page is opened.
