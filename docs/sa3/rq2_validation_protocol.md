# RQ2 real-LoRA validation — pre-registration (2026-08-21 17:52, Gabriel's GO)

**Status: pre-registration of the next phase, written before any data is fetched or any LoRA is
trained (rule S4: decision table before intervention). Gabriel's GO of 2026-08-21 authorises the
real-LoRA *validation* phase — NOT RQ3, NOT a CASE-C declaration, NOT stage-2.** All definitions are
grounded in `docs/sa3/analysis_protocol_rq1_rq2.md` (§3.4, §3.5, §4.1, §4.3, §5.1, §5.2, §8, §11);
nothing here invents a new statistic or threshold. Hard cap: **5 GPU credits all-time** (2.23 spent).

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
5. **`A_eco` + the prediction check (§7).** Compute `A_eco(g;L)` over the 20 single-block removals
   from the existing states. Primary pre-registered question: **does `A_tan` predict `A_eco` better
   than `D_P` does?** — using the already-defined rank correlation + set-disagreement (§4.1), no new
   threshold invented after seeing the adapters.
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
* **Pass (per control `L_b`), all three required:**
  1. `A_eco(b; L_b)` exceeds `A_eco(g; L_b)` for **every** `g ≠ b` beyond the bootstrap CI, on the
     **field**;
  2. the same on the **end-to-end task metric** `T(L_b; ·)` (§5.2);
  3. the `L_b`-specific greedy **never removes block `b`**.
* **STOP RQ2** if a control fails: a single-block adapter's *parameters* live in one block by
  construction, but if its *function* cannot be localised from outputs, the instrument cannot
  support any ecological reading. Report and close; do not train ecological adapters.

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
  [**primary scalar**], (b) CLAP text–audio to `prompts_L`, (c) `FD_openl3` to `eval_L`, (d)
  in-domain retrieval rate.
* **Contract check before any pruning:** `ΔT_L(dense post)` must be **positive** and within CI of
  `ΔT_L(dense base)`'s sign and order of magnitude. Otherwise the adapter does not transfer → the
  adapter is dropped from the axis (RQ2 "no contract to preserve"); RQ1 stands.
* **Compatibility band** `[ΔT_lo, ΔT_hi]` spanned by base/post uplifts; **pruning-induced extra
  loss** `ℓ(M;L)=max(0, ΔT_lo − ΔT_L(post^{−M}))`. **Only `ΔT` carries the function claim;** field
  `A_eco` is reported alongside as mechanism.

## 6. `A_eco` and the primary prediction check (§3.4, §4.1)

* **Field `A_eco`:** `A_eco(g; L) = ‖δF(L) − δF^{−g}(L)‖² / ‖δF(L)‖²` on `S_traj`
  (`research_sa3/metrics.py:a_eco`), reusing the already-persisted states; plus its end-to-end
  analogue (adapter effect on generated audio with/without `g`). `L` is **never** used to choose
  probes, `κ`, `N_main`, `n_u`, or any set.
* **Primary question (pre-registered):** does `A_tan` (blind, from `U_gen` r16) predict `A_eco`
  **better than `D_P`**? Statistics fixed in advance (§4.1): (i) rank correlation ρ(A_tan, A_eco) vs
  ρ(D_P, A_eco); (ii) decision-relevant set-disagreement δ_{A_tan,A_eco}(k) vs δ_{D_P,A_eco}(k) at
  k∈{2,4,6}, each against the bootstrap floor. "`A_tan` predicts" ⇔ δ_{A_tan,A_eco}(k) ≤ max floor.
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
* **To build (CPU, Step 0):** `scripts/sa3/train_control_loras.py` (backbone-only `--include`
  wrapper); the `A_eco`/`ΔT_L` driver (apply a trained `L` over the states + the with/without-`L`
  task generations); their tests + CPU dry-runs.
