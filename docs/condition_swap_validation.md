# M2 — Conditioning-path validation

**Milestone:** M2 (master plan `docs/master_plan_v3.md`, "M2 — Conditioning-path validation").
**Status:** PASS. The audio-conditioned and text-conditioned CLAP paths are proven
to be instrumented correctly and to feed the same FiLM interface. M3's fail
condition ("normalization or code-path differences cannot be resolved") is **not**
triggered.
**Environment:** `.venv/bin/python` (CPython 3.10.20, torch 1.13.1). CPU only.
**Config (frozen):** `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`.
**Weights (real):** `data/checkpoints/audioldm-m-full.ckpt` (base U-Net `[1,2,3,5]`),
`data/checkpoints/clap_music_speech_audioset_epoch_15_esc_89.98.pt` (CLAP HTSAT-base).
**Items (real):** AudioCaps `test` split (waveforms @ 16 kHz + captions).
**Upstream patch:** none. `git diff upstream-frozen -- audioldm_train/` stays empty.
All diagnostic code is in `research_pruning/diagnostics/conditioning.py` and
`tests/research/test_conditioning_paths.py`.

Every number below is backed by a saved artifact under
`artifacts/m2_condition_swap/`:
`test_conditioning_paths.log`, `condition_swap_metrics.json`,
`condition_swap_metrics.log`, `embedding_norm_hist.png`, `paired_cosine_hist.png`.

---

## Why direct instantiation is faithful

The diagnostic instantiates `UNetModel` + `CLAPAudioEmbeddingClassifierFreev2`
directly (full `LatentDiffusion` is heavy on CPU: VAE, vocoder, EMA, tokenizer
downloads). The wiring reproduced in `eps_pred`/`paired_eps` is byte-for-byte the
inference path of `LatentDiffusion.apply_model`:

| Step | Production path | file:line |
|---|---|---|
| CLAP conditioner returns `[B,1,512]` | `embed = embed.unsqueeze(1)` | `audioldm_train/conditional_models.py:1321` |
| `apply_model` → wrapper | `x_recon = self.model(x_noisy, t, cond_dict=cond)` | `audioldm_train/modules/latent_diffusion/ddpm.py:1398` |
| `"film"` cond → `y` | `y = cond_dict[key].squeeze(1)` | `audioldm_train/modules/latent_diffusion/ddpm.py:1996-2000` |
| wrapper → U-Net | `self.diffusion_model(xc, t, context_list=[], y=y, ...)` | `audioldm_train/modules/latent_diffusion/ddpm.py:2039-2041` |
| FiLM injection | `emb = th.cat([emb, self.film_emb(y)], dim=-1)` | `audioldm_train/modules/diffusionmodules/openaimodel.py:871-872` |
| FiLM projection | `self.film_emb = nn.Linear(extra_film_condition_dim, time_embed_dim)` | `audioldm_train/modules/diffusionmodules/openaimodel.py:555` |

`eps_pred` passes empty context lists exactly like `DiffusionWrapper.forward`
(`ddpm.py:1989`), because the U-Net does `[None] + context_list` internally
(`openaimodel.py:86`) and would crash on `None`.

---

## (a) How audio vs text is routed

**`embed_mode` switch (inside the CLAP conditioner):**
- text branch: `elif self.embed_mode == "text":` → `get_text_embedding` —
  `audioldm_train/conditional_models.py:1308-1319`.
- audio branch: `if self.embed_mode == "audio":` → resample+mel+`get_audio_embedding` —
  `audioldm_train/conditional_models.py:1280-1307`.

**`cond_stage_key` / `embed_mode` switch (inside `LatentDiffusion`):** during
training the CLAP modality is chosen per step,
```
if torch.randn(1).item() < 0.5:
    ...["cond_stage_key"] = "text";     embed_mode = "text"
else:
    ...["cond_stage_key"] = "waveform"; embed_mode = "audio"
```
`audioldm_train/modules/latent_diffusion/ddpm.py:678-683`. Validation/eval force
**text** (`on_validation_epoch_start`, `ddpm.py:685-713`). The frozen config sets
the default `embed_mode: text` and `cond_stage_key: text`
(`audioldm_original_medium.yaml:134-140`). Our diagnostic toggles the same
`embed_mode` flag (`clap_embed`, `conditioning.py`).

**FiLM injection point:** `openaimodel.py:871-872` — the `[B,512]` CLAP vector `y`
is linearly projected by `film_emb` (`openaimodel.py:555`) and concatenated to the
timestep embedding, which then conditions every ResBlock. `use_extra_film_by_concat`
is `True` because `extra_film_condition_dim=512` is set (`openaimodel.py:552`).

---

## (b) Audio-branch preprocessing and the 16 kHz mismatch

The audio branch (`conditional_models.py:1280-1307`):
1. resample `16000 → 48000` Hz: `torchaudio.functional.resample(...)`
   (`conditional_models.py:1291-1294`). The dataset waveform is
   `10.24 s @ 16 kHz = 163840` samples → `491520` samples @ 48 kHz.
2. mel via CLAP's **own** front end `self.mel_transform`
   (`conditional_models.py:1182-1195`, applied at `:1297`) — window/hop/fmin/fmax
   come from `model_cfg["audio_cfg"]`, independent of the AudioLDM 16 kHz/64-bin
   mel used for the latent.
3. `get_audio_features(audio_data, mel, 480000, data_truncating="fusion",
   data_filling="repeatpad", ...)` (`conditional_models.py:1298-1305`).

**Mismatch (documented, not blocking):** `get_audio_features` asserts
`audio_data.size(-1) > max_len` and then truncates to the first `max_len=480000`
samples (`training/data.py:452,460`), i.e. **10.0 s @ 48 kHz**. Every AudioCaps
clip is `10.24 s`, so the resampled signal is `491520` samples and the **last
`11520` samples ≈ 0.24 s of every clip is dropped** before CLAP audio embedding.
The text branch has no such truncation. This is consistent across all items (the
`>` assertion holds because every clip exceeds 10 s after resampling); a clip
shorter than 10 s at 48 kHz would trip the assertion, but none exist in this
manifest. The 0.24 s truncation is a fixed, deterministic property of the audio
path and is identical for the full and pruned models, so it does not confound the
modality-swap diagnostic; it is recorded here for reproducibility.

---

## (c) Are CLAP embeddings L2-normalized in this checkpoint? — YES

Both heads apply `F.normalize(..., dim=-1)`:
`get_text_embedding` (`audioldm_train/modules/clap/open_clip/model.py:749`) and
`get_audio_embedding` (`model.py:777`). The roberta pooler flagged "newly
initialized" by HuggingFace at `roberta-base` load is **overwritten** by the CLAP
checkpoint, which contains `text_branch.pooler.dense.{weight,bias}` and is loaded
with `model.load_state_dict(ckpt)` (`open_clip/factory.py:152-153`); text
embeddings are therefore the trained ones (the cosine sanity check in (d)
confirms this empirically).

Empirical `||e||_2` over **N = 48** real AudioCaps items
(`artifacts/m2_condition_swap/condition_swap_metrics.json`):

| modality | mean | std | min | max |
|---|---|---|---|---|
| audio | 0.99999999 | 4.2e-08 | 0.99999988 | 1.00000012 |
| text  | 0.99999998 | 5.2e-08 | 0.99999988 | 1.00000012 |

Both are unit-norm to floating-point precision. **No normalization mismatch
between modalities** → the FiLM input magnitude is identical in scale for audio
and text.

---

## (d) Paired vs cross-item audio/text cosine

Cosine between `e_a` and `e_t` of the **same** item vs **different** items
(N = 48; `condition_swap_metrics.json`):

| set | n | mean | std | p05 | p50 | p95 |
|---|---|---|---|---|---|---|
| paired (same item) | 48 | **0.2479** | 0.1024 | 0.0625 | 0.2643 | 0.4295 |
| cross (next item) | 48 | -0.0896 | 0.1230 | -0.2693 | -0.1177 | 0.1486 |
| cross (all off-diagonal) | 2256 | -0.0789 | 0.1149 | -0.2491 | -0.0912 | 0.1227 |

**Sanity check PASS:** same-item cosine mean (0.248) ≫ cross-item mean (-0.079).
The paired cosine is modest in absolute terms (~0.25) because of the well-known
CLAP audio–text **modality gap** (unit-norm embeddings occupy partly separated
cones); crucially it is strongly and consistently above the cross-item baseline,
so audio and text encode aligned-but-distinct conditioning signals. This is
exactly the regime in which a modality swap is a meaningful, non-trivial
intervention. Figure: `artifacts/m2_condition_swap/paired_cosine_hist.png`.

---

## (e) CPU forward cost and the M3A plan-B estimate

U-Net forward on CPU (base `[1,2,3,5]`, latent `[B,8,256,16]`, 5 reps;
`condition_swap_metrics.json`):

| batch | sec/forward (mean) | sec/forward (min) | sec/item |
|---|---|---|---|
| 1 | 1.524 | 1.500 | 1.524 |
| 4 | 4.941 | 4.361 | 1.235 |

**M3A plan-B estimate** (forward-only diagnostics: full + L1 + 20 random masks =
22 models × 2 modalities × N_eval items × S timestep strata, at the batch-4
per-item rate; excludes CLAP/VAE/backward):

| N_eval | strata | U-Net forwards | CPU-hours |
|---|---|---:|---:|
| 64 | 1 | 2 816 | 0.97 |
| 64 | 5 | 14 080 | 4.83 |
| 200 | 1 | 8 800 | 3.02 |
| 200 | 5 | 44 000 | 15.10 |

A full M3A diagnostic pass is therefore ~1–15 CPU-hours depending on the frozen
`N_eval`/strata (plus CLAP embedding and VAE encode, done once per item). This is
feasible on CPU as a fallback, but the GPU benchmark is still required before M3
runs (Compute Gate CG). These are CPU forward costs, **not** the GPU numbers
`docs/compute_budget.md` must hold — that file stays `TBD_MEASURED`.

---

## (f) Figures

- `artifacts/m2_condition_swap/embedding_norm_hist.png` — `||e_a||` and `||e_t||`
  histograms (both spike at 1.0).
- `artifacts/m2_condition_swap/paired_cosine_hist.png` — paired vs cross-item
  cosine histograms.

---

## Test results (`tests/research/test_conditioning_paths.py`, CPU)

Log: `artifacts/m2_condition_swap/test_conditioning_paths.log`.

| Test | Result | Key evidence |
|---|---|---|
| T1 DIMENSIONS | PASS | `e_a`, `e_t` = `(4,1,512)`; `film_emb` `512→768`; `extra_film_condition_dim==512`; `use_extra_film_by_concat==True` |
| T2 DROPOUT-OFF | PASS | upstream default `unconditional_prob=0.1`; frozen config does **not** override it; diagnostic forces `0.0`; text & audio embeds bit-identical across calls |
| T3 DETERMINISM | PASS | same seed → `z_t`, `t` identical; `max|Δ| eps_a = 0.0`, `eps_t = 0.0` |
| T4 PAIRING | PASS | `hash(z_t) == hash(noise)` (`7c4ad16b…`); shared `z_t`,`t`; control: perturbing `z_t` changes hash and epsilon |
| T5 NON-DEGENERATION | PASS | `mean|eps_a − eps_t| = 1.148e-02`, `max = 3.520e-01` → swap is real |

**Unconditional dropout (T2 detail):** `unconditional_prob` defaults to `0.1`
(`conditional_models.py:1151`) and the frozen config's
`cond_stage_config.film_clap_cond1.params` sets only `pretrained_path`,
`sampling_rate`, `embed_mode`, `amodel` — so **the raw upstream conditioner would
apply 10 % stochastic unconditional-token replacement** (`conditional_models.py:1322-1324`).
The diagnostic sets `unconditional_prob = 0.0` (`build_clap`), which is why
`clap_embed` is bit-identical across repeated calls.

---

## Reproduce

```bash
cd /teamspace/studios/this_studio/audioldm-modality-swap-pruning
.venv/bin/python tests/research/test_conditioning_paths.py            # T1..T5
.venv/bin/python scripts/research/m2_condition_swap.py --n 48         # norms, cosine, timing, figures
```

## Fail condition (master plan) — NOT triggered

No unresolved normalization or code-path difference was found: both modalities are
unit-norm, both enter the identical FiLM interface, the paths are deterministic and
correctly paired, and the swap is non-degenerate. **M3 is not blocked by M2.**
Remaining M3 gates (GPU benchmark, Compute Gate CG, frozen `pilot_protocol.md`,
disjoint val split) are independent of this milestone.
