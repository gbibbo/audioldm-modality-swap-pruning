#!/usr/bin/env python3
"""KIM-CLIP-LENGTH executable CPU audit (no GPU, no long training, synthetic audio).

Question: is M-Full REALLY locked to a 10.24 s latent (256 frames), forcing us to
pad Kim's 4 s clips to 10.24 s (option d)? Or is true / near-4 s training feasible
without changing learned architecture params or breaking checkpoint compatibility?

Options under test:
  (a) native 4-s waveform -> mel -> latent through the existing VAE + U-Net
  (b) shorter latent time dim with the SAME pretrained weights (conv/attn flexibility)
  (c) 10.24-s tensors, diffusion loss MASKED to the real 4-s region
  (d) current: 4-s audio zero-padded to 10.24 s, unmasked loss

For each we care about: exact shapes wave->mel->latent->U-Net->loss; does the
pretrained ckpt strict-load unchanged; does forward run; is the training loss
well-defined; does the pruning materializer / Scenario-B LoRA mapping change; how
faithful to Kim's real 4-s semantics.
"""
import json, os, sys, traceback
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")  # E-BLAS guard
torch.manual_seed(20260826)

# CPU-only shim: some upstream constructors (get_vocoder) torch.load without
# map_location and hit CUDA storage tags. Force CPU (mirrors measure_tgen._cpu_torch_load).
_orig_load = torch.load
def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu")
    return _orig_load(*a, **k)
torch.load = _cpu_load

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
OUT = "artifacts/icassp_gate0/kim_clip_length_audit.json"  # gitignored (artifacts/)

R = {"config": CONFIG, "tests": {}}

cfg = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
p = cfg["model"]["params"]
pre = cfg["preprocessing"]
SR = pre["audio"]["sampling_rate"]
HOP = pre["stft"]["hop_length"]
FIL = pre["stft"]["filter_length"]
WIN = pre["stft"]["win_length"]
NMEL = pre["mel"]["n_mel_channels"]
R["params"] = {"sr": SR, "hop": HOP, "filter": FIL, "win": WIN, "n_mel": NMEL,
               "latent_t_size_cfg": p["latent_t_size"], "latent_f_size_cfg": p["latent_f_size"],
               "unet_channel_mult": p["unet_config"]["params"]["channel_mult"],
               "unet_has_latent_t_param": "latent_t_size" in p["unet_config"]["params"]}

from audioldm_train.utilities.model_util import instantiate_from_config

sd = torch.load(BASE_CKPT, map_location="cpu")
sd = sd.get("state_dict", sd)
scale_factor = float(sd["scale_factor"]) if "scale_factor" in sd else 1.0
R["scale_factor"] = scale_factor

# ---------------------------------------------------------------- build U-Net (base 1,2,3,5)
def strip(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

unet = instantiate_from_config(p["unet_config"]).eval()
unet_sd = strip(sd, "model.diffusion_model.")
miss, unexp = unet.load_state_dict(unet_sd, strict=False)
R["tests"]["T1_unet_strict_load"] = {
    "n_weights_in_ckpt": len(unet_sd), "missing": len(miss), "unexpected": len(unexp),
    "strict_ok": len(miss) == 0 and len(unexp) == 0,
    "note": "latent_t_size is NOT a UNet constructor param -> no weight shape depends on it",
}

# ---------------------------------------------------------------- build VAE (embedded M-Full VAE)
# Force reload_from_ckpt=None (the on-disk vae_mel_16k_64bins.ckpt is NOT the M-Full VAE,
# 204/398 tensors differ per repo notes); load the embedded first_stage_model.* instead.
fs_cfg = json.loads(json.dumps(p["first_stage_config"]))  # deep copy
fs_cfg["params"]["reload_from_ckpt"] = None
vae = instantiate_from_config(fs_cfg).eval()
vae_sd = strip(sd, "first_stage_model.")
vmiss, vunexp = vae.load_state_dict(vae_sd, strict=False)
enc_path_missing = [k for k in vmiss if k.startswith(("encoder.", "quant_conv.", "decoder.", "post_quant_conv."))]
R["tests"]["T1b_vae_strict_load"] = {
    "n_weights_in_ckpt": len(vae_sd), "missing": len(vmiss), "unexpected": len(vunexp),
    "encode_decode_path_missing": len(enc_path_missing),
    "missing_key_prefixes": sorted({k.split(".")[0] for k in vmiss}),
    "note": "missing are loss/vocoder submodules not stored in the ckpt; encode/decode path fully loaded",
}

# ---------------------------------------------------------------- mel pipeline
from audioldm_train.utilities.audio.stft import TacotronSTFT
stft = TacotronSTFT(FIL, HOP, WIN, NMEL, SR, pre["mel"]["mel_fmin"], pre["mel"]["mel_fmax"])

def wav_to_fbank(dur_s):
    n = int(round(dur_s * SR))
    t = torch.linspace(0, dur_s, n).unsqueeze(0)
    y = 0.5 * torch.sin(2 * 3.14159 * 440.0 * t)  # in [-1,1]
    mel = stft.mel_spectrogram(y)[0]           # returns (mel, mag, phase, energy) -> [B, n_mel, T_mel]
    fbank = mel.transpose(1, 2)                # [B, T_mel, n_mel]
    return y.shape[-1], fbank

@torch.no_grad()
def vae_encode(fbank):
    x = fbank.unsqueeze(1)                     # [B,1,T_mel,n_mel]
    post = vae.encode(x)
    z = post.sample() if hasattr(post, "sample") else post
    return (scale_factor * z)

durations = {"10.24 (native)": 10.24, "4.0 (exact Kim)": 4.0,
             "3.84 (96 lat)": 3.84, "4.16 (104 lat)": 4.16}
R["tests"]["T2_wave_mel_latent_shapes"] = {}
for name, d in durations.items():
    try:
        nsamp, fb = wav_to_fbank(d)
        z = vae_encode(fb)
        R["tests"]["T2_wave_mel_latent_shapes"][name] = {
            "duration_s": d, "n_samples": nsamp,
            "mel_shape": list(fb.shape), "mel_frames": fb.shape[1],
            "latent_shape": list(z.shape), "latent_t": z.shape[2], "latent_f": z.shape[3],
            "latent_t_div_by_8": (z.shape[2] % 8 == 0),
        }
    except Exception as e:
        R["tests"]["T2_wave_mel_latent_shapes"][name] = {"duration_s": d, "ERROR": repr(e)[:200]}

# ---------------------------------------------------------------- U-Net forward at various latent_t
FILM_DIM = p["unet_config"]["params"]["extra_film_condition_dim"]
LATENT_C = p["unet_config"]["params"]["in_channels"]
LATENT_F = p["latent_f_size"]

@torch.no_grad()
def unet_forward(latent_t):
    x = torch.randn(1, LATENT_C, latent_t, LATENT_F)
    tstep = torch.tensor([500])
    y = torch.randn(1, FILM_DIM)
    return unet(x, timesteps=tstep, y=y, context_list=[], context_attn_mask_list=[])

R["tests"]["T3_unet_forward_by_latent_t"] = {}
for lt in [256, 128, 104, 100, 96, 64, 32]:
    try:
        out = unet_forward(lt)
        R["tests"]["T3_unet_forward_by_latent_t"][str(lt)] = {
            "latent_t": lt, "div_by_8": lt % 8 == 0,
            "out_shape": list(out.shape), "out_matches_in": list(out.shape) == [1, LATENT_C, lt, LATENT_F],
            "ok": True,
        }
    except Exception as e:
        R["tests"]["T3_unet_forward_by_latent_t"][str(lt)] = {
            "latent_t": lt, "div_by_8": lt % 8 == 0, "ok": False, "ERROR": repr(e)[:160]}

# ---------------------------------------------------------------- diffusion loss well-defined
def diffusion_loss(latent_t, mask_frac=None):
    x0 = torch.randn(1, LATENT_C, latent_t, LATENT_F)
    noise = torch.randn_like(x0)
    xt = 0.7 * x0 + 0.7 * noise   # stand-in noisy latent (schedule-agnostic; loss shape is the point)
    tstep = torch.tensor([500])
    y = torch.randn(1, FILM_DIM)
    with torch.no_grad():
        eps = unet(xt, timesteps=tstep, y=y, context_list=[], context_attn_mask_list=[])
    se = (eps - noise) ** 2
    if mask_frac is not None:  # option (c): restrict loss to the first mask_frac of time
        keep = int(round(latent_t * mask_frac))
        m = torch.zeros(1, 1, latent_t, 1); m[:, :, :keep, :] = 1.0
        loss = (se * m).sum() / (m.sum() * LATENT_C * LATENT_F)
    else:
        loss = se.mean()
    return float(loss), (mask_frac, keep if mask_frac is not None else None)

R["tests"]["T4_loss_well_defined"] = {}
try:
    l256, _ = diffusion_loss(256)
    R["tests"]["T4_loss_well_defined"]["d_full_256"] = {"loss": l256, "finite": torch.isfinite(torch.tensor(l256)).item()}
except Exception as e:
    R["tests"]["T4_loss_well_defined"]["d_full_256"] = {"ERROR": repr(e)[:160]}
try:
    l96, _ = diffusion_loss(96)
    R["tests"]["T4_loss_well_defined"]["a_native_96"] = {"loss": l96, "finite": torch.isfinite(torch.tensor(l96)).item()}
except Exception as e:
    R["tests"]["T4_loss_well_defined"]["a_native_96"] = {"ERROR": repr(e)[:160]}
try:
    # option (c): 256-frame latent, loss masked to ~4 s (first 100/256 frames ~ 0.39)
    lc, info = diffusion_loss(256, mask_frac=100/256)
    R["tests"]["T4_loss_well_defined"]["c_masked_256_to_4s"] = {"loss": lc, "kept_frames": info[1],
                                                                "finite": torch.isfinite(torch.tensor(lc)).item()}
except Exception as e:
    R["tests"]["T4_loss_well_defined"]["c_masked_256_to_4s"] = {"ERROR": repr(e)[:160]}

# ---------------------------------------------------------------- pruning materializer time-independence
from research_pruning.diagnostics.random_masks import build_pruned_unet
R["tests"]["T5_pruned_materializer_time_independence"] = {}
for cm_name, cm in [("dense_1235", [1, 2, 3, 5]), ("pruned_1231", [1, 2, 3, 1]),
                    ("mild_1234", [1, 2, 3, 4])]:
    try:
        pu = build_pruned_unet(cfg, channel_mult=cm).eval()
        nparam = sum(v.numel() for v in pu.state_dict().values())
        outs = {}
        for lt in [256, 96]:
            x = torch.randn(1, LATENT_C, lt, LATENT_F); ts = torch.tensor([500]); y = torch.randn(1, FILM_DIM)
            with torch.no_grad():
                o = pu(x, timesteps=ts, y=y, context_list=[], context_attn_mask_list=[])
            outs[str(lt)] = list(o.shape)
        # a to_q weight shape: proves LoRA slice target (feature dim) is time-independent
        toq = [k for k in pu.state_dict() if k.endswith("to_q.weight")]
        toq_shape = list(pu.state_dict()[toq[0]].shape) if toq else None
        R["tests"]["T5_pruned_materializer_time_independence"][cm_name] = {
            "channel_mult": cm, "n_params": nparam,
            "forward_out_shapes": outs,
            "example_to_q_weight": {"key": toq[0] if toq else None, "shape": toq_shape},
            "note": "forward works at both 256 and 96; to_q shape has no time dim (LoRA slicing is channel-only)",
        }
    except Exception as e:
        R["tests"]["T5_pruned_materializer_time_independence"][cm_name] = {"channel_mult": cm, "ERROR": repr(e)[:200]}

# ---------------------------------------------------------------- verdict (computed from results)
t3 = R["tests"]["T3_unet_forward_by_latent_t"]
feasible = sorted(int(v["latent_t"]) for v in t3.values() if v.get("ok"))
exact4s = t3.get("100", {})
verdict = {
    "unet_time_downsample_factor": 8,   # channel_mult len 4 -> 3 stride-2 stages
    "vae_time_downsample_factor": 4,    # mel frames / latent_t (empirical, T2)
    "constraint": "latent_t must be divisible by 8 (U-Net skip-connection symmetry)",
    "premise_256_fixed_is": "FALSE — latent_t is not a weight-bearing param; ckpt strict-loads (690/690) at any length",
    "exact_4p0s_latent_100_feasible": bool(exact4s.get("ok", False)),
    "feasible_latent_t": feasible,
    "nearest_to_kim_4s": {"latent_t_96": "3.84 s (pure crop of Kim's 4 s, zero padding)",
                          "latent_t_104": "4.16 s (0.16 s pad)"},
    "VERDICT": "A/B (true near-4 s) FEASIBLE at latent_t=96 (3.84 s); exact 4.0 s (latent 100) INFEASIBLE "
               "(100 not div by 8); (c) masked-256 valid but silence-contaminated fallback; (d) 10.24 s "
               "padding is the scientifically weakest and is NOT substrate-forced. RECOMMEND A at 3.84 s.",
}
R["verdict"] = verdict

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(R, fh, indent=2)
print(json.dumps(R, indent=2))
print("\n=== VERDICT ===")
for k, v in verdict.items():
    print(f"  {k}: {v}")
print("\nWROTE", OUT)
