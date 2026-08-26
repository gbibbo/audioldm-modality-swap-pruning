#!/usr/bin/env python3
"""WIRING REGRESSION ONLY — superseded as the scientific path by gate0_generator.py (production
generator: EMA convention, guidance 2.5, eta 0.0, deterministic paired noise, provenance) +
gate0_smoke.py. Kept because it exercises the trainer->generator->scorer->bootstrap chain end to end.
DO NOT use for experimental generation.

End-to-end WIRING test for the Gate-0 pipeline (NOT science):
    generator (DENSE M-Full + LoRA, 3.84 s/latent 96) -> fused-CLAP scorer -> prompt-clustered
    paired ΔCLAP + cluster bootstrap -> Gate0Verdict object.

Tiny/local only: 2 battery prompts x 1 seed x few DDIM steps, two systems (base = LoRA-B zero,
"adapter" = LoRA-B perturbed) so ΔCLAP is nonzero and the whole chain produces a verdict. CPU-only.
Do NOT infer any science from the numbers — this only proves the components connect.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch
import yaml

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research")

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
PREREG = "configs/research/icassp_gate0_prereg.yaml"
BATTERY = "configs/research/icassp_gate0_battery.json"
KIM_MANIFEST = "artifacts/icassp_gate0/kim193_train_manifest.json"
OUTDIR = "artifacts/icassp_gate0/e2e_smoke"
METRICS_PY = ".venv-metrics/bin/python"

_orig_load = torch.load
def _cpu_load(*a, **k):
    k.setdefault("map_location", "cpu"); return _orig_load(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--ddim", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(20260826)
    torch.load = _cpu_load

    # CPU shim: DDIMSampler defaults device=cuda (ddim.py); force CPU (mirrors measure_tgen).
    import audioldm_train.modules.latent_diffusion.ddim as _ddim
    _orig_ddim_init = _ddim.DDIMSampler.__init__
    def _patched(inner, model, schedule="linear", device=None, **kw):
        _orig_ddim_init(inner, model, schedule=schedule, device=torch.device("cpu"), **kw)
    _ddim.DDIMSampler.__init__ = _patched

    from measure_tgen import build_model
    from audioldm_peft import PeftConfig, setup_peft
    from audioldm_peft.layers import LoRALinear
    from research_pruning.eval.cluster_bootstrap import gate0_verdict

    pre = yaml.safe_load(open(PREREG))["gate0"]
    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = pre["data"]["train_clip_seconds"]
    dev = torch.device("cpu")
    model, _ = build_model(config, dev)
    model.latent_t_size = pre["data"]["latent_t_size"]   # 96 -> generate at 3.84 s
    model.eval()

    cfg = PeftConfig(rank=pre["lora"]["rank"], alpha=pre["lora"]["alpha"], dropout=0.0,
                     target_linear=True, target_conv2d=False, train_bias=False,
                     train_groupnorm_affine=False, train_layernorm_affine=False,
                     root_path="model.diffusion_model",
                     include_name_substrings=tuple(pre["lora"]["target_substrings"]))
    setup_peft(model, cfg)

    prompts = [p["caption"] for p in json.load(open(BATTERY))["prompts"]][: args.n_prompts]
    dummy_wav = json.load(open(KIM_MANIFEST))["data"][0]["wav"]  # audio ignored for text->audio gen

    def gen_one(caption):
        from audioldm_train.utilities.data.dataset import AudioDataset
        from torch.utils.data import DataLoader
        ds = AudioDataset(config=config, split="test", waveform_only=False,
                          dataset_json={"data": [{"wav": dummy_wav, "caption": caption}]})
        batch = next(iter(DataLoader(ds, batch_size=1)))
        with torch.no_grad():
            _, cond = model.get_input(batch, model.first_stage_key, unconditional_prob_cfg=0.0)
            samples, _ = model.sample_log(cond=cond, batch_size=1, ddim=True,
                                          ddim_steps=args.ddim, unconditional_guidance_scale=1.0)
            mel = model.decode_first_stage(samples)
            wav = model.mel_spectrogram_to_waveform(mel)
        w = np.asarray(wav).squeeze()
        return w

    def save_wav(w, path):
        import soundfile as sf
        sf.write(path, w.astype(np.float32), 16000, subtype="PCM_16")

    # --- two systems: base (LoRA-B zero) and adapter (LoRA-B perturbed) ---
    systems = {}
    # base first (fresh LoRA has lora_B == 0 -> identity)
    base_items = []
    for i, cap in enumerate(prompts):
        w = gen_one(cap); p = f"{OUTDIR}/base_p{i}.wav"; save_wav(w, p)
        base_items.append({"caption": cap, "wav": p})
    systems["base"] = base_items

    # perturb LoRA-B -> nonzero adapter
    with torch.no_grad():
        for _, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.lora_B.add_(torch.randn_like(m.lora_B) * 0.05)
    adap_items = []
    for i, cap in enumerate(prompts):
        w = gen_one(cap); p = f"{OUTDIR}/adapter_p{i}.wav"; save_wav(w, p)
        adap_items.append({"caption": cap, "wav": p})
    systems["adapter"] = adap_items

    # --- score each system with fused-CLAP in .venv-metrics (subprocess) ---
    def score(items):
        man = f"{OUTDIR}/_score_in.json"; out = f"{OUTDIR}/_score_out.json"
        json.dump({"items": items}, open(man, "w"))
        subprocess.run([METRICS_PY, "scripts/research/gate0_clap_scorer.py",
                        "--score-json", man, out], check=True,
                       env={**os.environ, "OPENBLAS_CORETYPE": "Haswell"},
                       stdout=subprocess.DEVNULL)
        return np.array(json.load(open(out))["cosines"], dtype=np.float64)

    base_cos = score(systems["base"])
    adap_cos = score(systems["adapter"])

    # --- prompt-clustered paired ΔCLAP + cluster bootstrap (n_seeds=1 here) ---
    base_arr = base_cos.reshape(-1, 1)     # (n_prompts, n_seeds=1)
    adap_arr = adap_cos.reshape(-1, 1)
    verdict = gate0_verdict(adap_arr, base_arr)
    res = {
        "note": "WIRING TEST ONLY — tiny data, 1 seed, ddim=%d; not a scientific result" % args.ddim,
        "n_prompts": len(prompts),
        "base_cosines": [round(float(x), 4) for x in base_cos],
        "adapter_cosines": [round(float(x), 4) for x in adap_cos],
        "gate0_verdict": verdict.as_dict(),
        "chain_ok": True,
    }
    print(json.dumps(res, indent=2))
    print("\nE2E WIRING PASS — generator -> fused-CLAP -> paired ΔCLAP/bootstrap -> verdict all connected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
