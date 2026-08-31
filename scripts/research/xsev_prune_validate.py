#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — validate the A'/B' pruned operators before freeze (CPU, 0 cr).

Oracle re-validation (bit-exact), construct pruned2_A / pruned2_B from dense EMA, prove they differ in
EXACTLY the 3 documented decoder-seam tensors (+ norms), and forward-validate all three severity-2
systems through the production pipeline (proper y FiLM conditioning). NO GPU, NO scientific scoring.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_prune_validate.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, "scripts/research"); sys.path.insert(0, os.getcwd())
import torch, yaml
torch.set_grad_enabled(False)
import research_pruning.diagnostics.random_masks as rm
from research_pruning.diagnostics import prune_operator as po
from research_pruning.eval.ema_weights import ema_unet_state_dict, materialize_ema_into_unet

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"
DENSE = "data/checkpoints/audioldm-m-full.ckpt"
PUB_P1 = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
PUB_DP1 = "data/checkpoints/l1_audioldm-m-full_p1_dp1.ckpt"
REC_DP1 = "data/checkpoints/l1_p1_dp1_finetuned_global_step_999999.ckpt"


def unet_rel(path):
    d = torch.load(path, map_location="cpu"); d = d.get("state_dict", d)
    return {k[len("model.diffusion_model."):]: v for k, v in d.items() if k.startswith("model.diffusion_model.")}


def bit_equal(a_sd, b_sd):
    keys = [k for k in b_sd if k in a_sd]
    bit = sum(1 for k in keys if a_sd[k].shape == b_sd[k].shape and torch.equal(a_sd[k], b_sd[k]))
    mism = [k for k in keys if not (a_sd[k].shape == b_sd[k].shape and torch.equal(a_sd[k], b_sd[k]))]
    return bit, len(keys), mism


def main():
    config = yaml.load(open(CONFIG), Loader=yaml.FullLoader)
    config["preprocessing"]["audio"]["duration"] = 3.84
    ranking = rm.load_l1_ranking(PKL)
    dsd = torch.load(DENSE, map_location="cpu"); dsd = dsd.get("state_dict", dsd)
    raw = {k[len("model.diffusion_model."):]: v for k, v in dsd.items() if k.startswith("model.diffusion_model.")}
    ema_base, _ = ema_unet_state_dict(dsd)
    report = {"artifact": "xsev_prune_validate", "oracles": {}, "diff_proof": {}, "forward": {}}

    def target(cm):
        return rm.build_pruned_unet(config, cm).float().state_dict()

    # ---- ORACLES ----
    a_raw131 = po.prune_general(raw, target([1, 2, 3, 1]), ranking, po.LAYER_MAP_APRIME)
    b, n, m = bit_equal(a_raw131, unet_rel(PUB_P1)); report["oracles"]["A_RAW_131_vs_pub_p1"] = f"{b}/{n} mism={len(m)}"
    frozen = rm.materialize(ema_base, ranking, config, channel_mult=[1, 2, 3, 1]).float().state_dict()
    a_ema131 = po.prune_general(ema_base, target([1, 2, 3, 1]), ranking, po.LAYER_MAP_APRIME)
    b2, n2, m2 = bit_equal(a_ema131, frozen); report["oracles"]["A_EMA_131_vs_frozen_sev1"] = f"{b2}/{n2} mism={len(m2)}"
    b_raw111 = po.prune_general(raw, target([1, 2, 1, 1]), ranking, po.LAYER_MAP_BPRIME)
    b3, n3, m3 = bit_equal(b_raw111, unet_rel(PUB_DP1)); report["oracles"]["B_RAW_111_vs_pub_dp1"] = f"{b3}/{n3} mism={len(m3)}"
    oracles_ok = (len(m) == 0 and len(m2) == 0 and len(m3) == 0)

    # ---- CONSTRUCT severity-2 EMA baselines ----
    p2A = po.prune_general(ema_base, target([1, 2, 1, 1]), ranking, po.LAYER_MAP_APRIME)
    p2B = po.prune_general(ema_base, target([1, 2, 1, 1]), ranking, po.LAYER_MAP_BPRIME)

    # ---- DIFF PROOF: exactly 3 tensors ----
    keys = list(p2A)
    diff = [k for k in keys if not (p2A[k].shape == p2B[k].shape and torch.equal(p2A[k], p2B[k]))]
    report["diff_proof"]["n_tensors"] = len(keys)
    report["diff_proof"]["n_differing"] = len(diff)
    report["diff_proof"]["differing"] = diff
    report["diff_proof"]["expected"] = list(po.SEAM_TENSORS)
    report["diff_proof"]["exactly_expected_three"] = (sorted(diff) == sorted(po.SEAM_TENSORS))
    per = {}
    for k in diff:
        d = (p2A[k].float() - p2B[k].float())
        per[k] = {"shape": list(p2A[k].shape), "max_abs_diff": float(d.abs().max()),
                  "l2_diff": float(d.norm()), "rel_norm_diff": float(d.norm() / (p2A[k].float().norm() + 1e-12))}
    report["diff_proof"]["per_tensor"] = per

    # persist oracle+diff results BEFORE the (slower, failure-prone) forward stage
    json.dump(report, open("configs/research/xsev_prune_validate.json", "w"), indent=2)
    print("ORACLES:", report["oracles"])
    print("DIFF exactly-3:", report["diff_proof"]["exactly_expected_three"], report["diff_proof"]["differing"])

    # ---- FORWARD via production pipeline (proper y FiLM) ----
    from measure_tgen import build_model
    import gate0_generator as G0
    torch.load = G0._cpu_load          # patch: get_vocoder/etc. load without map_location on a CPU box
    dev = torch.device("cpu")
    import audioldm_train.modules.latent_diffusion.ddim as _ddim   # patch: bind DDIMSampler to CPU
    _oi = _ddim.DDIMSampler.__init__
    _ddim.DDIMSampler.__init__ = lambda s, m, schedule="linear", device=None, **k: _oi(
        s, m, schedule=schedule, device=dev, **k)
    def fwd_check(name, unet):
        model, _ = build_model(config, dev); model = model.float()
        model.model.diffusion_model = unet.to(dev).eval()
        model._gate0_config = config
        model.latent_t_size = 96
        model.use_ema = False; model.eval()
        x_T = torch.randn(1, 8, 96, 16)
        w = G0.generate(model, "a short test sound", x_T, 2, 2.5, 0.0)
        import numpy as np
        w = np.asarray(w).squeeze()
        return {"n_samples": int(w.shape[-1]), "finite": bool(np.isfinite(w).all())}

    unetA = rm.build_pruned_unet(config, [1, 2, 1, 1]).float(); unetA.load_state_dict(p2A, strict=True)
    unetB = rm.build_pruned_unet(config, [1, 2, 1, 1]).float(); unetB.load_state_dict(p2B, strict=True)
    rsd = torch.load(REC_DP1, map_location="cpu"); rsd = rsd.get("state_dict", rsd)
    unetR = rm.build_pruned_unet(config, [1, 2, 1, 1]).float()
    rel = {k[len("model.diffusion_model."):]: v for k, v in rsd.items() if k.startswith("model.diffusion_model.")}
    unetR.load_state_dict(rel, strict=True); materialize_ema_into_unet(unetR, rsd, strict=True)
    report["forward"]["pruned2_A"] = fwd_check("A", unetA)
    report["forward"]["pruned2_B"] = fwd_check("B", unetB)
    report["forward"]["recovered2"] = fwd_check("R", unetR)

    fwd_ok = all(v["finite"] for v in report["forward"].values())
    verdict = ("SEAM SENSITIVITY DESIGN VALIDATED" if
               (oracles_ok and report["diff_proof"]["exactly_expected_three"] and fwd_ok) else "STOP")
    report["VERDICT"] = verdict
    json.dump(report, open("configs/research/xsev_prune_validate.json", "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("oracles", "diff_proof", "forward", "VERDICT")}, indent=2)[:2000])
    print("VERDICT:", verdict)
    return 0 if verdict.startswith("SEAM") else 2


if __name__ == "__main__":
    sys.exit(main())
