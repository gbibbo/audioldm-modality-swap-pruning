#!/usr/bin/env python3
"""OPERATING-POINT-DISCRIMINATOR — DESIGN AUDIT (CPU, 0 cr; NO GPU, NO audio, NO prereg).

Quantitative inputs for choosing the smallest informative recovered-vs-pruned operating-point
experiment. Uses ONLY already-observed data:
  * settled V1.1 generation cost (1.262 cr / 576 WAV @ 3.84 s/DDIM50/single) -> cost model,
  * V1.1 CLAP grid (96x2 per system) -> between/within variance of the recovered-pruned contrast,
  * recomputed per-ytid PANN KL + top-10 capture grids (imported audit machinery) -> same for KL/PANN.
Projects the precision of the operating-point x recovery INTERACTION J = D_alt - D_ctrl under
candidate (n_prompts, n_replicates), exploiting ytid-level pairing (same ytids at both OPs).
Writes a design artifact (NOT a scientific result; changes nothing frozen).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/op_discriminator_design_audit.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np

V1 = "configs/research/reversal_v1_1_result.json"
OUT = "configs/research/op_discriminator_design_audit.json"
SESOI = 0.025

# ---- measured baseline (V1.1, settled) ----
BASE_CR = 1.2623          # settled cr for the whole job
BASE_NWAV = 576           # 3 systems x 96 x 2
BASE_DDIM = 50
BASE_LATENT_T = 96        # 3.84 s
BASE_PERWAV = BASE_CR / BASE_NWAV   # 0.002191 cr/wav (gross, incl. 3 model loads)
# conservative fixed per-JOB overhead (container spin-up + model loads + CLAP/VAE init);
# prior gate/phenom smokes settled ~0.10 cr for tiny jobs -> use 0.09 as the fixed floor.
FIXED_JOB_CR = 0.09
# latent_t for the verified Singh duration 10.24 s (1024 mel / 4)
LATENT_T_10s = 256


def perwav_cr(ddim, latent_t, best_of):
    """Marginal per-wav cr scales ~ ddim_steps x latent_t x best_of (U-Net forwards dominate)."""
    return BASE_PERWAV * (ddim / BASE_DDIM) * (latent_t / BASE_LATENT_T) * best_of


def arm_cost(n_prompts, n_rep, ddim, latent_t, best_of, n_systems=2):
    nwav = n_prompts * n_rep * best_of * n_systems
    # count generated candidates for cost (best_of multiplies compute even if 1 kept)
    marginal = n_prompts * n_rep * n_systems * perwav_cr(ddim, latent_t, best_of)
    return FIXED_JOB_CR + marginal, nwav


# ---- variance decomposition helpers ----
def decompose(dgrid):
    """dgrid: [n, r] paired per-(ytid,rep) contrast. Returns between/within components of d."""
    n, r = dgrid.shape
    ybar = dgrid.mean(axis=1)                      # per-ytid mean contrast (D_i)
    grand = dgrid.mean()
    ss_within = ((dgrid - ybar[:, None]) ** 2).sum()
    ss_between = r * ((ybar - grand) ** 2).sum()
    ms_within = ss_within / (n * (r - 1)) if r > 1 else 0.0
    ms_between = ss_between / (n - 1)
    # random-effects: MS_between = sig_w^2 + r*sig_b^2
    sig_w2 = ms_within
    sig_b2 = max((ms_between - sig_w2) / r, 0.0) if r > 1 else np.var(ybar, ddof=1)
    return dict(mean=float(grand), sig_b2=float(sig_b2), sig_w2=float(sig_w2),
                sd_between=float(sig_b2 ** 0.5), sd_within=float(sig_w2 ** 0.5),
                var_Di=float(np.var(ybar, ddof=1)))


def se_mean_contrast(dec, n, r):
    """SE of the mean paired contrast for n ytids x r replicates (cluster formula)."""
    return (dec["sig_b2"] / n + dec["sig_w2"] / (n * r)) ** 0.5


def se_interaction(dec, n, r, rho):
    """SE of J_hat = mean(D_alt) - mean(D_ctrl), same n ytids, ytid-paired.
    Control is already measured at 96x2; treat both arms with this variance structure.
    Between-ytid part of J has var 2*sig_b2*(1-rho); within parts add per arm.
    Control within uses its OWN (n_ctrl=96, r_ctrl=2)."""
    n_ctrl, r_ctrl = 96, 2
    var_between_J = 2 * dec["sig_b2"] * (1 - rho) / n     # shared-prompt pairing gain
    var_within = dec["sig_w2"] / (n * r) + dec["sig_w2"] / (n_ctrl * r_ctrl)
    # if the alt arm uses a subset, the shared ytids for pairing = n (<=96)
    return (var_between_J + var_within) ** 0.5


def main():
    d = json.load(open(V1))
    rc = d["raw_clap_scores"]
    pcv = np.array(d["PRIMARY"]["prompt_contrast_vector"], dtype=float)  # per-ytid rec-pru (control)

    # infer ordering of the 192 raw scores by matching prompt_contrast_vector
    rec = np.array(rc["p1_recovered"]); pru = np.array(rc["p1_pruned_ema_reconstructed"])
    cand = {}
    pm = (rec.reshape(96, 2) - pru.reshape(96, 2)).mean(1)      # prompt-major
    rm = (rec.reshape(2, 96) - pru.reshape(2, 96)).mean(0)      # rep-major
    order = "prompt_major" if np.allclose(pm, pcv, atol=1e-6) else (
            "rep_major" if np.allclose(rm, pcv, atol=1e-6) else "UNKNOWN")
    if order == "prompt_major":
        d_clap = rec.reshape(96, 2) - pru.reshape(96, 2)
    elif order == "rep_major":
        d_clap = (rec.reshape(2, 96) - pru.reshape(2, 96)).T
    else:
        raise SystemExit("could not match raw-score ordering to prompt_contrast_vector")

    dec_clap = decompose(d_clap)

    # ---- recompute per-ytid KL + capture grids (paired rec-pru), oriented so + = recovered better ----
    import recovery_metric_audit_1 as A
    import torch
    from audioldm_eval import EvaluationHelper
    man = json.load(open(A.MANIFEST))["prompts"]
    ytids = [p["ytid"] for p in man]; n = len(ytids)
    gt = A.load_gt_indices(); gt_idx = [gt[y] for y in ytids]
    scratch = os.path.join(os.environ.get("SCRATCH", "/tmp/claude-1000"), "op_design")
    os.makedirs(scratch, exist_ok=True)
    ref_files = [(A.find_ref(y), f"Y{y}.wav") for y in ytids]
    ref_dir = A.symlink_dir(scratch, "refs", ref_files)
    gen_all = []
    for s in ["pruned", "recovered"]:
        pref = A.SYS_PREFIX[s]
        for pi in range(n):
            for r in (0, 1):
                bn = f"{pref}_p{pi}_r{r}.wav"
                gen_all.append((os.path.join(A.GEN_DIR, bn), bn))
    gen_dir = A.symlink_dir(scratch, "gen_pr", gen_all)
    helper = EvaluationHelper(A.SR, torch.device("cpu"))
    fg = A.get_features(helper, gen_dir); fr = A.get_features(helper, ref_dir)
    ref_logits = {y: fr[f"Y{y}.wav"]["logits"] for y in ytids}

    kl = {s: np.zeros((n, 2)) for s in ["pruned", "recovered"]}
    cap = {s: np.zeros((n, 2)) for s in ["pruned", "recovered"]}
    for s in ["pruned", "recovered"]:
        pref = A.SYS_PREFIX[s]
        for pi in range(n):
            for r in (0, 1):
                lg = fg[f"{pref}_p{pi}_r{r}.wav"]["logits"]
                kl[s][pi, r] = A.kl_pair(lg, ref_logits[ytids[pi]])
                cap[s][pi, r] = len(set(np.argsort(lg)[::-1][:10].tolist()) & set(gt_idx[pi]))
    # orient + = recovered better: KL lower is better -> R_KL = KL_pruned - KL_recovered
    d_kl = kl["pruned"] - kl["recovered"]
    d_cap = cap["recovered"] - cap["pruned"]
    dec_kl = decompose(d_kl); dec_cap = decompose(d_cap)

    # ---- precision / MDE projection for the interaction J ----
    designs = [(96, 1), (64, 1), (48, 1), (32, 1), (48, 2), (32, 2), (24, 2)]
    rhos = [0.0, 0.5, 0.7]
    endpoints = {"CLAP": dec_clap, "KL(pru-rec)": dec_kl, "PANN_capture": dec_cap}
    proj = {}
    for name, dec in endpoints.items():
        proj[name] = {"control_grid_mean_contrast": dec["mean"],
                      "sd_between": dec["sd_between"], "sd_within": dec["sd_within"],
                      "designs": {}}
        for (npr, r) in designs:
            row = {}
            for rho in rhos:
                se = se_interaction(dec, npr, r, rho)
                row[f"rho={rho}"] = {"SE_J": round(se, 4), "MDE_J_80pct": round(2.8 * se, 4)}
            proj[name]["designs"][f"{npr}x{r}"] = row

    # ---- cost table ----
    arms = {
        "S_steps200_3.84s_single":   dict(ddim=200, latent_t=96,  best_of=1),
        "D_dur10.24s_ddim50_single": dict(ddim=50,  latent_t=256, best_of=1),
        "F_full_singh_bestof3":      dict(ddim=200, latent_t=256, best_of=3),
        "F_full_singh_single(no b3)":dict(ddim=200, latent_t=256, best_of=1),
    }
    ncfg = [(96, 1), (64, 1), (48, 1), (32, 1), (48, 2), (32, 2), (24, 2)]
    cost = {}
    for aname, a in arms.items():
        cost[aname] = {}
        for (npr, r) in ncfg:
            c, nw = arm_cost(npr, r, a["ddim"], a["latent_t"], a["best_of"], n_systems=2)
            cost[aname][f"{npr}x{r}"] = {"cr": round(c, 3), "n_wav_kept": npr * r * 2,
                                         "n_gen_candidates": nw}

    out = {
        "artifact": "op_discriminator_design_audit",
        "status": "DESIGN-ONLY, CPU, 0 cr, 0 GPU — not a scientific result; nothing frozen changed",
        "raw_order_inferred": order,
        "cost_model": {"measured_base_cr": BASE_CR, "base_nwav": BASE_NWAV,
                       "base_perwav_cr": round(BASE_PERWAV, 6), "fixed_job_cr": FIXED_JOB_CR,
                       "latent_t_10.24s": LATENT_T_10s,
                       "note": "marginal ~ ddim x latent_t x best_of x n_wav; fixed = conservative job overhead"},
        "variance_decomposition": {k: v for k, v in
                                   [("CLAP", dec_clap), ("KL(pru-rec)", dec_kl), ("PANN_capture", dec_cap)]},
        "interaction_precision": proj,
        "cost_table": cost,
        "SESOI_reference": SESOI,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    # console summary
    print("raw order:", order)
    print("\nVARIANCE (per-ytid recovered-advantage contrast, + = recovered better):")
    for k, dec in [("CLAP", dec_clap), ("KL(pru-rec)", dec_kl), ("PANN_cap", dec_cap)]:
        print(f"  {k:10s} mean={dec['mean']:+.4f}  sd_between={dec['sd_between']:.4f}  sd_within={dec['sd_within']:.4f}")
    print("\nINTERACTION MDE (80% power ~ 2.8*SE_J), CLAP endpoint:")
    for (npr, r) in designs:
        row = proj["CLAP"]["designs"][f"{npr}x{r}"]
        print(f"  {npr}x{r}: MDE rho0={row['rho=0.0']['MDE_J_80pct']}  rho.5={row['rho=0.5']['MDE_J_80pct']}  rho.7={row['rho=0.7']['MDE_J_80pct']}")
    print("\nINTERACTION MDE, KL endpoint (pru-rec, nats):")
    for (npr, r) in designs:
        row = proj["KL(pru-rec)"]["designs"][f"{npr}x{r}"]
        print(f"  {npr}x{r}: MDE rho0={row['rho=0.0']['MDE_J_80pct']}  rho.5={row['rho=0.5']['MDE_J_80pct']}  rho.7={row['rho=0.7']['MDE_J_80pct']}")
    print("\nCOST (cr) by arm x design (pruned+recovered only, 2 systems):")
    for aname in arms:
        cells = "  ".join(f"{k}:{cost[aname][k]['cr']}" for k in ["96x1","64x1","48x1","32x1","48x2","32x2","24x2"])
        print(f"  {aname:30s} {cells}")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
