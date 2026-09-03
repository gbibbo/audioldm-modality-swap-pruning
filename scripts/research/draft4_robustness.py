#!/usr/bin/env python3
"""DRAFT4-ROBUSTNESS — multiplicity, rank-scale and caption-length checks on the frozen per-prompt
CLAP scores (CPU, 0 cr). POST-HOC SENSITIVITY (Draft-4 review pass, 2026-09-02); changes no frozen verdict.

R1  Multiplicity. For every contrast the manuscript marks as resolved (CI excludes 0) or gated, compute the
    bootstrap achieved significance level (two-sided: 2*min(frac<=0, frac>=0) over the prompt bootstrap;
    resolution 1/B) and apply Holm across (a) the whole reported family and (b) per severity. Reported so
    the paper can state plainly which conclusions survive a family-wise correction.
R2  Rank-scale duration interaction. Per-prompt j_i = r_native_i - r_short_i; report the median with a prompt
    bootstrap CI and the Wilcoxon signed-rank test (two-sided) at both severities. Complements the win-rate
    interaction of draft3_sensitivity.py; addresses "J is a raw-scale difference-in-differences".
R3  Caption-length check for the domain contrast. Held-out music captions (MusicCaps) are ~7x longer than
    AudioCaps captions, so "domain" bundles content AND caption style. Within AudioCaps we test whether the
    per-prompt recovery gain depends on caption length (Spearman rho, prompt bootstrap CI; long- vs
    short-caption quartile gain) at both severities and durations. Descriptive: the pruned checkpoint's
    absolute music score vs its in-domain score at 10.24 s (floor argument).

Seed namespace "DRAFT4-ROBUSTNESS|BOOTSTRAP|2026-09-02" -> PCG64(int(sha256(ns)[:8],16) % 2**31), B = 10000.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft4_robustness.py
"""
from __future__ import annotations
import hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy import stats

NS = "DRAFT4-ROBUSTNESS|BOOTSTRAP|2026-09-02"
SEED = int(hashlib.sha256(NS.encode()).hexdigest()[:8], 16) % (2 ** 31)
B = 10000
SEV2_OUT = "artifacts/icassp_gate0/_score_tmp/xsev_sev2_groups_out.json"
SEV2_IN = "artifacts/icassp_gate0/_score_tmp/xsev_sev2_groups_in.json"
MUSNAT_OUT = "artifacts/icassp_gate0/_score_tmp/music_native_groups_out.json"
OPD = "configs/research/op_duration_discriminator_1_result.json"
SUBSET = "configs/research/op_duration_discriminator_1_subset.json"
REV11 = "configs/research/reversal_v1_1_result.json"
MUS1 = "configs/research/reversal_v1_r_music_clap.json"
XSEV = "configs/research/xsev_result.json"
OUT = "configs/research/draft4_robustness_result.json"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def boot_mean(a, rng):
    n = len(a)
    return np.array([a[rng.integers(0, n, n)].mean() for _ in range(B)])


def boot_two_sample(a, b_, rng):
    return np.array([a[rng.integers(0, len(a), len(a))].mean() - b_[rng.integers(0, len(b_), len(b_))].mean() for _ in range(B)])


def summarise(name, point, boots, n):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fle, fge = float(np.mean(boots <= 0)), float(np.mean(boots >= 0))
    p2 = min(1.0, 2 * min(fle, fge))
    return {"contrast": name, "point": float(point), "lo": float(lo), "hi": float(hi), "n": n,
            "asl_two_sided": float(p2), "asl_floor": 1.0 / B}


def holm(tests, alpha=0.05):
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i]["asl_two_sided"])
    out = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * tests[i]["asl_two_sided"]))
        prev = adj
        out[i] = adj
    for i, t in enumerate(tests):
        t["holm_adjusted"] = float(out[i])
        t["holm_pass_005"] = bool(out[i] < alpha)
    return tests


def main():
    rng = np.random.default_rng(np.random.PCG64(SEED))
    s2 = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(SEV2_OUT))["results"]}
    s2in = {g["name"]: g["items"] for g in json.load(open(SEV2_IN))["groups"]}
    mn = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(MUSNAT_OUT))["results"]}
    opd = json.load(open(OPD)); rc = {k: np.asarray(v, float) for k, v in opd["raw_cosines"].items()}
    rev = json.load(open(REV11))["PRIMARY"]
    mus1 = json.load(open(MUS1))
    xsev = json.load(open(XSEV))["PRIMARY_A"]

    # ---------------- per-prompt contrasts
    r2_nat = s2["recovered2__ac_native"] - s2["pruned2_A__ac_native"]
    r2_short = s2["recovered2__ac_short"] - s2["pruned2_A__ac_short"]
    j2 = r2_nat - r2_short
    r2_mus = (s2["recovered2__music"] - s2["pruned2_A__music"]).reshape(64, 3).mean(1)   # order (prompt, replicate)
    r2_mus_nat = mn["recovered2__music_native"] - mn["pruned2_A__music_native"]
    r1_nat = rc["recovered_alt"] - rc["pruned_alt"]; r1_short = rc["recovered_ctrl"] - rc["pruned_ctrl"]; j1 = r1_nat - r1_short
    r1_ac96 = np.asarray(rev["prompt_contrast_vector"], float)
    r1_mus = np.asarray([p["prompt_mean_diff"] for p in sorted(mus1["prompts"], key=lambda p: p["prompt_index"])], float)
    # guards: reproduce frozen points
    for nm, mine, frozen in (("sev2 J", j2.mean(), xsev["J"]["point"]), ("sev2 R_music", r2_mus.mean(), xsev["R_music"]["point"]),
                             ("sev1 J", j1.mean(), opd["PRIMARY_clap"]["J"]["point"]), ("sev1 R_AC", r1_ac96.mean(), rev["R_AC"]["point"]),
                             ("sev1 R_music", r1_mus.mean(), mus1["R_music"]["point"])):
        if abs(mine - frozen) > 1e-9:
            raise SystemExit(f"guard failed {nm}: {mine} vs frozen {frozen}")

    # ---------------- R1 multiplicity
    fam = []
    fam.append(summarise("sev1 R_native (n=80)", r1_nat.mean(), boot_mean(r1_nat, rng), 80))
    fam.append(summarise("sev1 R_short (n=80)", r1_short.mean(), boot_mean(r1_short, rng), 80))
    fam.append(summarise("sev1 J (n=80)", j1.mean(), boot_mean(j1, rng), 80))
    fam.append(summarise("sev1 R_AC pre-specified (n=96)", r1_ac96.mean(), boot_mean(r1_ac96, rng), 96))
    fam.append(summarise("sev1 R_music (n=64)", r1_mus.mean(), boot_mean(r1_mus, rng), 64))
    fam.append(summarise("sev1 domain 3.84 s = R_AC - R_music", r1_ac96.mean() - r1_mus.mean(), boot_two_sample(r1_ac96, r1_mus, rng), "96/64"))
    fam.append(summarise("sev2 R_native (n=192)", r2_nat.mean(), boot_mean(r2_nat, rng), 192))
    fam.append(summarise("sev2 R_short (n=192)", r2_short.mean(), boot_mean(r2_short, rng), 192))
    fam.append(summarise("sev2 J (n=192)", j2.mean(), boot_mean(j2, rng), 192))
    fam.append(summarise("sev2 R_music 3.84 s (n=64)", r2_mus.mean(), boot_mean(r2_mus, rng), 64))
    fam.append(summarise("sev2 R_music 10.24 s (n=64)", r2_mus_nat.mean(), boot_mean(r2_mus_nat, rng), 64))
    fam.append(summarise("sev2 domain 3.84 s = R_short - R_music", r2_short.mean() - r2_mus.mean(), boot_two_sample(r2_short, r2_mus, rng), "192/64"))
    fam.append(summarise("sev2 domain 10.24 s = R_native - R_music_nat", r2_nat.mean() - r2_mus_nat.mean(), boot_two_sample(r2_nat, r2_mus_nat, rng), "192/64"))
    import copy
    fam_all = holm(copy.deepcopy(fam))
    fam_s1 = holm(copy.deepcopy([t for t in fam if t["contrast"].startswith("sev1")]))
    fam_s2 = holm(copy.deepcopy([t for t in fam if t["contrast"].startswith("sev2")]))

    # ---------------- R2 rank-scale J
    def robust(j, n):
        med_boots = np.array([np.median(j[rng.integers(0, n, n)]) for _ in range(B)])
        w = stats.wilcoxon(j, alternative="two-sided", zero_method="wilcox")
        return {"median_j": {"point": float(np.median(j)), "lo": float(np.percentile(med_boots, 2.5)), "hi": float(np.percentile(med_boots, 97.5)), "n": n},
                "wilcoxon_signed_rank": {"statistic": float(w.statistic), "p_two_sided": float(w.pvalue)},
                "frac_j_pos": float(np.mean(j > 0)),
                "mean_j": float(j.mean())}
    r2 = {"sev1_armd80": robust(j1, 80), "sev2_xsev192": robust(j2, 192)}
    # rank-transformed interaction: pool the four cells per severity, rank all scores, recompute J on ranks/n
    def rank_J(ps, fs, pn, fn):
        allv = np.concatenate([ps, fs, pn, fn]); rk = stats.rankdata(allv) / len(allv)
        k = len(ps); rps, rfs, rpn, rfn = rk[:k], rk[k:2 * k], rk[2 * k:3 * k], rk[3 * k:]
        j = (rfn - rpn) - (rfs - rps)
        boots = boot_mean(j, rng)
        return {"point": float(j.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "n": k,
                "note": "J recomputed on pooled-rank-transformed scores (0..1), paired per prompt"}
    r2["rank_transformed_J"] = {
        "sev1_armd80": rank_J(rc["pruned_ctrl"], rc["recovered_ctrl"], rc["pruned_alt"], rc["recovered_alt"]),
        "sev2_xsev192": rank_J(s2["pruned2_A__ac_short"], s2["recovered2__ac_short"], s2["pruned2_A__ac_native"], s2["recovered2__ac_native"]),
    }

    # ---------------- R3 caption length
    def wc(items):
        return np.asarray([len(it["caption"].split()) for it in items], float)
    len2 = wc(s2in["recovered2__ac_native"])
    sub = sorted(json.load(open(SUBSET))["prompts"], key=lambda p: p["subset_prompt_index"])
    len1 = np.asarray([len(p["caption"].split()) for p in sub], float)
    mus_len = wc(s2in["recovered2__music"][::3])

    def spearman_ci(x, y):
        n = len(x); rho = stats.spearmanr(x, y).correlation
        boots = np.array([stats.spearmanr(x[i], y[i]).correlation for i in (rng.integers(0, n, n) for _ in range(2000))])
        return {"rho": float(rho), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "n": n, "boot_B": 2000}

    def quartiles(x, y):
        q1, q3 = np.percentile(x, [25, 75])
        lo_m, hi_m = y[x <= q1], y[x >= q3]
        return {"short_caption_quartile_gain": float(lo_m.mean()), "n_short": int(len(lo_m)),
                "long_caption_quartile_gain": float(hi_m.mean()), "n_long": int(len(hi_m)),
                "caption_words_q1_q3": [float(q1), float(q3)]}
    r3 = {
        "caption_words": {"audiocaps_sev2_median": float(np.median(len2)), "audiocaps_sev1_median": float(np.median(len1)),
                          "music_sev2_median": float(np.median(mus_len)), "music_sev2_min_max": [float(mus_len.min()), float(mus_len.max())]},
        "sev2_native_gain_vs_caption_length": {**spearman_ci(len2, r2_nat), **quartiles(len2, r2_nat)},
        "sev2_short_gain_vs_caption_length": {**spearman_ci(len2, r2_short), **quartiles(len2, r2_short)},
        "sev1_native_gain_vs_caption_length": {**spearman_ci(len1, r1_nat), **quartiles(len1, r1_nat)},
        "pruned_absolute_music_vs_indomain_native_sev2": {
            "pruned2_A_music_10p24s_mean": float(mn["pruned2_A__music_native"].mean()),
            "pruned2_A_audiocaps_10p24s_mean": float(s2["pruned2_A__ac_native"].mean()),
            "postft_music_10p24s_mean": float(mn["recovered2__music_native"].mean()),
            "postft_audiocaps_10p24s_mean": float(s2["recovered2__ac_native"].mean()),
            "note": "descriptive; different prompt sets; the pruned checkpoint is not lower on music than in-domain at native duration"},
    }

    res = {"artifact": "draft4_robustness_result",
           "class": "POST-HOC SENSITIVITY (Draft-4 review pass, 2026-09-02); changes no frozen verdict",
           "bootstrap": {"B": B, "seed_namespace": NS, "seed_pcg64": SEED, "unit": "prompt", "ci": "percentile 95%"},
           "inputs": {p: sha(p) for p in (SEV2_OUT, SEV2_IN, MUSNAT_OUT, OPD, SUBSET, REV11, MUS1, XSEV)},
           "R1_multiplicity": {"family_all": fam_all, "family_sev1": fam_s1, "family_sev2": fam_s2,
                               "method": "Holm step-down on two-sided bootstrap ASLs (resolution 1/B)"},
           "R2_rank_scale_J": r2, "R3_caption_length": r3}
    json.dump(res, open(OUT, "w"), indent=1)
    for t in fam_all:
        print(f"{t['contrast']:<45s} {t['point']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}] asl={t['asl_two_sided']:.4f} holm={t['holm_adjusted']:.4f} pass={t['holm_pass_005']}")
    print("per-severity Holm:", [(t["contrast"], t["holm_pass_005"]) for t in fam_s1 + fam_s2])
    print(json.dumps(r2, indent=1)); print(json.dumps(r3, indent=1)); print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
