#!/usr/bin/env python3
"""DRAFT5 — Holm family extended with the sweep and published-recipe contrasts (CPU, 0 cr, post-hoc).

Reproduces the 13-contrast family of draft4_robustness.py EXACTLY (same per-prompt vectors, same seed
namespace, same draw order -> guard: the 13 achieved significance levels equal the committed artifact),
then appends the DRAFT5-OPSWEEP-1 / DRAFT5-PUBRECIPE-1 contrasts and re-applies Holm over the whole
family and per severity. Changes no verdict; answers "do the new contrasts survive the family-wise
correction, and does adding them cost any existing conclusion?".

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/draft5_holm_extension.py
"""
from __future__ import annotations
import copy, hashlib, json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/research")
import numpy as np
import draft4_robustness as D4

SWEEP_OUT = "artifacts/icassp_gate0/_score_tmp/draft5_sweep_groups_out.json"
PUB_OUT = "artifacts/icassp_gate0/_score_tmp/draft5_pubrecipe_groups_out.json"
SWEEP_RES = "configs/research/draft5_opsweep_result.json"
PUB_RES = "configs/research/draft5_pubrecipe_result.json"
OUT = "configs/research/draft5_holm_extension.json"


def main():
    rng = np.random.default_rng(np.random.PCG64(D4.SEED))
    s2 = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(D4.SEV2_OUT))["results"]}
    mn = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(D4.MUSNAT_OUT))["results"]}
    opd = json.load(open(D4.OPD)); rc = {k: np.asarray(v, float) for k, v in opd["raw_cosines"].items()}
    rev = json.load(open(D4.REV11))["PRIMARY"]; mus1 = json.load(open(D4.MUS1))
    r2_nat = s2["recovered2__ac_native"] - s2["pruned2_A__ac_native"]
    r2_short = s2["recovered2__ac_short"] - s2["pruned2_A__ac_short"]
    j2 = r2_nat - r2_short
    r2_mus = (s2["recovered2__music"] - s2["pruned2_A__music"]).reshape(64, 3).mean(1)
    r2_mus_nat = mn["recovered2__music_native"] - mn["pruned2_A__music_native"]
    r1_nat = rc["recovered_alt"] - rc["pruned_alt"]; r1_short = rc["recovered_ctrl"] - rc["pruned_ctrl"]; j1 = r1_nat - r1_short
    r1_ac96 = np.asarray(rev["prompt_contrast_vector"], float)
    r1_mus = np.asarray([p["prompt_mean_diff"] for p in sorted(mus1["prompts"], key=lambda p: p["prompt_index"])], float)
    S = D4.summarise; bm = D4.boot_mean; b2 = D4.boot_two_sample
    fam = [S("sev1 R_native (n=80)", r1_nat.mean(), bm(r1_nat, rng), 80),
           S("sev1 R_short (n=80)", r1_short.mean(), bm(r1_short, rng), 80),
           S("sev1 J (n=80)", j1.mean(), bm(j1, rng), 80),
           S("sev1 R_AC pre-specified (n=96)", r1_ac96.mean(), bm(r1_ac96, rng), 96),
           S("sev1 R_music (n=64)", r1_mus.mean(), bm(r1_mus, rng), 64),
           S("sev1 domain 3.84 s = R_AC - R_music", r1_ac96.mean() - r1_mus.mean(), b2(r1_ac96, r1_mus, rng), "96/64"),
           S("sev2 R_native (n=192)", r2_nat.mean(), bm(r2_nat, rng), 192),
           S("sev2 R_short (n=192)", r2_short.mean(), bm(r2_short, rng), 192),
           S("sev2 J (n=192)", j2.mean(), bm(j2, rng), 192),
           S("sev2 R_music 3.84 s (n=64)", r2_mus.mean(), bm(r2_mus, rng), 64),
           S("sev2 R_music 10.24 s (n=64)", r2_mus_nat.mean(), bm(r2_mus_nat, rng), 64),
           S("sev2 domain 3.84 s = R_short - R_music", r2_short.mean() - r2_mus.mean(), b2(r2_short, r2_mus, rng), "192/64"),
           S("sev2 domain 10.24 s = R_native - R_music_nat", r2_nat.mean() - r2_mus_nat.mean(), b2(r2_nat, r2_mus_nat, rng), "192/64")]
    # guard: identical to the committed 13-contrast family
    com = json.load(open(D4.OUT))["R1_multiplicity"]["family_all"]
    worst = max(abs(a["asl_two_sided"] - b["asl_two_sided"]) + abs(a["point"] - b["point"]) for a, b in zip(fam, com))
    assert [a["contrast"] for a in fam] == [b["contrast"] for b in com] and worst < 1e-12, worst
    # new contrasts (draws AFTER the 13, so the 13 are unchanged)
    sw = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(SWEEP_OUT))["results"]}
    pb = {r["name"]: np.asarray(r["cosines"], float) for r in json.load(open(PUB_OUT))["results"]}
    R = {3.84: r2_short, 5.12: sw["recovered2__ac_d128"] - sw["pruned2_A__ac_d128"],
         7.68: sw["recovered2__ac_d192"] - sw["pruned2_A__ac_d192"], 10.24: r2_nat}
    jpub = ((pb["recovered2__ac_native_pub"] - pb["pruned2_A__ac_native_pub"])
            - (pb["recovered2__ac_short_pub"] - pb["pruned2_A__ac_short_pub"]))
    new = [S("sev2 R@5.12 s sweep (n=192)", R[5.12].mean(), bm(R[5.12], rng), 192),
           S("sev2 R@7.68 s sweep (n=192)", R[7.68].mean(), bm(R[7.68], rng), 192),
           S("sev2 sweep D1 = R(5.12)-R(3.84)", (R[5.12] - R[3.84]).mean(), bm(R[5.12] - R[3.84], rng), 192),
           S("sev2 sweep D2 = R(7.68)-R(5.12)", (R[7.68] - R[5.12]).mean(), bm(R[7.68] - R[5.12], rng), 192),
           S("sev2 sweep D3 = R(10.24)-R(7.68)", (R[10.24] - R[7.68]).mean(), bm(R[10.24] - R[7.68], rng), 192),
           S("sev2 J_pub published recipe (n=64)", jpub.mean(), bm(jpub, rng), 64)]
    # guard vs the committed sweep / pub points
    swr = json.load(open(SWEEP_RES)); pbr = json.load(open(PUB_RES))
    for t, ref in zip(new, (swr["R_by_duration"]["5.12"], swr["R_by_duration"]["7.68"], swr["steps"]["D1"],
                            swr["steps"]["D2"], swr["steps"]["D3"], pbr["J_pub"])):
        assert abs(t["point"] - ref["point"]) < 1e-12, t["contrast"]
    fam19 = fam + new
    fam_all = D4.holm(copy.deepcopy(fam19))
    fam_s1 = D4.holm(copy.deepcopy([t for t in fam19 if t["contrast"].startswith("sev1")]))
    fam_s2 = D4.holm(copy.deepcopy([t for t in fam19 if t["contrast"].startswith("sev2")]))
    old = {t["contrast"]: t["holm_pass_005"] for t in com}
    changed = [t["contrast"] for t in fam_all if t["contrast"] in old and old[t["contrast"]] != t["holm_pass_005"]]
    out = {"artifact": "draft5_holm_extension", "class": "POST-HOC multiplicity extension; changes no verdict",
           "bootstrap": {"B": D4.B, "seed_namespace": D4.NS, "seed_pcg64": D4.SEED,
                         "note": "same draw order as draft4_robustness.py; the 13 original ASLs reproduce exactly"},
           "guard_13_reproduced": {"max_absdiff": worst, "PASS": True},
           "family_size": len(fam19), "family_all": fam_all, "family_sev1": fam_s1, "family_sev2": fam_s2,
           "existing_conclusions_changed_by_extension": changed,
           "new_contrasts_survive_holm_all": {t["contrast"]: t["holm_pass_005"] for t in fam_all if t in fam_all[13:]},
           "method": "Holm step-down at alpha 0.05 on two-sided bootstrap ASLs (resolution 1/B)"}
    txt = json.dumps(out, indent=1, sort_keys=True); out["artifact_sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    json.dump(out, open(OUT, "w"), indent=1)
    for t in fam_all:
        print(f"{t['contrast']:<45s} {t['point']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}] asl={t['asl_two_sided']:.4f} holm={t['holm_adjusted']:.4f} pass={t['holm_pass_005']}")
    print("changed existing conclusions:", changed); print("wrote", OUT)


if __name__ == "__main__":
    main()
