#!/usr/bin/env python3
"""Fill the Draft-5 `@@` placeholders in icassp/icassp_operating_point.tex from the committed
configs/research/draft5_floor_ceiling_result.json (CPU, 0 cr). Idempotent: placeholders are replaced
in place; any placeholder without a mapping is reported and left untouched (exit 1). The two prose
placeholders @@MUSIC_AFTER_FT_PHRASE / @@MUSIC_FLOOR_SENTENCES are written by hand (they depend on the
sign pattern) and are checked by verify_draft5_numbers.py through the numbers they contain.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/fill_draft5.py
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
FC = json.load(open(os.path.join(ROOT, "configs/research/draft5_floor_ceiling_result.json")))
cells, s1, s2 = FC["cells"], FC["sev1_armd80"], FC["sev2_xsev192"]


def pct(c):
    return f"${100*c['point']:.0f}\\%$"


def pct_ci(c):
    return f"${100*c['point']:.0f}\\%$~\\ci{{{100*c['lo']:+.0f}\\%}}{{{100*c['hi']:+.0f}\\%}}"


def pt(c, nd=3):
    return f"${c['point']:+.{nd}f}$"


def pt_ci(c, nd=3):
    return f"${c['point']:+.{nd}f}$~\\ci{{{c['lo']:+.{nd}f}}}{{{c['hi']:+.{nd}f}}}"


def lvl(name):
    return f"${cells[name]['matched_mean']:.3f}$"


def floor_pair(p, q):
    return f"${cells[p]['floor']['point']:+.3f}\\,/\\,{cells[q]['floor']['point']:+.3f}$"


shifts = [abs(s1[f"floor_shift_{k}"]["point"]) for k in ("dense", "pruned", "postft")] + \
         [abs(s2[f"floor_shift_{k}"]["point"]) for k in ("pruned", "postft")]
floors_ac = [cells[n]["floor"]["point"] for n in ("pruned2_A__ac_short", "pruned2_A__ac_native", "recovered2__ac_short",
                                                    "recovered2__ac_native", "pruned_short_sev1__armd80", "dense10s__pruned_sev1",
                                                    "postft_short_sev1__armd80", "dense10s__recovered_sev1",
                                                    "dense_short_sev1__armd80", "dense10s__dense")]
tok = FC["caption_tokens_vs_conditioner_limit"]["music64"]["frac_over_77"]

MAP = {
    "@@RHO_REAL2_NATIVE_CI": pct_ci(s2["rho_real_native"]),
    "@@RHO_REAL2_SHORT_CI": pct_ci(s2["rho_real_short"]),
    "@@RHO_REAL1_NATIVE_CI": pct_ci(s1["rho_real_native"]),
    "@@RHO_REAL1_SHORT_CI": pct_ci(s1["rho_real_short"]),
    "@@RHO_REAL2_NATIVE": pct(s2["rho_real_native"]),
    "@@RHO_REAL2_SHORT": pct(s2["rho_real_short"]),
    "@@RHO_REAL1_NATIVE": pct(s1["rho_real_native"]),
    "@@RHO_REAL1_SHORT": pct(s1["rho_real_short"]),
    "@@FLOOR_SHIFT_MAX": f"${max(shifts):.3f}$",
    "@@FLOOR_RANGE": f"from ${min(floors_ac):+.3f}$ to ${max(floors_ac):+.3f}$ across the AudioCaps cells",
    "@@JC2": pt_ci(s2["J_c"]),
    "@@REAL2_SHORT": lvl("real_crop__sev2_192"),
    "@@REAL2_NATIVE": lvl("real_full__sev2_192"),
    "@@REAL1_SHORT": lvl("real_crop__sev1_80"),
    "@@REAL1_NATIVE": lvl("real_full__sev1_80"),
    "@@S_REAL2": pt_ci(s2["s_raw_real"]),
    "@@TOK_MUS_FRAC": f"${100*tok:.0f}\\%$",
    "@@FLOOR1_SHORT": floor_pair("pruned_short_sev1__armd80", "postft_short_sev1__armd80"),
    "@@FLOOR1_NATIVE": floor_pair("dense10s__pruned_sev1", "dense10s__recovered_sev1"),
    "@@FLOOR2_SHORT": floor_pair("pruned2_A__ac_short", "recovered2__ac_short"),
    "@@FLOOR2_NATIVE": floor_pair("pruned2_A__ac_native", "recovered2__ac_native"),
    "@@FLOORM1_SHORT": floor_pair("p1_pruned_ema_reconstructed__off", "p1_recovered__off"),
    "@@FLOORM2_SHORT": floor_pair("pruned2_A__music", "recovered2__music"),
    "@@FLOORM2_NATIVE": floor_pair("pruned2_A__music_native", "recovered2__music_native"),
}

tex = open(TEX, encoding="utf-8").read()
# longest keys first so @@X_CI is replaced before @@X
for k in sorted(MAP, key=len, reverse=True):
    tex = tex.replace(k, MAP[k])
left = sorted(set(re.findall(r"@@[A-Z0-9_]+", tex)))
open(TEX, "w", encoding="utf-8").write(tex)
print("filled", len(MAP), "placeholders")
for k in sorted(MAP, key=len, reverse=True):
    print(f"  {k:26s} -> {MAP[k]}")
if left:
    print("LEFT (hand-written prose placeholders):", left)
    sys.exit(1 if any(k not in ("@@MUSIC_AFTER_FT_PHRASE", "@@MUSIC_FLOOR_SENTENCES") for k in left) else 0)
