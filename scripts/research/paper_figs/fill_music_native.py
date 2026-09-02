#!/usr/bin/env python3
"""Fill the [[MN-*]] placeholders of icassp/icassp_operating_point.tex from the frozen
XSEV-MUSIC-NATIVE-1 result (configs/research/xsev_music_native_1_result.json). CPU, 0 cr.

Branch-dependent wording follows the frozen protocol docs/xsev_music_native_1.md §7. The script only
touches the placeholders; if the branch is not (a), the abstract/intro/conclusion sentences that say
"absent on held-out music" must be revised by hand (the script prints a reminder).
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TEX = os.path.join(ROOT, "icassp", "icassp_operating_point.tex")
RES = os.path.join(ROOT, "configs", "research", "xsev_music_native_1_result.json")

r = json.load(open(RES))
R = r["PRIMARY_R_music_native"]; D = r["secondary_D_native_domain_contrast_AC_minus_music"]
m = r["means"]; W = r["win_rate_music_native"]; branch = r["branch"]


def ci(c, nd=3):
    return f"\\ci{{{c['lo']:+.{nd}f}}}{{{c['hi']:+.{nd}f}}}"


dag = "^{\\dagger}" if (R["lo"] > 0 or R["hi"] < 0) else ""
dagD = "^{\\dagger}" if (D["lo"] > 0 or D["hi"] < 0) else ""
sub = {
    "[[MN-P]]": f"${m['pruned_music_native']:.3f}$",
    "[[MN-PFT]]": f"${m['rec_music_native']:.3f}$",
    "[[MN-R]]": f"${R['point']:+.3f}{dag}$~{ci(R)}",
    "[[MN-W]]": f"${W:.2f}$",
    "[[MN-D]]": f"${D['point']:+.3f}{dagD}$~{ci(D)}",
    "[[MN-CELLS]]": ", plus $10.24$\\,s at severity~2",
}
if branch.startswith("a"):
    sub["[[MN-INTRO]]"] = "and at both durations"
    sub["[[MN-RESULT]]"] = (f"At the native duration the music contrast is likewise null (${R['point']:+.3f}$ {ci(R)}; "
                            f"$W={W:.2f}$), so the large native gain is confined to the fine-tuning domain: the "
                            f"$10.24$\\,s domain contrast is ${D['point']:+.3f}$ {ci(D)}.")
elif branch.startswith("b"):
    sub["[[MN-INTRO]]"] = "at $3.84$\\,s, while a smaller music gain appears at $10.24$\\,s"
    sub["[[MN-RESULT]]"] = (f"At the native duration a music gain does appear (${R['point']:+.3f}$ {ci(R)}; "
                            f"$W={W:.2f}$), smaller than in-domain: the $10.24$\\,s domain contrast is "
                            f"${D['point']:+.3f}$ {ci(D)}, so part of the duration gain is domain-generic.")
elif branch.startswith("c"):
    sub["[[MN-INTRO]]"] = "and negative at $10.24$\\,s"
    sub["[[MN-RESULT]]"] = (f"At the native duration the music contrast is negative (${R['point']:+.3f}$ {ci(R)}; "
                            f"$W={W:.2f}$): the $10.24$\\,s domain contrast is ${D['point']:+.3f}$ {ci(D)}.")
else:
    sub["[[MN-INTRO]]"] = "at $3.84$\\,s and unresolved at $10.24$\\,s"
    sub["[[MN-RESULT]]"] = (f"At the native duration the music contrast is unresolved at $n{{=}}64$ (${R['point']:+.3f}$ "
                            f"{ci(R)}; $W={W:.2f}$); the $10.24$\\,s domain contrast is ${D['point']:+.3f}$ {ci(D)}.")

s = open(TEX, encoding="utf-8").read()
for k, v in sub.items():
    if k not in s:
        raise SystemExit(f"placeholder {k} not found")
    s = s.replace(k, v)
left = sorted(set(re.findall(r"\[\[[A-Z-]+\]\]", s)))
if left:
    raise SystemExit(f"unfilled placeholders remain: {left}")
open(TEX, "w", encoding="utf-8").write(s)
print("branch", branch)
for k, v in sub.items():
    print(f"  {k} -> {v}")
if not branch.startswith("a"):
    print("REMINDER: branch != (a): revise abstract / intro bullet / conclusion ('absent on held-out music').")
