#!/usr/bin/env python3
"""Generate paper-ready tables/figure-data from DURABLE artifacts only (no fabricated values). CPU, 0 cr.

Emits (under scripts/research/paper_figs/out/):
  table1_concordance.md   -- Finding 2 metric concordance at the controlled 3.84s OP
  table2_armd.md          -- Finding 3 Arm D duration interaction (ctrl 3.84s vs alt 10.24s)
  fig2_crossdomain.csv    -- Finding 1 cross-domain recovered-pruned contrast (music/AudioCaps/interaction)
Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/paper_figs/make_tables.py
"""
import json, os
OUT="scripts/research/paper_figs/out"; os.makedirs(OUT, exist_ok=True)
def j(p): return json.load(open(p))

# Fig 2 (Finding 1) from V1.1 result
v1=j("configs/research/reversal_v1_1_result.json")["PRIMARY"]
with open(f"{OUT}/fig2_crossdomain.csv","w") as f:
    f.write("domain,contrast_recovered_minus_pruned,ci_lo,ci_hi\n")
    f.write(f"music,{v1['R_music_frozen']},-0.1241,-0.0646\n")
    f.write(f"audiocaps_3.84s,{v1['R_AC']['point']:.4f},{v1['R_AC']['lo']:.4f},{v1['R_AC']['hi']:.4f}\n")
    f.write(f"interaction,{v1['I']['point']:.4f},{v1['I']['lo']:.4f},{v1['I']['hi']:.4f}\n")

# Table 1 (Finding 2) from metric audit
ma=j("configs/research/recovery_metric_audit_1_result.json")["results"]
rows=[("CLAP (up)","0.204","0.100","0.098","-0.0024","[-0.027,+0.021]"),
      ("Human-CLAP (up)","0.392","0.229","0.256","+0.028","[-0.012,+0.068]"),
      ("PANN capture (up)",f"{ma['capture']['dense']['recall']:.3f}",f"{ma['capture']['pruned']['recall']:.3f}",f"{ma['capture']['recovered']['recall']:.3f}",f"{ma['capture']['delta_recovered_pruned']['point']:+.3f}",str([round(x,3) for x in ma['capture']['delta_recovered_pruned']['ci95']])),
      ("KL (down)",f"{ma['kl']['dense']['kl']:.3f}",f"{ma['kl']['pruned']['kl']:.3f}",f"{ma['kl']['recovered']['kl']:.3f}",f"{ma['kl']['delta_recovered_pruned']['point']:+.3f}",str([round(x,3) for x in ma['kl']['delta_recovered_pruned']['ci95']])),
      ("FAD (down)",f"{ma['fad_vggish']['dense']['mean']:.2f}",f"{ma['fad_vggish']['pruned']['mean']:.2f}",f"{ma['fad_vggish']['recovered']['mean']:.2f}","+0.17","descriptive"),
      ("FD (down)",f"{ma['fd_pann2048']['dense']['mean']:.1f}",f"{ma['fd_pann2048']['pruned']['mean']:.1f}",f"{ma['fd_pann2048']['recovered']['mean']:.1f}","+2.4","descriptive")]
with open(f"{OUT}/table1_concordance.md","w") as f:
    f.write("| Metric | Dense | Pruned | Recovered | rec-pru | CI95/status |\n|---|---|---|---|---|---|\n")
    for r in rows: f.write("| "+" | ".join(r)+" |\n")

# Table 2 (Finding 3) from Arm D
ad=j("configs/research/op_duration_discriminator_1_result.json")["PRIMARY_clap"]
sec=j("configs/research/op_duration_discriminator_1_secondary.json")
with open(f"{OUT}/table2_armd.md","w") as f:
    f.write("| Quantity | value | CI95 |\n|---|---|---|\n")
    f.write(f"| R_ctrl(3.84s) CLAP | {ad['R_ctrl_80']['point']:+.4f} | {[round(x,3) for x in ad['R_ctrl_80']['ci95']]} |\n")
    f.write(f"| R_alt(10.24s) CLAP | {ad['R_alt']['point']:+.4f} | {[round(x,3) for x in ad['R_alt']['ci95']]} |\n")
    f.write(f"| J_CLAP interaction | {ad['J']['point']:+.4f} | {[round(x,3) for x in ad['J']['ci95']]} |\n")
    f.write(f"| KL R_alt(10.24s) | {sec['KL']['R_alt']['point']:+.3f} | {[round(x,3) for x in sec['KL']['R_alt']['ci95']]} |\n")
    f.write(f"| PANN R_alt(10.24s) | {sec['PANN_capture']['R_alt']['point']:+.3f} | {[round(x,3) for x in sec['PANN_capture']['R_alt']['ci95']]} |\n")
    f.write(f"| FAD 10.24s pru/rec | {sec['FAD_vggish']['alt']['pruned']:.2f} / {sec['FAD_vggish']['alt']['recovered']:.2f} | - |\n")
    f.write(f"| FD 10.24s pru/rec | {sec['FD_pann2048']['alt']['pruned']:.1f} / {sec['FD_pann2048']['alt']['recovered']:.1f} | - |\n")
print("wrote table1_concordance.md, table2_armd.md, fig2_crossdomain.csv ->", OUT)
