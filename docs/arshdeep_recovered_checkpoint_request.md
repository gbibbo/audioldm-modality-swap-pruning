# Arshdeep recovered-checkpoint request — CANCELLED (checkpoint is public)

**Status: CANCELLED / DO-NOT-SEND (2026-08-26). No email is needed.**

Reason: the recovered/finetuned **(1,2,3,1)** checkpoint is publicly downloadable at
**Zenodo 21977996** ("Pruned and finetuned Models", published 2026-08-17, CC-BY-4.0), which
supersedes 21376822 as a strict superset (all 6 old files keep identical md5s):

* recovered **(1,2,3,1)** — `l1_p1_finetuned_global_step_999999.ckpt`, md5
  `cfb7ca3f8c712850f5a4bfe2162f5d1c`, 4,446,514,762 B,
  `https://zenodo.org/api/records/21977996/files/l1_p1_finetuned_global_step_999999.ckpt/content`
* recovered **(1,2,1,1)** — `l1_p1_dp1_finetuned_global_step_999999.ckpt`, md5
  `5d7da1504280913a6a91c76b13d0ff79`, 3,244,099,100 B
* no recovered checkpoint for **(1,2,2,2) p2_dp2** (only a pruned-only U-Net there)

**Caveat before use:** the `_finetuned_global_step_999999` files are larger than their pruned-only
counterparts; they may be full Lightning checkpoints with optimizer/EMA state rather than
weights-only exports — verify on download.

**Provenance corrected in:** `PROGRESS.md` OPEN ITEM #1, `docs/HANDOFF.md` §9.3, memo §5 and
amendment §D recovered-backbone baseline, ledger ARSHDEEP-RECOVERED-PUBLIC.

**Still open (separate, non-blocking, ask later — NOT bundled with anything):** the AUDIT-M3-001
seam-convention question (4 tensors deviate from Arshdeep's public script; `output_blocks.0/1`
positional-vs-ranked; `output_blocks.2.0.in_layers.2` weight-positional/bias-ranked).

*(Historical: this file previously drafted a request for a checkpoint believed absent; the belief
was based on the pre-2026-08-17 state of the upstream record.)*
