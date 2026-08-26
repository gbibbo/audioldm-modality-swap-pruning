# DRAFT — request to Arshdeep Singh for the recovered pruned checkpoint

**Status: DRAFT for Gabriel to review and send. Not sent by the agent.** Zero compute cost;
high scientific upside (memo §5 recovered-backbone baseline, correction 5). Two asks bundled:
the recovered checkpoint(s), and confirmation of the published-artifact seam conventions
(AUDIT-M3-001, already logged as an open question in PROGRESS.md).

---

**Subject:** AudioLDM-M pruning — request for the recovered (post-finetuning) checkpoint + a seam question

Hi Arshdeep,

We're building on your structured-pruning work on AudioLDM-M-Full (Zenodo 21376822 /
arXiv 2607.13330). We've reproduced your published `l1_audioldm-m-full_p1.ckpt` exactly: our
materializer equals it bit-for-bit on all 690 U-Net tensors, and we've confirmed it is the
**pre-recovery** (pruned-only, never finetuned) artifact — all same-shape tensors are identical
to the dense checkpoint.

Two requests, if it's easy on your side:

1. **The recovered / post-finetuning checkpoint(s).** You report recovering quality with
   lightweight finetuning after pruning; the released artifact is pre-recovery. Would you be able
   to share the **recovered** pruned model — at minimum the `(1,2,3,1)` (−65 %) severity, and
   ideally any other severities you finetuned (e.g. the `p2_dp2` point)? Full-pipeline `.ckpt` or
   U-Net-only both work for us. It would let us study a downstream question your paper doesn't
   touch, and we'd of course cite and credit your checkpoints.

2. **A seam-convention sanity check.** Reproducing your `p1` artifact bit-exactly, we found a few
   tensors that don't follow the ranking convention used elsewhere, and one internal
   inconsistency we wanted to confirm is intentional rather than a quirk of the release:
   - `output_blocks.0/1.0.in_layers.2.weight` are kept **positionally** (not by L1 ranking);
   - `input_blocks.10.0.in_layers.2.weight` keeps input columns in **identity** order;
   - `output_blocks.2.0.in_layers.2`: the **weight** rows are positional but the **bias** rows are
     ranked (so bias values attach to different channels than the kept weight rows).

   Are these the intended conventions from your pruning script, or an artifact of how the release
   was exported? No urgency — it only affects how we describe the exact pruning map.

Happy to share what we find. Thanks a lot,

Gabriel

---

*Internal note:* if (1) arrives, run the frozen dense-trained/sliced LoRA on the recovered
backbone as an extra system in the phenomenon test; the target result is "recovery restores
standalone CLAP but not legacy-adapter uplift." If it does not arrive by the phenomenon stage,
the paper states the primary experiment is pre-recovery pruning (limitation), per the memo's
time-box.
