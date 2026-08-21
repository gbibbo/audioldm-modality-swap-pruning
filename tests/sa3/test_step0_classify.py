#!/usr/bin/env python3
"""C1: classify_config_diff separates architecture / objective-or-sampling / text-encoder-location / other."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "sa3"))
from step0_verify_pair import classify_config_diff  # noqa: E402

cd = {"only_a": [], "only_b": ["model.diffusion.sampling_distribution_shift_options.type", "training.demo.x"],
      "changed": [("model.diffusion.diffusion_objective", "rectified_flow", "rf_denoiser"),
                  ("model.diffusion.config.depth", 20, 16),
                  ("model.conditioning.configs.[0].config.repo_id", "a", "b"),
                  ("weird.key", 1, 2)]}
c = classify_config_diff(cd)
ok = ([x[1] for x in c["architecture"]] == ["model.diffusion.config.depth"]
      and sorted(x[1] for x in c["objective_or_sampling"]) == sorted(["model.diffusion.sampling_distribution_shift_options.type", "training.demo.x", "model.diffusion.diffusion_objective"])
      and [x[1] for x in c["text_encoder_location"]] == ["model.conditioning.configs.[0].config.repo_id"]
      and [x[1] for x in c["other"]] == ["weird.key"] and c["resolved_at_build"] == [])
print("  C1 classify:", c); print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
