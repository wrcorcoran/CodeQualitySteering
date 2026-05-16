#!/usr/bin/env bash
set -euo pipefail

# Example steered BigCodeBench run.
# Replace steering.vector with the actual vector you want to test.
# Start with LIMIT=2 for a quick smoke test, then remove it for the full hard subset.

python scripts/10_run_bigcodebench.py \
  model.config=configs/models/qwen2.5_coder_14b.yaml \
  run.name=qwen_cc_l16_alpha1_hard \
  subset=hard \
  generation.n_samples=10 \
  generation.temperature=0.8 \
  generation.max_new_tokens=1280 \
  generation.limit=2 \
  steering.vector=steering_vectors/qwen2.5_coder_14b/vectors/layer_16_mean_cc.npy \
  steering.layer=16 \
  steering.alpha=1.0
