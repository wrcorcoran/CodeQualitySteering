#!/usr/bin/env bash
set -euo pipefail

# Example base-model BigCodeBench run.
# Start with LIMIT=2 for a quick smoke test, then remove it for the full hard subset.

python scripts/10_run_bigcodebench.py \
  model.config=configs/models/qwen2.5_coder_14b.yaml \
  run.name=qwen_base_hard \
  subset=hard \
  generation.n_samples=10 \
  generation.temperature=0.8 \
  generation.max_new_tokens=1280 \
  generation.limit=2
