#!/bin/bash
set -e

MODELS=(llama3.1_8b gemma2_9b qwen2.5_coder_14b)

for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    echo "=== $model ==="
    if [ $i -eq $(( ${#MODELS[@]} - 1 )) ]; then
        uv run python scripts/05_report_probes.py probes/$model --compare
    else
        uv run python scripts/05_report_probes.py probes/$model
    fi
done