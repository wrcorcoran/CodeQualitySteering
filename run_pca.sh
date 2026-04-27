#!/usr/bin/env bash
#SBATCH --job-name=pca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --output=logs/pca_%j.out
#SBATCH --error=logs/pca_%j.err
# Usage: sbatch run_pca.sh <activations_dir> [pool]
#    or: ./run_pca.sh <activations_dir> [pool]
# Example: sbatch run_pca.sh activations/llama3.1_8b mean
set -euo pipefail

mkdir -p logs

ACTIVATIONS_DIR="${1:?Usage: $0 <activations_dir> [pool]}"
POOL="${2:-mean}"

MODEL_NAME="$(basename "$ACTIVATIONS_DIR")"
N_LAYERS="$(python3 -c "import json; print(json.load(open('$ACTIVATIONS_DIR/meta.json'))['n_layers'])")"

echo "Model: $MODEL_NAME  layers: $N_LAYERS  pool: $POOL"

for layer in $(seq 1 "$N_LAYERS"); do
    PLOT_PATH="plots/${MODEL_NAME}/layer$(printf '%02d' "$layer")"
    echo "Layer $layer → $PLOT_PATH"
    uv run python scripts/03_load_sample.py "$ACTIVATIONS_DIR" "$layer" "$POOL" "$PLOT_PATH"
done
