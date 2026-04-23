"""
Sample script: load one activation layer and join with code metrics.

Usage:
    uv run python scripts/03_load_sample.py [activations_dir] [layer] [pool]

    activations_dir : e.g. activations/llama3.1_8b   (default)
    layer           : 1-indexed layer number          (default: 16)
    pool            : "mean" or "last"                (default: mean)

NOTE: Rows skipped during extraction (token overflow) are left as all-zero
vectors in the memmap. These are filtered out via `valid_mask` below before
any downstream use. Remove this filter step only if you need to preserve
original row indices.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path("data/processed/stackv2_python")
METRICS_COLS = ["id", "cc", "mi", "comment_ratio", "h_volume", "h_difficulty", "h_effort", "sloc"]


def load_layer(
    activations_dir: Path,
    layer: int,
    pool: str = "mean",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns:
        vecs : float16 ndarray [N_valid, d_model]  (bf16 bits stored as float16)
        df   : DataFrame [N_valid] with columns:
               id, split, cc, mi, comment_ratio, h_volume, h_difficulty, h_effort, sloc
    Rows skipped during extraction (all-zero vectors) are excluded from both.
    """
    meta = json.loads((activations_dir / "meta.json").read_text())
    d_model = meta["d_model"]
    n_layers = meta["n_layers"]
    assert 1 <= layer <= n_layers, f"layer must be in 1..{n_layers}, got {layer}"

    # index.parquet: id (str), split (str) — one row per memmap row
    index = pd.read_parquet(activations_dir / "index.parquet")

    # load metrics from parquet shards; split lives in index, not needed from shards
    shards = sorted(DATASET_DIR.glob("shard_*.parquet"))
    assert shards, f"No parquet shards found in {DATASET_DIR}"
    dataset = pd.concat(
        [pd.read_parquet(p, columns=METRICS_COLS) for p in shards],
        ignore_index=True,
    )

    # left join so row order matches the memmap exactly
    df = index.merge(dataset, on="id", how="left")
    assert len(df) == len(index), (
        f"id join changed row count ({len(index)} → {len(df)}). "
        "Dataset and activations may be out of sync."
    )

    memmap_path = activations_dir / f"layer_{layer:02d}_{pool}.memmap"
    assert memmap_path.exists(), f"Memmap not found: {memmap_path}"

    n_rows = len(index)
    vecs = np.memmap(memmap_path, dtype=np.float16, mode="r", shape=(n_rows, d_model))

    # NOTE: skipped batches (token overflow) leave all-zero rows — remove them.
    valid_mask = np.any(vecs != 0, axis=1)
    n_skipped = (~valid_mask).sum()
    if n_skipped:
        print(f"[warn] {n_skipped} all-zero rows filtered (token overflow skips during extraction)")

    vecs = np.array(vecs[valid_mask])   # copy out of memmap into RAM
    df = df[valid_mask].reset_index(drop=True)

    return vecs, df


if __name__ == "__main__":
    activations_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("activations/llama3.1_8b")
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    pool = sys.argv[3] if len(sys.argv) > 3 else "mean"

    print(f"Loading {activations_dir.name}  layer={layer}  pool={pool}")
    vecs, df = load_layer(activations_dir, layer, pool)

    print(f"\nActivations : {vecs.shape}  dtype={vecs.dtype}")
    print(f"Rows        : {len(df)}")
    print(f"Columns     : {list(df.columns)}")
    print(f"\nMetrics sample:")
    print(df[["id", "split", "cc", "mi", "comment_ratio", "sloc"]].head(10).to_string(index=False))
    print(f"\nActivation norms (first 5 rows):")
    print(np.linalg.norm(vecs[:5].astype(np.float32), axis=1))
    print(f"\nMetric ranges:")
    print(df[["cc", "mi", "comment_ratio", "h_volume", "h_difficulty", "h_effort", "sloc"]].describe())
