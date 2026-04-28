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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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

def compute_pca(
        vecs: np.ndarray,
        df: pd.DataFrame,
        plot_path: str,
) -> pd.DataFrame:
    """
    Reduces the dimensionality of the activations to 2 using principal component analysis. For each element of
    METRICS_COLS, excluding the id, saves a scatter plot to plot_path/<metric>.svg of the activations along the
    first two principal components, colored by by the metric.
    Returns:
        df: DataFrame [N_valid] with columns:
            id, pc_0, pc_1, cc, mi, comment_ratio, h_volume, h_difficulty, h_effort, sloc
    """
    plot_dir = Path(plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    coords = PCA(n_components=2).fit_transform(vecs.astype(np.float32))  # [N, 2]

    metric_cols = [c for c in METRICS_COLS if c != "id"]
    result = df[["id"] + metric_cols].copy().reset_index(drop=True)
    result["pc_0"] = coords[:, 0]
    result["pc_1"] = coords[:, 1]

    for metric in metric_cols:
        raw = result[metric].to_numpy(dtype=np.float32)
        raw_min = raw.min()
        log_vals = np.log1p(raw - raw_min)  # shift so minimum is 0 before log

        metric_dir = plot_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # variants = [
        #     (raw,      metric,              "raw"),
        #     (log_vals, f"log1p({metric})",  "log"),
        # ]
        # for vals, label, filename in variants:
        #     fig, ax = plt.subplots(figsize=(8, 6))
        #     sc = ax.scatter(result["pc_0"], result["pc_1"], c=vals, cmap="viridis", s=1, alpha=0.5)
        #     plt.colorbar(sc, ax=ax, label=label)
        #     ax.set_xlabel("PC 1")
        #     ax.set_ylabel("PC 2")
        #     ax.set_title(label)
        #     fig.savefig(metric_dir / f"{filename}.png", dpi=150)
        #     plt.close(fig)

        p5 = np.percentile(raw, 5)
        p95 = np.percentile(raw, 95)
        bottom_mask = raw <= p5
        top_mask = raw >= p95
        pct_mask = bottom_mask | top_mask

        pct_x = result["pc_0"].to_numpy()[pct_mask]
        pct_y = result["pc_1"].to_numpy()[pct_mask]
        pct_labels = np.where(bottom_mask[pct_mask], "bottom 5%", "top 5%")

        colors = {"bottom 5%": "steelblue", "top 5%": "tomato"}
        fig, ax = plt.subplots(figsize=(8, 6))
        for group, color in colors.items():
            sel = pct_labels == group
            ax.scatter(pct_x[sel], pct_y[sel], c=color, label=group, s=4, alpha=0.6)
        ax.legend(title=metric, markerscale=3)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"{metric} — bottom/top 5th percentile")
        fig.savefig(metric_dir / "percentile_only.png", dpi=150)
        plt.close(fig)

    return result[["id", "pc_0", "pc_1"] + metric_cols]


if __name__ == "__main__":
    activations_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("activations/llama3.1_8b")
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    pool = sys.argv[3] if len(sys.argv) > 3 else "mean"
    plot_path = sys.argv[4] if len(sys.argv) > 4 else "plots"

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

    print(f"\nRunning PCA, saving plots to {plot_path}/")
    result = compute_pca(vecs, df, plot_path)
    print(f"PCA result: {result.shape}")
    print(result[["id", "pc_0", "pc_1", "cc", "mi"]].head(5).to_string(index=False))
