"""
Extract steering vectors via difference of means (CAA).

For each metric and each layer:
    v = mean(top p% activations) - mean(bottom p% activations)

Direction is always high-value minus low-value; interpretation is metric-dependent.
Uses mean-pooled activations. Train split only.
Saves per-metric .npz files (float32, no normalization) via save_steering_vectors().
Load back with load_steering_vectors() from src.utils.io.

Usage:
    uv run python scripts/06_extract_steering_vectors.py --model llama3.1_8b
    uv run python scripts/06_extract_steering_vectors.py --model llama3.1_8b --p 25
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from src.utils.io import boundary_mask, load_metrics_dataset, save_steering_vectors

METRICS = ["cc", "mi", "comment_ratio", "h_volume", "h_difficulty", "h_effort", "sloc"]
POOL = "mean"


def top_bottom_indices(values: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, int(len(values) * p / 100))
    sorted_idx = np.argsort(values)
    return sorted_idx[-k:], sorted_idx[:k]  # top k, bottom k


def main(
    model: str, p: int, data_dir: Path, activations_dir: Path, output_dir: Path
) -> None:
    acts_path = activations_dir / model
    assert acts_path.exists(), f"Activations directory not found: {acts_path}"

    meta = json.loads((acts_path / "meta.json").read_text())
    d_model: int = meta["d_model"]
    n_layers: int = meta["n_layers"]
    logger.info(f"Model: {model}  layers={n_layers}  d_model={d_model}")

    index = pd.read_parquet(acts_path / "index.parquet")
    dataset = load_metrics_dataset(data_dir, cols=["id"] + METRICS)

    df = index.merge(dataset, on="id", how="left")
    assert len(df) == len(
        index
    ), "id join changed row count — dataset and activations may be out of sync"

    n_rows = len(index)

    # zero-vector mask (token overflow rows left as all-zero during extraction)
    # check on a single layer to build the mask — all layers share the same skipped rows
    sample_path = acts_path / f"layer_01_{POOL}.memmap"
    sample_vecs = np.memmap(
        sample_path, dtype=np.float16, mode="r", shape=(n_rows, d_model)
    )
    valid_mask = np.any(sample_vecs != 0, axis=1)
    n_skipped = (~valid_mask).sum()
    if n_skipped:
        logger.warning(f"{n_skipped} all-zero rows excluded (token overflow)")
    del sample_vecs

    train_mask = (df["split"].to_numpy() == "train") & valid_mask
    df_train = df[train_mask].reset_index(drop=True)
    train_row_indices = np.where(train_mask)[0]  # global memmap indices for train rows
    n_train = len(df_train)
    logger.info(
        f"Train rows after filtering: {n_train}  (p={p}% → k={max(1, int(n_train * p / 100))} per class)"
    )

    out_path = output_dir / model
    out_path.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        logger.info(f"  metric={metric}")
        values = df_train[metric].to_numpy(dtype=np.float32)

        # drop boundary values (point mass at floor/ceiling are degenerate files)
        interior_mask = boundary_mask(values)
        n_dropped = (~interior_mask).sum()
        if n_dropped:
            logger.info(
                f"    dropped {n_dropped} boundary rows (val={values.min():.3f} or {values.max():.3f})"
            )
        interior_values = values[interior_mask]
        interior_row_indices = train_row_indices[interior_mask]

        top_local, bot_local = top_bottom_indices(interior_values, p)

        top_global = interior_row_indices[top_local]
        bot_global = interior_row_indices[bot_local]

        logger.info(
            f"    {metric}: top [{interior_values[top_local].min():.3f}, {interior_values[top_local].max():.3f}]  "
            f"bottom [{interior_values[bot_local].min():.3f}, {interior_values[bot_local].max():.3f}]  "
            f"k={len(top_local)}"
        )

        layer_dict: dict[int, np.ndarray] = {}
        for layer in range(1, n_layers + 1):
            memmap_path = acts_path / f"layer_{layer:02d}_{POOL}.memmap"
            vecs = np.memmap(
                memmap_path, dtype=np.float16, mode="r", shape=(n_rows, d_model)
            )
            layer_dict[layer] = vecs[top_global].astype(np.float32).mean(axis=0) - vecs[
                bot_global
            ].astype(np.float32).mean(axis=0)
            del vecs

        save_path = out_path / f"{metric}_p{p}.npz"
        save_steering_vectors(save_path, layer_dict)
        norms = np.stack(list(layer_dict.values())).T  # [d_model, n_layers]
        layer_norms = np.linalg.norm(norms, axis=0)
        logger.info(
            f"    saved {save_path}  norm range=[{layer_norms.min():.3f}, {layer_norms.max():.3f}]"
        )

    logger.info(f"Done. Steering vectors written to {out_path}/")


def cli(
    model: str = typer.Argument(..., help="Model subdirectory name, e.g. llama3.1_8b"),
    p: int = typer.Option(10, help="Percentile cutoff"),
    data_dir: Path = typer.Option(Path("data/processed/stackv2_python")),
    activations_dir: Path = typer.Option(Path("activations")),
    output_dir: Path = typer.Option(Path("data/steering_vectors")),
) -> None:
    main(
        model=model,
        p=p,
        data_dir=data_dir,
        activations_dir=activations_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    typer.run(cli)
