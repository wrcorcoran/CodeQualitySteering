#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import read_json
from src.utils.logging import setup_logging


def main(activations_dir: Path = typer.Argument(..., help="Per-model activations directory")) -> None:
    setup_logging()

    meta = read_json(activations_dir / "meta.json")
    n_layers: int = meta["n_layers"]
    d_model: int = meta["d_model"]
    n_rows: int = _infer_n_rows(activations_dir, n_layers)

    logger.info(f"Model: {meta['model_name']}  layers={n_layers}  d_model={d_model}  rows={n_rows}")
    logger.info(f"dtype={meta['dtype']}  pool_strategies={meta['pool_strategies']}")

    # norm growth check across layers for last-token pooling
    logger.info("--- Mean L2 norm per layer (last-token pool) ---")
    for layer in range(1, n_layers + 1):
        path = activations_dir / f"layer_{layer:02d}_last.memmap"
        mm = np.memmap(path, dtype=np.float16, mode="r", shape=(n_rows, d_model))
        norms = np.linalg.norm(mm[:].astype(np.float32), axis=1)
        logger.info(f"  layer {layer:02d}: mean_norm={norms.mean():.2f}  std={norms.std():.2f}")

    # NaN check on a random layer
    mid = n_layers // 2
    for pool in meta["pool_strategies"]:
        path = activations_dir / f"layer_{mid:02d}_{pool}.memmap"
        mm = np.memmap(path, dtype=np.float16, mode="r", shape=(n_rows, d_model))
        nan_count = int(np.isnan(mm[:].astype(np.float32)).sum())
        logger.info(f"NaN check layer {mid} {pool}: {nan_count} NaNs")

    # metric distributions from index
    index = pq.read_table(activations_dir / "index.parquet").to_pydict()
    splits = index["split"]
    for s in ["train", "val", "test"]:
        count = splits.count(s)
        logger.info(f"Split {s}: {count} rows ({100*count/len(splits):.1f}%)")


def _infer_n_rows(activations_dir: Path, n_layers: int) -> int:
    path = activations_dir / f"layer_01_last.memmap"
    # file size / (d_model * 2 bytes) gives n_rows; read from meta instead
    meta = read_json(activations_dir / "meta.json")
    d_model = meta["d_model"]
    size = path.stat().st_size
    return size // (d_model * 2)


if __name__ == "__main__":
    typer.run(main)
