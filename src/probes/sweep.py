import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.probes.ridge import ProbeResult, fit_probe

METRICS = ["cc", "mi", "comment_ratio", "h_volume", "h_difficulty", "h_effort", "sloc"]
POOLS = ["last", "mean"]
DATASET_DIR = Path("data/processed/stackv2_python")


def _load_dataset(dataset_dir: Path) -> pd.DataFrame:
    shards = sorted(dataset_dir.glob("shard_*.parquet"))
    assert shards, f"No parquet shards found in {dataset_dir}"
    cols = ["id"] + METRICS
    return pd.concat(
        [pd.read_parquet(p, columns=cols) for p in shards],
        ignore_index=True,
    )


def _load_index(activations_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(activations_dir / "index.parquet")


def _load_vecs(activations_dir: Path, layer: int, pool: str, n_rows: int, d_model: int) -> np.ndarray:
    path = activations_dir / f"layer_{layer:02d}_{pool}.memmap"
    vecs = np.memmap(path, dtype=np.float16, mode="r", shape=(n_rows, d_model))
    return np.array(vecs)  # copy to RAM


def sweep_model(
    activations_dir: Path,
    dataset_dir: Path = DATASET_DIR,
    alpha: float = 1.0,
    pool: Optional[str] = None,
) -> Iterator[ProbeResult]:
    meta = json.loads((activations_dir / "meta.json").read_text())
    d_model: int = meta["d_model"]
    n_layers: int = meta["n_layers"]

    index = _load_index(activations_dir)
    dataset = _load_dataset(dataset_dir)

    df = index.merge(dataset, on="id", how="left")
    assert len(df) == len(index), "id join changed row count — dataset and activations may be out of sync"

    n_rows = len(index)
    train_mask = df["split"].to_numpy() == "train"
    test_mask = df["split"].to_numpy() == "test"

    pools = [pool] if pool is not None else POOLS
    total = n_layers * len(pools) * len(METRICS)
    done = 0

    for pool in pools:
        for layer in range(1, n_layers + 1):
            logger.info(f"Loading layer {layer}/{n_layers} pool={pool}")
            vecs = _load_vecs(activations_dir, layer, pool, n_rows, d_model)

            # exclude token-overflow rows (left as all-zero during extraction)
            valid_mask = np.any(vecs != 0, axis=1)
            n_skipped = (~valid_mask).sum()
            if n_skipped:
                logger.debug(f"  {n_skipped} all-zero rows excluded (token overflow)")

            for metric in METRICS:
                y = df[metric].to_numpy(dtype=np.float32)

                row_train = valid_mask & train_mask
                row_test = valid_mask & test_mask

                X_train = vecs[row_train]
                y_train = y[row_train]
                X_test = vecs[row_test]
                y_test = y[row_test]

                result = fit_probe(
                    X_train, y_train,
                    X_test, y_test,
                    metric=metric,
                    layer=layer,
                    pool=pool,
                    alpha=alpha,
                )
                done += 1
                logger.info(
                    f"[{done}/{total}] layer={layer:>2} pool={pool} metric={metric:<16} "
                    f"R2_train={result.r2_train:.3f}  R2_test={result.r2_test:.3f}"
                )

                yield result