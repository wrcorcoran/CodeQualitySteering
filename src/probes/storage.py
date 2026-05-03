import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.probes.ridge import ProbeResult


def save_results(results: list[ProbeResult], output_dir: Path, model_name: str, ridge_alpha: float, pool: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "layer": r.layer,
            "pool": r.pool,
            "metric": r.metric,
            "r2_train": r.r2_train,
            "r2_test": r.r2_test,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "intercept": r.intercept,
        })
        weight_path = weights_dir / f"layer_{r.layer:02d}_{r.pool}_{r.metric}.npy"
        np.save(weight_path, r.weights)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_dir / f"results_{pool}.parquet")

    meta = {
        "model_name": model_name,
        "ridge_alpha": ridge_alpha,
        "n_probes": len(results),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_results(output_dir: Path) -> pd.DataFrame:
    files = sorted(output_dir.glob("results_*.parquet"))
    assert files, f"No results parquet files found in {output_dir}"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_weights(output_dir: Path, layer: int, pool: str, metric: str) -> np.ndarray:
    path = output_dir / "weights" / f"layer_{layer:02d}_{pool}_{metric}.npy"
    return np.load(path)
