import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def boundary_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask that is True for rows not at the exact min or max of values."""
    vmin, vmax = values.min(), values.max()
    return (values > vmin) & (values < vmax)


def save_steering_vectors(path: Path, layer_dict: dict[int, np.ndarray]) -> None:
    """Save {layer_idx: vector} dict as .npz. Keys stored as 'layer_{L:02d}'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"layer_{l:02d}": v for l, v in layer_dict.items()})


def load_steering_vectors(path: Path) -> dict[int, np.ndarray]:
    """Load .npz saved by save_steering_vectors. Returns {layer_idx: vector}."""
    data = np.load(path)
    return {int(k.split("_")[1]): data[k] for k in data.files}


def load_metrics_dataset(dataset_dir: Path, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Concatenate all shard parquets from dataset_dir. Optionally restrict columns."""
    shards = sorted(dataset_dir.glob("shard_*.parquet"))
    assert shards, f"No parquet shards found in {dataset_dir}"
    return pd.concat(
        [pd.read_parquet(p, columns=cols) for p in shards],
        ignore_index=True,
    )
