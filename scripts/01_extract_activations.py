#!/usr/bin/env python3
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.types import DatasetRow
from src.extract.activations import extract_activations
from src.models.loader import ModelConfig, load_model
from src.utils.io import read_json, read_yaml
from src.utils.logging import setup_logging


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_rows(manifest_path: Path) -> list[DatasetRow]:
    manifest = read_json(manifest_path)
    rows: list[DatasetRow] = []
    for shard in manifest["shards"]:
        for record in pq.read_table(shard["path"]).to_pylist():
            rows.append(DatasetRow(**record))
    return rows


def main(
    model_config: Path = typer.Argument(..., help="Path to model config YAML"),
    data_manifest: Path = typer.Argument(..., help="Path to dataset manifest.json"),
    activations_dir: Path = typer.Option(Path("activations"), help="Root output directory"),
    seed: int = typer.Option(42),
) -> None:
    setup_logging()
    _seed_all(seed)

    raw = read_yaml(model_config)
    cfg = ModelConfig(**raw)
    logger.info(f"Loading model: {cfg.hf_id}")
    loaded = load_model(cfg)

    logger.info(f"Loading dataset rows from {data_manifest}")
    rows = _load_rows(data_manifest)
    logger.info(f"Loaded {len(rows)} rows")

    extract_activations(loaded, rows, activations_dir)


if __name__ == "__main__":
    typer.run(main)
