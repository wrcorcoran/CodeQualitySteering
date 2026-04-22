#!/usr/bin/env python3
import sys
from pathlib import Path

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_builder import build_dataset
from src.data.types import DataConfig
from src.utils.io import read_yaml
from src.utils.logging import setup_logging


def main(config: Path = typer.Argument(..., help="Path to data config YAML")) -> None:
    setup_logging()
    raw = read_yaml(config)
    cfg = DataConfig(**raw)
    logger.info(f"Building dataset: pull_count={cfg.pull_count}, output={cfg.output_dir}")
    build_dataset(cfg)


if __name__ == "__main__":
    typer.run(main)
