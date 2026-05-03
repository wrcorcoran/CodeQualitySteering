#!/usr/bin/env python3
from pathlib import Path

import typer
from loguru import logger

from src.probes.storage import save_results
from src.probes.sweep import DATASET_DIR, sweep_model
from src.utils.logging import setup_logging

RIDGE_ALPHA = 1.0


def main(
    activations_dir: Path = typer.Argument(..., help="Path to model activations dir, e.g. activations/llama3.1_8b"),
    pool: str = typer.Argument(..., help="Pooling strategy: 'last' or 'mean'"),
    dataset_dir: Path = typer.Option(DATASET_DIR, help="Path to processed dataset dir"),
    output_dir: Path = typer.Option(Path("probes"), help="Root output directory"),
) -> None:
    setup_logging()
    assert pool in ("last", "mean"), f"pool must be 'last' or 'mean', got {pool!r}"

    model_name = activations_dir.name
    probe_dir = output_dir / model_name
    logger.info(f"Training probes for {model_name} pool={pool} to {probe_dir}")

    results = list(sweep_model(activations_dir, dataset_dir, alpha=RIDGE_ALPHA, pool=pool))
    logger.info(f"Trained {len(results)} probes")

    save_results(results, probe_dir, model_name=model_name, ridge_alpha=RIDGE_ALPHA, pool=pool)
    logger.info(f"Saved results to {probe_dir}/results_{pool}.parquet and {probe_dir}/weights/")


if __name__ == "__main__":
    typer.run(main)
