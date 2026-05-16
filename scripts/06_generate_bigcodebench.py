#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.bigcodebench import generate_bigcodebench_samples, load_bigcodebench_tasks, make_output_stem
from src.eval.steering import steering_hook
from src.models.loader import ModelConfig, load_model
from src.utils.io import read_yaml, write_json
from src.utils.logging import setup_logging


def main(
    model_config: Path = typer.Argument(..., help="Path to model config YAML"),
    output_dir: Path = typer.Option(Path("bcb_results"), help="Directory for generated samples"),
    run_name: str | None = typer.Option(None, help="Optional stable output stem"),
    split: str = typer.Option("instruct", help="'instruct' or 'complete'"),
    subset: str = typer.Option("hard", help="'hard' or 'full'"),
    n_samples: int = typer.Option(10, help="Samples per task; keep >=10 for pass@1/pass@5 estimates"),
    temperature: float = typer.Option(0.8, help="Sampling temperature"),
    max_new_tokens: int = typer.Option(1280, help="Maximum generated tokens per sample"),
    limit: int | None = typer.Option(None, help="Optional task limit for smoke tests"),
    steering_vector: Path | None = typer.Option(None, help="Optional .npy steering vector"),
    steering_layer: int | None = typer.Option(None, help="1-indexed layer for steering"),
    steering_alpha: float = typer.Option(1.0, help="Steering multiplier"),
) -> None:
    setup_logging()
    if n_samples < 10:
        raise typer.BadParameter("n_samples should be at least 10 for pass@1/pass@5 reporting")
    if (steering_vector is None) != (steering_layer is None):
        raise typer.BadParameter("--steering-vector and --steering-layer must be provided together")

    cfg = ModelConfig(**read_yaml(model_config))
    tasks = load_bigcodebench_tasks(split=split, subset=subset, limit=limit)
    loaded = load_model(cfg)

    steered = steering_vector is not None
    stem = run_name or make_output_stem(cfg.model_name, split, subset, temperature, n_samples, steered=steered)
    samples_path = output_dir / f"{stem}.jsonl"
    meta_path = output_dir / f"{stem}.meta.json"

    logger.info(f"Writing samples to {samples_path}")
    if steering_vector is not None and steering_layer is not None:
        vector = torch.from_numpy(np.load(steering_vector).astype(np.float32))
        with steering_hook(loaded.model, steering_layer, vector, steering_alpha):
            generate_bigcodebench_samples(loaded, tasks, split, n_samples, temperature, max_new_tokens, samples_path)
    else:
        generate_bigcodebench_samples(loaded, tasks, split, n_samples, temperature, max_new_tokens, samples_path)

    write_json(meta_path, {
        "model_name": cfg.model_name,
        "hf_id": cfg.hf_id,
        "split": split,
        "subset": subset,
        "n_samples": n_samples,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "limit": limit,
        "samples_path": str(samples_path),
        "steering_vector": str(steering_vector) if steering_vector else None,
        "steering_layer": steering_layer,
        "steering_alpha": steering_alpha if steering_vector else None,
    })


if __name__ == "__main__":
    typer.run(main)
