#!/usr/bin/env python3
"""Run a batch of BigCodeBench generation configs from a YAML file.

Each model is loaded once and all its runs execute before unloading.
Writes one JSONL per named run to output_dir.

Usage:
    uv run python scripts/11_run_bcb_batch.py configs/bcb/smoke.yaml
    uv run python scripts/11_run_bcb_batch.py configs/bcb/smoke.yaml --model llama3.1_8b --p 10
"""
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.bigcodebench import load_bigcodebench_tasks, write_jsonl
from src.models.loader import ModelConfig, load_model
from src.steer.generate import SteerSpec, generate
from src.utils.io import read_yaml
from src.utils.logging import setup_logging


def _expand_sweeps(runs: list[dict]) -> list[dict]:
    expanded = []
    for run in runs:
        if "alpha_sweep" not in run:
            expanded.append(run)
            continue
        metric = run["metric"]
        layer = int(run["layer"])
        for alpha in run["alpha_sweep"]:
            expanded.append({
                "name": f"{run['name']}_a{alpha:g}",
                "model": run["model"],
                "steering": [{"metric": metric, "layer": layer, "alpha": float(alpha)}],
            })
    return expanded


def main(
    config: Path = typer.Argument(..., help="Path to BCB batch config YAML"),
    model: Optional[str] = typer.Option(None, help="Only run entries for this model (e.g. llama3.1_8b)"),
    output_dir: Path = typer.Option(Path("bcb_results_clean"), help="Directory for output JSONL files"),
    sv_dir: Path = typer.Option(Path("data/steering_vectors"), help="Steering vector directory"),
    config_dir: Path = typer.Option(Path("configs/models"), help="Model config directory"),
    subset: str = typer.Option("full", help="'full' or 'hard'"),
    n_samples: int = typer.Option(10, help="Samples per task"),
    temperature: float = typer.Option(0.2, help="Sampling temperature"),
    limit: Optional[int] = typer.Option(None, help="Limit number of tasks"),
    batch_size: int = typer.Option(64, help="Generation batch size"),
    max_new_tokens: int = typer.Option(2048, help="Max tokens to generate"),
    p: int = typer.Option(default=10, help="Steering vector percentile"),
) -> None:
    setup_logging()
    cfg = read_yaml(config)

    subset = cfg.get("subset", subset)
    n_samples = cfg.get("n_samples", n_samples)
    temperature = cfg.get("temperature", temperature)
    limit = cfg.get("limit", limit)
    batch_size = cfg.get("batch_size", batch_size)
    max_new_tokens = cfg.get("max_new_tokens", max_new_tokens)
    p = cfg.get("p", p)
    runs = cfg["runs"]

    if model is not None:
        runs = [r for r in runs if r["model"] == model]
        if not runs:
            raise SystemExit(f"No runs found for model={model!r}")

    tasks = load_bigcodebench_tasks(subset=subset, limit=limit)
    logger.info(f"Loaded {len(tasks)} tasks (subset={subset})")

    runs = _expand_sweeps(runs)
    logger.info(f"Expanded to {len(runs)} runs")

    runs_by_model: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        runs_by_model[run["model"]].append(run)

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_runs in runs_by_model.items():
        model_cfg = ModelConfig(**read_yaml(config_dir / f"{model_name}.yaml"))
        logger.info(f"Loading {model_name}...")
        loaded = load_model(model_cfg)

        for run in model_runs:
            run_name = run["name"]
            steer_specs: list[SteerSpec] = [
                (s["metric"], int(s["layer"]), float(s["alpha"]))
                for s in run.get("steering", [])
            ]
            label = steer_specs if steer_specs else "baseline"
            logger.info(f"  {run_name} | steering={label} | n_samples={n_samples} | p={p}")

            # Expand tasks n_samples times so all samples generate in one sweep
            expanded_tasks = tasks * n_samples
            results = generate(
                loaded,
                expanded_tasks,
                steer_specs,
                sv_dir=sv_dir,
                p=p,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                temperature=temperature,
            )
            n_tasks = len(tasks)
            for i, r in enumerate(results):
                r["generation_id"] = i // n_tasks
            all_results = results

            rows = [
                {
                    "task_id": r["task_id"],
                    "generation_id": r["generation_id"],
                    "solution": r["raw_output"],
                    "model": r["model"],
                    "steerings": r["steerings"],
                    "code": r["code"],
                    "metrics": r["metrics"],
                }
                for r in all_results
            ]
            out_path = output_dir / f"{run_name}_p{p}.jsonl"
            write_jsonl(out_path, rows)
            logger.info(f"  Wrote {len(rows)} rows to {out_path}")

        del loaded.model
        torch.cuda.empty_cache()
        logger.info(f"Unloaded {model_name}")


if __name__ == "__main__":
    typer.run(main)