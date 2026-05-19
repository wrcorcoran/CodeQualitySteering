#!/usr/bin/env python3
"""Generate BigCodeBench samples using run_sweep from src/steer/generate.py.

Baseline:
    uv run python scripts/09_generate_bigcodebench.py configs/models/gemma2_9b.yaml

Steered:
    uv run python scripts/09_generate_bigcodebench.py configs/models/gemma2_9b.yaml \
        --steering comment_ratio,20,30.0

Multiple steering specs (compound):
    uv run python scripts/09_generate_bigcodebench.py configs/models/gemma2_9b.yaml \
        --steering comment_ratio,20,30.0 --steering cc,15,-5.0
"""
import json
import sys
from pathlib import Path
from typing import Optional
from src.utils.io import read_yaml
from src.models.loader import ModelConfig
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.bigcodebench import load_bigcodebench_tasks, make_output_stem, write_jsonl
from src.steer.generate import SteerSpec, run_sweep
from src.utils.logging import setup_logging


def parse_steering(value: str) -> SteerSpec:
    parts = value.split(",")
    if len(parts) != 3:
        raise typer.BadParameter(f"Expected metric,layer,alpha — got {value!r}")
    metric, layer_s, alpha_s = parts
    return metric.strip(), int(layer_s), float(alpha_s)


def main(
    model_config: Path = typer.Argument(..., help="Path to model config YAML"),
    output_dir: Path = typer.Option(Path("bcb_results_clean"), help="Directory for generated samples"),
    run_name: Optional[str] = typer.Option(None, help="Override output filename stem"),
    subset: str = typer.Option("hard", help="'hard' or 'full'"),
    n_samples: int = typer.Option(5, help="Samples per task"),
    temperature: float = typer.Option(0.2, help="Sampling temperature; 0 = greedy"),
    max_new_tokens: int = typer.Option(2048, help="Maximum generated tokens"),
    batch_size: int = typer.Option(64, help="Batch size for generation"),
    limit: Optional[int] = typer.Option(None, help="Limit tasks (smoke test)"),
    steering: list[str] = typer.Option([], help="Steering spec as metric,layer,alpha (repeatable; omit for baseline)"),
    sv_dir: Path = typer.Option(Path("data/steering_vectors"), help="Steering vector directory"),
    p: int = typer.Option(25, help="Percentile used when extracting steering vectors"),
) -> None:
    setup_logging()
    cfg = ModelConfig(**read_yaml(model_config))

    steer_specs: list[SteerSpec] = [parse_steering(s) for s in steering]
    steered = bool(steer_specs)

    tasks = load_bigcodebench_tasks(subset=subset, limit=limit)
    logger.info(f"Loaded {len(tasks)} tasks (subset={subset})")

    stem = run_name or make_output_stem(
        cfg.model_name, "instruct", subset, temperature, n_samples, steered=steered
    )
    samples_path = output_dir / f"{stem}.jsonl"
    meta_path = output_dir / f"{stem}.meta.json"

    results = run_sweep(
        model_names=[cfg.model_name],
        tasks=tasks,
        steering_configs=[steer_specs],
        sv_dir=sv_dir,
        p=p,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        n_samples=n_samples,
        temperature=temperature,
    )

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
        for r in results
    ]
    write_jsonl(samples_path, rows)
    logger.info(f"Wrote {len(rows)} samples to {samples_path}")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "model_name": cfg.model_name,
        "hf_id": cfg.hf_id,
        "subset": subset,
        "n_samples": n_samples,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "limit": limit,
        "steerings": [{"metric": m, "layer": l, "alpha": a} for m, l, a in steer_specs],
        "sv_dir": str(sv_dir),
        "p": p,
        "samples_path": str(samples_path),
    }, indent=2))


if __name__ == "__main__":
    typer.run(main)