#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.bigcodebench import compute_quality_metrics, summarize_quality
from src.utils.logging import setup_logging


def main(
    samples: Path = typer.Argument(..., help="Generated or sanitized JSONL samples"),
    eval_results: Path | None = typer.Option(None, help="BigCodeBench *_eval_results.json for per-sample pass/fail"),
    pass_at_k: Path | None = typer.Option(None, help="BigCodeBench *_pass_at_k.json"),
    output_dir: Path = typer.Option(Path("bcb_reports"), help="Directory for quality reports"),
    run_name: str | None = typer.Option(None, help="Optional name for output files"),
) -> None:
    setup_logging()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or samples.stem

    df = compute_quality_metrics(samples, eval_results)
    sample_path = output_dir / f"{name}_quality.parquet"
    summary_path = output_dir / f"{name}_quality_summary.csv"
    combined_path = output_dir / f"{name}_summary.json"

    df.to_parquet(sample_path, index=False)
    summarize_quality(df).to_csv(summary_path, index=False)

    combined = {
        "samples": str(samples),
        "eval_results": str(eval_results) if eval_results else None,
        "quality_rows": int(len(df)),
        "quality_path": str(sample_path),
        "quality_summary_path": str(summary_path),
    }
    if pass_at_k is not None:
        combined["pass_at_k"] = json.loads(pass_at_k.read_text())
        combined["pass_at_k_path"] = str(pass_at_k)
    combined_path.write_text(json.dumps(combined, indent=2))

    logger.info(f"Wrote {sample_path}, {summary_path}, and {combined_path}")


if __name__ == "__main__":
    typer.run(main)
