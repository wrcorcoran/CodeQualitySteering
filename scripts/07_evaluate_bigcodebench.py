#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging


def main(
    samples: Path = typer.Argument(..., help="Generated JSONL samples"),
    split: str = typer.Option("instruct", help="'instruct' or 'complete'"),
    subset: str = typer.Option("hard", help="'hard' or 'full'"),
    work_dir: Path = typer.Option(Path("."), help="Working directory where BigCodeBench creates bcb_results"),
    pass_k: str = typer.Option("1,5", help="Comma-separated pass@k values"),
    parallel: int = typer.Option(8, help="Parallel execution workers"),
    gradio: bool = typer.Option(False, "--gradio", help="Use Gradio remote execution instead of local"),
    skip_sanitize: bool = typer.Option(False, help="Evaluate samples directly without sanitize/calibrate first"),
) -> None:
    setup_logging()
    samples = samples.resolve()
    execution = "gradio" if gradio else "local"
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    eval_samples = samples
    if not skip_sanitize:
        sanitize_cmd = [
            "bigcodebench.sanitize",
            "--samples", str(samples),
            "--calibrate",
        ]
        logger.info("Running: " + " ".join(sanitize_cmd))
        subprocess.run(sanitize_cmd, check=True)
        eval_samples = samples.with_name(f"{samples.stem}-sanitized-calibrated{samples.suffix}")

    eval_cmd = [
        "bigcodebench.evaluate",
        "--execution", execution,
        "--split", split,
        "--subset", subset,
        "--samples", str(eval_samples),
        "--pass_k", pass_k,
        "--parallel", str(parallel),
    ]
    logger.info("Running: " + " ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True, cwd=work_dir)

    logger.info(f"Evaluated {eval_samples} with execution={execution}; BigCodeBench wrote results under {work_dir / 'bcb_results'}")


if __name__ == "__main__":
    typer.run(main)
