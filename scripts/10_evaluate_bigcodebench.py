#!/usr/bin/env python3
"""Sanitize, evaluate, and compare BigCodeBench samples locally.

Basic evaluation:
    uv run python scripts/10_evaluate_bigcodebench.py <samples.jsonl>

With metric comparison against a baseline:
    uv run python scripts/10_evaluate_bigcodebench.py <samples.jsonl> --baseline <baseline.jsonl>
"""
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, cast

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.bigcodebench import METRICS
from src.utils.logging import setup_logging


def _load_metrics(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("metrics")
            if m is None:
                continue
            row = {"task_id": r["task_id"], "generation_id": r.get("generation_id", 0)}
            for metric in METRICS:
                row[metric] = m.get(metric)
            rows.append(row)
    empty = pd.DataFrame(columns=["task_id", "generation_id"] + METRICS)
    return pd.DataFrame(rows) if rows else empty


def _print_delta(samples: Path, baseline: Path) -> None:
    base_df = _load_metrics(baseline).dropna(subset=METRICS, how="any")
    steered_df = _load_metrics(samples).dropna(subset=METRICS, how="any")

    if base_df.empty:
        print(f"No valid metrics in baseline {baseline.name} — skipping comparison.")
        return
    if steered_df.empty:
        print(f"No valid metrics in {samples.name} — skipping comparison.")
        return

    # Average across generations per task, then merge on task_id
    base_task    = base_df.groupby("task_id")[METRICS].mean().reset_index()
    steered_task = steered_df.groupby("task_id")[METRICS].mean().reset_index()
    merged = base_task.merge(steered_task, on="task_id", suffixes=("_base", "_steered"))

    pd.set_option("display.float_format", "{:+.4f}".format)
    print(f"\n── Metric change vs baseline ({samples.stem} vs {baseline.stem}) ──")
    rows: dict[str, dict] = {}
    for m in METRICS:
        base_col    = cast(pd.Series, merged[f"{m}_base"])
        steered_col = cast(pd.Series, merged[f"{m}_steered"])
        delta       = steered_col - base_col
        mean_base   = float(base_col.mean())

        pct_per_row = (delta / base_col.abs().replace(0, float("nan"))) * 100
        norm     = (float(delta.mean())   / mean_base) * 100 if mean_base != 0 else float("nan")
        norm_med = (float(delta.median()) / mean_base) * 100 if mean_base != 0 else float("nan")
        norm_std = (float(delta.std())    / mean_base) * 100 if mean_base != 0 else float("nan")

        rows[m] = {
            "mean %Δ (per-row)":    float(pct_per_row.mean()),
            "median %Δ (per-row)":  float(pct_per_row.median()),
            "std %Δ (per-row)":     float(pct_per_row.std()),
            "mean Δ/mean(base)%":   norm,
            "median Δ/mean(base)%": norm_med,
            "std Δ/mean(base)%":    norm_std,
            "n":                    int(delta.notna().sum()),
        }
    summary = pd.DataFrame(rows).T
    summary["n"] = summary["n"].astype(int)
    print(summary.to_string())


def main(
    samples: Path = typer.Argument(..., help="Generated JSONL samples"),
    baseline: Optional[Path] = typer.Option(None, help="Baseline JSONL to compare metrics against"),
    subset: str = typer.Option("full", help="'hard' or 'full'"),
    pass_k: str = typer.Option("1,5", help="Comma-separated pass@k values"),
    parallel: int = typer.Option(8, help="Parallel execution workers"),
    min_time_limit: float = typer.Option(10.0, help="Min seconds per task execution"),
    skip_sanitize: bool = typer.Option(False, help="Skip sanitize/calibrate step"),
    skip_eval: bool = typer.Option(False, help="Skip pass@k evaluation (metrics comparison only)"),
) -> None:
    setup_logging()
    from bigcodebench.sanitize import script as sanitize_script
    from bigcodebench.evaluate import evaluate

    samples = samples.resolve()

    if not skip_eval:
        processed_dir = samples.parent / "processed"
        processed_dir.mkdir(exist_ok=True)

        eval_samples = processed_dir / samples.name
        shutil.copy2(samples, eval_samples)

        if not skip_sanitize:
            sanitize_script(str(eval_samples), calibrate=True)
            eval_samples = processed_dir / f"{samples.stem}-sanitized-calibrated{samples.suffix}"

        task_ids = {json.loads(line)["task_id"] for line in eval_samples.read_text().splitlines() if line.strip()}
        evaluate(
            split="instruct",
            subset=subset,
            samples=str(eval_samples),
            execution="local",
            pass_k=pass_k,
            parallel=parallel,
            min_time_limit=min_time_limit,
            selective_evaluate=",".join(sorted(task_ids)),
        )

    if baseline is not None:
        _print_delta(samples, baseline.resolve())


if __name__ == "__main__":
    typer.run(main)
