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
from collections import Counter
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


def _compute_deltas(samples: Path, baseline: Path) -> dict:
    base_df = _load_metrics(baseline).dropna(subset=METRICS, how="any")
    steered_df = _load_metrics(samples).dropna(subset=METRICS, how="any")

    if base_df.empty or steered_df.empty:
        return {}

    base_task = base_df.groupby("task_id")[METRICS].mean().reset_index()
    steered_task = steered_df.groupby("task_id")[METRICS].mean().reset_index()
    merged = base_task.merge(steered_task, on="task_id", suffixes=("_base", "_steered"))

    out = {}
    for m in METRICS:
        base_col = cast(pd.Series, merged[f"{m}_base"])
        steered_col = cast(pd.Series, merged[f"{m}_steered"])
        delta = steered_col - base_col
        mean_base = float(base_col.mean())
        pct_per_row = (delta / base_col.abs().replace(0, float("nan"))) * 100

        out[m] = {
            "mean_pct_delta":                  float(pct_per_row.mean()),
            "median_pct_delta":                float(pct_per_row.median()),
            "std_pct_delta":                   float(pct_per_row.std()),
            "mean_delta_over_base_mean_pct":   (float(delta.mean())   / mean_base * 100) if mean_base != 0 else None,
            "median_delta_over_base_mean_pct": (float(delta.median()) / mean_base * 100) if mean_base != 0 else None,
            "n": int(delta.notna().sum()),
        }
    return out


def _print_delta(samples: Path, baseline: Path) -> None:
    deltas = _compute_deltas(samples, baseline)
    if not deltas:
        print(f"No valid metrics to compare ({samples.stem} vs {baseline.stem})")
        return

    pd.set_option("display.float_format", "{:+.4f}".format)
    print(f"\n── Metric change vs baseline ({samples.stem} vs {baseline.stem}) ──")
    rows: dict[str, dict] = {}
    for m, d in deltas.items():
        rows[m] = {
            "mean %Δ (per-row)":    d["mean_pct_delta"],
            "median %Δ (per-row)":  d["median_pct_delta"],
            "std %Δ (per-row)":     d["std_pct_delta"],
            "mean Δ/mean(base)%":   d["mean_delta_over_base_mean_pct"],
            "median Δ/mean(base)%": d["median_delta_over_base_mean_pct"],
            "n":                    d["n"],
        }
    summary = pd.DataFrame(rows).T
    summary["n"] = summary["n"].astype(int)
    print(summary.to_string())


def _build_summary(
    samples: Path,
    eval_results_path: Path,
    pass_at_k_path: Path,
    baseline: Optional[Path],
) -> dict:
    out: dict = {"samples": samples.name, "baseline": baseline.name if baseline else None}

    if pass_at_k_path.exists():
        pak = json.loads(pass_at_k_path.read_text())
        out["pass_at_k"]    = {k: v for k, v in pak.items() if k.startswith("pass@")}
        out["gt_pass_rate"] = pak.get("gt_pass_rate")
        out["failed_tasks"] = pak.get("failed_tasks", [])

    if eval_results_path.exists():
        eval_results = json.loads(eval_results_path.read_text())
        per_task_raw = eval_results.get("eval", {})

        all_statuses: list[str] = []
        per_task: dict = {}
        for task_id, completions in per_task_raw.items():
            task_statuses = [c["status"] for c in completions]
            passed = sum(1 for s in task_statuses if s == "pass")
            per_task[task_id] = {
                "passed":   passed,
                "total":    len(completions),
                "statuses": task_statuses,
            }
            all_statuses.extend(task_statuses)

        status_counts = dict(Counter(all_statuses))
        out["num_total"]     = len(per_task)
        out["num_correct"]   = sum(1 for t in per_task.values() if t["passed"] > 0)
        out["num_invalid"]   = sum(v for k, v in status_counts.items()
                                   if any(w in k.lower() for w in ("syntax", "invalid", "empty")))
        out["status_counts"] = status_counts
        out["per_task"]      = per_task

    if baseline is not None:
        out["metric_deltas"] = _compute_deltas(samples, baseline)

    return out


def main(
    samples: Path = typer.Argument(..., help="Generated JSONL samples"),
    baseline: Optional[Path] = typer.Option(None, help="Baseline JSONL to compare metrics against"),
    subset: str = typer.Option("full", help="'hard' or 'full'"),
    pass_k: str = typer.Option("1,5", help="Comma-separated pass@k values"),
    parallel: int = typer.Option(104, help="Parallel execution workers"),
    min_time_limit: float = typer.Option(20.0, help="Min seconds per task execution"),
    skip_sanitize: bool = typer.Option(False, help="Skip sanitize/calibrate step"),
    skip_eval: bool = typer.Option(False, help="Skip pass@k evaluation (metrics comparison only)"),
) -> None:
    setup_logging()
    import os as _os
    import bigcodebench.eval as _bcb_eval
    _orig_unsafe = _bcb_eval.unsafe_execute
    def _isolated_unsafe(*args, **kwargs):
        _os.setsid()
        return _orig_unsafe(*args, **kwargs)
    _bcb_eval.unsafe_execute = _isolated_unsafe

    from bigcodebench.sanitize import script as sanitize_script
    from bigcodebench.evaluate import evaluate

    samples = samples.resolve()
    baseline_resolved = baseline.resolve() if baseline is not None else None

    if not skip_eval:
        processed_dir = samples.parent / "processed"
        processed_dir.mkdir(exist_ok=True)

        eval_samples = processed_dir / samples.name
        shutil.copy2(samples, eval_samples)

        if not skip_sanitize:
            sanitize_script(str(eval_samples), calibrate=True)
            eval_samples = processed_dir / f"{samples.stem}-sanitized-calibrated{samples.suffix}"

        eval_results_path = Path(str(eval_samples).replace(".jsonl", "_eval_results.json"))
        pass_at_k_path    = Path(str(eval_samples).replace(".jsonl", "_pass_at_k.json"))

        # Remove stale result files to avoid interactive overwrite prompts.
        for p in [eval_results_path, pass_at_k_path]:
            if p.exists():
                p.unlink()

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

        summary = _build_summary(samples, eval_results_path, pass_at_k_path, baseline_resolved)
        summary_path = samples.with_name(samples.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved → {summary_path}")

    if baseline_resolved is not None:
        _print_delta(samples, baseline_resolved)
        summary_path = samples.with_name(samples.stem + "_summary.json")
        existing = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        existing["baseline"] = baseline_resolved.name
        existing["metric_deltas"] = _compute_deltas(samples, baseline_resolved)
        summary_path.write_text(json.dumps(existing, indent=2))
        print(f"Summary updated → {summary_path}")


if __name__ == "__main__":
    typer.run(main)
