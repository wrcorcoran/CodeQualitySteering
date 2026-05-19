import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from bigcodebench.data import get_bigcodebench
from src.data.metrics import CodeMetrics
METRICS = ["cc", "mi", "comment_ratio", "h_volume", "h_difficulty", "h_effort", "sloc"]


def load_bigcodebench_tasks(
    split: str = "instruct",
    subset: str = "hard",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    assert split in {"instruct", "complete"}, f"Unsupported split: {split}"
    assert subset in {"hard", "full"}, f"Unsupported subset: {subset}"

    

    problems = get_bigcodebench(subset=subset)
    tasks = [problems[k] for k in sorted(problems)]

    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def task_prompt(task: dict[str, Any], split: str) -> str:
    key = "instruct_prompt" if split == "instruct" else "complete_prompt"
    prompt = task.get(key)
    if not prompt:
        raise KeyError(f"Task {task.get('task_id', '<unknown>')} has no {key}")
    return str(prompt)


def make_output_stem(model_name: str, split: str, subset: str, temperature: float, n_samples: int, steered: bool) -> str:
    benchmark = "bigcodebench-hard" if subset == "hard" else "bigcodebench"
    mode = "steered" if steered else "base"
    return f"{model_name}--{benchmark}-{split}--repo-{mode}-{temperature:g}-{n_samples}"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]



def compute_quality_metrics(samples_path: Path, eval_results_path: Path | None = None) -> pd.DataFrame:
    rows = read_jsonl(samples_path)
    pass_lookup = _load_pass_fail(eval_results_path) if eval_results_path else {}
    metric_rows: list[dict[str, Any]] = []
    seen_by_task: dict[str, int] = {}

    for row in rows:
        task_id = row["task_id"]
        generation_id = row.get("generation_id")
        if generation_id is None:
            generation_id = seen_by_task.get(task_id, 0)
            seen_by_task[task_id] = int(generation_id) + 1
        generation_id = int(generation_id)
        source = str(row.get("solution") or row.get("raw_solution") or "")
        item = {
            "task_id": task_id,
            "generation_id": generation_id,
            "passed": pass_lookup.get((task_id, generation_id)),
        }
        try:
            metrics = CodeMetrics(source)
            for metric in METRICS:
                item[metric] = getattr(metrics, metric)
            item["metrics_error"] = None
        except Exception as exc:
            for metric in METRICS:
                item[metric] = np.nan
            item["metrics_error"] = type(exc).__name__
        metric_rows.append(item)

    return pd.DataFrame(metric_rows)


def _load_pass_fail(eval_results_path: Path) -> dict[tuple[str, int], bool]:
    data = json.loads(eval_results_path.read_text())
    eval_by_task = data.get("eval", data)
    lookup: dict[tuple[str, int], bool] = {}
    for task_id, results in eval_by_task.items():
        if not isinstance(results, list):
            continue
        for generation_id, result in enumerate(results):
            status = str(result.get("status", "")).lower()
            lookup[(task_id, generation_id)] = status == "pass"
    return lookup


def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    if "passed" not in df or not df["passed"].notna().any():
        return df[METRICS].agg(["count", "mean", "median", "std"]).reset_index(names="stat")

    frames = []
    for passed, group in df.groupby("passed", dropna=False):
        summary = group[METRICS].agg(["count", "mean", "median", "std"]).reset_index(names="stat")
        summary.insert(0, "passed", passed)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)
