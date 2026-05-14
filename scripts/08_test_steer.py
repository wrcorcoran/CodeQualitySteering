"""
Toy smoke test for steered generation.

Runs baseline + one CC-steering config on two simple tasks using Llama.
Prints results to stdout. Expects steering vectors to exist at data/steering_vectors/.

Usage:
    uv run python scripts/09_test_steer.py
    uv run python scripts/09_test_steer.py --model gemma2_9b --layer 20 --alpha 30
"""
import json
from pathlib import Path

import typer

from src.steer.generate import run_sweep

TASKS = [
    {
        "task_id": "toy/0",
        "instruct_prompt": (
            "Write a Python function called `task_func` that takes a list of integers "
            "and returns the sum of all even numbers."
        ),
    },
    {
        "task_id": "toy/1",
        "instruct_prompt": (
            "Write a Python function called `task_func` that takes a string and "
            "returns the number of vowels in it."
        ),
    },
]


def main(
    model: str = typer.Option("llama3.1_8b"),
    layer: int = typer.Option(10),
    alpha: float = typer.Option(0.1),
    p: int = typer.Option(25),
) -> None:
    steering_configs = [
        [],                          # baseline
        [("comment_ratio", layer, alpha)],      # steer toward low CC
        [("comment_ratio", layer, -alpha)],     # steer toward high CC
    ]

    results = run_sweep(
        model_names=[model],
        tasks=TASKS,
        steering_configs=steering_configs,
        p=p,
        max_new_tokens=256,
        batch_size=2,
    )

    for r in results:
        steer_label = r["steerings"] or "baseline"
        print(f"\n=== task={r['task_id']}  steer={steer_label} ===")
        print(r["code"])
        print(f"metrics: {json.dumps(r['metrics'], indent=2)}")


if __name__ == "__main__":
    typer.run(main)