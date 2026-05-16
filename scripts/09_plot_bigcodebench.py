#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.bigcodebench import METRICS
from src.utils.logging import setup_logging


def _load_run_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    row = {"run": path.stem.replace("_summary", "")}
    pass_at_k = data.get("pass_at_k", {})
    for key, value in pass_at_k.items():
        if str(key).startswith("pass@"):
            row[str(key)] = value
    row["quality_path"] = data.get("quality_path")
    return row


def main(
    report_dir: Path = typer.Argument(..., help="Directory containing *_summary.json and *_quality.parquet files"),
    figures_dir: Path = typer.Option(Path("figures/bigcodebench"), help="Output figure directory"),
) -> None:
    setup_logging()
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = sorted(report_dir.glob("*_summary.json"))
    if summaries:
        runs = pd.DataFrame([_load_run_summary(p) for p in summaries])
        pass_cols = [c for c in runs.columns if c.startswith("pass@")]
        if pass_cols:
            ax = runs.set_index("run")[pass_cols].plot(kind="bar", figsize=(10, 5))
            ax.set_ylabel("pass@k")
            ax.set_xlabel("")
            ax.set_title("BigCodeBench pass@k")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(figures_dir / "pass_at_k.png", dpi=150)
            plt.close()

    quality_files = sorted(report_dir.glob("*_quality.parquet"))
    frames = []
    for path in quality_files:
        df = pd.read_parquet(path)
        df["run"] = path.stem.replace("_quality", "")
        frames.append(df)
    if not frames:
        return

    quality = pd.concat(frames, ignore_index=True)
    for metric in METRICS:
        if metric not in quality:
            continue
        ax = quality.boxplot(column=metric, by="passed", figsize=(7, 5))
        ax.set_title(f"{metric} by pass/fail")
        ax.set_xlabel("passed")
        ax.set_ylabel(metric)
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{metric}_by_passed.png", dpi=150)
        plt.close()

    corr_cols = [m for m in METRICS if m in quality]
    if "passed" in quality and quality["passed"].notna().any():
        corr_df = quality[corr_cols + ["passed"]].copy()
        corr_df["passed"] = corr_df["passed"].astype(float)
        corr = corr_df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Spearman correlation")
        plt.tight_layout()
        plt.savefig(figures_dir / "spearman_quality_passed.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    typer.run(main)
