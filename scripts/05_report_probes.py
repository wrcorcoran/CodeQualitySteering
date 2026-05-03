#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import typer

from src.probes.storage import load_results, load_weights
from src.probes.sweep import METRICS, POOLS

plt.style.use(["science", "no-latex"])


def _plot_metric(
    df_pool: pd.DataFrame,
    probe_dir: Path,
    pool: str,
    metric: str,
    figures_dir: Path,
) -> None:
    rows = df_pool[df_pool["metric"] == metric].sort_values("layer")
    layers = rows["layer"].to_numpy()
    r2 = rows["r2_test"].to_numpy()
    norms = np.array([
        float(np.linalg.norm(load_weights(probe_dir, int(l), pool, metric)))
        for l in layers
    ])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(layers, r2, color="steelblue", marker="o", markersize=3, label="R² test")
    ax2.plot(layers, norms, color="tomato", marker="s", markersize=3, linestyle="--", label="weight norm")

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("R² test", color="steelblue")
    ax2.set_ylabel("Weight L2 norm", color="tomato")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax1.set_title(f"{probe_dir.name} · {pool} · {metric}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=True)

    out_path = figures_dir / probe_dir.name / pool / f"{metric}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_all_metrics(
    df_pool: pd.DataFrame,
    probe_dir: Path,
    pool: str,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, metric in enumerate(METRICS):
        rows = df_pool[df_pool["metric"] == metric].sort_values("layer")
        layers = rows["layer"].to_numpy()
        r2 = rows["r2_test"].to_numpy()
        ax.plot(layers, r2, marker="o", markersize=3, color=colors[i], label=metric)

    ax.set_xlabel("Layer")
    ax.set_ylabel("R² test")
    ax.set_title(f"{probe_dir.name} · {pool} · all metrics")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(METRICS),
              fontsize=8, frameon=True)

    out_path = figures_dir / probe_dir.name / pool / "all_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_comparison(probes_root: Path) -> None:
    frames = []
    for model_dir in sorted(probes_root.iterdir()):
        if not (model_dir / "meta.json").exists():
            continue
        df = load_results(model_dir)
        df["model"] = model_dir.name
        frames.append(df)

    if len(frames) < 2:
        return

    all_df = pd.concat(frames, ignore_index=True)
    best = all_df.groupby(["model", "pool", "metric"])["r2_test"].max().reset_index()

    for pool in POOLS:
        subset = best[best["pool"] == pool]
        if subset.empty:
            continue

        print(f"\n{'='*60}")
        print(f"Cross-model comparison — pool={pool}")
        print(f"{'='*60}\n")
        pivot = subset.pivot(index="model", columns="metric", values="r2_test")[METRICS]
        print(pivot.to_string())

        print(f"\nWinner per metric:\n")
        for metric in METRICS:
            col = subset[subset["metric"] == metric].sort_values("r2_test", ascending=False)
            winner = col.iloc[0]
            model_df = all_df[
                (all_df["model"] == winner["model"]) &
                (all_df["pool"] == pool) &
                (all_df["metric"] == metric)
            ]
            best_layer = int(model_df.sort_values("r2_test", ascending=False).iloc[0]["layer"])
            print(f"  {metric:<16}  {winner['model']:<24}  R²={winner['r2_test']:.3f}  (layer={best_layer})")


def main(
    probe_dir: Path = typer.Argument(..., help="Path to probe output dir, e.g. probes/llama3.1_8b"),
    figures_dir: Path = typer.Option(Path("figures"), help="Root directory for saved plots"),
    compare: bool = typer.Option(False, "--compare", help="Print cross-model comparison using sibling dirs"),
) -> None:
    df = load_results(probe_dir)

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 140)

    for pool in POOLS:
        df_pool = df[df["pool"] == pool].copy()
        if df_pool.empty:
            continue

        print(f"\nR² test scores — {probe_dir.name}  pool={pool}\n")
        pivot = df_pool.pivot(index="layer", columns="metric", values="r2_test")[METRICS]
        print(pivot.to_string())

        print(f"\nBest layer per metric (R² test):\n")
        for metric in METRICS:
            best = df_pool[df_pool["metric"] == metric].sort_values("r2_test", ascending=False).iloc[0]
            print(f"  {metric:<16}  layer={int(best['layer']):>2}  R²={best['r2_test']:.3f}")

        print(f"\nSaving plots to {figures_dir / probe_dir.name / pool}/")
        _plot_all_metrics(df_pool, probe_dir, pool, figures_dir)
        for metric in METRICS:
            _plot_metric(df_pool, probe_dir, pool, metric, figures_dir)

    if compare:
        _print_comparison(probe_dir.parent)


if __name__ == "__main__":
    typer.run(main)