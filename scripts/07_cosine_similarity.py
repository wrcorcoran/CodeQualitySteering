"""
Plot pairwise principal angles of steering vectors across layers, gram matrix heatmaps,
and metric correlation heatmap from raw data.

Line plots: one figure per model, 21 lines (one per metric pair), x=layer, y=principal angle.
Gram matrix: 7x7 cosine similarity heatmap at a specified layer per model.
Metric correlations: 7x7 Spearman correlation heatmap of raw dataset metrics.

Usage:
    uv run python scripts/07_cosine_similarity.py
    uv run python scripts/07_cosine_similarity.py --p 10
    uv run python scripts/07_cosine_similarity.py --gram-layers gemma2_9b:40 llama3.1_8b:29 qwen2.5_coder_14b:46
"""

import itertools
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401
import typer

from src.utils.io import boundary_mask, load_metrics_dataset, load_steering_vectors

plt.style.use(["science", "no-latex"])

METRICS = ["comment_ratio", "h_volume", "h_difficulty", "h_effort", "cc", "sloc", "mi"]
METRICS_GRAM = METRICS


def load_all_svs(model_dir: Path, p: int) -> dict[str, dict[int, np.ndarray]]:
    return {m: load_steering_vectors(model_dir / f"{m}_p{p}.npz") for m in METRICS}


def normalized_gram(svs: dict[str, dict[int, np.ndarray]], layer: int) -> np.ndarray:
    vecs = np.stack([svs[m][layer] for m in METRICS])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    return vecs @ vecs.T


def compute_principal_angles(
    model_dir: Path, p: int
) -> tuple[np.ndarray, list[tuple[str, str]], list[int]]:
    svs = load_all_svs(model_dir, p)
    layers = sorted(next(iter(svs.values())).keys())
    pairs = list(itertools.combinations(METRICS, 2))
    cos_sims = np.zeros((len(pairs), len(layers)), dtype=np.float32)

    for li, layer in enumerate(layers):
        gram = normalized_gram(svs, layer)
        for pi, (ma, mb) in enumerate(pairs):
            ia, ib = METRICS.index(ma), METRICS.index(mb)
            cos_sims[pi, li] = gram[ia, ib]

    return np.degrees(np.arccos(np.clip(np.abs(cos_sims), 0.0, 1.0))), pairs, layers


def plot_principal_angles(
    model_name: str,
    angles: np.ndarray,
    pairs: list[tuple[str, str]],
    layers: list[int],
    out_path: Path,
) -> None:
    colors = {
        "cc": "#e41a1c",
        "mi": "#377eb8",
        "comment_ratio": "#4daf4a",
        "h_volume": "#984ea3",
        "h_difficulty": "#ff7f00",
        "h_effort": "#a65628",
        "sloc": "#f781bf",
    }
    linestyles = {
        "cc": "-",
        "mi": "--",
        "comment_ratio": "-.",
        "h_volume": ":",
        "h_difficulty": (0, (5, 1)),
        "h_effort": (0, (3, 1, 1, 1)),
        "sloc": (0, (1, 1)),
    }
    labels = {
        "cc": "CC",
        "mi": "MI",
        "comment_ratio": "CR",
        "h_volume": "HV",
        "h_difficulty": "HD",
        "h_effort": "HE",
        "sloc": "SLOC",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for pi, (ma, mb) in enumerate(pairs):
        ax.plot(
            layers,
            angles[pi],
            color=colors[ma],
            linestyle=linestyles[mb],
            linewidth=0.9,
        )
    ax.axhline(90, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_ylim(0, 90)
    ax.set_title(f"Principal Angles Over Layers: {model_name}")
    ax.set_xlim(layers[0], layers[-1])


    color_handles = [
        Line2D([0], [0], color=colors[m], linewidth=1.5, label=labels[m])
        for m in METRICS
    ]
    ls_handles = [
        Line2D([0], [0], color="black", linestyle=linestyles[m], linewidth=1.5, label=labels[m])
        for m in METRICS
    ]
    legend_kw = dict(ncol=7, loc="upper center", fontsize=7, title_fontsize=8,
                     frameon=True, handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    color_legend = ax.legend(
        handles=color_handles, title="Color (first metric)",
        bbox_to_anchor=(0.25, -0.12), **legend_kw,
    )
    ax.add_artist(color_legend)
    ax.legend(
        handles=ls_handles, title="Linestyle (second metric)",
        bbox_to_anchor=(0.75, -0.12), **legend_kw,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap(
    matrix: np.ndarray, title: str, colorbar_label: str, out_path: Path
) -> None:
    labels = {
        "h_volume": "HV",
        "h_difficulty": "HD",
        "h_effort": "HE",
        "cc": "CC",
        "mi": "MI",
        "comment_ratio": "CR",
        "sloc": "SLOC",
    }
    tick_labels = [labels[m] for m in METRICS_GRAM]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_xticks(range(len(METRICS_GRAM)))
    ax.set_yticks(range(len(METRICS_GRAM)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title(title, size=9)
    for i in range(len(METRICS_GRAM)):
        for j in range(len(METRICS_GRAM)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(matrix[i, j]) > 0.6 else "black",
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_metric_correlations(dataset_dir: Path, out_path: Path) -> None:
    df = load_metrics_dataset(dataset_dir, cols=METRICS_GRAM)
    mask = np.ones(len(df), dtype=bool)
    for m in METRICS:
        mask &= boundary_mask(df[m].to_numpy(dtype=np.float32))
    corr = df[mask].corr(method="spearman").to_numpy()
    plot_heatmap(corr, "Metric correlations", "Spearman (rho)", out_path)


def main(
    p: int,
    steering_dir: Path,
    dataset_dir: Path,
    figures_dir: Path,
    gram_layers: dict[str, int],
) -> None:
    models = sorted(d.name for d in steering_dir.iterdir() if d.is_dir())
    assert models, f"No model directories found in {steering_dir}"

    for model_name in models:
        model_dir = steering_dir / model_name
        print(f"Processing {model_name}...")

        angles, pairs, layers = compute_principal_angles(model_dir, p)
        plot_principal_angles(
            model_name,
            angles,
            pairs,
            layers,
            figures_dir / "principal_angle" / f"{model_name}_p{p}.png",
        )

        if model_name in gram_layers:
            layer = gram_layers[model_name]
            svs = load_all_svs(model_dir, p)
            out = figures_dir / "gram_matrix" / f"{model_name}_layer{layer:02d}_p{p}.png"
            plot_heatmap(normalized_gram(svs, layer), f"Steering Vector Cosine Similarity ({model_name}, Layer {layer}, p={p})", "Cosine similarity", out)

    plot_metric_correlations(dataset_dir, figures_dir / "metric_correlations.png")


def cli(
    p: int = typer.Option(10, help="Percentile cutoff used during SV extraction"),
    steering_dir: Path = typer.Option("data/steering_vectors"),
    dataset_dir: Path = typer.Option("data/processed/stackv2_python"),
    figures_dir: Path = typer.Option("figures/svs"),
    gram_layers: Optional[List[str]] = typer.Option(
        ["gemma2_9b:40", "llama3.1_8b:29", "qwen2.5_coder_14b:46"],
        help="Gram matrix layers as MODEL:LAYER, e.g. llama3.1_8b:29",
    ),
) -> None:
    parsed = {}
    for entry in gram_layers or []:
        model, layer = entry.rsplit(":", 1)
        parsed[model] = int(layer)
    main(p, steering_dir, dataset_dir, figures_dir, parsed)


if __name__ == "__main__":
    typer.run(cli)
