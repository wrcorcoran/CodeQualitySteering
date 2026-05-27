import json
import re
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "bcb_results_clean"

MODELS = ["llama", "gemma", "qwen"]
METRICS = ["comment_ratio", "mi"]
METRIC_LABELS = {"comment_ratio": "Comment Ratio", "mi": "Maint. Index"}
METRIC_DELTA_KEY = {"comment_ratio": "comment_ratio", "mi": "mi"}
MODEL_LABELS = {"llama": "Llama", "gemma": "Gemma", "qwen": "Qwen"}


def parse_filename(name: str):
    """
    Return (pass_k, model, metric, alpha) or None.
    Base files (alpha=0) return metric=None; handled separately in load_all.
    """
    # steered run
    m = re.match(
        r"pass(\d)_full_(llama|gemma|qwen)_(comment_ratio|mi)_a([+-]?\d+\.?\d*)_p10_summary\.json",
        name,
    )
    if m:
        return int(m.group(1)), m.group(2), m.group(3), float(m.group(4))

    # base run
    m = re.match(r"pass(\d)_full_(llama|gemma|qwen)_base_p10_summary\.json", name)
    if m:
        return int(m.group(1)), m.group(2), None, 0.0

    return None


def load_all():
    """Returns dict: pass_k -> model -> metric -> alpha -> {pass, mean_delta, std_delta}"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in RESULTS_DIR.glob("*_summary.json"):
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        pass_k, model, metric, alpha = parsed
        with open(path) as f:
            d = json.load(f)
        pass_key = f"pass@{pass_k}"
        pass_val = d["pass_at_k"].get(pass_key)

        if metric is None:
            # base run: insert as alpha=0 for every metric with delta=0
            for met in METRICS:
                data[pass_k][model][met][0.0] = {
                    "pass": pass_val,
                    "mean_delta": 0.0,
                    "std_delta": 0.0,
                }
        else:
            delta_key = METRIC_DELTA_KEY[metric]
            deltas = d.get("metric_deltas", {}).get(delta_key, {})
            data[pass_k][model][metric][alpha] = {
                "pass": pass_val,
                "mean_delta": deltas.get("mean_pct_delta"),
                "std_delta": deltas.get("std_pct_delta"),
            }
    return data


def build_alpha_index(data_for_k):
    """
    Returns per-model sorted alpha lists (all same length).
    Verify they're the same length across (model, metric) combos.
    """
    alpha_map = {}  # model -> sorted list of alphas
    for model in MODELS:
        alphas_per_metric = []
        for metric in METRICS:
            alphas_per_metric.append(sorted(data_for_k[model][metric].keys()))
        # all metrics for a model share the same alphas
        assert all(a == alphas_per_metric[0] for a in alphas_per_metric), (
            f"Alpha mismatch within {model}: {alphas_per_metric}"
        )
        alpha_map[model] = alphas_per_metric[0]
    return alpha_map


def fmt_cell(mean_delta, std_delta, pass_val):
    """Format one (Δ% ± std, pass) pair."""
    if mean_delta is None or pass_val is None:
        return "   —       —   "
    delta_str = f"{mean_delta:+6.1f}±{std_delta:5.1f}"
    pass_str = f"{pass_val:.3f}"
    return f"{delta_str} {pass_str}"


def print_table(pass_k: int, data_for_k: dict, alpha_map: dict):
    # Determine number of alpha steps (same for all models)
    n_alpha = len(next(iter(alpha_map.values())))

    # ---- column widths ----
    CELL_W = 18   # "±999.9±999.9 0.999" ~ 18 chars
    MODEL_COL = CELL_W * 2 + 3  # two metrics + separator

    # ---- header rows ----
    # Row 1: model groups
    model_header = " " * 6 + " | ".join(
        f"{MODEL_LABELS[m]:^{MODEL_COL}}" for m in MODELS
    )

    # Row 2: metric sub-headers
    def metric_sub(model):
        parts = []
        for metric in METRICS:
            parts.append(f"{METRIC_LABELS[metric]:^{CELL_W}}")
        return " | ".join(parts)

    metric_header = " " * 6 + " || ".join(metric_sub(m) for m in MODELS)

    # Row 3: column detail
    def detail_sub(model):
        parts = []
        for metric in METRICS:
            short = "CR" if metric == "comment_ratio" else "MI"
            parts.append(f"{'Δ'+short+'% ± std':>12} {'P@'+str(pass_k):>5}")
        return "  | ".join(parts)

    detail_header = "alpha  | " + "  || ".join(detail_sub(m) for m in MODELS)

    sep = "-" * len(detail_header)

    print(f"\n{'='*len(detail_header)}")
    print(f"  pass@{pass_k} results")
    print(f"{'='*len(detail_header)}")
    print(model_header)
    print(metric_header)
    print(sep)
    print(detail_header)
    print(sep)

    # ---- data rows ----
    for i in range(n_alpha):
        row_label = f"α{i+1:<4}"
        cells = []
        for model in MODELS:
            model_cells = []
            for metric in METRICS:
                alpha = alpha_map[model][i]
                entry = data_for_k[model][metric].get(alpha, {})
                model_cells.append(fmt_cell(
                    entry.get("mean_delta"),
                    entry.get("std_delta"),
                    entry.get("pass"),
                ))
            cells.append("  | ".join(model_cells))
        print(f"{row_label} | " + "  || ".join(cells))

    print(sep)

    # ---- alpha legend ----
    print(f"\nAlpha legend (α₁ = most negative, α{n_alpha} = most positive):")
    legend_header = f"{'':6}" + "".join(f"  {MODEL_LABELS[m]:<10}" for m in MODELS)
    print(legend_header)
    for i in range(n_alpha):
        row = f"α{i+1:<5}"
        for model in MODELS:
            row += f"  {alpha_map[model][i]:<10.4g}"
        print(row)


def main():
    data = load_all()
    for pass_k in sorted(data.keys()):
        alpha_map = build_alpha_index(data[pass_k])
        print_table(pass_k, data[pass_k], alpha_map)


if __name__ == "__main__":
    main()
