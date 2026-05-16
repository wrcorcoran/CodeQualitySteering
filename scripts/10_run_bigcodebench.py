#!/usr/bin/env python3
import json
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class BigCodeBenchRunConfig:
    model_config: Path
    run_name: str | None = None
    python: str = sys.executable

    split: str = "instruct"
    subset: str = "hard"
    n_samples: int = 10
    temperature: float = 0.8
    max_new_tokens: int = 1280
    limit: int | None = None
    parallel: int = 8

    steering_vector: Path | None = None
    steering_layer: int | None = None
    steering_alpha: float = 1.0

    gradio: bool = False
    skip_sanitize: bool = False
    dry_run: bool = False

    results_dir: Path = Path("bcb_results")
    reports_dir: Path = Path("bcb_reports")
    figures_dir: Path = Path("figures/bigcodebench")


def main(argv: list[str] | None = None) -> None:
    cfg = parse_overrides(sys.argv[1:] if argv is None else argv)
    run_bigcodebench(cfg)


def run_bigcodebench(cfg: BigCodeBenchRunConfig) -> None:
    run_name = cfg.run_name or cfg.model_config.with_suffix("").name
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    samples = cfg.results_dir / f"{run_name}.jsonl"

    print(f"=== Generate BigCodeBench samples: {run_name} ===", flush=True)
    generate_cmd = [
        cfg.python,
        "scripts/06_generate_bigcodebench.py",
        str(cfg.model_config),
        "--output-dir", str(cfg.results_dir),
        "--run-name", run_name,
        "--split", cfg.split,
        "--subset", cfg.subset,
        "--n-samples", str(cfg.n_samples),
        "--temperature", str(cfg.temperature),
        "--max-new-tokens", str(cfg.max_new_tokens),
    ]
    if cfg.limit is not None:
        generate_cmd += ["--limit", str(cfg.limit)]
    if cfg.steering_vector is not None:
        if cfg.steering_layer is None:
            raise ValueError("steering_layer is required when steering_vector is set")
        generate_cmd += [
            "--steering-vector", str(cfg.steering_vector),
            "--steering-layer", str(cfg.steering_layer),
            "--steering-alpha", str(cfg.steering_alpha),
        ]
    run_cmd(generate_cmd, dry_run=cfg.dry_run)
    print(f"Samples: {samples}", flush=True)

    print("=== Evaluate BigCodeBench samples ===", flush=True)
    evaluate_cmd = [
        cfg.python,
        "scripts/07_evaluate_bigcodebench.py",
        str(samples),
        "--split", cfg.split,
        "--subset", cfg.subset,
        "--parallel", str(cfg.parallel),
    ]
    if cfg.gradio:
        evaluate_cmd.append("--gradio")
    if cfg.skip_sanitize:
        evaluate_cmd.append("--skip-sanitize")
    run_cmd(evaluate_cmd, dry_run=cfg.dry_run)

    report_samples = samples.with_name(f"{samples.stem}-sanitized-calibrated{samples.suffix}")
    if not report_samples.exists():
        report_samples = samples

    eval_results = None if cfg.dry_run else newest_result("*eval_results.json", newer_than=samples)
    pass_at_k = None if cfg.dry_run else newest_result("*pass_at_k.json", newer_than=samples)

    print("=== Compute code-quality metrics ===", flush=True)
    report_cmd = [
        cfg.python,
        "scripts/08_report_bigcodebench_quality.py",
        str(report_samples),
        "--output-dir", str(cfg.reports_dir),
        "--run-name", run_name,
    ]
    if eval_results is not None:
        report_cmd += ["--eval-results", str(eval_results)]
    if pass_at_k is not None:
        report_cmd += ["--pass-at-k", str(pass_at_k)]
    run_cmd(report_cmd, dry_run=cfg.dry_run)

    print("=== Plot reports ===", flush=True)
    run_cmd([
        cfg.python,
        "scripts/09_plot_bigcodebench.py",
        str(cfg.reports_dir),
        "--figures-dir", str(cfg.figures_dir),
    ], dry_run=cfg.dry_run)

    print("\nDone.")
    print(f"  samples: {samples}")
    print(f"  eval results: {eval_results or 'not found'}")
    print(f"  pass@k: {pass_at_k or 'not found'}")
    if pass_at_k is not None:
        print_pass_at_k(pass_at_k)
    print(f"  reports: {cfg.reports_dir}")
    print(f"  figures: {cfg.figures_dir}")


def parse_overrides(argv: list[str]) -> BigCodeBenchRunConfig:
    argv = ["dry_run=true" if arg == "--dry-run" else arg for arg in argv]
    if not argv:
        raise SystemExit(
            "Usage: python scripts/10_run_bigcodebench.py "
            "model_config=configs/models/qwen2.5_coder_14b.yaml [run_name=qwen_base] [key=value ...]"
        )

    values: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(BigCodeBenchRunConfig)}

    # Compatibility with positional style:
    #   python scripts/10_run_bigcodebench.py configs/models/qwen.yaml qwen_base
    positional = [arg for arg in argv if "=" not in arg]
    overrides = [arg for arg in argv if "=" in arg]
    if positional:
        values["model_config"] = positional[0]
    if len(positional) > 1:
        values["run_name"] = positional[1]
    if len(positional) > 2:
        raise SystemExit(f"Unexpected positional arguments: {positional[2:]}")

    for arg in overrides:
        key, raw_value = arg.split("=", 1)
        key = normalize_key(key)
        if key not in field_map:
            raise SystemExit(f"Unknown override {key!r}. Valid keys: {', '.join(field_map)}")
        values[key] = raw_value

    if "model_config" not in values:
        raise SystemExit("Missing required override: model_config=<path>")

    casted = {}
    for field in fields(BigCodeBenchRunConfig):
        if field.name in values:
            casted[field.name] = cast_value(field.name, values[field.name], field.default)
    return BigCodeBenchRunConfig(**casted)


def normalize_key(key: str) -> str:
    key = key.replace("-", "_")
    aliases = {
        "model.config": "model_config",
        "model_config": "model_config",
        "run.name": "run_name",
        "generation.n_samples": "n_samples",
        "generation.temperature": "temperature",
        "generation.max_new_tokens": "max_new_tokens",
        "generation.limit": "limit",
        "eval.parallel": "parallel",
        "eval.gradio": "gradio",
        "eval.skip_sanitize": "skip_sanitize",
        "steering.vector": "steering_vector",
        "steering.layer": "steering_layer",
        "steering.alpha": "steering_alpha",
        "paths.results_dir": "results_dir",
        "paths.reports_dir": "reports_dir",
        "paths.figures_dir": "figures_dir",
    }
    return aliases.get(key, key.replace(".", "_"))


def cast_value(name: str, value: str, default: Any) -> Any:
    path_fields = {"model_config", "steering_vector", "results_dir", "reports_dir", "figures_dir"}
    int_fields = {"n_samples", "max_new_tokens", "limit", "parallel", "steering_layer"}
    float_fields = {"temperature", "steering_alpha"}
    bool_fields = {"gradio", "skip_sanitize", "dry_run"}

    if name in path_fields:
        return None if value.lower() in {"none", "null", ""} else Path(value)
    if name in int_fields:
        return None if value.lower() in {"none", "null", ""} else int(value)
    if name in float_fields:
        return float(value)
    if name in bool_fields:
        return value.lower() in {"1", "true", "yes", "y", "on"}
    if default is None:
        if value.lower() in {"none", "null", ""}:
            return None
        return infer_optional_value(value)
    if isinstance(default, bool):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, Path):
        return Path(value)
    return value


def infer_optional_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return Path(value) if "/" in value or value.endswith(".npy") or value.endswith(".yaml") else value


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def newest_result(pattern: str, newer_than: Path, root: Path = Path("bcb_results")) -> Path | None:
    if not root.exists() or not newer_than.exists():
        return None
    candidates = [p for p in root.rglob(pattern) if p.stat().st_mtime >= newer_than.stat().st_mtime]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def print_pass_at_k(path: Path) -> None:
    data = json.loads(path.read_text())
    pass_values = {k: v for k, v in data.items() if str(k).startswith("pass@")}
    if pass_values:
        formatted = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in pass_values.items())
        print(f"  {formatted}")


if __name__ == "__main__":
    main()
