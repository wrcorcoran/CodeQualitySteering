# BigCodeBench Evaluation

This repo includes a small pipeline for testing base and steered models on BigCodeBench, then reporting `pass@1`, `pass@5`, and code-quality metrics.

## Files

- `scripts/10_run_bigcodebench.py`  
  Main runner. It calls generation, evaluation, quality reporting, and plotting.

- `scripts/06_generate_bigcodebench.py`  
  Loads a model and generates BigCodeBench samples. Supports optional steering.

- `scripts/07_evaluate_bigcodebench.py`  
  Runs the official BigCodeBench evaluator. Local execution is the default; Gradio is optional.

- `scripts/08_report_bigcodebench_quality.py`  
  Computes code-quality metrics for generated samples: `cc`, `mi`, `comment_ratio`, `h_volume`, `h_difficulty`, `h_effort`, and `sloc`.

- `scripts/09_plot_bigcodebench.py`  
  Makes plots for pass rates, quality metrics by pass/fail, and Spearman correlations.

- `src/eval/bigcodebench.py`  
  Shared helpers for loading tasks, writing samples, and computing quality metrics.

- `src/eval/steering.py`  
  Context-managed activation steering hook.

- `scripts/run_bigcodebench_base_example.sh`  
  Example smoke-test script for a base model.

- `scripts/run_bigcodebench_steered_example.sh`  
  Example smoke-test script for a steered model.

Both example scripts include:

```bash
generation.limit=2
```

Remove that line to run the full hard subset.


## Main Commands

Base run:

```bash
python scripts/10_run_bigcodebench.py \
  model.config=configs/models/qwen2.5_coder_14b.yaml \
  run.name=qwen_base
```

Steered run:

```bash
python scripts/10_run_bigcodebench.py \
  model.config=configs/models/qwen2.5_coder_14b.yaml \
  run.name=qwen_cc_l16_alpha1 \
  steering.vector=steering_vectors/qwen2.5_coder_14b/vectors/layer_16_mean_cc.npy \
  steering.layer=16 \
  steering.alpha=1.0
```

Dry run, which prints commands without executing them:

```bash
python scripts/10_run_bigcodebench.py \
  model.config=configs/models/qwen2.5_coder_14b.yaml \
  run.name=debug \
  generation.limit=2 \
  dry_run=true
```

## Useful Arguments

- `model.config=...` model YAML file
- `run.name=...` output run name
- `subset=hard` or `subset=full`
- `generation.limit=2` small smoke-test task limit
- `generation.n_samples=10` samples per task
- `generation.temperature=0.8` sampling temperature
- `generation.max_new_tokens=1280` max generated tokens
- `steering.vector=...` path to `.npy` steering vector
- `steering.layer=16` 1-indexed layer to steer
- `steering.alpha=1.0` steering strength
- `eval.gradio=true` use Gradio remote evaluation instead of local execution
- `dry_run=true` print commands only

## Outputs

- `bcb_results/` generated samples and official BigCodeBench outputs
- `bcb_reports/` quality metrics and summary JSON
- `figures/bigcodebench/` plots
