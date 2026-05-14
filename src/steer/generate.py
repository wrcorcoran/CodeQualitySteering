import re
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from steering_vectors import SteeringVector
from steering_vectors.steering_vector import PatchDeltaOperator, SteeringPatchHandle

from src.data.metrics import CodeMetrics
from src.models.loader import LoadedModel, ModelConfig, load_model
from src.utils.io import load_steering_vectors as load_sv_npz, read_yaml

SteerSpec = tuple[str, int, float]  # (metric, layer, alpha)
SteeringConfig = list[SteerSpec]    # one configuration = zero or more specs; [] = baseline


def _generation_only(verbose: bool = False) -> PatchDeltaOperator:
    def op(activation: torch.Tensor, steering_vec: torch.Tensor) -> torch.Tensor:
        if activation.shape[1] > 1:
            if verbose:
                print("adding nothing -- prefill", flush=True)
            return torch.zeros_like(activation)
        if verbose:
            print(f"adding {steering_vec.norm()} to {activation.norm()}", flush=True)
        return steering_vec.to(dtype=activation.dtype)
    return op


def _build_steering_vector(
    model_name: str, metric: str, layer: int, sv_dir: Path, p: int
) -> SteeringVector:
    path = sv_dir / model_name / f"{metric}_p{p}.npz"
    layer_dict = load_sv_npz(path)
    vec = torch.tensor(layer_dict[layer], dtype=torch.float32)
    return SteeringVector(layer_activations={layer: vec}, layer_type="decoder_block")


def _extract_code(raw: str) -> str:
    """Extract the first Python code block, or return the full text."""
    match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    idx = raw.find("def task_func")
    if idx != -1:
        return raw[idx:].strip()
    return raw.strip()


def _compute_metrics(code: str) -> dict | None:
    try:
        m = CodeMetrics(source=code)
        d = asdict(m)
        d.pop("source")
        return d
    except Exception:
        return None


def generate(
    loaded: LoadedModel,
    tasks: list[dict],
    steerings: list[SteerSpec],
    *,
    sv_dir: Path = Path("data/steering_vectors"),
    p: int = 25,
    max_new_tokens: int = 2048,
    batch_size: int = 8,
) -> list[dict]:
    model = loaded.model
    tokenizer = loaded.tokenizer
    model_name = loaded.cfg.model_name

    svs = [
        (_build_steering_vector(model_name, metric, layer, sv_dir, p), alpha)
        for metric, layer, alpha in steerings
    ]

    steer_meta = [{"metric": m, "layer": l, "alpha": a} for m, l, a in steerings]
    operator = _generation_only()
    results = []

    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]

        prompts = []
        for task in batch:
            messages = [{"role": "user", "content": task["instruct_prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        handles: list[SteeringPatchHandle] = [
            sv.patch_activations(model, multiplier=alpha, operator=operator)
            for sv, alpha in svs
        ]
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            for h in handles:
                h.remove()

        for i, task in enumerate(batch):
            new_ids = output_ids[i][prompt_len:]
            raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)
            code = _extract_code(raw_output)
            metrics = _compute_metrics(code)
            results.append(
                {
                    "task_id": task["task_id"],
                    "model": model_name,
                    "steerings": steer_meta,
                    "raw_output": raw_output,
                    "code": code,
                    "metrics": metrics,
                }
            )

    return results


def run_sweep(
    model_names: list[str],
    tasks: list[dict],
    steering_configs: list[SteeringConfig],
    *,
    config_dir: Path = Path("configs/models"),
    sv_dir: Path = Path("data/steering_vectors"),
    p: int = 25,
    max_new_tokens: int = 2048,
    batch_size: int = 8,
) -> list[dict]:
    all_results = []

    for model_name in model_names:
        cfg = ModelConfig(**read_yaml(config_dir / f"{model_name}.yaml"))
        logger.info(f"Loading {model_name}...")
        loaded = load_model(cfg)

        for config in steering_configs:
            label = config if config else "baseline"
            logger.info(f"  Running {model_name} | steering={label}")
            results = generate(
                loaded,
                tasks,
                config,
                sv_dir=sv_dir,
                p=p,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            all_results.extend(results)

        del loaded.model
        torch.cuda.empty_cache()
        logger.info(f"Unloaded {model_name}")

    return all_results