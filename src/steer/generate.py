import re
import time
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
    temperature: float = 0.0,
) -> list[dict]:
    model = loaded.model
    tokenizer = loaded.tokenizer
    model_name = loaded.cfg.model_name

    svs = [
        (_build_steering_vector(model_name, metric, layer, sv_dir, p), alpha)
        for metric, layer, alpha in steerings
    ]

    steer_meta = [{"metric": m, "layer": layer, "alpha": a} for m, layer, a in steerings]
    operator = _generation_only()
    do_sample = temperature > 0.0
    model.generation_config.use_cache = True
    logger.info(
        f"use_cache={model.config.use_cache} | "
        f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()} | "
        f"mem_eff_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()} | "
        f"math_sdp={torch.backends.cuda.math_sdp_enabled()}"
    )

    # Pre-tokenize and sort by prompt length to minimize padding waste per batch
    all_prompts = []
    for task in tasks:
        messages = [{"role": "user", "content": task["instruct_prompt"]}]
        all_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    prompt_lengths = [len(tokenizer.encode(p)) for p in all_prompts]
    sorted_indices = sorted(range(len(tasks)), key=lambda i: prompt_lengths[i])
    tasks = [tasks[i] for i in sorted_indices]
    all_prompts = [all_prompts[i] for i in sorted_indices]

    results = []
    n_batches = (len(tasks) + batch_size - 1) // batch_size

    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]
        prompts = all_prompts[batch_start : batch_start + batch_size]
        batch_idx = batch_start // batch_size
        t0 = time.perf_counter()

        t_tok = time.perf_counter()
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]  # type: ignore[union-attr]
        t_tok = time.perf_counter() - t_tok

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,

        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None

        handles: list[SteeringPatchHandle] = [
            sv.patch_activations(model, multiplier=alpha, operator=operator)
            for sv, alpha in svs
        ]
        t_gen = time.perf_counter()
        try:
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)  # type: ignore[operator]
        finally:
            for h in handles:
                h.remove()
        t_gen = time.perf_counter() - t_gen

        n_new_tokens = (output_ids[:, prompt_len:] != tokenizer.pad_token_id).sum().item()
        tok_per_sec = n_new_tokens / t_gen if t_gen > 0 else 0

        for i, task in enumerate(batch):
            new_ids = output_ids[i][prompt_len:]
            raw_output: str = tokenizer.decode(new_ids, skip_special_tokens=True)
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

        t_total = time.perf_counter() - t0
        logger.info(
            f"batch {batch_idx+1}/{n_batches} | prompt_len={prompt_len} "
            f"| tok={t_tok:.2f}s gen={t_gen:.2f}s total={t_total:.2f}s "
            f"| {tok_per_sec:.0f} new_tok/s"
        )

        del inputs, output_ids
        if batch_idx % 50 == 49:
            torch.cuda.empty_cache()

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
    batch_size: int = 64,
    n_samples: int = 1,
    temperature: float = 0.0,
) -> list[dict]:
    all_results = []

    for model_name in model_names:
        cfg = ModelConfig(**read_yaml(config_dir / f"{model_name}.yaml"))
        logger.info(f"Loading {model_name}...")
        loaded = load_model(cfg)

        for config in steering_configs:
            label = config if config else "baseline"
            logger.info(f"  Running {model_name} | steering={label} | n_samples={n_samples}")
            for sample_idx in range(n_samples):
                results = generate(
                    loaded,
                    tasks,
                    config,
                    sv_dir=sv_dir,
                    p=p,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                    temperature=temperature,
                )
                for r in results:
                    r["generation_id"] = sample_idx
                all_results.extend(results)

        del loaded.model
        torch.cuda.empty_cache()
        logger.info(f"Unloaded {model_name}")

    return all_results
