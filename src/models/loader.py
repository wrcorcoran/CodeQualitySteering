from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class ModelConfig(BaseModel):
    model_name: str
    hf_id: str
    n_layers: int
    d_model: int | None = None
    batch_size: int = 8
    load_in_8bit: bool = False


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    cfg: ModelConfig


def load_model(cfg: ModelConfig) -> LoadedModel:
    load_kwargs: dict[str, Any] = dict(
        dtype=torch.bfloat16,  # type: ignore[attr-defined]
        attn_implementation="sdpa",
        device_map="cuda:0",
        use_cache=False,
    )

    if cfg.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **load_kwargs)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.d_model is None:
        cfg = cfg.model_copy(update={"d_model": model.config.hidden_size})

    return LoadedModel(model=model, tokenizer=tokenizer, cfg=cfg)
