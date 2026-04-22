from dataclasses import dataclass

import torch
from packaging.version import Version
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    __version__ as transformers_version,
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
    load_kwargs = dict(
        dtype=torch.bfloat16,  # type: ignore[attr-defined]
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        use_cache=False,
    )

    if cfg.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    if "gemma-4" in cfg.hf_id.lower():
        _check_gemma4_support()
        tokenizer = AutoProcessor.from_pretrained(cfg.hf_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)

    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **load_kwargs)
    model.eval()

    tokenizer.padding_side = "left"

    if cfg.d_model is None:
        cfg = cfg.model_copy(update={"d_model": model.config.hidden_size})

    return LoadedModel(model=model, tokenizer=tokenizer, cfg=cfg)


def _check_gemma4_support() -> None:
    if Version(transformers_version) < Version("4.45"):
        raise RuntimeError(
            f"transformers {transformers_version} does not support the gemma4 architecture. "
            "Please upgrade: uv add 'transformers>=4.45'"
        )
