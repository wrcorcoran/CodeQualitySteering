from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def get_transformer_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for first, second in candidates:
        parent = getattr(model, first, None)
        layers = getattr(parent, second, None) if parent is not None else None
        if layers is not None:
            return layers
    raise ValueError("Could not find transformer layers on model")


@contextmanager
def steering_hook(model: nn.Module, layer: int, vector: torch.Tensor, alpha: float) -> Iterator[None]:
    layers = get_transformer_layers(model)
    if layer < 1 or layer > len(layers):
        raise ValueError(f"layer must be in [1, {len(layers)}], got {layer}")

    def hook_fn(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            return output

        steer = vector.to(device=hidden.device, dtype=hidden.dtype)
        if steer.ndim != 1 or steer.shape[0] != hidden.shape[-1]:
            raise ValueError(f"Steering vector shape {tuple(steer.shape)} does not match hidden size {hidden.shape[-1]}")

        modified = hidden.clone()
        modified[:, -1, :] = modified[:, -1, :] + alpha * steer
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = layers[layer - 1].register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
