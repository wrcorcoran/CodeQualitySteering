import torch
from torch import Tensor


def pool_last(hidden: Tensor, mask: Tensor) -> Tensor:
    # hidden: [B, T, D], mask: [B, T]
    # left-padding guarantees the last non-pad token is always at position T-1
    return hidden[:, -1, :]


def pool_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    # hidden: [B, T, D], mask: [B, T]
    # zero out pad positions, divide by real token count per sequence
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1)
