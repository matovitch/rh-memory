"""Flow-matching losses."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def interpolate_linear(start: Tensor, end: Tensor, t: Tensor) -> Tensor:
    view_t = t.view(t.shape[0], 1, 1)
    return (1.0 - view_t) * start + view_t * end


def flow_matching_loss(model, start: Tensor, end: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x_t = interpolate_linear(start, end, t)
    target_velocity = end - start
    pred_velocity = model(x_t, t)
    loss = F.mse_loss(pred_velocity, target_velocity)
    return loss, pred_velocity, target_velocity