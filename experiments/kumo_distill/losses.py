"""Hard-label and output-distillation objectives."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DistillationLoss:
    """Components of a distillation objective."""

    total: Tensor
    hard: Tensor
    soft: Tensor


def classification_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    *,
    alpha: float = 0.7,
    temperature: float = 2.0,
) -> DistillationLoss:
    """Mix classification cross entropy with temperature-scaled KL."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have equal shapes")

    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    hard = F.cross_entropy(student_logits.flatten(0, 1), targets.flatten())
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    soft = (
        F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="batchmean",
        )
        * temperature**2
    )
    total = (1 - alpha) * hard + alpha * soft
    return DistillationLoss(total=total, hard=hard, soft=soft)


def regression_distillation_loss(
    student_quantiles: Tensor,
    teacher_quantiles: Tensor,
    targets: Tensor,
    *,
    alpha: float = 0.5,
) -> DistillationLoss:
    """Mix quantile pinball loss with teacher quantile SmoothL1."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if student_quantiles.shape != teacher_quantiles.shape:
        raise ValueError(
            "teacher and student quantiles must have equal shapes"
        )

    student_quantiles = student_quantiles.float()
    teacher_quantiles = teacher_quantiles.detach().float()
    levels = torch.linspace(
        0.001,
        0.999,
        student_quantiles.size(-1),
        device=student_quantiles.device,
        dtype=student_quantiles.dtype,
    )
    diff = targets.float().unsqueeze(-1) - student_quantiles
    hard = torch.maximum(levels * diff, (levels - 1) * diff).mean()
    soft = F.smooth_l1_loss(student_quantiles, teacher_quantiles)
    total = (1 - alpha) * hard + alpha * soft
    return DistillationLoss(total=total, hard=hard, soft=soft)
