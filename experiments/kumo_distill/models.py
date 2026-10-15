"""Pretrained Kumo teacher and student construction."""

from dataclasses import dataclass

import torch

from experiments.kumo_distill.config import ModelSize
from experiments.kumo_distill.tasks import TaskName
from sdm.models import KumoTabular


@dataclass(frozen=True)
class ModelPair:
    """Raw teacher and student networks with their public wrappers retained."""

    teacher: torch.nn.Module
    student: torch.nn.Module
    teacher_wrapper: KumoTabular
    student_wrapper: KumoTabular


def build_model_pair(
    task: TaskName,
    teacher_size: ModelSize,
    student_size: ModelSize,
    device: torch.device,
) -> ModelPair:
    """Load released checkpoints and expose aligned raw-output networks."""
    teacher_wrapper = KumoTabular(
        task=task,
        size=teacher_size,
        pretrained=True,
        device=device,
    )
    student_wrapper = KumoTabular(
        task=task,
        size=student_size,
        pretrained=True,
        device=device,
    )
    teacher = teacher_wrapper.models[task]
    student = student_wrapper.models[task]
    teacher.eval()
    teacher.requires_grad_(False)
    student.train()
    return ModelPair(
        teacher=teacher,
        student=student,
        teacher_wrapper=teacher_wrapper,
        student_wrapper=student_wrapper,
    )
