"""Output distillation experiments for Kumo Tabular."""

from experiments.kumo_distill.losses import (
    DistillationLoss,
    classification_distillation_loss,
    regression_distillation_loss,
)
from experiments.kumo_distill.tasks import (
    EpisodeBatch,
    RandomSCMTaskSource,
    TaskShape,
    TaskSource,
)

__all__ = [
    "DistillationLoss",
    "EpisodeBatch",
    "RandomSCMTaskSource",
    "TaskShape",
    "TaskSource",
    "classification_distillation_loss",
    "regression_distillation_loss",
]
