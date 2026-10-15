"""Configuration for Kumo Tabular output distillation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from experiments.kumo_distill.tasks import TaskName, TaskShape

ModelSize = Literal["small", "medium", "large"]
Precision = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class DistillConfig:
    """Serializable training configuration."""

    task: TaskName
    output_dir: str
    teacher_size: ModelSize = "large"
    student_size: ModelSize = "small"
    steps: int = 10_000
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    alpha: float | None = None
    temperature: float = 2.0
    seed: int = 0
    batch_size: int = 1
    gradient_accumulation: int = 1
    context_rows: tuple[int, ...] = (64, 128, 256)
    query_rows: tuple[int, ...] = (32, 64, 128)
    num_features: tuple[int, ...] = (4, 8, 16, 32)
    min_classes: int = 2
    max_classes: int = 10
    noise_std: float = 0.1
    max_grad_norm: float = 1.0
    precision: Precision = "auto"
    log_interval: int = 10
    checkpoint_interval: int = 500
    validation_interval: int = 500
    validation_episodes: int = 4
    validation_seed: int = 10_000
    revision: str = "v1.0.1"
    resume: str | None = None
    fixed_shape: bool = False
    _resolved_alpha: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.gradient_accumulation <= 0:
            raise ValueError(
                "steps and gradient_accumulation must be positive"
            )
        if self.validation_interval <= 0 or self.validation_episodes <= 0:
            raise ValueError("validation settings must be positive")
        if (
            not self.context_rows
            or not self.query_rows
            or not self.num_features
        ):
            raise ValueError("shape choices cannot be empty")
        alpha = 0.7 if self.task == "classification" else 0.5
        if self.alpha is not None:
            alpha = self.alpha
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one")
        object.__setattr__(self, "_resolved_alpha", alpha)

    @property
    def resolved_alpha(self) -> float:
        """Return the explicit or task-specific default KD weight."""
        return self._resolved_alpha

    @property
    def warmup_steps(self) -> int:
        """Return the smaller of 500 steps and two percent of training."""
        return min(500, max(1, round(0.02 * self.steps)))

    @property
    def output_path(self) -> Path:
        """Return the experiment output directory."""
        return Path(self.output_dir)

    def sample_shape(self, generator: torch.Generator) -> TaskShape:
        """Sample one supported episode shape from the configured choices."""

        def choose(values: tuple[int, ...]) -> int:
            if self.fixed_shape:
                return values[0]
            index = torch.randint(len(values), (), generator=generator).item()
            return values[index]

        num_classes = None
        if self.task == "classification":
            num_classes = int(
                torch.randint(
                    self.min_classes,
                    self.max_classes + 1,
                    (),
                    generator=generator,
                ).item()
            )
        return TaskShape(
            task=self.task,
            batch_size=self.batch_size,
            context_rows=choose(self.context_rows),
            query_rows=choose(self.query_rows),
            num_features=choose(self.num_features),
            num_classes=num_classes,
            noise_std=self.noise_std,
        )
