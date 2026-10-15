"""Pluggable episode sources for Kumo output distillation."""

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

TaskName = Literal["classification", "regression"]


@dataclass(frozen=True)
class TaskShape:
    """Shape and target configuration for one homogeneous task batch."""

    task: TaskName
    batch_size: int
    context_rows: int
    query_rows: int
    num_features: int
    num_classes: int | None = None
    noise_std: float = 0.1

    def __post_init__(self) -> None:
        if (
            min(
                self.batch_size,
                self.context_rows,
                self.query_rows,
                self.num_features,
            )
            <= 0
        ):
            raise ValueError("task shape dimensions must be positive")
        if self.task == "classification":
            if self.num_classes is None or not 2 <= self.num_classes <= 10:
                raise ValueError("classification requires 2 to 10 classes")
            if self.context_rows + self.query_rows < self.num_classes:
                raise ValueError(
                    "classification requires at least one row per class"
                )
        elif self.num_classes is not None:
            raise ValueError("regression does not use num_classes")


@dataclass(frozen=True)
class EpisodeBatch:
    """Processed context/query episode consumed by raw Kumo networks."""

    x: Tensor
    y_context: Tensor
    y_query: Tensor
    categorical_mask: Tensor
    num_classes: int | None

    @property
    def batch_size(self) -> int:
        """Return the number of tasks in this episode batch."""
        return self.x.size(0)

    @property
    def query_rows(self) -> int:
        """Return the number of query examples per task."""
        return self.y_query.size(1)

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        """Move all episode tensors to a device."""
        return EpisodeBatch(
            x=self.x.to(device),
            y_context=self.y_context.to(device),
            y_query=self.y_query.to(device),
            categorical_mask=self.categorical_mask.to(device),
            num_classes=self.num_classes,
        )


class TaskSource(Protocol):
    """Source of already-preprocessed Kumo episodes."""

    def sample(
        self, config: TaskShape, generator: torch.Generator
    ) -> EpisodeBatch:
        """Sample one homogeneous batch using only the supplied RNG."""
        ...


class RandomSCMTaskSource:
    """Small random structural model for mechanical distillation tests."""

    def sample(
        self, config: TaskShape, generator: torch.Generator
    ) -> EpisodeBatch:
        """Generate and context-normalize a deterministic synthetic episode."""
        rows = config.context_rows + config.query_rows
        roots = torch.randn(
            config.batch_size,
            rows,
            config.num_features,
            generator=generator,
        )
        transformed = self._transform(roots, generator)
        weights = torch.randn(
            config.batch_size,
            config.num_features,
            generator=generator,
        )
        target = (transformed * weights.unsqueeze(1)).sum(dim=-1)
        target = target + config.noise_std * torch.randn(
            target.shape, generator=generator
        )

        x = torch.empty_like(transformed)
        ordered_target = torch.empty_like(target)
        for batch_index in range(config.batch_size):
            row_order = torch.randperm(rows, generator=generator)
            column_order = torch.randperm(
                config.num_features, generator=generator
            )
            x[batch_index] = transformed[batch_index, row_order][
                :, column_order
            ]
            ordered_target[batch_index] = target[batch_index, row_order]

        context_x = x[:, : config.context_rows]
        x_mean = context_x.mean(dim=1, keepdim=True)
        x_std = context_x.std(dim=1, keepdim=True, unbiased=False).clamp_min(
            1e-6
        )
        x = (x - x_mean) / x_std

        if config.task == "classification":
            assert config.num_classes is not None
            labels = self._quantile_bin(ordered_target, config.num_classes)
            y_context = labels[:, : config.context_rows]
            y_query = labels[:, config.context_rows :]
        else:
            context_target = ordered_target[:, : config.context_rows]
            y_mean = context_target.mean(dim=1, keepdim=True)
            y_std = context_target.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            standardized = (ordered_target - y_mean) / y_std
            y_context = standardized[:, : config.context_rows]
            y_query = standardized[:, config.context_rows :]

        return EpisodeBatch(
            x=x,
            y_context=y_context,
            y_query=y_query,
            categorical_mask=torch.zeros(
                config.batch_size, config.num_features, dtype=torch.bool
            ),
            num_classes=config.num_classes,
        )

    @staticmethod
    def _transform(roots: Tensor, generator: torch.Generator) -> Tensor:
        batch_size, _, num_features = roots.shape
        operation = torch.randint(
            0, 5, (batch_size, num_features), generator=generator
        )
        scale = torch.randn(batch_size, 1, num_features, generator=generator)
        offset = torch.randn(batch_size, 1, num_features, generator=generator)
        linear = scale * roots + offset
        polynomial = roots + 0.5 * scale * roots.square()
        threshold = (roots > offset).to(roots.dtype)
        periodic = torch.sin((scale.abs() + 0.5) * roots + offset)
        mixing = (
            torch.randn(
                batch_size, num_features, num_features, generator=generator
            )
            / num_features**0.5
        )
        mlp_like = torch.tanh(torch.bmm(roots, mixing) + offset)
        candidates = torch.stack(
            [linear, polynomial, threshold, periodic, mlp_like], dim=0
        )
        selector = operation.transpose(0, 1).unsqueeze(1).unsqueeze(0)
        selector = selector.transpose(1, 3).expand(
            1, batch_size, roots.size(1), -1
        )
        return candidates.gather(0, selector).squeeze(0)

    @staticmethod
    def _quantile_bin(target: Tensor, num_classes: int) -> Tensor:
        order = target.argsort(dim=1)
        ranks = torch.empty_like(order)
        values = torch.arange(target.size(1)).expand_as(order)
        ranks.scatter_(1, order, values)
        return torch.div(
            ranks * num_classes, target.size(1), rounding_mode="floor"
        ).clamp_max(num_classes - 1)
