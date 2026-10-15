from pathlib import Path

import pytest
import torch
from experiments.kumo_distill.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from experiments.kumo_distill.config import DistillConfig
from experiments.kumo_distill.tasks import EpisodeBatch
from experiments.kumo_distill.train import (
    forward_losses,
    make_scheduler,
    train_step,
)
from torch import nn

from sdm.models.kumo.tabular.model import _KumoTabular


class TinyEpisodeModel(nn.Module):
    def __init__(self, outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3, outputs)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        categorical_mask: torch.Tensor,
    ) -> torch.Tensor:
        del categorical_mask
        return self.linear(x[:, y.size(1) :])


def classification_batch() -> EpisodeBatch:
    x = torch.tensor(
        [
            [
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [2.0, 1.0, 0.0],
                [-2.0, -1.0, 0.0],
            ]
        ]
    )
    return EpisodeBatch(
        x=x,
        y_context=torch.tensor([[0, 1]]),
        y_query=torch.tensor([[1, 0]]),
        categorical_mask=torch.zeros(1, 3, dtype=torch.bool),
        num_classes=2,
    )


def tiny_kumo(task: str) -> _KumoTabular:
    return _KumoTabular(
        num_classes=10 if task == "classification" else 0,
        num_quantiles=999 if task == "regression" else 0,
        cell_channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=2,
        num_frequencies=2,
        num_readout_tokens=2,
        icl_channels=16,
        num_icl_layers=1,
        num_icl_heads=2,
    )


@pytest.mark.parametrize(
    ("task", "output_channels"),
    [("classification", 2), ("regression", 999)],
)
def test_raw_kumo_teacher_student_outputs_align(
    task: str, output_channels: int
) -> None:
    teacher = tiny_kumo(task).requires_grad_(False)
    student = tiny_kumo(task)
    batch = classification_batch()
    if task == "regression":
        batch = EpisodeBatch(
            x=batch.x,
            y_context=torch.tensor([[0.25, -0.25]]),
            y_query=torch.tensor([[0.5, -0.5]]),
            categorical_mask=batch.categorical_mask,
            num_classes=None,
        )

    losses, student_output, teacher_output = forward_losses(
        teacher,
        student,
        batch,
        task=task,
        alpha=0.5,
        temperature=2.0,
    )

    assert student_output.shape == (1, 2, output_channels)
    assert teacher_output.shape == student_output.shape
    assert torch.isfinite(losses.total)
    assert not teacher_output.requires_grad


def test_train_step_freezes_teacher_and_updates_student() -> None:
    torch.manual_seed(4)
    teacher = TinyEpisodeModel(10).requires_grad_(False)
    student = TinyEpisodeModel(10)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    before = {
        name: value.detach().clone()
        for name, value in student.named_parameters()
    }

    metrics = train_step(
        teacher,
        student,
        classification_batch(),
        optimizer,
        task="classification",
        alpha=0.7,
    )

    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(
        not torch.equal(before[name], value)
        for name, value in student.named_parameters()
    )
    assert metrics["loss"] > 0


def test_loss_decreases_on_repeated_fixed_batch() -> None:
    torch.manual_seed(8)
    teacher = TinyEpisodeModel(10).requires_grad_(False)
    student = TinyEpisodeModel(10)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.2)
    losses = []
    for _ in range(20):
        metrics = train_step(
            teacher,
            student,
            classification_batch(),
            optimizer,
            task="classification",
            alpha=0.0,
        )
        losses.append(metrics["loss"])
    assert losses[-1] < losses[0]


def test_checkpoint_restores_exact_next_step(tmp_path: Path) -> None:
    torch.manual_seed(12)
    teacher = TinyEpisodeModel(10).requires_grad_(False)
    student = TinyEpisodeModel(10)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    scheduler = make_scheduler(optimizer, 5, 1)
    generator = torch.Generator().manual_seed(9)
    config = DistillConfig(
        task="classification", output_dir=str(tmp_path), steps=5
    )
    train_step(
        teacher,
        student,
        classification_batch(),
        optimizer,
        task="classification",
        alpha=0.7,
    )
    scheduler.step()
    checkpoint = tmp_path / "state.pt"
    save_checkpoint(
        checkpoint,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        config=config,
        task_generator=generator,
        metrics={"loss": 1.0},
    )

    expected_random = torch.randn(4, generator=generator)
    train_step(
        teacher,
        student,
        classification_batch(),
        optimizer,
        task="classification",
        alpha=0.7,
    )
    expected = {
        name: value.detach().clone()
        for name, value in student.named_parameters()
    }

    restored = TinyEpisodeModel(10)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = make_scheduler(restored_optimizer, 5, 1)
    restored_generator = torch.Generator()
    step, metrics = load_checkpoint(
        checkpoint,
        student=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        task_generator=restored_generator,
        device=torch.device("cpu"),
    )
    actual_random = torch.randn(4, generator=restored_generator)
    train_step(
        teacher,
        restored,
        classification_batch(),
        restored_optimizer,
        task="classification",
        alpha=0.7,
    )

    assert step == 1
    assert metrics == {"loss": 1.0}
    assert torch.equal(expected_random, actual_random)
    for name, value in restored.named_parameters():
        assert torch.equal(value, expected[name])
