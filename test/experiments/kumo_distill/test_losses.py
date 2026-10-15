import pytest
import torch
from experiments.kumo_distill.losses import (
    classification_distillation_loss,
    regression_distillation_loss,
)


def test_classification_kd_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 3, 5)
    result = classification_distillation_loss(
        logits,
        logits,
        torch.randint(0, 5, (2, 3)),
    )
    assert result.soft.item() == pytest.approx(0.0, abs=1e-6)


def test_classification_kd_is_normalized_per_query_example() -> None:
    student = torch.tensor([[[2.0, -1.0, 0.5]]])
    teacher = torch.tensor([[[-0.5, 1.5, 0.0]]])
    target = torch.tensor([[1]])
    single = classification_distillation_loss(student, teacher, target)
    repeated = classification_distillation_loss(
        student.repeat(1, 7, 1),
        teacher.repeat(1, 7, 1),
        target.repeat(1, 7),
    )

    assert repeated.hard.item() == pytest.approx(single.hard.item())
    assert repeated.soft.item() == pytest.approx(single.soft.item())


def test_regression_kd_is_zero_for_identical_quantiles() -> None:
    quantiles = torch.randn(2, 3, 999)
    result = regression_distillation_loss(
        quantiles,
        quantiles,
        torch.randn(2, 3),
    )
    assert result.soft.item() == 0.0


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_losses_are_finite_and_only_student_receives_gradients(
    task: str,
) -> None:
    if task == "classification":
        student = torch.tensor(
            [[[1e4, -1e4], [-1e4, 1e4]]], requires_grad=True
        )
        teacher = student.detach().clone().requires_grad_()
        result = classification_distillation_loss(
            student, teacher, torch.tensor([[0, 1]])
        )
    else:
        student = torch.full((1, 2, 999), 1e4, requires_grad=True)
        teacher = torch.full((1, 2, 999), -1e4, requires_grad=True)
        result = regression_distillation_loss(
            student, teacher, torch.tensor([[1e4, -1e4]])
        )

    assert torch.isfinite(result.total)
    result.total.backward()
    assert student.grad is not None
    assert teacher.grad is None
