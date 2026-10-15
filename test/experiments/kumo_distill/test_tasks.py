import pytest
import torch
from experiments.kumo_distill.tasks import RandomSCMTaskSource, TaskShape


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_task_shapes_dtypes_and_context_normalization(task: str) -> None:
    shape = TaskShape(
        task=task,
        batch_size=2,
        context_rows=20,
        query_rows=7,
        num_features=6,
        num_classes=5 if task == "classification" else None,
    )
    batch = RandomSCMTaskSource().sample(
        shape, torch.Generator().manual_seed(3)
    )

    assert batch.x.shape == (2, 27, 6)
    assert batch.y_context.shape == (2, 20)
    assert batch.y_query.shape == (2, 7)
    assert batch.categorical_mask.shape == (2, 6)
    assert batch.categorical_mask.dtype == torch.bool
    assert batch.x[:, :20].mean(dim=1) == pytest.approx(
        torch.zeros(2, 6), abs=1e-6
    )
    assert batch.x[:, :20].std(dim=1, unbiased=False) == pytest.approx(
        torch.ones(2, 6), abs=1e-5
    )
    if task == "classification":
        labels = torch.cat([batch.y_context, batch.y_query], dim=1)
        assert labels.dtype == torch.int64
        assert torch.equal(torch.unique(labels), torch.arange(5))
        assert batch.num_classes == 5
    else:
        assert batch.y_context.dtype == torch.float32
        assert batch.y_context.mean(dim=1) == pytest.approx(
            torch.zeros(2), abs=1e-6
        )
        assert batch.y_context.std(dim=1, unbiased=False) == pytest.approx(
            torch.ones(2), abs=1e-6
        )
        assert batch.num_classes is None


def test_task_source_is_deterministic_for_fixed_seed() -> None:
    shape = TaskShape(
        task="classification",
        batch_size=2,
        context_rows=16,
        query_rows=8,
        num_features=4,
        num_classes=3,
    )
    source = RandomSCMTaskSource()
    first = source.sample(shape, torch.Generator().manual_seed(17))
    second = source.sample(shape, torch.Generator().manual_seed(17))

    assert torch.equal(first.x, second.x)
    assert torch.equal(first.y_context, second.y_context)
    assert torch.equal(first.y_query, second.y_query)
