"""Evaluate pretrained and continued Kumo models on fixed synthetic shards."""

import argparse
import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.kumo_distill.models import build_model_pair
from experiments.kumo_distill.tasks import (
    EpisodeBatch,
    RandomSCMTaskSource,
    TaskShape,
)


def evaluate_model(
    model: torch.nn.Module,
    episodes: Sequence[EpisodeBatch],
    teacher_outputs: Sequence[torch.Tensor],
    *,
    task: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Evaluate one raw Kumo network against labels and teacher outputs."""
    totals: dict[str, float] = {}
    examples = 0
    model.eval()
    with torch.no_grad():
        for cpu_batch, cpu_teacher in zip(
            episodes, teacher_outputs, strict=True
        ):
            batch = cpu_batch.to(device)
            with _autocast(device, autocast_dtype):
                output = model(
                    x=batch.x,
                    y=batch.y_context,
                    categorical_mask=batch.categorical_mask,
                ).float()
            teacher = cpu_teacher.to(device).float()
            if task == "classification":
                assert batch.num_classes is not None
                output = output[..., : batch.num_classes]
                teacher = teacher[..., : batch.num_classes]
                probability = output.softmax(dim=-1)
                metrics = {
                    "hard_loss": F.cross_entropy(
                        output.flatten(0, 1), batch.y_query.flatten()
                    ),
                    "nll": F.nll_loss(
                        probability.clamp_min(1e-9).log().flatten(0, 1),
                        batch.y_query.flatten(),
                    ),
                    "accuracy": (output.argmax(dim=-1) == batch.y_query)
                    .float()
                    .mean(),
                    "teacher_student_kl": F.kl_div(
                        output.log_softmax(dim=-1),
                        teacher.softmax(dim=-1),
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean(),
                }
            else:
                levels = torch.linspace(
                    0.001, 0.999, output.size(-1), device=device
                )
                diff = batch.y_query.unsqueeze(-1) - output
                metrics = {
                    "hard_loss": torch.maximum(
                        levels * diff, (levels - 1) * diff
                    ).mean(),
                    "pinball": torch.maximum(
                        levels * diff, (levels - 1) * diff
                    ).mean(),
                    "rmse": (output.mean(dim=-1) - batch.y_query)
                    .square()
                    .mean()
                    .sqrt(),
                    "teacher_student_smooth_l1": F.smooth_l1_loss(
                        output, teacher
                    ),
                }
            count = batch.batch_size * batch.query_rows
            examples += count
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * count
    return {name: value / examples for name, value in totals.items()}


def make_validation_episodes(
    *,
    task: str,
    count: int,
    seed: int,
    batch_size: int,
    context_rows: int,
    query_rows: int,
    num_features: int,
    num_classes: int,
) -> list[EpisodeBatch]:
    """Create a fixed validation shard independent from the training RNG."""
    generator = torch.Generator().manual_seed(seed)
    source = RandomSCMTaskSource()
    shape = TaskShape(
        task=task,
        batch_size=batch_size,
        context_rows=context_rows,
        query_rows=query_rows,
        num_features=num_features,
        num_classes=num_classes if task == "classification" else None,
    )
    return [source.sample(shape, generator) for _ in range(count)]


def teacher_predictions(
    teacher: torch.nn.Module,
    episodes: Sequence[EpisodeBatch],
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """Cache raw teacher outputs for a fixed episode shard."""
    predictions = []
    teacher.eval()
    with torch.no_grad():
        for cpu_batch in episodes:
            batch = cpu_batch.to(device)
            with _autocast(device, autocast_dtype):
                output = teacher(
                    x=batch.x,
                    y=batch.y_context,
                    categorical_mask=batch.categorical_mask,
                )
            predictions.append(output.cpu())
    return predictions


def _autocast(device: torch.device, dtype: torch.dtype | None) -> Any:
    if dtype is None:
        return nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def _load_student_checkpoint(
    model: torch.nn.Module, path: Path, device: torch.device
) -> None:
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["student"])


def main() -> None:
    """Run all configured conditions and print/write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("classification", "regression"), required=True
    )
    parser.add_argument("--hard-label-checkpoint", type=Path)
    parser.add_argument("--distilled-checkpoint", type=Path)
    parser.add_argument("--teacher-size", default="large")
    parser.add_argument("--student-size", default="small")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-rows", type=int, default=128)
    parser.add_argument("--query-rows", type=int, default=64)
    parser.add_argument("--num-features", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = None
    if device.type == "cuda":
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    pair = build_model_pair(
        args.task, args.teacher_size, args.student_size, device
    )
    episodes = make_validation_episodes(
        task=args.task,
        count=args.episodes,
        seed=args.seed,
        batch_size=args.batch_size,
        context_rows=args.context_rows,
        query_rows=args.query_rows,
        num_features=args.num_features,
        num_classes=args.num_classes,
    )
    teacher_outputs = teacher_predictions(
        pair.teacher, episodes, device, autocast_dtype
    )
    results = {
        "large_teacher": evaluate_model(
            pair.teacher,
            episodes,
            teacher_outputs,
            task=args.task,
            device=device,
            autocast_dtype=autocast_dtype,
        ),
        "baseline_small": evaluate_model(
            pair.student,
            episodes,
            teacher_outputs,
            task=args.task,
            device=device,
            autocast_dtype=autocast_dtype,
        ),
    }
    checkpoints = {
        "hard_label_continuation": args.hard_label_checkpoint,
        "distilled_small": args.distilled_checkpoint,
    }
    for name, path in checkpoints.items():
        if path is None:
            continue
        _load_student_checkpoint(pair.student, path, device)
        results[name] = evaluate_model(
            pair.student,
            episodes,
            teacher_outputs,
            task=args.task,
            device=device,
            autocast_dtype=autocast_dtype,
        )

    report = json.dumps(results, indent=2)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
